from __future__ import annotations

import json
from pathlib import Path

import pytest

import equivoice.check_cuda_environment as preflight
from equivoice.check_cuda_environment import (
    artifact_hashes,
    check_project_data,
    check_python_and_packages,
    inspect_cuda,
    load_contract,
)
from equivoice.orchestrate_loso_training import (
    build_commands,
    selected_folds,
    validate_output_targets,
    verify_preflight_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_contract_matches_local_pinned_package_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract(ROOT / "configs/cuda_environment.json")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    versions = check_python_and_packages(contract)

    assert versions["torch"].split("+")[0] == "2.13.0"
    assert versions["transformers"] == "4.57.6"


def test_cuda_requirements_match_contract() -> None:
    contract = load_contract(ROOT / "configs/cuda_environment.json")
    pins = {}
    for raw_line in (ROOT / "requirements-cuda.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package, version = line.split("==", maxsplit=1)
        pins[package] = version

    assert pins == contract["required_packages"]


def test_cuda_contract_rejects_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract(ROOT / "configs/cuda_environment.json")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")

    with pytest.raises(ValueError, match="Linux is required"):
        check_python_and_packages(contract)


def test_cuda_check_rejects_unavailable_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract(ROOT / "configs/cuda_environment.json")
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        inspect_cuda(contract)


def test_project_data_handoff_contains_all_fold_inputs() -> None:
    report = check_project_data(ROOT)

    assert report["records"] == 4_365
    assert report["processed_wavs_present"] == 4_365
    assert report["transcripts_present"] == 4_365
    assert set(report["folds"]) == {
        "held_out_aba",
        "held_out_ska",
        "held_out_ybaa",
        "held_out_zhaa",
    }


def test_launcher_builds_one_correct_command_per_fold(tmp_path: Path) -> None:
    commands = build_commands(
        selected_folds("all"),
        tmp_path / "outputs",
        "/environment/bin/python",
    )

    assert len(commands) == 4
    for code, command in zip(("aba", "ska", "ybaa", "zhaa"), commands):
        assert f"results/manifests/l2_arctic_v5_loso/fold_{code}.csv" in command
        assert str(tmp_path / "outputs" / f"held_out_{code}") in command
        assert command[-2:] == ["--device", "cuda"]


def test_launcher_refuses_nonempty_output_without_resume(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output = output_root / "held_out_aba"
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        validate_output_targets(("aba",), output_root, None)


def test_preflight_report_is_bound_to_current_artifacts(tmp_path: Path) -> None:
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "cuda": {"available": True},
                "project_artifact_hashes": artifact_hashes(ROOT),
            }
        ),
        encoding="utf-8",
    )

    report = verify_preflight_report(report_path, ROOT)

    assert report["status"] == "passed"


def test_resume_is_restricted_to_selected_fold_output(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    wrong_checkpoint = tmp_path / "checkpoint-100"
    wrong_checkpoint.mkdir()

    with pytest.raises(ValueError, match="directly inside"):
        validate_output_targets(("aba",), output_root, wrong_checkpoint)
