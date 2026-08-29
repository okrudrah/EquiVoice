"""Verify a cloud CUDA host before any EquiVoice fine-tuning begins."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch

from equivoice.train_whisper_loso import load_fold_data
from equivoice.validate_l2_arctic import sha256_file


FOLD_CODES = ("aba", "ska", "ybaa", "zhaa")
ARTIFACT_PATHS = {
    "cuda_environment": Path("configs/cuda_environment.json"),
    "cuda_requirements": Path("requirements-cuda.txt"),
    "training_config": Path("configs/whisper_small_en_loso.json"),
    "fold_summary": Path("results/manifests/l2_arctic_v5_loso/folds_summary.json"),
    "processed_manifest": Path(
        "results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv"
    ),
    "baseline_metrics": Path(
        "results/baseline/whisper_small_en/"
        "l2_arctic_v5_arabic_utterance_metrics.csv"
    ),
    **{
        f"fold_{code}": Path(
            f"results/manifests/l2_arctic_v5_loso/fold_{code}.csv"
        )
        for code in FOLD_CODES
    },
}


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported CUDA-environment contract schema")
    if contract.get("operating_system") != "Linux":
        raise ValueError("CUDA training contract must target Linux")
    if not contract.get("required_packages"):
        raise ValueError("CUDA training contract has no package pins")
    return contract


def check_python_and_packages(contract: dict[str, Any]) -> dict[str, Any]:
    actual_system = platform.system()
    if actual_system != contract["operating_system"]:
        raise ValueError(
            f"{contract['operating_system']} is required; found {actual_system}"
        )
    actual_python = platform.python_version()
    expected_python = contract["python_major_minor"]
    if ".".join(actual_python.split(".")[:2]) != expected_python:
        raise ValueError(
            f"Python {expected_python} is required; found {actual_python}"
        )
    actual_torch = torch.__version__.split("+")[0]
    if actual_torch != contract["torch_version"]:
        raise ValueError(
            f"torch {contract['torch_version']} is required; found {torch.__version__}"
        )

    versions: dict[str, str] = {"torch": torch.__version__}
    mismatches: list[str] = []
    for package, expected in contract["required_packages"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package} is missing")
            continue
        versions[package] = actual
        if actual != expected:
            mismatches.append(f"{package} expected {expected}, found {actual}")
    if mismatches:
        raise ValueError("package contract failed: " + "; ".join(mismatches))
    return versions


def inspect_cuda(contract: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to PyTorch")
    if torch.version.cuda != contract["pytorch_cuda_runtime"]:
        raise ValueError(
            "PyTorch CUDA runtime mismatch: "
            f"expected {contract['pytorch_cuda_runtime']}, found {torch.version.cuda}"
        )
    if torch.cuda.device_count() != 1:
        raise ValueError(
            f"exactly one CUDA GPU is required; found {torch.cuda.device_count()}"
        )

    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / (1024**3)
    minimum = float(contract["minimum_gpu_memory_gib"])
    recommended = float(contract["recommended_gpu_memory_gib"])
    if memory_gib < minimum:
        raise ValueError(
            f"GPU has {memory_gib:.2f} GiB; at least {minimum:.0f} GiB is required"
        )
    warnings: list[str] = []
    if memory_gib < recommended:
        warnings.append(
            f"GPU has {memory_gib:.2f} GiB; {recommended:.0f} GiB is recommended"
        )

    left = torch.ones((64, 64), device="cuda", dtype=torch.float16)
    product = left @ left
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product).all().item()):
        raise ValueError("CUDA float16 computation produced a non-finite value")
    del left, product
    return (
        {
            "available": True,
            "device_count": 1,
            "device_name": properties.name,
            "memory_gib": round(memory_gib, 3),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "pytorch_cuda_runtime": torch.version.cuda,
        },
        warnings,
    )


def artifact_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, relative_path in ARTIFACT_PATHS.items():
        path = project_root / relative_path
        if not path.is_file():
            raise ValueError(f"required project artifact is missing: {relative_path}")
        hashes[str(relative_path)] = sha256_file(path)
    return hashes


def check_project_data(project_root: Path) -> dict[str, Any]:
    fold_dir = project_root / "results/manifests/l2_arctic_v5_loso"
    fold_summary = fold_dir / "folds_summary.json"
    processed_manifest = (
        project_root
        / "results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv"
    )
    baseline_metrics = (
        project_root
        / "results/baseline/whisper_small_en/"
        "l2_arctic_v5_arabic_utterance_metrics.csv"
    )
    processed_root = (
        project_root
        / "data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq"
    )
    raw_root = project_root / "data/raw/l2_arctic/v5.0"

    folds: dict[str, Any] = {}
    for code in FOLD_CODES:
        fold = load_fold_data(
            fold_dir / f"fold_{code}.csv",
            fold_summary,
            processed_manifest,
            baseline_metrics,
        )
        folds[fold.fold_id] = {
            "held_out_speaker": fold.held_out_speaker,
            "train": len(fold.train),
            "validation": len(fold.validation),
            "test_sealed": fold.test_count,
        }

    missing_processed = 0
    missing_transcripts = 0
    row_count = 0
    with processed_manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            if not (processed_root / row["processed_wav_path"]).is_file():
                missing_processed += 1
            if not (raw_root / row["transcript_path"]).is_file():
                missing_transcripts += 1
    if row_count != 4_365:
        raise ValueError(f"processed manifest has {row_count} rows; expected 4365")
    if missing_processed or missing_transcripts:
        raise ValueError(
            "cloud data transfer is incomplete: "
            f"{missing_processed} processed WAVs and "
            f"{missing_transcripts} transcripts are missing"
        )
    return {
        "records": row_count,
        "processed_wavs_present": row_count,
        "transcripts_present": row_count,
        "folds": folds,
    }


def build_report(
    project_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    versions = check_python_and_packages(contract)
    cuda, warnings = inspect_cuda(contract)
    hashes = artifact_hashes(project_root)
    data = check_project_data(project_root)
    return {
        "schema_version": 1,
        "status": "passed",
        "purpose": contract["purpose"],
        "project_artifact_hashes": hashes,
        "environment_contract_sha256": sha256_file(contract_path),
        "python": platform.python_version(),
        "packages": versions,
        "cuda": cuda,
        "data": data,
        "warnings": warnings,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/cuda_environment.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/experiments/cuda_preflight.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args.project_root.resolve(), args.contract)
    write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
