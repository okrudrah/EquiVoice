from __future__ import annotations

import json
from pathlib import Path

import pytest

from equivoice.run_librispeech_control import require_prepared_control
from equivoice.validate_l2_arctic import sha256_file


def test_require_prepared_control_verifies_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("header\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "complete test-clean",
                "dataset": "LibriSpeech",
                "subset": "test-clean",
                "manifest_sha256": sha256_file(manifest),
                "speakers": [str(index) for index in range(40)],
            }
        ),
        encoding="utf-8",
    )

    assert len(require_prepared_control(report, manifest)) == 40


def test_require_prepared_control_rejects_changed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("changed\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "complete test-clean",
                "dataset": "LibriSpeech",
                "subset": "test-clean",
                "manifest_sha256": "0" * 64,
                "speakers": [str(index) for index in range(40)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        require_prepared_control(report, manifest)
