from __future__ import annotations

from pathlib import Path

import pytest

from equivoice.build_loso_manifests import (
    SPEAKERS,
    build_fold_rows,
    is_validation_utterance,
    write_immutable,
)


def _records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for speaker in SPEAKERS:
        for index in range(200):
            utterance_id = f"arctic_a{index:04d}"
            records.append(
                {
                    "speaker": speaker,
                    "utterance_id": utterance_id,
                    "wav_path": f"speakers/{speaker}/wav/{utterance_id}.wav",
                    "transcript_path": (
                        f"speakers/{speaker}/transcript/{utterance_id}.txt"
                    ),
                    "textgrid_path": (
                        f"speakers/{speaker}/textgrid/{utterance_id}.TextGrid"
                    ),
                    "annotation_path": "",
                    "duration_seconds": "1.0",
                    "audio_pcm_sha256": f"{speaker}-{utterance_id}",
                }
            )
    return records


def test_validation_assignment_is_deterministic() -> None:
    first = is_validation_utterance("arctic_a0042")
    second = is_validation_utterance("arctic_a0042")

    assert first == second


def test_fold_is_complete_and_speaker_independent() -> None:
    records = _records()

    rows = build_fold_rows(records, "ABA")

    assert len(rows) == len(records)
    assert {row["speaker"] for row in rows if row["split"] == "test"} == {"ABA"}
    assert not any(
        row["speaker"] == "ABA" and row["split"] != "test" for row in rows
    )
    train_ids = {row["utterance_id"] for row in rows if row["split"] == "train"}
    validation_ids = {
        row["utterance_id"] for row in rows if row["split"] == "validation"
    }
    assert train_ids
    assert validation_ids
    assert train_ids.isdisjoint(validation_ids)
    assert {row["speaker"] for row in rows if row["split"] == "train"} == {
        "SKA",
        "YBAA",
        "ZHAA",
    }
    assert {row["speaker"] for row in rows if row["split"] == "validation"} == {
        "SKA",
        "YBAA",
        "ZHAA",
    }


def test_fold_generation_is_deterministic() -> None:
    records = _records()

    assert build_fold_rows(records, "SKA") == build_fold_rows(records, "SKA")


def test_immutable_writer_refuses_changed_content(tmp_path: Path) -> None:
    target = tmp_path / "fold.csv"
    write_immutable(target, b"original\n")
    write_immutable(target, b"original\n")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_immutable(target, b"changed\n")
