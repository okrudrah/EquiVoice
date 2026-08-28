from __future__ import annotations

import csv
from pathlib import Path

import pytest

from equivoice.run_whisper_baseline import (
    build_public_results,
    load_processed_manifest,
    score_pair,
    select_device,
)


def test_score_pair_counts_word_errors() -> None:
    metrics = score_pair(
        "The quick brown fox",
        "the quick foxes",
        lambda text: text.lower(),
    )

    assert metrics["reference_words"] == 4
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 0
    assert metrics["errors"] == 2
    assert metrics["wer"] == 0.5


def test_score_pair_rejects_empty_normalized_reference() -> None:
    with pytest.raises(ValueError, match="reference became empty"):
        score_pair("...", "anything", lambda _: "")


def test_select_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        select_device("quantum")


def test_load_processed_manifest_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "processed.csv"
    fields = [
        "speaker",
        "utterance_id",
        "transcript_path",
        "processed_wav_path",
        "processed_sample_rate_hz",
        "processed_channels",
        "processed_format",
        "processed_subtype",
        "processed_pcm_sha256",
    ]
    row = {
        "speaker": "ABA",
        "utterance_id": "arctic_a0001",
        "transcript_path": "speakers/ABA/transcript/arctic_a0001.txt",
        "processed_wav_path": "speakers/ABA/wav/arctic_a0001.wav",
        "processed_sample_rate_hz": "16000",
        "processed_channels": "1",
        "processed_format": "WAV",
        "processed_subtype": "FLOAT",
        "processed_pcm_sha256": "a" * 64,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="duplicate processed row"):
        load_processed_manifest(path)


def test_public_results_compute_micro_and_macro_wer(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("private\n", encoding="utf-8")
    rows = [
        {
            "speaker": "ABA",
            "utterance_id": "one",
            "processed_pcm_sha256": "a",
            "reference_sha256": "b",
            "hypothesis_sha256": "c",
            "reference_words": 10,
            "hypothesis_words": 9,
            "hits": 8,
            "substitutions": 1,
            "deletions": 1,
            "insertions": 0,
            "errors": 2,
            "wer": 0.2,
        },
        {
            "speaker": "SKA",
            "utterance_id": "two",
            "processed_pcm_sha256": "d",
            "reference_sha256": "e",
            "hypothesis_sha256": "f",
            "reference_words": 5,
            "hypothesis_words": 6,
            "hits": 4,
            "substitutions": 1,
            "deletions": 0,
            "insertions": 1,
            "errors": 2,
            "wer": 0.4,
        },
    ]

    report, public_rows = build_public_results(
        rows, {"model_id": "test"}, predictions, limited=True
    )

    assert len(public_rows) == 2
    assert report["aggregate"]["reference_words"] == 15
    assert report["aggregate"]["errors"] == 4
    assert report["aggregate"]["wer"] == pytest.approx(4 / 15)
    assert report["aggregate"]["macro_speaker_wer"] == pytest.approx(0.3)
