from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from equivoice.prepare_librispeech_control import (
    discover_records,
    parse_test_clean_speakers,
    prepare_record,
)


def test_parse_test_clean_speakers_filters_other_subsets(tmp_path: Path) -> None:
    metadata = tmp_path / "SPEAKERS.TXT"
    metadata.write_text(
        "; comment\n61 | M | test-clean | 8.08 | Reader\n"
        "84 | F | dev-clean | 8.02 | Another reader\n",
        encoding="utf-8",
    )

    assert parse_test_clean_speakers(metadata) == {"61": "M"}


def test_discover_records_requires_audio_for_every_transcript(tmp_path: Path) -> None:
    chapter = tmp_path / "61" / "70968"
    chapter.mkdir(parents=True)
    (chapter / "61-70968.trans.txt").write_text(
        "61-70968-0000 SOME WORDS\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing FLAC"):
        discover_records(tmp_path)


def test_prepare_record_creates_identical_float32_audio(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = source_root / "61" / "70968" / "61-70968-0000.flac"
    source.parent.mkdir(parents=True)
    pcm = np.array([-32768, -100, 0, 100, 32767], dtype=np.int16)
    sf.write(source, pcm, 16_000, format="FLAC", subtype="PCM_16")
    record = {
        "speaker": "61",
        "chapter": "70968",
        "utterance_id": "61-70968-0000",
        "reference_text": "SOME WORDS",
        "source_flac_path": "61/70968/61-70968-0000.flac",
    }

    row = prepare_record(record, source_root, output_root)
    output, rate = sf.read(
        output_root / row["processed_wav_path"],
        dtype="float32",
        always_2d=False,
    )

    assert rate == 16_000
    assert np.array_equal(output, pcm.astype(np.float32) / 32_768.0)
    assert (output_root / row["transcript_path"]).read_text(encoding="utf-8") == (
        "SOME WORDS\n"
    )
    assert row["processed_frames"] == len(pcm)
    assert row["source_duration_seconds"] == row["processed_duration_seconds"]


def test_prepare_record_refuses_changed_output(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = source_root / "61" / "70968" / "61-70968-0000.flac"
    source.parent.mkdir(parents=True)
    sf.write(source, np.arange(20, dtype=np.int16), 16_000, format="FLAC")
    record = {
        "speaker": "61",
        "chapter": "70968",
        "utterance_id": "61-70968-0000",
        "reference_text": "SOME WORDS",
        "source_flac_path": "61/70968/61-70968-0000.flac",
    }
    row = prepare_record(record, source_root, output_root)
    sf.write(
        output_root / row["processed_wav_path"],
        np.zeros(20, dtype=np.float32),
        16_000,
        subtype="FLOAT",
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare_record(record, source_root, output_root)
