from __future__ import annotations

import struct
import wave
from pathlib import Path

from equivoice.validate_l2_arctic import SpeakerExpectation, validate_speaker


def _write_wav(path: Path, frames: int = 4_410, sample_rate: int = 44_100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = struct.pack(f"<{frames}h", *([0] * frames))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples)


def _write_textgrid(path: Path, duration: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n\n'
        "xmin = 0\n"
        f"xmax = {duration}\n",
        encoding="utf-8",
    )


def _complete_fixture(root: Path) -> Path:
    speaker = root / "speakers" / "TEST"
    _write_wav(speaker / "wav" / "arctic_a0001.wav")
    transcript = speaker / "transcript" / "arctic_a0001.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("A test sentence.", encoding="utf-8")
    _write_textgrid(speaker / "textgrid" / "arctic_a0001.TextGrid")
    _write_textgrid(speaker / "annotation" / "arctic_a0001.TextGrid")
    return speaker


def test_validate_speaker_accepts_complete_fixture(tmp_path: Path) -> None:
    speaker = _complete_fixture(tmp_path)

    summary, rows = validate_speaker(
        speaker,
        "TEST",
        SpeakerExpectation(scripted_wavs=1, annotations=1),
        tmp_path,
        scan_samples=True,
    )

    assert summary["counts"] == {
        "wav": 1,
        "transcript": 1,
        "textgrid": 1,
        "annotation": 1,
    }
    assert summary["errors"] == []
    assert len(rows) == 1
    assert rows[0]["duration_seconds"] == 0.1
    assert rows[0]["full_scale_sample_count"] == 0


def test_validate_speaker_reports_missing_transcript(tmp_path: Path) -> None:
    speaker = _complete_fixture(tmp_path)
    (speaker / "transcript" / "arctic_a0001.txt").unlink()

    summary, _ = validate_speaker(
        speaker,
        "TEST",
        SpeakerExpectation(scripted_wavs=1, annotations=1),
        tmp_path,
        scan_samples=False,
    )

    assert "transcript count is 0; expected 1" in summary["errors"]
    assert "transcripts missing for WAVs: arctic_a0001" in summary["errors"]


def test_validate_speaker_reports_textgrid_duration_mismatch(tmp_path: Path) -> None:
    speaker = _complete_fixture(tmp_path)
    _write_textgrid(speaker / "textgrid" / "arctic_a0001.TextGrid", duration=1.0)

    summary, _ = validate_speaker(
        speaker,
        "TEST",
        SpeakerExpectation(scripted_wavs=1, annotations=1),
        tmp_path,
        scan_samples=False,
    )

    assert any("duration mismatch" in error for error in summary["errors"])
