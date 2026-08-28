from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from equivoice.preprocess_l2_arctic import preprocess_record


def _fixture(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    source_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    wav_path = source_root / "speakers" / "ABA" / "wav" / "arctic_a0001.wav"
    wav_path.parent.mkdir(parents=True)
    frames = 44_100
    time = np.arange(frames, dtype=np.float64) / 44_100
    samples = np.rint(8_000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(struct.pack(f"<{frames}h", *samples))

    digest = hashlib.sha256(samples.astype("<i2").reshape(-1, 1).tobytes()).hexdigest()
    record = {
        "speaker": "ABA",
        "utterance_id": "arctic_a0001",
        "wav_path": "speakers/ABA/wav/arctic_a0001.wav",
        "transcript_path": "speakers/ABA/transcript/arctic_a0001.txt",
        "textgrid_path": "speakers/ABA/textgrid/arctic_a0001.TextGrid",
        "annotation_path": "",
        "frames": str(frames),
        "sample_rate_hz": "44100",
        "channels": "1",
        "format": "WAV",
        "subtype": "PCM_16",
        "duration_seconds": "1.0",
        "full_scale_sample_count": "0",
        "audio_pcm_sha256": digest,
    }
    return record, source_root, output_root


def test_preprocess_record_creates_verified_16khz_float_wav(tmp_path: Path) -> None:
    record, source_root, output_root = _fixture(tmp_path)

    row = preprocess_record(record, source_root, output_root)

    output = output_root / row["processed_wav_path"]
    info = sf.info(output)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.format == "WAV"
    assert info.subtype == "FLOAT"
    assert row["processed_frames"] == 16_000
    assert row["duration_delta_seconds"] == 0.0
    assert row["processed_out_of_unit_range_sample_count"] == 0


def test_preprocess_record_is_idempotent(tmp_path: Path) -> None:
    record, source_root, output_root = _fixture(tmp_path)

    first = preprocess_record(record, source_root, output_root)
    second = preprocess_record(record, source_root, output_root)

    assert first == second


def test_preprocess_record_refuses_changed_output(tmp_path: Path) -> None:
    record, source_root, output_root = _fixture(tmp_path)
    row = preprocess_record(record, source_root, output_root)
    output = output_root / row["processed_wav_path"]
    sf.write(output, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        preprocess_record(record, source_root, output_root)


def test_preprocess_record_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    record, source_root, output_root = _fixture(tmp_path)
    record["audio_pcm_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source decoded-audio hash mismatch"):
        preprocess_record(record, source_root, output_root)
