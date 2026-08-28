"""Validate and prepare the complete LibriSpeech test-clean control split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from equivoice.validate_l2_arctic import sha256_file


DATASET = "LibriSpeech"
RELEASE = "SLR12"
SUBSET = "test-clean"
ARCHIVE_NAME = "test-clean.tar.gz"
ARCHIVE_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
EXPECTED_UTTERANCES = 2_620
EXPECTED_SPEAKERS = 40
EXPECTED_CHAPTERS = 87
TARGET_SAMPLE_RATE = 16_000
PREPARATION_VERSION = "librispeech-test-clean-16khz-mono-float32-v1"

MANIFEST_FIELDS = [
    "speaker",
    "chapter",
    "utterance_id",
    "source_flac_path",
    "source_file_sha256",
    "source_pcm_sha256",
    "source_frames",
    "source_sample_rate_hz",
    "source_channels",
    "source_format",
    "source_subtype",
    "source_duration_seconds",
    "transcript_path",
    "transcript_sha256",
    "processed_wav_path",
    "processed_frames",
    "processed_sample_rate_hz",
    "processed_channels",
    "processed_format",
    "processed_subtype",
    "processed_duration_seconds",
    "processed_peak_abs",
    "processed_pcm_sha256",
    "processed_file_sha256",
]


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int16_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<i2", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _float32_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def parse_test_clean_speakers(path: Path) -> dict[str, str]:
    speakers: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 4:
            raise ValueError(f"malformed speaker metadata at line {line_number}")
        speaker, sex, subset = fields[:3]
        if subset != SUBSET:
            continue
        if speaker in speakers:
            raise ValueError(f"duplicate speaker metadata: {speaker}")
        if sex not in {"F", "M"}:
            raise ValueError(f"unexpected speaker sex metadata: {speaker}")
        speakers[speaker] = sex
    return speakers


def discover_records(source_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    transcript_files = sorted(source_root.glob("*/*/*.trans.txt"))
    for transcript_file in transcript_files:
        speaker = transcript_file.parent.parent.name
        chapter = transcript_file.parent.name
        expected_transcript_name = f"{speaker}-{chapter}.trans.txt"
        if transcript_file.name != expected_transcript_name:
            raise ValueError(f"unexpected transcript filename: {transcript_file}")
        for line_number, raw_line in enumerate(
            transcript_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                utterance_id, reference = raw_line.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    f"malformed transcript line {transcript_file}:{line_number}"
                ) from error
            parts = utterance_id.split("-")
            if len(parts) != 3 or parts[0] != speaker or parts[1] != chapter:
                raise ValueError(f"utterance ID/path mismatch: {utterance_id}")
            if utterance_id in seen:
                raise ValueError(f"duplicate utterance ID: {utterance_id}")
            seen.add(utterance_id)
            audio_path = transcript_file.parent / f"{utterance_id}.flac"
            if not audio_path.is_file():
                raise ValueError(f"missing FLAC for transcript: {utterance_id}")
            records.append(
                {
                    "speaker": speaker,
                    "chapter": chapter,
                    "utterance_id": utterance_id,
                    "reference_text": reference,
                    "source_flac_path": audio_path.relative_to(source_root).as_posix(),
                }
            )

    discovered_audio = {
        path.relative_to(source_root).as_posix()
        for path in source_root.glob("*/*/*.flac")
    }
    referenced_audio = {record["source_flac_path"] for record in records}
    extras = discovered_audio - referenced_audio
    if extras:
        raise ValueError(f"FLAC files without transcripts: {len(extras)}")
    if not records:
        raise ValueError("no LibriSpeech records discovered")
    return sorted(records, key=lambda row: row["utterance_id"])


def _write_text_new_or_verify(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing to overwrite changed transcript: {path}")
        return
    path.write_bytes(encoded)


def _write_audio_new_or_verify(path: Path, expected: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        info = sf.info(path)
        existing, rate = sf.read(path, dtype="float32", always_2d=False)
        if (
            rate != TARGET_SAMPLE_RATE
            or info.channels != 1
            or info.format != "WAV"
            or info.subtype != "FLOAT"
            or not np.array_equal(existing, expected)
        ):
            raise ValueError(f"refusing to overwrite changed processed WAV: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    sf.write(
        temporary,
        expected,
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="FLOAT",
    )
    temporary.replace(path)


def prepare_record(
    record: dict[str, str], source_root: Path, output_root: Path
) -> dict[str, Any]:
    source_path = source_root / record["source_flac_path"]
    info = sf.info(source_path)
    if (
        info.samplerate != TARGET_SAMPLE_RATE
        or info.channels != 1
        or info.format != "FLAC"
        or info.subtype != "PCM_16"
        or info.frames <= 0
    ):
        raise ValueError(f"unexpected LibriSpeech audio properties: {source_path}")

    pcm, rate = sf.read(source_path, dtype="int16", always_2d=False)
    if rate != TARGET_SAMPLE_RATE or pcm.ndim != 1 or pcm.size != info.frames:
        raise ValueError(f"invalid decoded LibriSpeech audio: {source_path}")
    processed = pcm.astype(np.float32) / 32_768.0
    if not np.all(np.isfinite(processed)):
        raise ValueError(f"nonfinite decoded audio: {source_path}")

    speaker = record["speaker"]
    chapter = record["chapter"]
    utterance_id = record["utterance_id"]
    transcript_relative = Path("transcripts") / speaker / chapter / f"{utterance_id}.txt"
    wav_relative = Path("wav") / speaker / chapter / f"{utterance_id}.wav"
    transcript_content = record["reference_text"] + "\n"
    _write_text_new_or_verify(output_root / transcript_relative, transcript_content)
    _write_audio_new_or_verify(output_root / wav_relative, processed)

    output_path = output_root / wav_relative
    output_info = sf.info(output_path)
    readback, output_rate = sf.read(output_path, dtype="float32", always_2d=False)
    if (
        output_rate != TARGET_SAMPLE_RATE
        or output_info.frames != info.frames
        or output_info.channels != 1
        or output_info.format != "WAV"
        or output_info.subtype != "FLOAT"
        or not np.array_equal(readback, processed)
    ):
        raise ValueError(f"processed LibriSpeech verification failed: {output_path}")

    duration = info.frames / info.samplerate
    return {
        "speaker": speaker,
        "chapter": chapter,
        "utterance_id": utterance_id,
        "source_flac_path": record["source_flac_path"],
        "source_file_sha256": sha256_file(source_path),
        "source_pcm_sha256": _int16_sha256(pcm),
        "source_frames": info.frames,
        "source_sample_rate_hz": info.samplerate,
        "source_channels": info.channels,
        "source_format": info.format,
        "source_subtype": info.subtype,
        "source_duration_seconds": round(duration, 9),
        "transcript_path": transcript_relative.as_posix(),
        "transcript_sha256": hashlib.sha256(
            transcript_content.encode("utf-8")
        ).hexdigest(),
        "processed_wav_path": wav_relative.as_posix(),
        "processed_frames": output_info.frames,
        "processed_sample_rate_hz": output_info.samplerate,
        "processed_channels": output_info.channels,
        "processed_format": output_info.format,
        "processed_subtype": output_info.subtype,
        "processed_duration_seconds": round(
            output_info.frames / output_info.samplerate, 9
        ),
        "processed_peak_abs": round(float(np.max(np.abs(readback))), 9),
        "processed_pcm_sha256": _float32_sha256(readback),
        "processed_file_sha256": sha256_file(output_path),
    }


def _csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite changed artifact: {path}")
        return
    path.write_bytes(content)


def prepare_control(
    source_root: Path,
    speakers_metadata: Path,
    archive: Path,
    output_root: Path,
    manifest_path: Path,
    report_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    if md5_file(archive) != ARCHIVE_MD5:
        raise ValueError("LibriSpeech archive MD5 does not match OpenSLR")
    speaker_metadata = parse_test_clean_speakers(speakers_metadata)
    records = discover_records(source_root)
    discovered_speakers = {record["speaker"] for record in records}
    discovered_chapters = {
        (record["speaker"], record["chapter"]) for record in records
    }
    if limit is None:
        if len(records) != EXPECTED_UTTERANCES:
            raise ValueError(f"unexpected utterance count: {len(records)}")
        if len(discovered_speakers) != EXPECTED_SPEAKERS:
            raise ValueError(f"unexpected speaker count: {len(discovered_speakers)}")
        if len(discovered_chapters) != EXPECTED_CHAPTERS:
            raise ValueError(f"unexpected chapter count: {len(discovered_chapters)}")
        if discovered_speakers != set(speaker_metadata):
            raise ValueError("discovered speakers do not match SPEAKERS.TXT")
    else:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rows.append(prepare_record(record, source_root, output_root))
        if index % 100 == 0 or index == len(records):
            print(f"Prepared {index}/{len(records)} control utterances...", flush=True)

    write_immutable(manifest_path, _csv_bytes(rows))
    duration = sum(float(row["processed_duration_seconds"]) for row in rows)
    selected_speakers = sorted({row["speaker"] for row in rows}, key=int)
    selected_chapters = {(row["speaker"], row["chapter"]) for row in rows}
    sex_counts = Counter(speaker_metadata[speaker] for speaker in selected_speakers)
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "limited smoke test" if limit is not None else "complete test-clean",
        "dataset": DATASET,
        "release": RELEASE,
        "subset": SUBSET,
        "selection_policy": "all official test-clean utterances",
        "preparation": {
            "version": PREPARATION_VERSION,
            "source_sample_rate_hz": TARGET_SAMPLE_RATE,
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "source_channels": 1,
            "target_channels": 1,
            "source_format": "FLAC",
            "source_subtype": "PCM_16",
            "target_format": "WAV",
            "target_subtype": "FLOAT",
            "resampling": False,
            "normalization": False,
            "trimming": False,
            "denoising": False,
        },
        "license": "CC BY 4.0",
        "source_url": "https://www.openslr.org/12/",
        "archive": {
            "name": ARCHIVE_NAME,
            "published_md5": ARCHIVE_MD5,
            "verified_md5": md5_file(archive),
            "sha256": sha256_file(archive),
        },
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "aggregate": {
            "utterances": len(rows),
            "speakers": len(selected_speakers),
            "chapters": len(selected_chapters),
            "female_speakers": sex_counts["F"],
            "male_speakers": sex_counts["M"],
            "duration_seconds": round(duration, 9),
            "duration_hours": round(duration / 3600, 6),
            "maximum_processed_peak_abs": max(
                row["processed_peak_abs"] for row in rows
            ),
        },
        "speakers": selected_speakers,
        "limitations": [
            "LibriSpeech is read audiobook speech, so it is not domain-matched to every use case.",
            "The corpus documentation says the clean/US-English classification was automated and is not completely reliable.",
        ],
    }
    write_immutable(report_path, _json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the complete LibriSpeech test-clean native control."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--speakers-metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = prepare_control(
        source_root=args.source_root,
        speakers_metadata=args.speakers_metadata,
        archive=args.archive,
        output_root=args.output_root,
        manifest_path=args.manifest,
        report_path=args.report,
        limit=args.limit,
    )
    aggregate = report["aggregate"]
    print(
        f"LibriSpeech preparation {report['status']}: "
        f"{aggregate['utterances']} utterances, "
        f"{aggregate['duration_hours']:.6f} hours.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
