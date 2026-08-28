"""Create deterministic 16 kHz mono derivatives from validated L2-ARCTIC WAVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import soxr

from equivoice.validate_l2_arctic import EXPECTED_SPEAKERS, sha256_file


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_FORMAT = "WAV"
TARGET_SUBTYPE = "FLOAT"
RESAMPLER_QUALITY = "HQ"
PREPROCESSING_VERSION = "l2-arctic-16khz-mono-float32-soxr-hq-v1"
DURATION_TOLERANCE_SECONDS = 1 / TARGET_SAMPLE_RATE + 1e-9

SOURCE_REQUIRED_FIELDS = {
    "speaker",
    "utterance_id",
    "wav_path",
    "transcript_path",
    "textgrid_path",
    "annotation_path",
    "frames",
    "sample_rate_hz",
    "channels",
    "format",
    "subtype",
    "duration_seconds",
    "full_scale_sample_count",
    "audio_pcm_sha256",
}

PROCESSED_MANIFEST_FIELDS = [
    "speaker",
    "utterance_id",
    "source_wav_path",
    "transcript_path",
    "textgrid_path",
    "annotation_path",
    "source_frames",
    "source_sample_rate_hz",
    "source_duration_seconds",
    "source_full_scale_sample_count",
    "source_audio_pcm_sha256",
    "processed_wav_path",
    "processed_frames",
    "processed_sample_rate_hz",
    "processed_channels",
    "processed_format",
    "processed_subtype",
    "processed_duration_seconds",
    "duration_delta_seconds",
    "processed_peak_abs",
    "processed_out_of_unit_range_sample_count",
    "processed_pcm_sha256",
    "processed_file_sha256",
]


def require_passed_validation(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("raw-data validation report did not pass")
    if report.get("aggregate", {}).get("errors") != 0:
        raise ValueError("raw-data validation report contains errors")
    report_speakers = {item["speaker"] for item in report.get("speakers", [])}
    if report_speakers != set(EXPECTED_SPEAKERS):
        raise ValueError("raw-data validation report has unexpected speakers")
    return report


def load_source_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = SOURCE_REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(
                f"source manifest is missing fields: {', '.join(sorted(missing))}"
            )
        records = list(reader)

    seen: set[tuple[str, str]] = set()
    for row_number, record in enumerate(records, start=2):
        key = (record["speaker"], record["utterance_id"])
        if key in seen:
            raise ValueError(f"duplicate source row {row_number}: {key}")
        seen.add(key)
        if record["speaker"] not in EXPECTED_SPEAKERS:
            raise ValueError(f"unexpected speaker at row {row_number}: {key[0]}")
        if not record["wav_path"] or not record["audio_pcm_sha256"]:
            raise ValueError(f"incomplete source row {row_number}: {key}")
    return records


def _decoded_int16_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<i2", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _decoded_float32_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _processed_relative_path(record: dict[str, str]) -> Path:
    return (
        Path("speakers")
        / record["speaker"]
        / "wav"
        / f"{record['utterance_id']}.wav"
    )


def _write_new_or_verify_identical(path: Path, expected_audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        info = sf.info(path)
        existing, existing_rate = sf.read(path, dtype="float32", always_2d=False)
        if (
            info.samplerate != TARGET_SAMPLE_RATE
            or info.channels != TARGET_CHANNELS
            or info.format != TARGET_FORMAT
            or info.subtype != TARGET_SUBTYPE
            or existing_rate != TARGET_SAMPLE_RATE
            or not np.array_equal(existing, expected_audio)
        ):
            raise ValueError(f"refusing to overwrite changed processed WAV: {path}")
        return

    temporary = path.with_suffix(path.suffix + ".tmp")
    sf.write(
        temporary,
        expected_audio,
        TARGET_SAMPLE_RATE,
        format=TARGET_FORMAT,
        subtype=TARGET_SUBTYPE,
    )
    temporary.replace(path)


def preprocess_record(
    record: dict[str, str],
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify one raw WAV, resample it, and return metadata for the derivative."""

    source_path = source_root / record["wav_path"]
    if not source_path.is_file():
        raise ValueError(f"missing source WAV: {source_path}")

    source_info = sf.info(source_path)
    if source_info.samplerate != int(record["sample_rate_hz"]):
        raise ValueError(f"source sample-rate mismatch: {source_path}")
    if source_info.channels != int(record["channels"]):
        raise ValueError(f"source channel mismatch: {source_path}")
    if source_info.frames != int(record["frames"]):
        raise ValueError(f"source frame-count mismatch: {source_path}")
    if source_info.format != record["format"]:
        raise ValueError(f"source format mismatch: {source_path}")
    if source_info.subtype != record["subtype"]:
        raise ValueError(f"source subtype mismatch: {source_path}")
    if source_info.channels != 1:
        raise ValueError(f"source is not mono: {source_path}")

    source_pcm, source_rate = sf.read(
        source_path, dtype="int16", always_2d=True
    )
    if source_rate != source_info.samplerate:
        raise ValueError(f"source decoder sample-rate mismatch: {source_path}")
    source_digest = _decoded_int16_sha256(source_pcm)
    if source_digest != record["audio_pcm_sha256"]:
        raise ValueError(f"source decoded-audio hash mismatch: {source_path}")

    source_duration = source_info.frames / source_info.samplerate
    if abs(source_duration - float(record["duration_seconds"])) > 1e-8:
        raise ValueError(f"source duration mismatch: {source_path}")

    source_float = source_pcm[:, 0].astype(np.float32) / 32_768.0
    processed = np.asarray(
        soxr.resample(
            source_float,
            source_info.samplerate,
            TARGET_SAMPLE_RATE,
            quality=RESAMPLER_QUALITY,
        ),
        dtype=np.float32,
    )
    if processed.ndim != 1 or processed.size == 0:
        raise ValueError(f"resampler produced invalid output: {source_path}")
    if not np.all(np.isfinite(processed)):
        raise ValueError(f"resampler produced nonfinite samples: {source_path}")

    relative_output = _processed_relative_path(record)
    output_path = output_root / relative_output
    _write_new_or_verify_identical(output_path, processed)

    processed_info = sf.info(output_path)
    readback, readback_rate = sf.read(
        output_path, dtype="float32", always_2d=False
    )
    if (
        readback_rate != TARGET_SAMPLE_RATE
        or processed_info.samplerate != TARGET_SAMPLE_RATE
        or processed_info.channels != TARGET_CHANNELS
        or processed_info.format != TARGET_FORMAT
        or processed_info.subtype != TARGET_SUBTYPE
        or not np.array_equal(readback, processed)
    ):
        raise ValueError(f"processed WAV verification failed: {output_path}")

    processed_duration = processed_info.frames / processed_info.samplerate
    duration_delta = processed_duration - source_duration
    if abs(duration_delta) > DURATION_TOLERANCE_SECONDS:
        raise ValueError(
            f"processed duration drift exceeds tolerance: {output_path} "
            f"({duration_delta:.9f}s)"
        )

    peak = float(np.max(np.abs(readback)))
    out_of_range_count = int(np.count_nonzero(np.abs(readback) > 1.0))
    return {
        "speaker": record["speaker"],
        "utterance_id": record["utterance_id"],
        "source_wav_path": record["wav_path"],
        "transcript_path": record["transcript_path"],
        "textgrid_path": record["textgrid_path"],
        "annotation_path": record["annotation_path"],
        "source_frames": source_info.frames,
        "source_sample_rate_hz": source_info.samplerate,
        "source_duration_seconds": round(source_duration, 9),
        "source_full_scale_sample_count": int(record["full_scale_sample_count"]),
        "source_audio_pcm_sha256": source_digest,
        "processed_wav_path": relative_output.as_posix(),
        "processed_frames": processed_info.frames,
        "processed_sample_rate_hz": processed_info.samplerate,
        "processed_channels": processed_info.channels,
        "processed_format": processed_info.format,
        "processed_subtype": processed_info.subtype,
        "processed_duration_seconds": round(processed_duration, 9),
        "duration_delta_seconds": round(duration_delta, 9),
        "processed_peak_abs": round(peak, 9),
        "processed_out_of_unit_range_sample_count": out_of_range_count,
        "processed_pcm_sha256": _decoded_float32_sha256(readback),
        "processed_file_sha256": sha256_file(output_path),
    }


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=PROCESSED_MANIFEST_FIELDS, lineterminator="\n"
    )
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


def build_report(
    rows: list[dict[str, Any]],
    source_manifest: Path,
    limited: bool,
) -> dict[str, Any]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    speaker_summaries: list[dict[str, Any]] = []
    for speaker in EXPECTED_SPEAKERS:
        speaker_rows = by_speaker.get(speaker, [])
        if not speaker_rows and limited:
            continue
        duration = sum(row["processed_duration_seconds"] for row in speaker_rows)
        speaker_summaries.append(
            {
                "speaker": speaker,
                "utterances": len(speaker_rows),
                "processed_duration_seconds": round(duration, 9),
                "processed_duration_hours": round(duration / 3600, 6),
                "source_full_scale_files": sum(
                    row["source_full_scale_sample_count"] > 0
                    for row in speaker_rows
                ),
                "processed_out_of_unit_range_files": sum(
                    row["processed_out_of_unit_range_sample_count"] > 0
                    for row in speaker_rows
                ),
            }
        )

    source_duration = sum(row["source_duration_seconds"] for row in rows)
    processed_duration = sum(row["processed_duration_seconds"] for row in rows)
    out_of_range_ids = [
        f"{row['speaker']}/{row['utterance_id']}"
        for row in rows
        if row["processed_out_of_unit_range_sample_count"] > 0
    ]
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "limited smoke test" if limited else "full Arabic cohort",
        "dataset": "L2-ARCTIC",
        "release": "v5.0",
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "source_sample_rate_hz": 44_100,
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "target_channels": TARGET_CHANNELS,
            "target_format": TARGET_FORMAT,
            "target_subtype": TARGET_SUBTYPE,
            "resampler": "soxr",
            "resampler_version": soxr.__version__,
            "resampler_quality": RESAMPLER_QUALITY,
            "normalization": False,
            "trimming": False,
            "denoising": False,
            "output_clipping": False,
            "duration_tolerance_seconds": DURATION_TOLERANCE_SECONDS,
        },
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": sha256_file(source_manifest),
        "aggregate": {
            "utterances": len(rows),
            "source_duration_seconds": round(source_duration, 9),
            "processed_duration_seconds": round(processed_duration, 9),
            "processed_duration_hours": round(processed_duration / 3600, 6),
            "maximum_absolute_duration_delta_seconds": round(
                max(abs(row["duration_delta_seconds"]) for row in rows), 9
            ),
            "source_full_scale_files": sum(
                row["source_full_scale_sample_count"] > 0 for row in rows
            ),
            "processed_out_of_unit_range_files": len(out_of_range_ids),
            "errors": 0,
            "warnings": 1 if out_of_range_ids else 0,
        },
        "processed_out_of_unit_range_ids": out_of_range_ids,
        "speakers": speaker_summaries,
    }


def run_preprocessing(
    source_root: Path,
    source_manifest: Path,
    validation_report: Path,
    output_root: Path,
    processed_manifest: Path,
    report_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    require_passed_validation(validation_report)
    records = load_source_manifest(source_manifest)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]

    rows: list[dict[str, Any]] = []
    current_speaker: str | None = None
    for index, record in enumerate(records, start=1):
        if record["speaker"] != current_speaker:
            current_speaker = record["speaker"]
            print(f"Processing {current_speaker}...", flush=True)
        if index % 250 == 0:
            print(f"Processed {index}/{len(records)} recordings...", flush=True)
        rows.append(preprocess_record(record, source_root, output_root))

    manifest_content = _manifest_bytes(rows)
    write_immutable(processed_manifest, manifest_content)
    report = build_report(rows, source_manifest, limited=limit is not None)
    report["processed_manifest"] = processed_manifest.name
    report["processed_manifest_sha256"] = hashlib.sha256(
        manifest_content
    ).hexdigest()
    write_immutable(report_path, _json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess validated L2-ARCTIC Arabic WAVs to 16 kHz mono."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_preprocessing(
        source_root=args.source_root,
        source_manifest=args.source_manifest,
        validation_report=args.validation_report,
        output_root=args.output_root,
        processed_manifest=args.processed_manifest,
        report_path=args.report,
        limit=args.limit,
    )
    print(
        f"Preprocessing {report['status']}: "
        f"{report['aggregate']['utterances']} recordings, "
        f"{report['aggregate']['errors']} errors, "
        f"{report['aggregate']['warnings']} warnings.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
