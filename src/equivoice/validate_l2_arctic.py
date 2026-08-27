"""Validate an extracted L2-ARCTIC Arabic cohort without modifying it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1
EXPECTED_FORMAT = "WAV"
EXPECTED_SUBTYPE = "PCM_16"
TEXTGRID_DURATION_TOLERANCE_SECONDS = 0.02


@dataclass(frozen=True)
class SpeakerExpectation:
    """Expected file counts from the official v5.0 documentation."""

    scripted_wavs: int
    annotations: int


EXPECTED_SPEAKERS = {
    "ABA": SpeakerExpectation(scripted_wavs=1_129, annotations=150),
    "SKA": SpeakerExpectation(scripted_wavs=974, annotations=150),
    "YBAA": SpeakerExpectation(scripted_wavs=1_130, annotations=149),
    "ZHAA": SpeakerExpectation(scripted_wavs=1_132, annotations=150),
}

MANIFEST_FIELDS = [
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
    "transcript_bytes",
    "transcript_characters",
    "forced_textgrid_duration_seconds",
    "annotation_duration_seconds",
    "peak_abs_pcm16",
    "full_scale_sample_count",
    "audio_pcm_sha256",
]

_TEXTGRID_XMAX = re.compile(
    r"(?m)^\s*xmax\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _files_by_stem(
    directory: Path, expected_suffix: str
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Index expected files and report unexpected or duplicate identifiers."""

    indexed: dict[str, Path] = {}
    unexpected: list[str] = []
    duplicate_ids: list[str] = []
    casefold_ids: dict[str, str] = {}

    if not directory.is_dir():
        return indexed, unexpected, duplicate_ids

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            unexpected.append(path.name)
            continue
        if path.suffix.casefold() != expected_suffix.casefold():
            unexpected.append(path.name)
            continue

        folded = path.stem.casefold()
        if folded in casefold_ids:
            duplicate_ids.extend([casefold_ids[folded], path.stem])
            continue
        casefold_ids[folded] = path.stem
        indexed[path.stem] = path

    return indexed, sorted(unexpected), sorted(set(duplicate_ids))


def _read_text(path: Path) -> tuple[str | None, int, list[str]]:
    """Read a required UTF-8 text file without normalizing its content."""

    errors: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, 0, [f"could not read {path.name}: {exc}"]

    if not raw:
        return "", 0, [f"empty file: {path.name}"]

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return None, len(raw), [f"invalid UTF-8 in {path.name}: {exc}"]

    if not text.strip():
        errors.append(f"whitespace-only file: {path.name}")
    return text, len(raw), errors


def _validate_textgrid(
    path: Path, audio_duration: float
) -> tuple[float | None, list[str]]:
    """Validate the TextGrid signature and its root duration."""

    text, _, errors = _read_text(path)
    if text is None:
        return None, errors
    if 'Object class = "TextGrid"' not in text:
        errors.append(f"missing TextGrid signature: {path.name}")

    match = _TEXTGRID_XMAX.search(text)
    if match is None:
        errors.append(f"missing root xmax in TextGrid: {path.name}")
        return None, errors

    grid_duration = float(match.group(1))
    if abs(grid_duration - audio_duration) > TEXTGRID_DURATION_TOLERANCE_SECONDS:
        errors.append(
            f"duration mismatch for {path.name}: "
            f"audio={audio_duration:.6f}s textgrid={grid_duration:.6f}s"
        )
    return grid_duration, errors


def _scan_audio_samples(path: Path) -> tuple[int, int, str]:
    """Measure PCM peak/full-scale samples and hash decoded PCM data."""

    peak = 0
    full_scale_count = 0
    digest = hashlib.sha256()

    for block in sf.blocks(path, blocksize=65_536, dtype="int16", always_2d=True):
        digest.update(block.tobytes(order="C"))
        samples = block.astype(np.int32, copy=False)
        if samples.size:
            peak = max(peak, int(np.max(np.abs(samples))))
            full_scale_count += int(
                np.count_nonzero((samples == -32_768) | (samples == 32_767))
            )

    return peak, full_scale_count, digest.hexdigest()


def _relative(path: Path | None, root: Path) -> str:
    return "" if path is None else path.relative_to(root).as_posix()


def validate_speaker(
    speaker_dir: Path,
    speaker: str,
    expectation: SpeakerExpectation,
    dataset_root: Path,
    scan_samples: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one speaker directory and return summary plus manifest rows."""

    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "wav": (".wav", expectation.scripted_wavs),
        "transcript": (".txt", expectation.scripted_wavs),
        "textgrid": (".TextGrid", expectation.scripted_wavs),
        "annotation": (".TextGrid", expectation.annotations),
    }
    indexed: dict[str, dict[str, Path]] = {}
    counts: dict[str, int] = {}

    if not speaker_dir.is_dir():
        return (
            {
                "speaker": speaker,
                "status": "failed",
                "counts": {},
                "errors": [f"missing speaker directory: {speaker_dir.name}"],
                "warnings": [],
            },
            [],
        )

    for folder_name, (suffix, expected_count) in required.items():
        folder = speaker_dir / folder_name
        if not folder.is_dir():
            errors.append(f"missing directory: {speaker}/{folder_name}")
        files, unexpected, duplicate_ids = _files_by_stem(folder, suffix)
        indexed[folder_name] = files
        counts[folder_name] = len(files)
        if len(files) != expected_count:
            errors.append(
                f"{folder_name} count is {len(files)}; expected {expected_count}"
            )
        if unexpected:
            warnings.append(
                f"unexpected entries in {folder_name}: {', '.join(unexpected)}"
            )
        if duplicate_ids:
            errors.append(
                f"duplicate case-insensitive IDs in {folder_name}: "
                f"{', '.join(duplicate_ids)}"
            )

    wav_ids = set(indexed["wav"])
    transcript_ids = set(indexed["transcript"])
    textgrid_ids = set(indexed["textgrid"])
    annotation_ids = set(indexed["annotation"])

    relationships = [
        ("transcripts missing for WAVs", wav_ids - transcript_ids),
        ("transcripts without WAVs", transcript_ids - wav_ids),
        ("forced TextGrids missing for WAVs", wav_ids - textgrid_ids),
        ("forced TextGrids without WAVs", textgrid_ids - wav_ids),
        ("annotations without WAVs", annotation_ids - wav_ids),
    ]
    for label, identifiers in relationships:
        if identifiers:
            errors.append(f"{label}: {', '.join(sorted(identifiers))}")

    manifest_rows: list[dict[str, Any]] = []
    property_counts: Counter[str] = Counter()
    durations: list[float] = []
    full_scale_files: list[str] = []
    pcm_hash_to_ids: dict[str, list[str]] = defaultdict(list)

    for utterance_id in sorted(wav_ids):
        wav_path = indexed["wav"][utterance_id]
        transcript_path = indexed["transcript"].get(utterance_id)
        textgrid_path = indexed["textgrid"].get(utterance_id)
        annotation_path = indexed["annotation"].get(utterance_id)
        row: dict[str, Any] = {
            "speaker": speaker,
            "utterance_id": utterance_id,
            "wav_path": _relative(wav_path, dataset_root),
            "transcript_path": _relative(transcript_path, dataset_root),
            "textgrid_path": _relative(textgrid_path, dataset_root),
            "annotation_path": _relative(annotation_path, dataset_root),
            "frames": "",
            "sample_rate_hz": "",
            "channels": "",
            "format": "",
            "subtype": "",
            "duration_seconds": "",
            "transcript_bytes": "",
            "transcript_characters": "",
            "forced_textgrid_duration_seconds": "",
            "annotation_duration_seconds": "",
            "peak_abs_pcm16": "",
            "full_scale_sample_count": "",
            "audio_pcm_sha256": "",
        }

        try:
            info = sf.info(wav_path)
        except (OSError, RuntimeError) as exc:
            errors.append(f"unreadable WAV {utterance_id}: {exc}")
            manifest_rows.append(row)
            continue

        duration = info.frames / info.samplerate if info.samplerate else 0.0
        durations.append(duration)
        property_key = (
            f"{info.samplerate}Hz/{info.channels}ch/{info.format}/{info.subtype}"
        )
        property_counts[property_key] += 1
        row.update(
            {
                "frames": info.frames,
                "sample_rate_hz": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
                "duration_seconds": round(duration, 9),
            }
        )

        if info.frames <= 0:
            errors.append(f"zero-length WAV: {utterance_id}")
        if info.samplerate != EXPECTED_SAMPLE_RATE:
            errors.append(
                f"unexpected sample rate for {utterance_id}: {info.samplerate}"
            )
        if info.channels != EXPECTED_CHANNELS:
            errors.append(f"unexpected channel count for {utterance_id}: {info.channels}")
        if info.format != EXPECTED_FORMAT:
            errors.append(f"unexpected audio format for {utterance_id}: {info.format}")
        if info.subtype != EXPECTED_SUBTYPE:
            errors.append(f"unexpected audio subtype for {utterance_id}: {info.subtype}")

        if scan_samples:
            try:
                peak, full_scale_count, pcm_hash = _scan_audio_samples(wav_path)
            except (OSError, RuntimeError) as exc:
                errors.append(f"could not scan WAV samples {utterance_id}: {exc}")
            else:
                row["peak_abs_pcm16"] = peak
                row["full_scale_sample_count"] = full_scale_count
                row["audio_pcm_sha256"] = pcm_hash
                pcm_hash_to_ids[pcm_hash].append(utterance_id)
                if full_scale_count:
                    full_scale_files.append(utterance_id)

        if transcript_path is not None:
            transcript, byte_count, transcript_errors = _read_text(transcript_path)
            errors.extend(
                f"{utterance_id}: {message}" for message in transcript_errors
            )
            row["transcript_bytes"] = byte_count
            row["transcript_characters"] = (
                "" if transcript is None else len(transcript)
            )

        if textgrid_path is not None:
            grid_duration, grid_errors = _validate_textgrid(textgrid_path, duration)
            errors.extend(f"{utterance_id}: {message}" for message in grid_errors)
            row["forced_textgrid_duration_seconds"] = (
                "" if grid_duration is None else grid_duration
            )

        if annotation_path is not None:
            annotation_duration, annotation_errors = _validate_textgrid(
                annotation_path, duration
            )
            errors.extend(
                f"{utterance_id} annotation: {message}"
                for message in annotation_errors
            )
            row["annotation_duration_seconds"] = (
                "" if annotation_duration is None else annotation_duration
            )

        manifest_rows.append(row)

    duplicate_audio = [
        ids for ids in pcm_hash_to_ids.values() if len(ids) > 1
    ]
    if duplicate_audio:
        errors.extend(
            f"duplicate decoded audio: {', '.join(ids)}" for ids in duplicate_audio
        )

    if full_scale_files:
        warnings.append(
            f"{len(full_scale_files)} WAV file(s) contain full-scale samples; "
            "this is a clipping indicator, not proof of audible clipping"
        )

    total_duration = sum(durations)
    summary = {
        "speaker": speaker,
        "status": "passed" if not errors else "failed",
        "counts": counts,
        "audio_properties": dict(sorted(property_counts.items())),
        "duration_seconds": round(total_duration, 9),
        "duration_hours": round(total_duration / 3600, 6),
        "minimum_utterance_seconds": round(min(durations), 9) if durations else None,
        "maximum_utterance_seconds": round(max(durations), 9) if durations else None,
        "mean_utterance_seconds": (
            round(total_duration / len(durations), 9) if durations else None
        ),
        "files_with_full_scale_samples": len(full_scale_files),
        "full_scale_utterance_ids": full_scale_files,
        "duplicate_audio_groups": duplicate_audio,
        "errors": errors,
        "warnings": warnings,
    }
    return summary, manifest_rows


def _archive_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    targets = [dataset_root / "archives" / "l2arctic_release_v5.0.zip"]
    targets.extend(
        dataset_root / "speaker_archives" / f"{speaker}.zip"
        for speaker in EXPECTED_SPEAKERS
    )
    metadata: list[dict[str, Any]] = []
    for path in targets:
        if path.is_file():
            metadata.append(
                {
                    "path": path.relative_to(dataset_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return metadata


def validate_dataset(
    dataset_root: Path,
    scan_samples: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the four documented Arabic speakers in L2-ARCTIC v5.0."""

    dataset_root = dataset_root.resolve()
    speaker_summaries: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for speaker, expectation in EXPECTED_SPEAKERS.items():
        print(f"Validating {speaker}...", flush=True)
        summary, rows = validate_speaker(
            dataset_root / "speakers" / speaker,
            speaker,
            expectation,
            dataset_root,
            scan_samples=scan_samples,
        )
        speaker_summaries.append(summary)
        manifest_rows.extend(rows)

    errors = [
        f"{summary['speaker']}: {error}"
        for summary in speaker_summaries
        for error in summary["errors"]
    ]
    warnings = [
        f"{summary['speaker']}: {warning}"
        for summary in speaker_summaries
        for warning in summary["warnings"]
    ]
    aggregate_seconds = sum(
        float(summary.get("duration_seconds", 0.0)) for summary in speaker_summaries
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "L2-ARCTIC",
        "release": "v5.0",
        "cohort": "Arabic L1 scripted speech",
        "validation_scope": {
            "speakers": list(EXPECTED_SPEAKERS),
            "read_only": True,
            "sample_scan_enabled": scan_samples,
            "expected_sample_rate_hz": EXPECTED_SAMPLE_RATE,
            "expected_channels": EXPECTED_CHANNELS,
            "expected_format": EXPECTED_FORMAT,
            "expected_subtype": EXPECTED_SUBTYPE,
            "textgrid_duration_tolerance_seconds": (
                TEXTGRID_DURATION_TOLERANCE_SECONDS
            ),
        },
        "status": "passed" if not errors else "failed",
        "aggregate": {
            "speakers": len(speaker_summaries),
            "wav_files": sum(s["counts"].get("wav", 0) for s in speaker_summaries),
            "transcript_files": sum(
                s["counts"].get("transcript", 0) for s in speaker_summaries
            ),
            "forced_textgrid_files": sum(
                s["counts"].get("textgrid", 0) for s in speaker_summaries
            ),
            "annotation_files": sum(
                s["counts"].get("annotation", 0) for s in speaker_summaries
            ),
            "duration_seconds": round(aggregate_seconds, 9),
            "duration_hours": round(aggregate_seconds / 3600, 6),
            "files_with_full_scale_samples": sum(
                s["files_with_full_scale_samples"] for s in speaker_summaries
            ),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "archives": _archive_metadata(dataset_root),
        "speakers": speaker_summaries,
        "errors": errors,
        "warnings": warnings,
    }
    return report, manifest_rows


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_manifest(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the extracted L2-ARCTIC v5.0 Arabic cohort."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing speakers/, archives/, and speaker_archives/.",
    )
    parser.add_argument("--report", type=Path, required=True, help="JSON report path.")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Per-utterance CSV manifest path."
    )
    parser.add_argument(
        "--skip-sample-scan",
        action="store_true",
        help="Skip decoded-audio hashes and full-scale sample checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, rows = validate_dataset(
        args.dataset_root,
        scan_samples=not args.skip_sample_scan,
    )
    write_report(report, args.report)
    write_manifest(rows, args.manifest)
    print(
        f"Validation {report['status']}: "
        f"{report['aggregate']['wav_files']} WAV files, "
        f"{report['aggregate']['errors']} errors, "
        f"{report['aggregate']['warnings']} warnings.",
        flush=True,
    )
    print(f"Report: {args.report}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
