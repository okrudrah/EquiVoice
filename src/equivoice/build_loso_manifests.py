"""Build deterministic leave-one-speaker-out manifests for L2-ARCTIC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from equivoice.validate_l2_arctic import EXPECTED_SPEAKERS, sha256_file


SCHEMA_VERSION = 1
SPLIT_ALGORITHM_VERSION = "sha256-prompt-group-v1"
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_NAMESPACE = "equivoice:l2-arctic:v5.0:arabic:validation:v1"
SPEAKERS = tuple(EXPECTED_SPEAKERS)
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}

SOURCE_REQUIRED_FIELDS = {
    "speaker",
    "utterance_id",
    "wav_path",
    "transcript_path",
    "textgrid_path",
    "annotation_path",
    "duration_seconds",
    "audio_pcm_sha256",
}

FOLD_FIELDS = [
    "fold_id",
    "split",
    "speaker",
    "utterance_id",
    "wav_path",
    "transcript_path",
    "textgrid_path",
    "annotation_path",
    "duration_seconds",
    "audio_pcm_sha256",
]


def is_validation_utterance(
    utterance_id: str,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    namespace: str = DEFAULT_NAMESPACE,
) -> bool:
    """Assign a prompt ID to validation using a stable SHA-256 threshold."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    payload = f"{namespace}\0{utterance_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    threshold = int(validation_fraction * (1 << 64))
    return value < threshold


def load_source_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate the metadata-only raw-data manifest."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_fields = SOURCE_REQUIRED_FIELDS - fields
        if missing_fields:
            raise ValueError(
                f"source manifest is missing fields: {', '.join(sorted(missing_fields))}"
            )
        records = list(reader)

    seen: set[tuple[str, str]] = set()
    speakers: set[str] = set()
    for row_number, record in enumerate(records, start=2):
        speaker = record["speaker"]
        utterance_id = record["utterance_id"]
        key = (speaker, utterance_id)
        if key in seen:
            raise ValueError(f"duplicate source record at row {row_number}: {key}")
        seen.add(key)
        speakers.add(speaker)
        for field in ("wav_path", "transcript_path", "textgrid_path"):
            if not record[field]:
                raise ValueError(f"empty {field} at row {row_number}")
        if not record["audio_pcm_sha256"]:
            raise ValueError(f"empty audio hash at row {row_number}")
        try:
            duration = float(record["duration_seconds"])
        except ValueError as exc:
            raise ValueError(f"invalid duration at row {row_number}") from exc
        if duration <= 0:
            raise ValueError(f"nonpositive duration at row {row_number}")

    if speakers != set(SPEAKERS):
        raise ValueError(
            f"source speakers are {sorted(speakers)}; expected {sorted(SPEAKERS)}"
        )
    return records


def require_passed_validation(path: Path) -> dict[str, Any]:
    """Require a successful raw-data report before freezing fold assignments."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("raw-data validation report did not pass")
    if report.get("aggregate", {}).get("errors") != 0:
        raise ValueError("raw-data validation report contains errors")
    report_speakers = {item["speaker"] for item in report.get("speakers", [])}
    if report_speakers != set(SPEAKERS):
        raise ValueError("raw-data validation report has unexpected speakers")
    return report


def build_fold_rows(
    records: Iterable[dict[str, str]],
    held_out_speaker: str,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    namespace: str = DEFAULT_NAMESPACE,
) -> list[dict[str, str]]:
    """Assign every source record to train, validation, or test for one fold."""

    if held_out_speaker not in SPEAKERS:
        raise ValueError(f"unknown held-out speaker: {held_out_speaker}")

    fold_id = f"held_out_{held_out_speaker.lower()}"
    rows: list[dict[str, str]] = []
    source_records = list(records)
    for record in source_records:
        if record["speaker"] == held_out_speaker:
            split = "test"
        elif is_validation_utterance(
            record["utterance_id"], validation_fraction, namespace
        ):
            split = "validation"
        else:
            split = "train"

        rows.append(
            {
                "fold_id": fold_id,
                "split": split,
                "speaker": record["speaker"],
                "utterance_id": record["utterance_id"],
                "wav_path": record["wav_path"],
                "transcript_path": record["transcript_path"],
                "textgrid_path": record["textgrid_path"],
                "annotation_path": record["annotation_path"],
                "duration_seconds": record["duration_seconds"],
                "audio_pcm_sha256": record["audio_pcm_sha256"],
            }
        )

    rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["speaker"],
            row["utterance_id"],
        )
    )
    validate_fold(rows, source_records, held_out_speaker)
    return rows


def validate_fold(
    rows: list[dict[str, str]],
    source_records: list[dict[str, str]],
    held_out_speaker: str,
) -> None:
    """Enforce speaker isolation, complete coverage, and prompt grouping."""

    source_keys = {
        (record["speaker"], record["utterance_id"]) for record in source_records
    }
    row_keys = {(row["speaker"], row["utterance_id"]) for row in rows}
    if len(row_keys) != len(rows):
        raise ValueError("fold contains duplicate speaker/utterance records")
    if row_keys != source_keys:
        raise ValueError("fold does not cover the source manifest exactly once")

    test_rows = [row for row in rows if row["split"] == "test"]
    training_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not training_rows or not validation_rows or not test_rows:
        raise ValueError("train, validation, and test splits must all be nonempty")

    if {row["speaker"] for row in test_rows} != {held_out_speaker}:
        raise ValueError("test split contains a non-held-out speaker")
    if any(
        row["speaker"] == held_out_speaker
        for row in training_rows + validation_rows
    ):
        raise ValueError("held-out speaker leaked into train or validation")

    expected_training_speakers = set(SPEAKERS) - {held_out_speaker}
    if {row["speaker"] for row in training_rows} != expected_training_speakers:
        raise ValueError("training split does not contain all three training speakers")
    if {row["speaker"] for row in validation_rows} != expected_training_speakers:
        raise ValueError("validation split does not contain all three training speakers")

    train_ids = {row["utterance_id"] for row in training_rows}
    validation_ids = {row["utterance_id"] for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            "prompt IDs overlap between train and validation: "
            f"{', '.join(sorted(overlap))}"
        )


def _split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split"] == split]
        by_speaker: dict[str, dict[str, Any]] = {}
        for speaker in SPEAKERS:
            speaker_rows = [row for row in split_rows if row["speaker"] == speaker]
            by_speaker[speaker] = {
                "utterances": len(speaker_rows),
                "duration_seconds": round(
                    sum(float(row["duration_seconds"]) for row in speaker_rows), 9
                ),
            }
        duration = sum(float(row["duration_seconds"]) for row in split_rows)
        summary[split] = {
            "utterances": len(split_rows),
            "duration_seconds": round(duration, 9),
            "duration_hours": round(duration / 3600, 6),
            "speakers": by_speaker,
        }
    return summary


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FOLD_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    """Write a new artifact, or verify that an existing artifact is identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(
                f"refusing to overwrite changed immutable manifest: {path}"
            )
        return
    path.write_bytes(content)


def generate_manifests(
    source_manifest: Path,
    validation_report: Path,
    output_dir: Path,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """Generate and content-hash all four deterministic fold manifests."""

    require_passed_validation(validation_report)
    records = load_source_manifest(source_manifest)
    source_manifest_digest = sha256_file(source_manifest)
    unique_ids = sorted({record["utterance_id"] for record in records})
    validation_ids = [
        utterance_id
        for utterance_id in unique_ids
        if is_validation_utterance(utterance_id, validation_fraction, namespace)
    ]
    validation_ids_digest = hashlib.sha256(
        ("\n".join(validation_ids) + "\n").encode("utf-8")
    ).hexdigest()

    folds: list[dict[str, Any]] = []
    for held_out_speaker in SPEAKERS:
        print(f"Building fold held_out_{held_out_speaker.lower()}...", flush=True)
        rows = build_fold_rows(
            records,
            held_out_speaker,
            validation_fraction=validation_fraction,
            namespace=namespace,
        )
        filename = f"fold_{held_out_speaker.lower()}.csv"
        content = _csv_bytes(rows)
        write_immutable(output_dir / filename, content)
        folds.append(
            {
                "fold_id": f"held_out_{held_out_speaker.lower()}",
                "held_out_speaker": held_out_speaker,
                "manifest": filename,
                "manifest_sha256": hashlib.sha256(content).hexdigest(),
                "splits": _split_summary(rows),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "L2-ARCTIC",
        "release": "v5.0",
        "cohort": "Arabic L1 scripted speech",
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": source_manifest_digest,
        "source_records": len(records),
        "speakers": list(SPEAKERS),
        "split_policy": {
            "algorithm": SPLIT_ALGORITHM_VERSION,
            "namespace": namespace,
            "validation_fraction_threshold": validation_fraction,
            "validation_unit": "utterance_id (prompt-grouped across speakers)",
            "validation_utterance_id_count": len(validation_ids),
            "validation_utterance_ids_sha256": validation_ids_digest,
            "test_unit": "speaker",
            "held_out_speaker_usage": "test only",
        },
        "folds": folds,
    }
    write_immutable(output_dir / "folds_summary.json", _json_bytes(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable L2-ARCTIC Arabic leave-one-speaker-out manifests."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_manifests(
        source_manifest=args.source_manifest,
        validation_report=args.validation_report,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        namespace=args.namespace,
    )
    print(
        f"Created {len(summary['folds'])} immutable folds from "
        f"{summary['source_records']} validated recordings.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
