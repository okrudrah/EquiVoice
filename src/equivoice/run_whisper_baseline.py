"""Run a reproducible pretrained Whisper baseline on processed L2-ARCTIC audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import jiwer
import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from equivoice.validate_l2_arctic import EXPECTED_SPEAKERS, sha256_file


MODEL_ID = "openai/whisper-small.en"
MODEL_REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_SEED = 17
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 128

PROCESSED_REQUIRED_FIELDS = {
    "speaker",
    "utterance_id",
    "transcript_path",
    "processed_wav_path",
    "processed_sample_rate_hz",
    "processed_channels",
    "processed_format",
    "processed_subtype",
    "processed_pcm_sha256",
}

PREDICTION_FIELDS = [
    "speaker",
    "utterance_id",
    "processed_wav_path",
    "processed_pcm_sha256",
    "reference_text",
    "hypothesis_text",
    "normalized_reference",
    "normalized_hypothesis",
    "reference_sha256",
    "hypothesis_sha256",
    "reference_words",
    "hypothesis_words",
    "hits",
    "substitutions",
    "deletions",
    "insertions",
    "errors",
    "wer",
]

PUBLIC_METRIC_FIELDS = [
    "speaker",
    "utterance_id",
    "processed_pcm_sha256",
    "reference_sha256",
    "hypothesis_sha256",
    "reference_words",
    "hypothesis_words",
    "hits",
    "substitutions",
    "deletions",
    "insertions",
    "errors",
    "wer",
]


def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device: {requested}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_processed_manifest(
    path: Path,
    expected_speakers: Iterable[str] | None = EXPECTED_SPEAKERS,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = PROCESSED_REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(
                f"processed manifest is missing fields: {', '.join(sorted(missing))}"
            )
        records = list(reader)

    allowed_speakers = set(expected_speakers) if expected_speakers is not None else None
    seen: set[tuple[str, str]] = set()
    for row_number, record in enumerate(records, start=2):
        key = (record["speaker"], record["utterance_id"])
        if key in seen:
            raise ValueError(f"duplicate processed row {row_number}: {key}")
        seen.add(key)
        if allowed_speakers is not None and record["speaker"] not in allowed_speakers:
            raise ValueError(f"unexpected speaker at row {row_number}: {key[0]}")
        if int(record["processed_sample_rate_hz"]) != TARGET_SAMPLE_RATE:
            raise ValueError(f"unexpected sample rate at row {row_number}")
        if int(record["processed_channels"]) != 1:
            raise ValueError(f"unexpected channel count at row {row_number}")
        if record["processed_format"] != "WAV":
            raise ValueError(f"unexpected format at row {row_number}")
        if record["processed_subtype"] != "FLOAT":
            raise ValueError(f"unexpected subtype at row {row_number}")
    return records


def _float32_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def score_pair(
    reference: str,
    hypothesis: str,
    normalizer: Callable[[str], str],
) -> dict[str, Any]:
    normalized_reference = normalizer(reference).strip()
    normalized_hypothesis = normalizer(hypothesis).strip()
    if not normalized_reference:
        raise ValueError("reference became empty after normalization")

    output = jiwer.process_words(normalized_reference, normalized_hypothesis)
    errors = output.substitutions + output.deletions + output.insertions
    reference_words = output.hits + output.substitutions + output.deletions
    hypothesis_words = output.hits + output.substitutions + output.insertions
    return {
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "errors": errors,
        "wer": errors / reference_words,
    }


def _csv_bytes(rows: Iterable[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite changed result: {path}")
        return
    path.write_bytes(content)


def load_existing_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError("existing prediction file has an unexpected schema")
        rows = list(reader)
    keys = [(row["speaker"], row["utterance_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("existing prediction file contains duplicate rows")
    return rows


def build_run_config(
    device: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    processed_manifest: Path,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_language": "English-only",
        "language_detection": False,
        "task": "transcription",
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "return_timestamps": False,
        },
        "normalizer": "Transformers Whisper EnglishTextNormalizer",
        "sampling_rate_hz": TARGET_SAMPLE_RATE,
        "dtype": "float32",
        "attention_implementation": "eager",
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "processed_manifest": processed_manifest.name,
        "processed_manifest_sha256": sha256_file(processed_manifest),
        "record_limit": limit,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def load_model_and_processor(cache_dir: Path, local_files_only: bool):
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_safetensors=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    return processor, model


def _read_batch_inputs(
    records: list[dict[str, str]],
    processed_root: Path,
    raw_root: Path,
) -> tuple[list[np.ndarray], list[str]]:
    audio: list[np.ndarray] = []
    references: list[str] = []
    for record in records:
        audio_path = processed_root / record["processed_wav_path"]
        info = sf.info(audio_path)
        samples, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=False
        )
        if (
            sample_rate != TARGET_SAMPLE_RATE
            or info.samplerate != TARGET_SAMPLE_RATE
            or info.channels != 1
            or info.format != "WAV"
            or info.subtype != "FLOAT"
            or samples.ndim != 1
        ):
            raise ValueError(f"invalid processed audio: {audio_path}")
        if _float32_sha256(samples) != record["processed_pcm_sha256"]:
            raise ValueError(f"processed audio hash mismatch: {audio_path}")
        transcript_path = raw_root / record["transcript_path"]
        reference = transcript_path.read_text(encoding="utf-8-sig")
        if not reference.strip():
            raise ValueError(f"empty reference transcript: {transcript_path}")
        audio.append(samples)
        references.append(reference)
    return audio, references


def transcribe_batch(
    records: list[dict[str, str]],
    processed_root: Path,
    raw_root: Path,
    processor,
    model,
    device: str,
    max_new_tokens: int,
    normalizer: Callable[[str], str],
) -> list[dict[str, Any]]:
    audio, references = _read_batch_inputs(records, processed_root, raw_root)
    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device=device, dtype=torch.float32)
    attention_mask = inputs.attention_mask.to(device=device)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            return_timestamps=False,
        )
    hypotheses = processor.batch_decode(
        generated_ids.detach().cpu(), skip_special_tokens=True
    )

    rows: list[dict[str, Any]] = []
    for record, reference, hypothesis in zip(
        records, references, hypotheses, strict=True
    ):
        metrics = score_pair(reference, hypothesis, normalizer)
        rows.append(
            {
                "speaker": record["speaker"],
                "utterance_id": record["utterance_id"],
                "processed_wav_path": record["processed_wav_path"],
                "processed_pcm_sha256": record["processed_pcm_sha256"],
                "reference_text": reference,
                "hypothesis_text": hypothesis,
                "normalized_reference": metrics["normalized_reference"],
                "normalized_hypothesis": metrics["normalized_hypothesis"],
                "reference_sha256": hashlib.sha256(
                    reference.encode("utf-8")
                ).hexdigest(),
                "hypothesis_sha256": hashlib.sha256(
                    hypothesis.encode("utf-8")
                ).hexdigest(),
                "reference_words": metrics["reference_words"],
                "hypothesis_words": metrics["hypothesis_words"],
                "hits": metrics["hits"],
                "substitutions": metrics["substitutions"],
                "deletions": metrics["deletions"],
                "insertions": metrics["insertions"],
                "errors": metrics["errors"],
                "wer": round(metrics["wer"], 12),
            }
        )
    return rows


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_words = sum(int(row["reference_words"]) for row in rows)
    hits = sum(int(row["hits"]) for row in rows)
    substitutions = sum(int(row["substitutions"]) for row in rows)
    deletions = sum(int(row["deletions"]) for row in rows)
    insertions = sum(int(row["insertions"]) for row in rows)
    errors = substitutions + deletions + insertions
    return {
        "utterances": len(rows),
        "reference_words": reference_words,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "wer": round(errors / reference_words, 12),
    }


def build_public_results(
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    predictions_path: Path,
    limited: bool,
    *,
    speaker_order: Iterable[str] | None = EXPECTED_SPEAKERS,
    dataset: str = "L2-ARCTIC",
    release: str = "v5.0",
    full_scope: str = "full Arabic cohort",
    evaluation: str = "pretrained baseline",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    ordered_speakers = (
        list(speaker_order) if speaker_order is not None else sorted(by_speaker)
    )
    speakers = [
        {"speaker": speaker, **_metric_summary(by_speaker[speaker])}
        for speaker in ordered_speakers
        if by_speaker.get(speaker)
    ]
    aggregate = _metric_summary(rows)
    aggregate["macro_speaker_wer"] = round(
        sum(item["wer"] for item in speakers) / len(speakers), 12
    )
    public_rows = [
        {field: row[field] for field in PUBLIC_METRIC_FIELDS} for row in rows
    ]
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "limited smoke test" if limited else full_scope,
        "dataset": dataset,
        "release": release,
        "evaluation": evaluation,
        "run_config": run_config,
        "aggregate": aggregate,
        "speakers": speakers,
        "private_predictions_file": predictions_path.name,
        "private_predictions_sha256": sha256_file(predictions_path),
        "public_utterance_metrics_contains_text": False,
    }
    return report, public_rows


def run_baseline(
    processed_root: Path,
    raw_root: Path,
    processed_manifest: Path,
    model_cache: Path,
    predictions_path: Path,
    public_metrics_path: Path,
    report_path: Path,
    device_request: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    limit: int | None,
    local_files_only: bool,
    expected_speakers: Iterable[str] | None = EXPECTED_SPEAKERS,
    dataset: str = "L2-ARCTIC",
    release: str = "v5.0",
    full_scope: str = "full Arabic cohort",
    evaluation: str = "pretrained baseline",
    run_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    records = load_processed_manifest(processed_manifest, expected_speakers)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    device = select_device(device_request)
    run_config = build_run_config(
        device, batch_size, max_new_tokens, seed, processed_manifest, limit
    )
    if run_config_overrides:
        overlap = set(run_config) & set(run_config_overrides)
        if overlap:
            raise ValueError(
                "run-config overrides conflict with standard fields: "
                + ", ".join(sorted(overlap))
            )
        run_config.update(run_config_overrides)
    config_path = predictions_path.with_suffix(".config.json")
    write_immutable(config_path, _json_bytes(run_config))

    existing_rows = load_existing_predictions(predictions_path)
    expected_keys = {(row["speaker"], row["utterance_id"]) for row in records}
    existing_by_key = {
        (row["speaker"], row["utterance_id"]): row for row in existing_rows
    }
    if not set(existing_by_key).issubset(expected_keys):
        raise ValueError("existing predictions contain rows outside this run")
    record_by_key = {
        (record["speaker"], record["utterance_id"]): record for record in records
    }
    for key, row in existing_by_key.items():
        if row["processed_pcm_sha256"] != record_by_key[key]["processed_pcm_sha256"]:
            raise ValueError(f"existing prediction audio hash mismatch: {key}")

    pending = [
        record
        for record in records
        if (record["speaker"], record["utterance_id"]) not in existing_by_key
    ]
    if pending:
        set_inference_seed(seed)
        print(f"Loading {MODEL_ID} on {device}...", flush=True)
        processor, model = load_model_and_processor(model_cache, local_files_only)
        model.to(device)
        model.eval()
        normalizer = EnglishTextNormalizer(
            processor.tokenizer.english_spelling_normalizer
        )

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            new_rows = transcribe_batch(
                batch,
                processed_root,
                raw_root,
                processor,
                model,
                device,
                max_new_tokens,
                normalizer,
            )
            for row in new_rows:
                existing_by_key[(row["speaker"], row["utterance_id"])] = row
            ordered_partial = [
                existing_by_key[(record["speaker"], record["utterance_id"])]
                for record in records
                if (record["speaker"], record["utterance_id"]) in existing_by_key
            ]
            write_atomic(
                predictions_path,
                _csv_bytes(ordered_partial, PREDICTION_FIELDS),
            )
            completed = min(start + len(batch), len(pending))
            print(
                f"Generated {completed}/{len(pending)} pending predictions "
                f"({len(existing_by_key)}/{len(records)} total)...",
                flush=True,
            )

    missing_keys = expected_keys - set(existing_by_key)
    if missing_keys:
        raise ValueError(f"missing predictions after inference: {len(missing_keys)}")
    ordered_rows = [
        existing_by_key[(record["speaker"], record["utterance_id"])]
        for record in records
    ]
    write_atomic(predictions_path, _csv_bytes(ordered_rows, PREDICTION_FIELDS))
    report, public_rows = build_public_results(
        ordered_rows,
        run_config,
        predictions_path,
        limited=limit is not None,
        speaker_order=expected_speakers,
        dataset=dataset,
        release=release,
        full_scope=full_scope,
        evaluation=evaluation,
    )
    write_immutable(
        public_metrics_path,
        _csv_bytes(public_rows, PUBLIC_METRIC_FIELDS),
    )
    public_metrics_sha = sha256_file(public_metrics_path)
    report["public_utterance_metrics"] = public_metrics_path.name
    report["public_utterance_metrics_sha256"] = public_metrics_sha
    write_immutable(report_path, _json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned pretrained Whisper Small English baseline."
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public-metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_baseline(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        processed_manifest=args.processed_manifest,
        model_cache=args.model_cache,
        predictions_path=args.predictions,
        public_metrics_path=args.public_metrics,
        report_path=args.report,
        device_request=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )
    print(
        f"Baseline {report['status']}: "
        f"{report['aggregate']['utterances']} utterances, "
        f"WER={report['aggregate']['wer']:.6f}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
