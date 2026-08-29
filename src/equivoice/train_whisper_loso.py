"""Set up and run leakage-safe Whisper fine-tuning for one LOSO fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import accelerate
import jiwer
import numpy as np
import soundfile as sf
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from equivoice.run_whisper_baseline import _float32_sha256
from equivoice.validate_l2_arctic import EXPECTED_SPEAKERS, sha256_file


MODEL_ID = "openai/whisper-small.en"
MODEL_REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"
TARGET_SAMPLE_RATE = 16_000
EXPECTED_FOLD_RECORDS = 4_365
ALLOWED_SPLITS = {"train", "validation", "test"}
EXPECTED_ARCHITECTURAL_FROZEN_PARAMETERS = {
    "model.encoder.embed_positions.weight",
}

FOLD_REQUIRED_FIELDS = {
    "fold_id",
    "split",
    "speaker",
    "utterance_id",
    "transcript_path",
    "audio_pcm_sha256",
}
PROCESSED_REQUIRED_FIELDS = {
    "speaker",
    "utterance_id",
    "source_audio_pcm_sha256",
    "processed_wav_path",
    "processed_sample_rate_hz",
    "processed_channels",
    "processed_format",
    "processed_subtype",
    "processed_pcm_sha256",
}
BASELINE_REQUIRED_FIELDS = {
    "speaker",
    "utterance_id",
    "processed_pcm_sha256",
    "reference_sha256",
}


@dataclass(frozen=True)
class TrainingExample:
    split: str
    speaker: str
    utterance_id: str
    transcript_path: str
    processed_wav_path: str
    processed_pcm_sha256: str
    reference_sha256: str


@dataclass(frozen=True)
class FoldData:
    fold_id: str
    held_out_speaker: str
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test_count: int
    test_utterance_digest: str


def _load_csv(path: Path, required_fields: set[str], label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _index_unique(
    rows: Iterable[dict[str, str]], label: str
) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["speaker"], row["utterance_id"])
        if key in index:
            raise ValueError(f"duplicate {label} record: {key}")
        index[key] = row
    return index


def load_training_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported training-config schema")
    if config.get("model_id") != MODEL_ID:
        raise ValueError(f"training config must use {MODEL_ID}")
    if config.get("model_revision") != MODEL_REVISION:
        raise ValueError("training config does not pin the approved model revision")
    if config.get("sampling_rate_hz") != TARGET_SAMPLE_RATE:
        raise ValueError("training config must use 16 kHz audio")

    training = config.get("training", {})
    required_positive = (
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "learning_rate",
        "evaluation_steps",
        "checkpoint_steps",
    )
    if any(float(training.get(key, 0)) <= 0 for key in required_positive):
        raise ValueError("training config contains a nonpositive required value")
    calculated_batch = (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
    )
    if int(training["effective_batch_size"]) != calculated_batch:
        raise ValueError("effective batch size does not match batch x accumulation")
    if training.get("evaluation_strategy") != "steps":
        raise ValueError("validation must run at fixed optimizer-step intervals")
    if training.get("checkpoint_strategy") != "steps":
        raise ValueError("checkpointing must run at fixed optimizer-step intervals")
    if training.get("best_model_metric") != "wer":
        raise ValueError("best checkpoint must be selected by validation WER")
    if not training.get("load_best_model_at_end"):
        raise ValueError("best validation checkpoint must be restored at the end")
    if config.get("decoding", {}).get("strategy") != "greedy":
        raise ValueError("the frozen comparison uses greedy decoding")
    return config


def _validate_fold_summary(
    fold_manifest: Path, summary_path: Path, rows: Sequence[dict[str, str]]
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matching = [
        fold
        for fold in summary.get("folds", [])
        if fold.get("manifest") == fold_manifest.name
    ]
    if len(matching) != 1:
        raise ValueError("fold summary does not identify the selected manifest once")
    expected_hash = matching[0].get("manifest_sha256")
    if expected_hash != sha256_file(fold_manifest):
        raise ValueError("fold manifest hash differs from the frozen summary")
    split_counts = Counter(row["split"] for row in rows)
    for split in ALLOWED_SPLITS:
        expected = matching[0].get("splits", {}).get(split, {}).get("utterances")
        if expected != split_counts[split]:
            raise ValueError(f"{split} count differs from the frozen fold summary")


def load_fold_data(
    fold_manifest: Path,
    fold_summary: Path,
    processed_manifest: Path,
    baseline_metrics: Path,
) -> FoldData:
    """Join immutable metadata while enforcing the speaker/test boundary."""

    fold_rows = _load_csv(fold_manifest, FOLD_REQUIRED_FIELDS, "fold manifest")
    processed_rows = _load_csv(
        processed_manifest, PROCESSED_REQUIRED_FIELDS, "processed manifest"
    )
    baseline_rows = _load_csv(
        baseline_metrics, BASELINE_REQUIRED_FIELDS, "baseline metrics"
    )
    if len(fold_rows) != EXPECTED_FOLD_RECORDS:
        raise ValueError(
            f"fold has {len(fold_rows)} records; expected {EXPECTED_FOLD_RECORDS}"
        )
    _validate_fold_summary(fold_manifest, fold_summary, fold_rows)

    fold_ids = {row["fold_id"] for row in fold_rows}
    if len(fold_ids) != 1:
        raise ValueError("fold manifest contains multiple fold IDs")
    unknown_splits = {row["split"] for row in fold_rows} - ALLOWED_SPLITS
    if unknown_splits:
        raise ValueError(f"fold contains unknown splits: {sorted(unknown_splits)}")
    fold_index = _index_unique(fold_rows, "fold")
    processed_index = _index_unique(processed_rows, "processed")
    baseline_index = _index_unique(baseline_rows, "baseline")
    if set(fold_index) != set(processed_index) or set(fold_index) != set(
        baseline_index
    ):
        raise ValueError("fold, processed, and baseline utterance sets differ")

    test_speakers = {row["speaker"] for row in fold_rows if row["split"] == "test"}
    if len(test_speakers) != 1:
        raise ValueError("test split must contain exactly one speaker")
    held_out_speaker = next(iter(test_speakers))
    if held_out_speaker not in set(EXPECTED_SPEAKERS):
        raise ValueError("test split contains an unexpected speaker")
    adaptation_rows = [row for row in fold_rows if row["split"] != "test"]
    if any(row["speaker"] == held_out_speaker for row in adaptation_rows):
        raise ValueError("held-out speaker leaked into train or validation")
    expected_adaptation_speakers = set(EXPECTED_SPEAKERS) - {held_out_speaker}
    for split in ("train", "validation"):
        speakers = {row["speaker"] for row in fold_rows if row["split"] == split}
        if speakers != expected_adaptation_speakers:
            raise ValueError(f"{split} does not contain the three adaptation speakers")
    train_prompts = {
        row["utterance_id"] for row in fold_rows if row["split"] == "train"
    }
    validation_prompts = {
        row["utterance_id"] for row in fold_rows if row["split"] == "validation"
    }
    if train_prompts & validation_prompts:
        raise ValueError("prompt leakage exists between train and validation")

    examples: dict[str, list[TrainingExample]] = {
        "train": [],
        "validation": [],
    }
    for key, fold_row in fold_index.items():
        processed = processed_index[key]
        baseline = baseline_index[key]
        if fold_row["audio_pcm_sha256"] != processed["source_audio_pcm_sha256"]:
            raise ValueError(f"source audio hash mismatch in metadata: {key}")
        if processed["processed_pcm_sha256"] != baseline["processed_pcm_sha256"]:
            raise ValueError(f"processed audio hash mismatch in metadata: {key}")
        if int(processed["processed_sample_rate_hz"]) != TARGET_SAMPLE_RATE:
            raise ValueError(f"processed audio is not 16 kHz: {key}")
        if (
            processed["processed_channels"] != "1"
            or processed["processed_format"] != "WAV"
            or processed["processed_subtype"] != "FLOAT"
        ):
            raise ValueError(f"processed audio has an unexpected format: {key}")
        if fold_row["split"] in examples:
            examples[fold_row["split"]].append(
                TrainingExample(
                    split=fold_row["split"],
                    speaker=fold_row["speaker"],
                    utterance_id=fold_row["utterance_id"],
                    transcript_path=fold_row["transcript_path"],
                    processed_wav_path=processed["processed_wav_path"],
                    processed_pcm_sha256=processed["processed_pcm_sha256"],
                    reference_sha256=baseline["reference_sha256"],
                )
            )

    test_ids = sorted(
        row["utterance_id"] for row in fold_rows if row["split"] == "test"
    )
    test_digest = hashlib.sha256(("\n".join(test_ids) + "\n").encode()).hexdigest()
    return FoldData(
        fold_id=next(iter(fold_ids)),
        held_out_speaker=held_out_speaker,
        train=tuple(examples["train"]),
        validation=tuple(examples["validation"]),
        test_count=len(test_ids),
        test_utterance_digest=test_digest,
    )


class LosoSpeechDataset(Dataset):
    """Lazy, hash-validating access to adaptation audio and transcripts."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        processed_root: Path,
        raw_root: Path,
        processor: Any,
    ) -> None:
        self.examples = tuple(examples)
        self.processed_root = processed_root
        self.raw_root = raw_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        audio_path = self.processed_root / example.processed_wav_path
        info = sf.info(audio_path)
        samples, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=False
        )
        if (
            sample_rate != TARGET_SAMPLE_RATE
            or info.channels != 1
            or info.format != "WAV"
            or info.subtype != "FLOAT"
            or samples.ndim != 1
        ):
            raise ValueError(f"invalid processed audio: {audio_path}")
        if _float32_sha256(samples) != example.processed_pcm_sha256:
            raise ValueError(f"processed audio hash mismatch: {audio_path}")

        transcript_path = self.raw_root / example.transcript_path
        transcript = transcript_path.read_text(encoding="utf-8-sig")
        transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        if transcript_hash != example.reference_sha256:
            raise ValueError(f"transcript hash mismatch: {transcript_path}")
        if not transcript.strip():
            raise ValueError(f"empty transcript: {transcript_path}")

        extracted = self.processor.feature_extractor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer(transcript).input_ids
        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalizer: Callable[[str], str],
) -> float:
    normalized_references = [normalizer(text).strip() for text in references]
    normalized_hypotheses = [normalizer(text).strip() for text in hypotheses]
    if any(not text for text in normalized_references):
        raise ValueError("a validation reference became empty after normalization")
    output = jiwer.process_words(normalized_references, normalized_hypotheses)
    errors = output.substitutions + output.deletions + output.insertions
    reference_words = output.hits + output.substitutions + output.deletions
    return errors / reference_words


def make_compute_metrics(processor: Any):
    normalizer = EnglishTextNormalizer({})

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hypotheses = processor.batch_decode(prediction_ids, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(references, hypotheses, normalizer)}

    return compute_metrics


def select_device(requested: str, smoke_test: bool) -> str:
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if not smoke_test and requested != "cuda":
        raise ValueError("full runs are intentionally restricted to a CUDA environment")
    return requested


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_full_fine_tuning_parameters(model: Any) -> dict[str, int]:
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if frozen != EXPECTED_ARCHITECTURAL_FROZEN_PARAMETERS:
        raise ValueError(
            "unexpected frozen parameters for full fine-tuning: "
            f"expected {sorted(EXPECTED_ARCHITECTURAL_FROZEN_PARAMETERS)}, "
            f"found {sorted(frozen)}"
        )
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "architectural_frozen": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def load_model_and_processor(cache_dir: Path, local_files_only: bool):
    source: str | Path = MODEL_ID
    if local_files_only:
        snapshot = (
            cache_dir
            / "models--openai--whisper-small.en"
            / "snapshots"
            / MODEL_REVISION
        )
        if not snapshot.is_dir():
            raise ValueError(f"pinned local model snapshot is missing: {snapshot}")
        source = snapshot
    processor = AutoProcessor.from_pretrained(
        source,
        revision=None if local_files_only else MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source,
        revision=None if local_files_only else MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_safetensors=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.return_timestamps = False
    return processor, model


def build_training_arguments(
    output_dir: Path,
    config: dict[str, Any],
    device: str,
    smoke_test: bool,
) -> Seq2SeqTrainingArguments:
    training = config["training"]
    common = {
        "output_dir": str(output_dir),
        "report_to": [],
        "push_to_hub": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "predict_with_generate": True,
        "generation_num_beams": int(config["decoding"]["num_beams"]),
        "use_cpu": device == "cpu",
        "fp16": device == "cuda",
        "dataloader_pin_memory": device == "cuda",
    }
    if smoke_test:
        smoke = config["smoke_test"]
        return Seq2SeqTrainingArguments(
            **common,
            max_steps=int(smoke["max_optimizer_steps"]),
            per_device_train_batch_size=int(smoke["per_device_batch_size"]),
            per_device_eval_batch_size=int(smoke["per_device_batch_size"]),
            gradient_accumulation_steps=1,
            learning_rate=float(training["learning_rate"]),
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            gradient_checkpointing=False,
        )
    return Seq2SeqTrainingArguments(
        **common,
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        max_grad_norm=float(training["max_grad_norm"]),
        optim=training["optimizer"],
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        eval_strategy=training["evaluation_strategy"],
        eval_steps=int(training["evaluation_steps"]),
        save_strategy=training["checkpoint_strategy"],
        save_steps=int(training["checkpoint_steps"]),
        save_total_limit=int(training["checkpoint_limit"]),
        logging_strategy="steps",
        logging_steps=int(training["logging_steps"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
        metric_for_best_model=training["best_model_metric"],
        greater_is_better=bool(training["greater_is_better"]),
    )


def build_run_metadata(
    config_path: Path,
    fold_manifest: Path,
    fold_summary: Path,
    processed_manifest: Path,
    baseline_metrics: Path,
    fold_data: FoldData,
    device: str,
    smoke_test: bool,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "setup" if not smoke_test else "smoke_test",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fine_tuning_method": "full_fine_tuning",
        "fold_id": fold_data.fold_id,
        "held_out_test_speaker": fold_data.held_out_speaker,
        "held_out_test_records_sealed": fold_data.test_count,
        "held_out_test_utterance_digest": fold_data.test_utterance_digest,
        "train_records_used": train_count,
        "validation_records_used": validation_count,
        "smoke_test": smoke_test,
        "device": device,
        "artifact_hashes": {
            "training_config": sha256_file(config_path),
            "fold_manifest": sha256_file(fold_manifest),
            "fold_summary": sha256_file(fold_summary),
            "processed_manifest": sha256_file(processed_manifest),
            "baseline_metrics": sha256_file(baseline_metrics),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
        },
    }


def write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite changed run metadata: {path}")
    if not path.exists():
        path.write_bytes(content)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(args.config)
    device = select_device(args.device, args.smoke_test)
    set_training_seed(int(config["seed"]))
    fold_data = load_fold_data(
        args.fold_manifest,
        args.fold_summary,
        args.processed_manifest,
        args.baseline_metrics,
    )
    processor, model = load_model_and_processor(args.cache_dir, args.local_files_only)
    parameter_counts = validate_full_fine_tuning_parameters(model)
    model.generation_config.max_new_tokens = int(
        config["decoding"]["max_new_tokens"]
    )

    train_examples = fold_data.train
    validation_examples = fold_data.validation
    if args.smoke_test:
        train_examples = train_examples[: int(config["smoke_test"]["train_examples"])]
        validation_examples = validation_examples[
            : int(config["smoke_test"]["validation_examples"])
        ]
    train_dataset = LosoSpeechDataset(
        train_examples, args.processed_root, args.raw_root, processor
    )
    validation_dataset = LosoSpeechDataset(
        validation_examples, args.processed_root, args.raw_root, processor
    )

    metadata = build_run_metadata(
        args.config,
        args.fold_manifest,
        args.fold_summary,
        args.processed_manifest,
        args.baseline_metrics,
        fold_data,
        device,
        args.smoke_test,
        len(train_dataset),
        len(validation_dataset),
    )
    metadata["model_parameters"] = parameter_counts
    write_immutable_json(args.output_dir / "run_metadata.json", metadata)

    training_args = build_training_arguments(
        args.output_dir, config, device, args.smoke_test
    )
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    report = {
        **metadata,
        "status": "completed",
        "train_metrics": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float, str, bool))
        },
        "validation_metrics": {
            key: value
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float, str, bool))
        },
    }
    report_name = "smoke_report.json" if args.smoke_test else "training_report.json"
    write_immutable_json(args.output_dir / report_name, report)
    if not args.smoke_test:
        best_model_dir = args.output_dir / "best_model"
        trainer.save_model(best_model_dir)
        processor.save_pretrained(best_model_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/whisper_small_en_loso.json"),
    )
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("results/manifests/l2_arctic_v5_loso/folds_summary.json"),
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=Path(
            "results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv"
        ),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path(
            "results/baseline/whisper_small_en/"
            "l2_arctic_v5_arabic_utterance_metrics.csv"
        ),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw/l2_arctic/v5.0")
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            "data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/models/huggingface")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
