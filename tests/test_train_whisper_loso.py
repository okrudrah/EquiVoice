from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import equivoice.train_whisper_loso as training
from equivoice.train_whisper_loso import (
    DataCollatorSpeechSeq2SeqWithPadding,
    LosoSpeechDataset,
    TrainingExample,
    compute_wer,
    load_fold_data,
    load_training_config,
    select_device,
    validate_full_fine_tuning_parameters,
    write_immutable_json,
)
from equivoice.validate_l2_arctic import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_has_expected_effective_batch() -> None:
    config = load_training_config(ROOT / "configs/whisper_small_en_loso.json")

    assert config["model_revision"] == training.MODEL_REVISION
    assert config["training"]["effective_batch_size"] == 16
    assert config["training"]["best_model_metric"] == "wer"


def test_real_fold_keeps_held_out_speaker_sealed() -> None:
    fold = load_fold_data(
        ROOT / "results/manifests/l2_arctic_v5_loso/fold_aba.csv",
        ROOT / "results/manifests/l2_arctic_v5_loso/folds_summary.json",
        ROOT / "results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv",
        ROOT
        / "results/baseline/whisper_small_en/l2_arctic_v5_arabic_utterance_metrics.csv",
    )

    assert fold.fold_id == "held_out_aba"
    assert fold.held_out_speaker == "ABA"
    assert len(fold.train) == 2_944
    assert len(fold.validation) == 292
    assert fold.test_count == 1_129
    assert {example.speaker for example in fold.train} == {"SKA", "YBAA", "ZHAA"}
    assert {example.speaker for example in fold.validation} == {
        "SKA",
        "YBAA",
        "ZHAA",
    }
    assert all(example.speaker != "ABA" for example in fold.train + fold.validation)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_fold_loader_rejects_test_speaker_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "EXPECTED_FOLD_RECORDS", 7)
    fold_path = tmp_path / "fold_aba.csv"
    fold_fields = [
        "fold_id",
        "split",
        "speaker",
        "utterance_id",
        "transcript_path",
        "audio_pcm_sha256",
    ]
    fold_rows = [
        {
            "fold_id": "held_out_aba",
            "split": "test",
            "speaker": "ABA",
            "utterance_id": "test",
            "transcript_path": "test.txt",
            "audio_pcm_sha256": "a" * 64,
        }
    ]
    for split, suffix in (("train", "train"), ("validation", "validation")):
        for speaker in ("SKA", "YBAA", "ZHAA"):
            fold_rows.append(
                {
                    "fold_id": "held_out_aba",
                    "split": split,
                    "speaker": "ABA" if speaker == "SKA" and split == "train" else speaker,
                    "utterance_id": f"{speaker.lower()}_{suffix}",
                    "transcript_path": f"{speaker.lower()}_{suffix}.txt",
                    "audio_pcm_sha256": "a" * 64,
                }
            )
    _write_csv(fold_path, fold_fields, fold_rows)
    summary_path = tmp_path / "summary.json"
    counts = {split: sum(row["split"] == split for row in fold_rows) for split in training.ALLOWED_SPLITS}
    summary_path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "manifest": fold_path.name,
                        "manifest_sha256": sha256_file(fold_path),
                        "splits": {
                            split: {"utterances": count}
                            for split, count in counts.items()
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    processed_path = tmp_path / "processed.csv"
    baseline_path = tmp_path / "baseline.csv"
    processed_fields = sorted(training.PROCESSED_REQUIRED_FIELDS)
    baseline_fields = sorted(training.BASELINE_REQUIRED_FIELDS)
    processed_rows = []
    baseline_rows = []
    for row in fold_rows:
        processed_rows.append(
            {
                "speaker": row["speaker"],
                "utterance_id": row["utterance_id"],
                "source_audio_pcm_sha256": "a" * 64,
                "processed_wav_path": "unused.wav",
                "processed_sample_rate_hz": "16000",
                "processed_channels": "1",
                "processed_format": "WAV",
                "processed_subtype": "FLOAT",
                "processed_pcm_sha256": "b" * 64,
            }
        )
        baseline_rows.append(
            {
                "speaker": row["speaker"],
                "utterance_id": row["utterance_id"],
                "processed_pcm_sha256": "b" * 64,
                "reference_sha256": "c" * 64,
            }
        )
    _write_csv(processed_path, processed_fields, processed_rows)
    _write_csv(baseline_path, baseline_fields, baseline_rows)

    with pytest.raises(ValueError, match="held-out speaker leaked"):
        load_fold_data(fold_path, summary_path, processed_path, baseline_path)


class _FakeFeatureExtractor:
    def __call__(self, samples, sampling_rate, return_attention_mask):
        assert sampling_rate == 16_000
        assert return_attention_mask is True
        return SimpleNamespace(
            input_features=[np.asarray(samples[:4])],
            attention_mask=[np.ones(4, dtype=np.int64)],
        )

    def pad(self, features, return_tensors):
        assert return_tensors == "pt"
        return {
            "input_features": torch.tensor(
                [feature["input_features"] for feature in features]
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] for feature in features]
            ),
        }


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text):
        return SimpleNamespace(input_ids=[1, len(text), 2])

    def pad(self, features, return_tensors):
        assert return_tensors == "pt"
        width = max(len(feature["input_ids"]) for feature in features)
        ids = []
        masks = []
        for feature in features:
            padding = width - len(feature["input_ids"])
            ids.append(feature["input_ids"] + [0] * padding)
            masks.append([1] * len(feature["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


class _FakeProcessor:
    feature_extractor = _FakeFeatureExtractor()
    tokenizer = _FakeTokenizer()


def test_dataset_validates_audio_and_transcript_hashes(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    raw_root = tmp_path / "raw"
    processed_root.mkdir()
    raw_root.mkdir()
    samples = np.asarray([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
    sf.write(processed_root / "sample.wav", samples, 16_000, subtype="FLOAT")
    transcript = "A short sentence.\n"
    (raw_root / "sample.txt").write_text(transcript, encoding="utf-8")
    example = TrainingExample(
        split="train",
        speaker="SKA",
        utterance_id="sample",
        transcript_path="sample.txt",
        processed_wav_path="sample.wav",
        processed_pcm_sha256=training._float32_sha256(samples),
        reference_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
    )

    item = LosoSpeechDataset(
        [example], processed_root, raw_root, _FakeProcessor()
    )[0]

    assert list(item) == ["input_features", "attention_mask", "labels"]
    assert item["labels"] == [1, len(transcript), 2]


def test_collator_masks_padding_and_removes_decoder_start() -> None:
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=_FakeProcessor(), decoder_start_token_id=1
    )
    batch = collator(
        [
            {
                "input_features": [0.0, 1.0],
                "attention_mask": [1, 1],
                "labels": [1, 8, 2],
            },
            {
                "input_features": [2.0, 3.0],
                "attention_mask": [1, 1],
                "labels": [1, 9],
            },
        ]
    )

    assert batch["labels"].tolist() == [[8, 2], [9, -100]]


def test_compute_wer_uses_corpus_level_counts() -> None:
    wer = compute_wer(
        ["one two three", "four"],
        ["one two", "five"],
        lambda text: text.lower(),
    )

    assert wer == pytest.approx(0.5)


def test_full_training_is_restricted_to_cuda() -> None:
    with pytest.raises(ValueError, match="restricted to a CUDA"):
        select_device("cpu", smoke_test=False)


def test_full_fine_tuning_rejects_unexpected_frozen_parameter() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model[0].weight.requires_grad = False

    with pytest.raises(ValueError, match="unexpected frozen parameters"):
        validate_full_fine_tuning_parameters(model)


def test_immutable_metadata_rejects_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_immutable_json(path, {"fold": "held_out_aba"})

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_immutable_json(path, {"fold": "held_out_ska"})
