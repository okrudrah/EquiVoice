# Whisper fine-tuning setup

EquiVoice now has a tested training entry point for full fine-tuning of the
pinned English-only `openai/whisper-small.en` checkpoint. The setup is complete,
but no full adaptation run or held-out-speaker evaluation has been performed.

## Experimental boundary

Each run uses one frozen leave-one-speaker-out fold:

| Fold | Adaptation speakers | Sealed test speaker | Train | Validation | Test |
| --- | --- | --- | ---: | ---: | ---: |
| `held_out_aba` | SKA, YBAA, ZHAA | ABA | 2,944 | 292 | 1,129 |
| `held_out_ska` | ABA, YBAA, ZHAA | SKA | 3,085 | 306 | 974 |
| `held_out_ybaa` | ABA, SKA, ZHAA | YBAA | 2,943 | 292 | 1,130 |
| `held_out_zhaa` | ABA, SKA, YBAA | ZHAA | 2,941 | 292 | 1,132 |

Training code constructs datasets only for the train and validation rows. It
validates the test-speaker identity and count from metadata, but it does not read
test audio or transcripts. Held-out evaluation remains a separate future step.
The train/validation policy also keeps each prompt ID in only one of those two
splits, preventing prompt-text leakage.

Before model loading, the entry point requires exact agreement among the frozen
fold manifest, its content hash and counts, the processed-audio manifest, and the
text-free pretrained-baseline metrics. At data access time, it recomputes the
processed float32 PCM hash and transcript text hash. Any mismatch stops the run.

## Frozen configuration

The complete machine-readable configuration is
[`configs/whisper_small_en_loso.json`](../configs/whisper_small_en_loso.json).
The same configuration is to be used for all four folds before any held-out
adaptation result is inspected.

- Model: `openai/whisper-small.en`
- Revision: `e8727524f962ee844a7319d92be39ac1bd25655a`
- Method: full fine-tuning of every normally trainable model parameter
- Epochs: 5
- Learning rate: `1e-5`, linear schedule, 5% warmup
- Per-device train batch: 4
- Gradient accumulation: 4
- Effective batch: 16 on one GPU
- Per-device validation batch: 8
- Weight decay: 0.01
- Maximum gradient norm: 1.0
- Validation and checkpoint interval: 100 optimizer steps
- Retained checkpoints: 2
- Best checkpoint: lowest validation WER
- Decoding: deterministic greedy transcription, no timestamps, at most 128 new tokens
- Seed: 17
- Precision: float16 on CUDA; the smoke test uses float32 on MPS

Gradient checkpointing is enabled for full runs. The trainer records training
loss every 25 optimizer steps and validation loss/WER every 100 steps, so
overfitting can be monitored rather than inferred after the fact. The model's
fixed sinusoidal encoder positional embedding is the only parameter marked
non-trainable by the architecture; the entry point fails if it finds any other
frozen parameter.

These values are an a priori experiment configuration, not a claim that they are
optimal. In particular, the held-out speakers must not be used to tune them.

## Data collation and metric

Audio is loaded lazily from the verified 16 kHz mono float32 WAV collection.
The Whisper feature extractor produces log-Mel input features and an explicit
attention mask. Transcripts are tokenized without rewriting their corpus text.
The custom collator pads input features and labels independently, converts label
padding to `-100` for loss masking, and removes the duplicated decoder-start
token.

Validation predictions are decoded with the same Transformers
`EnglishTextNormalizer` used for the pretrained baseline. WER is computed from
corpus-level substitutions, deletions, and insertions rather than averaging
utterance WERs.

## Local smoke test

The setup was exercised locally on Apple MPS with two ABA-fold training examples,
two validation examples, and one optimizer step. The run completed a forward
pass, backward pass, optimizer update, generated validation transcripts, and WER
calculation. The smoke artifacts are local under `data/experiments/` and ignored
by Git.

The smoke loss and two-example WER are deliberately not reported as research
results. They are too small and were produced only to test the pipeline. No model
checkpoint is saved in smoke mode.

To repeat the mechanical check:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python -m equivoice.train_whisper_loso \
  --fold-manifest results/manifests/l2_arctic_v5_loso/fold_aba.csv \
  --output-dir data/experiments/fine_tuning/smoke_held_out_aba \
  --device mps \
  --smoke-test \
  --local-files-only
```

## Full-run command

Full runs are intentionally blocked on CPU and MPS. They should be performed on
a CUDA GPU in a recorded cloud environment, one fold at a time. For example:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
python -m equivoice.train_whisper_loso \
  --fold-manifest results/manifests/l2_arctic_v5_loso/fold_aba.csv \
  --output-dir data/experiments/fine_tuning/held_out_aba \
  --device cuda
```

Change both `fold_aba.csv` and `held_out_aba` consistently for the other three
folds. A full run saves periodic Trainer checkpoints, restores the checkpoint
with the lowest validation WER, and saves that model and processor under
`best_model/`. To resume an interrupted run, add:

```text
--resume-from-checkpoint data/experiments/fine_tuning/held_out_aba/checkpoint-N
```

Every output directory receives immutable run metadata containing the model
revision, fold identity, sealed test count and digest, artifact hashes, parameter
counts, device, and package versions. The `data/` boundary keeps checkpoints and
text-bearing local artifacts out of GitHub.

## What remains

1. Reproduce the environment on a suitable CUDA host and record its lock file.
2. Run all four folds with the frozen configuration, retaining validation logs
   and checkpoints.
3. Add a separate evaluation entry point that loads each best checkpoint and
   scores only its matching sealed speaker plus the fixed LibriSpeech control.
4. Compare pretrained and adapted predictions with paired statistics and report
   all folds, including negative or mixed outcomes.

The collator and validation-WER pattern follows Hugging Face's official
[Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper), and
checkpoint/evaluation behavior uses the documented
[`Seq2SeqTrainer`](https://huggingface.co/docs/transformers/main_classes/trainer)
interfaces.
