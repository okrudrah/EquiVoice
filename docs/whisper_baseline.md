# Pretrained Whisper baseline

This milestone measures an unadapted automatic-speech-recognition baseline for
the four verified Arabic L1 speakers in L2-ARCTIC v5.0. It does not train or
fine-tune a model.

## Why this checkpoint

The baseline uses [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en)
at immutable Hugging Face revision
`e8727524f962ee844a7319d92be39ac1bd25655a`. Whisper Small is large enough to
provide a meaningful pretrained reference while remaining practical for local
inference on the project's 16 GB Apple Silicon machine. The `.en` checkpoint is
English-only, so the experiment does not perform unnecessary language
detection on English L2 speech.

The model runs in float32 on Apple's MPS backend with eager attention. Decoding
is deterministic greedy transcription: sampling disabled, one beam, no
timestamps, and at most 128 new tokens. The seed is 17 and the batch size is 8.
The exact model revision, library versions, manifest digest, and decoding
configuration are recorded in the machine-readable report.

## Evaluation scope

The baseline covers all 4,365 validated recordings from ABA, SKA, YBAA, and
ZHAA. Because the pretrained model is identical in every future
leave-one-speaker-out fold, one full pass is equivalent to evaluating that
unchanged checkpoint on each fold's held-out speaker. Later adapted models must
be compared with the matching speaker row below; no training data are used in
this baseline.

Before inference, every input is checked against the preprocessing manifest for
16 kHz, mono, float32 WAV structure and decoded-audio SHA-256. Reference and
hypothesis strings are normalized with Transformers' Whisper
`EnglishTextNormalizer`. Word error rate is computed as
`(substitutions + deletions + insertions) / reference words`.

## Measured results

| Scope | Utterances | Reference words | Errors | WER |
|---|---:|---:|---:|---:|
| ABA | 1,129 | 10,138 | 1,245 | 12.2805% |
| SKA | 974 | 8,754 | 1,868 | 21.3388% |
| YBAA | 1,130 | 10,134 | 1,015 | 10.0158% |
| ZHAA | 1,132 | 10,158 | 1,069 | 10.5237% |
| **Micro aggregate** | **4,365** | **39,184** | **5,197** | **13.2631%** |

The unweighted macro average across the four speaker WERs is **13.5397%**.
SKA's higher baseline WER shows substantial speaker variation, but this result
alone does not identify its linguistic or acoustic cause. Pronunciation-linked
analysis remains a separate later milestone.

## Reproduction

The model must first be present in the ignored local cache. With the Python 3.12
environment active, the verified full run is:

```bash
PYTHONPATH=src python -m equivoice.run_whisper_baseline \
  --processed-root data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq \
  --raw-root data/raw/l2_arctic/v5.0 \
  --processed-manifest results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv \
  --model-cache data/models/huggingface \
  --predictions data/experiments/baseline/whisper_small_en_l2_arctic/predictions.csv \
  --public-metrics results/baseline/whisper_small_en/l2_arctic_v5_arabic_utterance_metrics.csv \
  --report results/baseline/whisper_small_en/l2_arctic_v5_arabic_baseline.json \
  --device mps \
  --batch-size 8 \
  --local-files-only
```

The prediction CSV is atomically updated after every batch and can resume after
an interruption. A saved run configuration prevents an incompatible run from
silently reusing those predictions. Public outputs are immutable: the runner
refuses to overwrite them with changed content.

## Artifact and privacy policy

- `data/experiments/.../predictions.csv` contains corpus references,
  hypotheses, and normalized text. It is local-only and ignored by Git.
- `results/baseline/.../l2_arctic_v5_arabic_utterance_metrics.csv` contains no
  transcript or hypothesis text. It stores identifiers, content hashes, word
  counts, and error counts for auditing.
- `results/baseline/.../l2_arctic_v5_arabic_baseline.json` contains aggregate
  and per-speaker measurements plus a complete run configuration and artifact
  hashes.
- The downloaded model cache is local-only and ignored by Git.

The native-English control baseline has not yet been run. These measurements
therefore establish Arabic-accented English performance only; they do not yet
support a fairness comparison or a claim that accent-specific adaptation helps.
