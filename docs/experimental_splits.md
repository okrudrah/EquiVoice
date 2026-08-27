# L2-ARCTIC Arabic leave-one-speaker-out splits

**Status:** Four deterministic fold manifests frozen on 2026-08-26. No audio
preprocessing or model work was performed.

## Primary test design

Each fold holds out one complete Arabic L1 speaker for testing. That speaker has
no rows in either training or validation, so model selection cannot access the
test speaker's recordings, transcripts, annotations, durations, or predictions.

| Fold | Training/validation speakers | Test speaker |
|---|---|---|
| `held_out_aba` | SKA, YBAA, ZHAA | ABA |
| `held_out_ska` | ABA, YBAA, ZHAA | SKA |
| `held_out_ybaa` | ABA, SKA, ZHAA | YBAA |
| `held_out_zhaa` | ABA, SKA, YBAA | ZHAA |

Every manifest covers all 4,365 validated utterances exactly once. The held-out
speaker is test-only, and every non-held-out recording belongs to either training
or validation.

## Training/validation policy

Validation is selected from the three non-held-out speakers using a deterministic
SHA-256 threshold:

- algorithm version: `sha256-prompt-group-v1`
- namespace: `equivoice:l2-arctic:v5.0:arabic:validation:v1`
- threshold: 10% of the unsigned 64-bit value formed from the first eight digest bytes
- assignment unit: `utterance_id`, which represents the shared CMU ARCTIC prompt ID

The selector assigns 102 of the 1,132 unique prompt IDs to validation. Because
assignment happens by prompt ID rather than recording, all available versions of
the same prompt across the three training speakers remain together. Train and
validation therefore have no prompt-ID overlap. SKA has fewer validation rows
because it did not record every selected prompt.

The assignment is independent of audio content, pronunciation annotations,
model output, and held-out test performance. All folds use the same hash policy.

This validation split is prompt-independent but not speaker-independent: the
three adaptation speakers appear in both training and validation on different
prompts. It is intended only for checkpoint selection and early stopping. The
entire fourth speaker remains the speaker-independent primary test set.

## Frozen split sizes

| Test speaker | Train utterances | Train hours | Validation utterances | Validation hours | Test utterances | Test hours |
|---|---:|---:|---:|---:|---:|---:|
| ABA | 2,944 | 2.844088 | 292 | 0.278222 | 1,129 | 1.262864 |
| SKA | 3,085 | 3.207467 | 306 | 0.314259 | 974 | 0.863448 |
| YBAA | 2,943 | 2.896375 | 292 | 0.285057 | 1,130 | 1.203741 |
| ZHAA | 2,941 | 3.032626 | 292 | 0.297427 | 1,132 | 1.055121 |

Unequal fold sizes are expected because the speakers recorded different numbers
of prompts. Later reporting must retain per-speaker results rather than relying
only on a pooled score.

## Reproduction and immutability

The split builder consumes only the passed raw-data validation report and its
metadata-only utterance manifest:

```bash
PYTHONPATH=src python -m equivoice.build_loso_manifests \
  --source-manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv \
  --validation-report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --output-dir results/manifests/l2_arctic_v5_loso
```

Generated files:

- `fold_aba.csv`
- `fold_ska.csv`
- `fold_ybaa.csv`
- `fold_zhaa.csv`
- `folds_summary.json`

Each CSV contains the fold and split assignment, speaker and utterance IDs,
relative local data paths, measured duration, and decoded-audio SHA-256. It does
not contain transcript text or audio. The summary records the source-manifest
digest, split-policy version, exact split counts and durations, and the SHA-256
digest of every fold CSV.

The generator writes a new artifact only when the target path does not exist. If
the path already exists, it verifies byte-for-byte equality and refuses to
overwrite different content. Any future split-policy change must use a new
algorithm version, namespace, and output location instead of silently altering
these folds.
