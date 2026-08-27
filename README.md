# EquiVoice

EquiVoice is a research project investigating whether accent-specific adaptation can improve automatic speech recognition for Arabic-accented English on unseen speakers without substantially degrading recognition of native English.

The project combines speaker-independent machine-learning evaluation with phonetic analysis of Arabic-to-English transfer patterns. It is being developed incrementally; no performance results are available yet.

## Research question

Can accent-specific adaptation improve Whisper's recognition of Arabic-accented English on unseen speakers without substantially degrading recognition of native English, and are any measured improvements concentrated around predictable L1 phonetic-transfer phenomena?

## Planned experimental design

- Use the four verified Arabic L1 speakers in L2-ARCTIC: ABA, SKA, YBAA, and ZHAA.
- Run four leave-one-speaker-out folds, adapting on three speakers and testing on the unseen fourth speaker.
- Compare the same held-out speech under pretrained and adapted Whisper models.
- Evaluate both models on a native-English control subset from LibriSpeech.
- Save utterance-level predictions and analyze recognition errors alongside L2-ARCTIC's pronunciation annotations where the evidence supports that linkage.
- Report per-speaker and aggregate results, including negative or mixed findings.

## Current status

- [x] Reproducible Python 3.12 environment and dependency lock
- [x] Local Apple Silicon/MPS availability verified
- [x] Official L2-ARCTIC documentation reviewed
- [x] Arabic speaker IDs, file counts, annotations, structure, and license documented
- [x] Licensed dataset acquisition and extracted-file validation
- [x] Immutable leave-one-speaker-out fold manifests
- [ ] Audio preprocessing
- [ ] Pretrained Whisper baseline
- [ ] Accent-specific adaptation
- [ ] Native-English control evaluation
- [ ] Linguistic error analysis

No dataset audio, model checkpoints, predictions, or model-performance results are included.

## Verified dataset documentation

See [L2-ARCTIC dataset verification](docs/l2_arctic_dataset_verification.md) for:

- release-version distinctions
- exact Arabic speaker and utterance counts
- annotation semantics
- expected directory structure
- licensing and acquisition constraints
- leave-one-speaker-out implications
- limitations and post-download validation requirements

See [L2-ARCTIC raw-data validation](docs/l2_arctic_raw_validation.md) for the
measured file correspondence, audio properties, durations, and validation caveats.

See [Experimental splits](docs/experimental_splits.md) for the frozen
leave-one-speaker-out folds, prompt-grouped validation policy, exact split sizes,
and leakage controls.

## Repository structure

```text
EquiVoice/
├── data/          # local-only datasets; ignored by Git
├── docs/          # research and dataset documentation
├── results/       # lightweight validation metadata; no corpus content
├── src/equivoice/ # maintainable project code
└── tests/         # automated tests
```

## Raw-data validation

The read-only validator checks archive digests, file correspondence, WAV
readability and properties, transcript and TextGrid readability, TextGrid/audio
duration agreement, duplicate decoded audio, and full-scale PCM samples.

```bash
PYTHONPATH=src python -m equivoice.validate_l2_arctic \
  --dataset-root data/raw/l2_arctic/v5.0 \
  --report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv
```

The completed local v5.0 validation passed for 4,365 scripted recordings with
zero errors. The machine-readable report records one caution category: 39 ABA
recordings contain one or more full-scale PCM samples. Those files remain in the
corpus because this is a clipping indicator, not proof that the recordings are
unusable.

## Experimental split manifests

Each fold places one complete speaker in the test split and uses only the other
three speakers for training and validation. A stable SHA-256 rule assigns about
10% of prompt IDs to validation; every recording of the same prompt stays in the
same train/validation split. This avoids train/validation prompt overlap while
keeping the test speaker completely sealed.

```bash
PYTHONPATH=src python -m equivoice.build_loso_manifests \
  --source-manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv \
  --validation-report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --output-dir results/manifests/l2_arctic_v5_loso
```

The four content-hashed manifests and their summary are under
`results/manifests/l2_arctic_v5_loso/`. Regeneration is deterministic, and the
builder refuses to overwrite an existing manifest with different content.

## Environment

EquiVoice currently targets Python 3.12. The exact verified local environment is recorded in `requirements-lock.txt`; direct project dependencies are listed in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

Platform-specific training environments, including any later cloud-GPU environment, will be recorded separately rather than assumed to match the Apple Silicon lock file.

## Data access and licensing

L2-ARCTIC is distributed by the Texas A&M PSI Lab under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Access requires reviewing and accepting the official terms through the [L2-ARCTIC corpus page](https://psi.engr.tamu.edu/l2-arctic-corpus/).

The corpus itself is not redistributed in this repository. A license for EquiVoice's future original source code has not yet been selected.

## Results

Raw-data validation metadata is available under `results/data_validation/`. No
baseline, fine-tuned, control, or linguistic-analysis results have been measured
yet. This section will be updated only after the corresponding experiments are
run and verified.
