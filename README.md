# EquiVoice

EquiVoice is a research project investigating whether accent-specific adaptation can improve automatic speech recognition for Arabic-accented English on unseen speakers without substantially degrading recognition of native English.

The project combines speaker-independent machine-learning evaluation with phonetic analysis of Arabic-to-English transfer patterns. It is being developed incrementally; the pretrained Arabic-accented English and native-English control baselines are measured, and the adaptation pipeline is now set up but has not completed full training.

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
- [x] Deterministic 16 kHz mono audio preprocessing
- [x] Pretrained Whisper baseline for the Arabic L2-ARCTIC cohort
- [x] Native-English LibriSpeech control baseline
- [x] Leakage-safe fine-tuning pipeline and one-step local smoke test
- [ ] Accent-specific adaptation
- [ ] Linguistic error analysis

No dataset audio, model checkpoints, or text-bearing predictions are included.
The repository contains only aggregate baseline results and text-free
utterance-level error statistics.

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

See [Audio preprocessing](docs/audio_preprocessing.md) for the deterministic
44.1-to-16 kHz transformation, output validation, measured duration drift, and
amplitude-range caveat.

See [Pretrained Whisper baseline](docs/whisper_baseline.md) for the pinned model
revision, deterministic inference configuration, privacy boundary, and measured
per-speaker and aggregate WER.

See [Native-English control baseline](docs/native_english_control.md) for the
complete LibriSpeech `test-clean` selection, equivalent preparation policy,
measured native-control WER, and cross-corpus comparison limitations.

See [Whisper fine-tuning setup](docs/fine_tuning_setup.md) for the frozen
four-fold training configuration, data-integrity and leakage safeguards, local
smoke test, CUDA command, checkpoint policy, and remaining work.

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

## Audio preprocessing

Validated raw WAVs are decoded as their original signed 16-bit PCM, checked
against the raw-data hashes, converted to float32, and resampled from 44.1 kHz to
16 kHz with SoXR high quality. The output remains mono and is stored as float32
WAV so resampler output is preserved without normalization, clipping, or another
integer quantization step.

```bash
PYTHONPATH=src python -m equivoice.preprocess_l2_arctic \
  --source-root data/raw/l2_arctic/v5.0 \
  --source-manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv \
  --validation-report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --output-root data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq \
  --processed-manifest results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv \
  --report results/preprocessing/l2_arctic_v5_arabic_16khz_report.json
```

All 4,365 processed files passed verification. The ignored processed-audio
directory is local-only; the repository contains only the metadata manifest and
aggregate report.

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

LibriSpeech SLR12 is distributed by OpenSLR under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). EquiVoice uses the
complete official [`test-clean` split](https://www.openslr.org/12/) as its fixed
native-English control.

The corpus itself is not redistributed in this repository. A license for EquiVoice's future original source code has not yet been selected.

## Results

The pinned pretrained `openai/whisper-small.en` checkpoint achieved **13.2631%
micro-WER** across all 4,365 Arabic-accented English recordings. Per-speaker WER
was 12.2805% for ABA, 21.3388% for SKA, 10.0158% for YBAA, and 10.5237% for
ZHAA; the unweighted macro-speaker WER was 13.5397%.

On the complete 2,620-utterance LibriSpeech `test-clean` control, the identical
checkpoint and evaluation policy achieved **3.0532% micro-WER** and **3.0413%
macro-speaker WER**. The descriptive cross-corpus difference is 10.2099
percentage points. It must not be interpreted as a pure causal accent effect
because the two corpora also differ in speakers, prompts, recording conditions,
and speech domain.

These are inference-only baseline measurements, not adaptation results. The
speaker variation has not yet been attributed to pronunciation transfer, audio
conditions, or another cause. Fine-tuned and linguistic-analysis results remain
unmeasured.

Machine-readable baseline artifacts are under
`results/baseline/whisper_small_en/`; raw-data validation metadata remains under
`results/data_validation/`.
