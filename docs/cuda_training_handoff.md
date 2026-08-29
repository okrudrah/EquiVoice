# CUDA training handoff

This handoff prepares EquiVoice for full training on a private Linux cloud host
with one NVIDIA GPU. It does not create a cloud account, rent a machine, upload
licensed data, or start training. Those actions can cost money and require the
project owner's explicit choice and approval.

## Plain-language checkpoint

The project is currently at the boundary between preparation and the real
experiment:

- The data has been checked and converted.
- The four fair speaker splits are frozen.
- Pretrained Whisper and the native-English control have been measured.
- The fine-tuning program has passed a tiny local test.
- No complete model has been trained yet.
- No fine-tuned model has seen or been scored on its held-out speaker.

The next expensive action is to rent a CUDA GPU and run about 3,725 optimizer
steps across four separate models. The exact duration and price depend on the
chosen provider and GPU, so neither is assumed here.

## Frozen cloud environment

[`configs/cuda_environment.json`](../configs/cuda_environment.json) defines the
portable environment contract:

- Linux and Python 3.12
- PyTorch 2.13.0 with its official CUDA 12.6 wheel
- exactly one CUDA GPU
- a nominal 16 GB GPU at minimum; a nominal 24 GB GPU is recommended
- exact versions for Transformers, Accelerate, audio libraries, and tests

The CUDA 12.6 build is used instead of silently accepting whatever PyTorch a
provider preinstalls. PyTorch must be installed separately so pip does not
replace the selected CUDA wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-cuda.txt
python -m pip check
```

The current macOS lock file remains unchanged because CUDA wheels are
Linux-specific. After the cloud preflight passes, save the complete Linux lock:

```bash
python -m pip freeze --all > data/experiments/cuda-requirements-lock.txt
```

That generated lock belongs with the private experiment artifacts until it has
been reviewed for platform-specific entries.

## Private data transfer

GitHub intentionally does not contain corpus audio, transcripts, model caches,
or checkpoints. The training host needs only the 2.1 GiB processed WAV
collection and approximately 17 MiB of transcripts; the original 10 GiB raw
archive and pronunciation annotations are not required for training.

A local transfer archive can be created from the project root:

```bash
tar -cf data/equivoice_training_data.tar \
  data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq \
  data/raw/l2_arctic/v5.0/speakers/ABA/transcript \
  data/raw/l2_arctic/v5.0/speakers/SKA/transcript \
  data/raw/l2_arctic/v5.0/speakers/YBAA/transcript \
  data/raw/l2_arctic/v5.0/speakers/ZHAA/transcript
```

Upload that archive only to private storage controlled by the project owner.
Do not publish it or commit it. After cloning the GitHub repository on the
private host, extract it from the repository root:

```bash
tar -xf /private/upload/location/equivoice_training_data.tar
```

A persistent volume of at least 60 GiB is recommended for the environment,
model cache, transferred data, two retained Trainer checkpoints per fold, and
four final best-model copies. Checkpoint storage should be synchronized back to
private durable storage after every fold.

## Mandatory preflight

The preflight uses 14 and 22 binary GiB as the acceptance thresholds because
GPU products advertised as 16 and 24 GB report slightly smaller binary-GiB
values to PyTorch. It refuses to pass if Python or package versions differ,
CUDA is missing, the CUDA runtime is wrong, GPU memory is below the nominal
16 GB tier, more than one GPU is exposed, project metadata hashes have changed,
or any required processed WAV or transcript is absent.

```bash
PYTHONPATH=src python -m equivoice.check_cuda_environment
```

Its report is written to `data/experiments/cuda_preflight.json`. That report is
ignored by Git and cryptographically binds the launcher to the current frozen
configuration and manifests.

## CUDA smoke test

Before paying for a long run, repeat the two-example mechanical test on CUDA:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
python -m equivoice.train_whisper_loso \
  --fold-manifest results/manifests/l2_arctic_v5_loso/fold_aba.csv \
  --output-dir data/experiments/fine_tuning/smoke_cuda_held_out_aba \
  --device cuda \
  --smoke-test
```

This must finish one optimizer step and two-example validation generation. Its
loss and WER are not research results and must not be compared with the baseline.

## Dry-run training plan

The launcher defaults to printing commands. It cannot start training unless the
`--execute` flag is explicitly supplied:

```bash
PYTHONPATH=src python -m equivoice.orchestrate_loso_training
```

The dry run should show four commands with matching fold and output names. To
run only the first full fold after preflight and the CUDA smoke test:

```bash
PYTHONPATH=src python -m equivoice.orchestrate_loso_training \
  --fold aba \
  --preflight-report data/experiments/cuda_preflight.json \
  --execute
```

After confirming that its checkpoints, training loss, validation loss, and
validation WER are finite, run the remaining folds without changing the frozen
hyperparameters. All four can be launched sequentially with `--fold all`, but
running one at a time makes storage synchronization and interruption recovery
clearer.

An interrupted single fold can be resumed only from a checkpoint directly
inside that fold's output directory:

```bash
PYTHONPATH=src python -m equivoice.orchestrate_loso_training \
  --fold aba \
  --resume-from-checkpoint \
    data/experiments/fine_tuning/held_out_aba/checkpoint-100 \
  --preflight-report data/experiments/cuda_preflight.json \
  --execute
```

The launcher refuses nonempty output directories unless this restricted resume
mode is used. It also refuses to execute if any protected manifest or config has
changed since preflight.

## After training

Completing the four training runs will still not answer the research question.
The next code milestone is a separate evaluator that:

1. loads each fold's best model;
2. evaluates it only on that fold's previously sealed speaker;
3. evaluates it on the unchanged LibriSpeech native control;
4. reruns the pinned pretrained model in the same CUDA environment for backend
   parity;
5. stores paired utterance-level predictions privately; and
6. reports aggregate, per-speaker, uncertainty, and negative results.

This separation prevents validation observations from silently turning into
test-set tuning.
