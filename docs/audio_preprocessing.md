# L2-ARCTIC Arabic audio preprocessing

**Status:** Passed for all 4,365 validated scripted recordings on 2026-08-27.
No Whisper model was downloaded or run during this step.

## Purpose and boundaries

This step creates a deterministic 16 kHz mono derivative for later Whisper
experiments. The validated 44.1 kHz source corpus remains unchanged. The pipeline
does not trim silence, denoise, normalize volume, clip amplitudes, alter
transcripts, or exclude any recording.

Processed audio remains under the Git-ignored local `data/processed/` directory.
Only metadata, hashes, aggregate measurements, code, and tests are versioned.

## Transformation contract

For each row in the passed raw-data manifest, the preprocessor:

1. Opens the source WAV and verifies its frames, sample rate, channels, format,
   subtype, duration, and decoded-PCM SHA-256 against the validated metadata.
2. Decodes the original mono signed 16-bit PCM samples.
3. Converts PCM to float32 by dividing by 32,768 without peak normalization.
4. Resamples from 44,100 Hz to 16,000 Hz using SoXR 1.1.0 at `HQ` quality.
5. Writes a mono WAV with the `FLOAT` subtype.
6. Reads the saved file back and verifies format, waveform equality, duration,
   decoded-waveform hash, and complete-file SHA-256.

Float32 WAV was selected to preserve the resampler output without applying
integer requantization or silently clipping values at the signed 16-bit limits.
The later model loader will receive the same float32 samples recorded in the
processed manifest.

## Validated output

| Speaker | Files | Processed seconds | Processed hours |
|---|---:|---:|---:|
| ABA | 1,129 | 4,546.308375 | 1.262863 |
| SKA | 974 | 3,108.412000 | 0.863448 |
| YBAA | 1,130 | 4,333.468813 | 1.203741 |
| ZHAA | 1,132 | 3,798.435750 | 1.055121 |
| **Total** | **4,365** | **15,786.624938** | **4.385174** |

Every processed recording is:

- readable
- 16,000 Hz
- mono
- WAV format with float32 samples
- linked one-to-one with its source speaker and utterance ID
- within 0.000062501 seconds of its source duration

The measured maximum absolute duration difference is 0.000031179 seconds, less
than one 16 kHz sample period. The small aggregate duration difference results
from rounding each resampled waveform to a whole number of output samples; no
speech was intentionally removed.

## Amplitude-range warning

The raw validation identified 39 ABA recordings containing at least one signed
16-bit full-scale sample. After high-quality resampling, 36 ABA derivatives have
at least one float sample with magnitude slightly above 1.0:

- 190 samples across the entire 4.385-hour cohort
- maximum absolute value: 1.035439372
- no affected SKA, YBAA, or ZHAA files

These samples were preserved rather than clipped. The warning is metadata, not a
claim that the recordings are unusable. No file was removed. Any later clipping,
exclusion, or sensitivity analysis must be specified before examining held-out
model performance and applied consistently.

## Reproduction and immutability

From the repository root with the Python 3.12 environment activated:

```bash
PYTHONPATH=src python -m equivoice.preprocess_l2_arctic \
  --source-root data/raw/l2_arctic/v5.0 \
  --source-manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv \
  --validation-report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --output-root data/processed/l2_arctic/v5.0/16khz_mono_float32_soxr_hq \
  --processed-manifest results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv \
  --report results/preprocessing/l2_arctic_v5_arabic_16khz_report.json
```

The generated metadata artifacts are:

- `results/preprocessing/l2_arctic_v5_arabic_16khz_manifest.csv`
- `results/preprocessing/l2_arctic_v5_arabic_16khz_report.json`

The manifest contains no transcript text or audio. It maps each speaker and
utterance ID to its raw and processed relative paths, source and processed audio
properties, duration difference, amplitude-range count, decoded-waveform hash,
and file hash. Downstream code can join it to the frozen fold assignments using
`speaker` and `utterance_id` without changing any split.

Existing processed WAVs are never silently overwritten. A rerun recomputes the
expected waveform and requires exact float32 equality; a mismatch causes the
pipeline to stop. Existing report and manifest files likewise must be
byte-for-byte identical.
