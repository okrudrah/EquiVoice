# L2-ARCTIC v5.0 Arabic cohort raw-data validation

**Status:** Passed on 2026-08-26 with zero validation errors and one documented
caution category.

## Scope

This validation covered the scripted-speech directories for ABA, SKA, YBAA,
and ZHAA from the licensed L2-ARCTIC v5.0 archive. It was read-only: no audio was
resampled, normalized, trimmed, converted, or otherwise modified. Transcript and
TextGrid contents remain local and are not reproduced in the repository.

The validator checked:

- SHA-256 digests for the complete release archive and four extracted speaker archives
- expected directory names, extensions, and official v5.0 file counts
- one-to-one WAV/transcript/forced-TextGrid correspondence
- annotation files as a valid subset of the speaker's WAV identifiers
- WAV readability, nonzero length, sample rate, channels, format, and subtype
- UTF-8 transcript readability and nonempty content without normalization
- Praat TextGrid signatures and agreement between root `xmax` and WAV duration
- duplicate decoded PCM content within each speaker
- full-scale 16-bit PCM samples as a reproducible clipping indicator

Synthetic automated tests cover a complete fixture, a missing transcript, and a
TextGrid/audio duration mismatch.

## Validated inventory and duration

| Speaker | WAV | Transcripts | Forced TextGrids | Annotations | Duration (seconds) | Duration (hours) |
|---|---:|---:|---:|---:|---:|---:|
| ABA | 1,129 | 1,129 | 1,129 | 150 | 4,546.309592 | 1.262864 |
| SKA | 974 | 974 | 974 | 150 | 3,108.411927 | 0.863448 |
| YBAA | 1,130 | 1,130 | 1,130 | 149 | 4,333.468526 | 1.203741 |
| ZHAA | 1,132 | 1,132 | 1,132 | 150 | 3,798.435193 | 1.055121 |
| **Total** | **4,365** | **4,365** | **4,365** | **599** | **15,786.625238** | **4.385174** |

Every recording is readable and has the same measured properties:

- 44,100 Hz sample rate
- one channel
- WAV container
- signed 16-bit PCM subtype

No missing counterparts, empty required text files, invalid UTF-8 transcripts,
malformed TextGrid headers, TextGrid/audio duration mismatches, or duplicate
decoded recordings were detected.

## Full-scale-sample caution

Thirty-nine ABA recordings contain at least one sample at either endpoint of the
signed 16-bit PCM range. The other three speakers have no such files. This is a
clipping indicator, not sufficient evidence of audible clipping or unusable
speech. The files therefore remain in the validated corpus and are identified in
the machine-readable report and manifest. They must not be silently removed; any
later exclusion or sensitivity analysis requires a documented rule applied
without consulting held-out model performance.

## Archive identity

The locally acquired complete archive is
`l2arctic_release_v5.0.zip` (7,515,454,814 bytes), with SHA-256:

```text
490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda
```

The JSON report records the size and SHA-256 digest of this archive and each of
the four extracted speaker archives. These are measurements of the acquired
files, not checksums asserted by the corpus publisher.

## Reproduction

From the repository root, with the Python 3.12 environment activated:

```bash
PYTHONPATH=src python -m equivoice.validate_l2_arctic \
  --dataset-root data/raw/l2_arctic/v5.0 \
  --report results/data_validation/l2_arctic_v5_arabic_validation.json \
  --manifest results/data_validation/l2_arctic_v5_arabic_manifest.csv
```

The generated artifacts are:

- `results/data_validation/l2_arctic_v5_arabic_validation.json`: aggregate and per-speaker findings plus archive digests
- `results/data_validation/l2_arctic_v5_arabic_manifest.csv`: one metadata-only row per utterance, without transcript text or audio

The raw corpus and ZIP files remain under the ignored `data/` directory and are
not included in Git.
