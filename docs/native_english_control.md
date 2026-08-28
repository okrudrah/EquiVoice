# Native-English control baseline

This milestone establishes EquiVoice's native/standard-English reference using
the complete LibriSpeech `test-clean` split. It uses the same pretrained model,
audio representation, decoding policy, text normalization, and WER calculation
as the Arabic-accented English baseline. No model training or fine-tuning occurs.

## Control selection

LibriSpeech is an approximately 1,000-hour corpus of segmented 16 kHz read
English derived from LibriVox audiobooks. OpenSLR distributes it under CC BY 4.0
and designates `test-clean` as clean test speech. See the official
[OpenSLR SLR12 page](https://www.openslr.org/12/).

EquiVoice uses every official `test-clean` utterance rather than selecting a
smaller custom sample:

- 2,620 utterances
- 40 speakers: 20 listed as female and 20 as male
- 87 speaker/chapter combinations
- 5.403467 hours

This remains locally manageable while avoiding post-hoc selection based on
recognition results. Its duration is similar in scale to the 4.385174-hour
Arabic cohort. The official archive's published MD5,
`32fa31d27d2e1cad72775fee3f4849a9`, was verified before extraction; its SHA-256
is `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.

The LibriSpeech documentation describes the clean subsets as predominantly US
English but warns that the automated clean/accent classification is crude and
not completely reliable. EquiVoice therefore calls this a native/standard-
English control cohort, not a perfect demographic ground truth.

## Validation and preparation

The preparation command validates one-to-one correspondence between FLAC files
and transcript entries, utterance IDs and directory structure, the 40 speakers
listed for `test-clean` in `SPEAKERS.TXT`, and the expected utterance and chapter
counts. Every source file must be 16 kHz, mono, FLAC, and signed 16-bit PCM.

The source is already at Whisper's target sample rate. Preparation losslessly
decodes PCM-16 FLAC to mono float32 WAV without resampling, normalization,
trimming, denoising, or clipping. Per-utterance source, transcript, decoded-PCM,
and output hashes are recorded in a text-free manifest.

```bash
PYTHONPATH=src python -m equivoice.prepare_librispeech_control \
  --source-root data/raw/librispeech/slr12/LibriSpeech/test-clean \
  --speakers-metadata data/raw/librispeech/slr12/LibriSpeech/SPEAKERS.TXT \
  --archive data/raw/librispeech/slr12/archives/test-clean.tar.gz \
  --output-root data/processed/librispeech/slr12/test-clean/16khz_mono_float32 \
  --manifest results/preprocessing/librispeech_slr12_test_clean_16khz_manifest.csv \
  --report results/preprocessing/librispeech_slr12_test_clean_16khz_report.json
```

## Model and evaluation settings

The control uses the pinned
[`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en)
revision `e8727524f962ee844a7319d92be39ac1bd25655a`, float32 MPS inference,
batch size 8, seed 17, and deterministic greedy decoding. Because this is the
English-only checkpoint, language detection is disabled. References and
hypotheses use the same Transformers Whisper `EnglishTextNormalizer` as the
Arabic cohort.

```bash
PYTHONPATH=src python -m equivoice.run_librispeech_control \
  --prepared-root data/processed/librispeech/slr12/test-clean/16khz_mono_float32 \
  --processed-manifest results/preprocessing/librispeech_slr12_test_clean_16khz_manifest.csv \
  --preparation-report results/preprocessing/librispeech_slr12_test_clean_16khz_report.json \
  --model-cache data/models/huggingface \
  --predictions data/experiments/baseline/whisper_small_en_librispeech_test_clean/predictions.csv \
  --public-metrics results/baseline/whisper_small_en/librispeech_slr12_test_clean_utterance_metrics.csv \
  --report results/baseline/whisper_small_en/librispeech_slr12_test_clean_baseline.json \
  --device mps \
  --batch-size 8 \
  --local-files-only
```

## Measured result

| Metric | Result |
|---|---:|
| Utterances | 2,620 |
| Reference words | 53,027 |
| Word errors | 1,619 |
| Micro-WER | **3.0532%** |
| Macro speaker WER | **3.0413%** |
| Lowest individual speaker WER | 0.6457% |
| Highest individual speaker WER | 6.2319% |

For context, the same checkpoint measured 13.2631% micro-WER on the four-speaker
Arabic-accented L2-ARCTIC cohort. The observed difference is **10.2099 percentage
points**, and the Arabic-cohort WER is 4.344 times the LibriSpeech control WER.

This is a descriptive cross-corpus baseline, not proof that accent alone caused
the entire difference. LibriSpeech and L2-ARCTIC differ in speakers, prompts,
recording pipelines, audiobook versus elicited-sentence domain, and other
conditions. The central later experiment is within-dataset: compare the same
held-out L2-ARCTIC speaker before and after adaptation, then measure whether the
same adapted checkpoint changes WER on this fixed control.

## Artifact and privacy policy

The archive, processed audio, individual transcripts, model cache, and detailed
predictions remain in ignored `data/` paths. The repository stores only:

- the text-free preparation manifest and aggregate report;
- the text-free utterance error-count CSV;
- the aggregate and per-speaker baseline report with artifact hashes.

Both preparation and evaluation are resumable. Existing public artifacts are
immutable and cannot be silently overwritten with changed content.
