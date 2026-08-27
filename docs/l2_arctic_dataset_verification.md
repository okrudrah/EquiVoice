# L2-ARCTIC dataset verification

**Status:** Pre-acquisition documentation completed 2026-08-26. Licensed local
acquisition and raw-file validation were subsequently completed; see the
[raw-data validation report](l2_arctic_raw_validation.md). No corpus content is
tracked in Git.

## Scope and source hierarchy

This note records facts needed before EquiVoice acquires or processes L2-ARCTIC. It prioritizes the current Texas A&M PSI Lab corpus page and documentation over third-party mirrors. The 2018 Interspeech paper is a primary source for corpus design, but it describes the initial v1.0 release rather than the current corpus.

## Release distinction

- The current official documentation identifies **v5.0** as the latest listed release. It contains 24 non-native English speakers: four speakers for each of six L1 groups (Arabic, Hindi, Korean, Mandarin, Spanish, and Vietnamese).[1][2]
- The 2018 paper describes **v1.0**, which contained 10 speakers from five L1 groups. Counts and demographics from that paper must not be treated as complete current-release metadata.[1][3]
- The current scripted corpus contains **26,867 WAV files** and **3,599 manual annotation files** across all 24 speakers.[2]
- v5.0 also added a spontaneous-speech "suitcase corpus" for 22 speakers. SKA and ASI did not participate in that task.[2]

## Verified Arabic L1 cohort

The official corpus page identifies two male and two female Arabic L1 speakers. The official file summary gives the following current counts.[1][2]

| Speaker | L1 | Gender | Scripted WAV files | Manual annotations |
|---|---|---:|---:|---:|
| ABA | Arabic | M | 1,129 | 150 |
| SKA | Arabic | F | 974 | 150 |
| YBAA | Arabic | M | 1,130 | 149 |
| ZHAA | Arabic | F | 1,132 | 150 |
| **Total** |  |  | **4,365** | **599** |

The v1.0 paper reports TOEFL iBT scores only for the two Arabic speakers present in that initial release: YBAA (100) and SKA (79).[3] Equivalent proficiency metadata for ABA and ZHAA was not found in the current official public documentation, so EquiVoice must not invent or infer it.

## Leave-one-speaker-out implications

The verified scripted-file counts imply these nominal fold sizes before file-level validation:

| Held-out test speaker | Training speakers | Nominal training utterances | Nominal test utterances |
|---|---|---:|---:|
| ABA | SKA, YBAA, ZHAA | 3,236 | 1,129 |
| SKA | ABA, YBAA, ZHAA | 3,391 | 974 |
| YBAA | ABA, SKA, ZHAA | 3,235 | 1,130 |
| ZHAA | ABA, SKA, YBAA | 3,233 | 1,132 |

These are planning counts, not yet validated manifests. After licensed acquisition, EquiVoice must verify that every WAV file has the expected transcript and inspect for unreadable or invalid files before freezing fold manifests.

The held-out speaker must remain completely absent from adaptation and validation data for that fold. A validation strategy using only the three training speakers still needs to be specified before training; no validation utterances may come from the held-out test speaker.

## Verified corpus organization

The official documentation describes this root-level material:[2]

- `README.md`
- `README.pdf`
- `LICENSE`
- `PROMPTS`, containing the original CMU ARCTIC prompts
- One directory per speaker
- `suitcase_corpus/`, containing the separate spontaneous-speech task

Each scripted speaker directory is documented as containing:

```text
<speaker_id>/
├── wav/          # WAV recordings, sampled at 44.1 kHz
├── transcript/   # orthographic TXT transcriptions
├── textgrid/     # forced-aligned word/phone TextGrid files
└── annotation/   # manually corrected and tagged TextGrid subset
```

The suitcase corpus follows a similar organization but omits the forced-aligned TextGrid files because its manual annotations provide the more accurate alignments.[2]

The official documentation does not establish channel count, exact file-to-file completeness, archive checksum, or the validity of every recording. Those properties must be measured after acquisition rather than assumed. Whisper-specific conversion to 16 kHz mono belongs to a later preprocessing step.

## Annotation content and semantics

The scripted corpus uses approximately 1,132 phonetically balanced CMU ARCTIC prompts per speaker. Most speakers did not necessarily record every prompt.[1][3]

Manual annotation covers a selected subset of up to about 150 utterances per speaker:

- 100 sentences shared across speakers
- 50 sentences selected to contain phonemes expected to be difficult for the speaker's L1
- Manually corrected word and phone boundaries
- ARPAbet phone labels
- Substitution, deletion, and addition tags
- Optional annotator comments, which may use IPA symbols[1][2][3]

The official tag patterns are:[2]

- Substitution: `CPL,PPL,s`, where `CPL` is the canonical phone and `PPL` is the perceived phone
- Addition: `sil,PPL,a`
- Deletion: `CPL,sil,d`
- `*` marks a perceived deviation from a standard American English phone; `err` is used when the perceived phone cannot be assigned within the phone set

The `textgrid/` alignments are forced alignments, whereas `annotation/` is the manually reviewed subset. They must not be treated as equivalent ground truth.

## Access, license, and repository policy

- The corpus is released under **CC BY-NC 4.0**. The official site requests citation of Zhao et al. (2018), and directs users to contact the corpus owner for uses outside that license.[1][4]
- Official access requires reviewing the license, submitting a form with name, email address, and affiliation, affirming agreement, and receiving a download link by email.[1]
- The user must personally review and accept those terms before EquiVoice downloads the corpus.
- Raw audio, transcripts, TextGrids, access links, and other corpus files must remain under the ignored local `data/` directory and must not be committed to the public GitHub repository.
- Redistribution of derived corpus material must be reviewed against the license. EquiVoice should publish code, manifests containing non-sensitive file identifiers, aggregate measurements, and reproducibility instructions—not the source corpus itself.

## Methodological consequences for EquiVoice

1. **Use the scripted corpus for the primary experiment.** It provides standardized prompts and orthographic transcripts for all four Arabic speakers. The suitcase corpus is a different spontaneous task and excludes SKA, so combining it with the primary four-fold experiment would make the folds structurally unequal.
2. **Preserve speaker independence.** Split and manifest by speaker ID first; never randomly divide the full Arabic utterance pool.
3. **Do not treat manual annotations as a random sample.** Fifty items per speaker were intentionally enriched for anticipated L1-dependent difficulties. Raw error frequencies from this subset cannot be generalized to all speech without accounting for its selection design.
4. **Separate pronunciation evidence from ASR errors.** A pronunciation tag describes perceived speech production; a Whisper word error describes model output. Claims about phonetic transfer require an explicit alignment/linking method rather than assuming every word error was caused by an annotated phone event.
5. **Report uneven fold sizes.** SKA has 974 scripted WAV files, substantially fewer than the other Arabic speakers. Per-speaker results and macro-level summaries are necessary; a pooled result alone could overweight larger test folds.
6. **Limit population claims.** Four speakers cannot represent the diversity of Arabic varieties, English proficiency, gender identities, ages, or speaking contexts. The official public documentation does not provide dialect labels for the four Arabic speakers.
7. **Keep the test speaker sealed.** File validation may remove unusable samples, but no modeling, hyperparameter, checkpoint, or early-stopping decision may use the held-out speaker.

## Items verified after licensed acquisition

- [x] Exact archive/release identifier and checksum
- [x] Actual extracted directory names and file extensions
- [x] WAV readability, sample rate, channel count, subtype, and duration
- [x] Full-scale PCM sample scan as a reproducible clipping indicator
- [x] One-to-one correspondence among WAV, transcript, forced-alignment, and annotation files
- [x] Transcript UTF-8 readability without normalization or content redistribution
- [x] Missing, duplicate, empty, or malformed records
- [x] Exact per-speaker durations
- [ ] Stable, versioned manifests for the four leave-one-speaker-out folds

The completed checks are documented in
[L2-ARCTIC raw-data validation](l2_arctic_raw_validation.md). Fold manifests
remain a separate next step. No preprocessing or model implementation was
performed during validation.

## Primary sources

1. [Texas A&M PSI Lab: L2-ARCTIC corpus homepage](https://psi.engr.tamu.edu/l2-arctic-corpus/)
2. [Texas A&M PSI Lab: L2-ARCTIC documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/)
3. [Zhao et al. (2018), "L2-ARCTIC: A Non-native English Speech Corpus," ISCA Archive](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.html), DOI: [10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110)
4. [Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/)

## Required dataset citation

```bibtex
@inproceedings{zhao2018l2arctic,
  author    = {Guanlong Zhao and Sinem Sonsaat and Alif Silpachai and
               Ivana Lucic and Evgeny Chukharev-Hudilainen and John Levis and
               Ricardo Gutierrez-Osuna},
  title     = {L2-ARCTIC: A Non-native English Speech Corpus},
  booktitle = {Proceedings of Interspeech},
  year      = {2018},
  pages     = {2783--2787},
  doi       = {10.21437/Interspeech.2018-1110}
}
```
