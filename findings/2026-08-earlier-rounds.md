# Earlier rounds — reconstructed, and incomplete

Everything here is recovered from code comments, CLAUDE.md and one session's
memory, after the fact. Where a docstring carries the reasoning it is
trustworthy; where a number appears without one, it is a number somebody chose
and the evidence is gone. Marked accordingly.

## Recoverable from the code, with reasoning intact

- **Emphatic pause cap, 0.59 s** (`baseline.EMPHATIC_CAP_PCT`, p80). Chosen from
  the distribution, not rounded: 85% of within-sentence gaps are exactly zero
  and the non-zero ones are bimodal with an empty band between 0.80 and 1.00 s.
- **Emphasis weights** — loudness 0.45, duration 0.25, pitch 0.20, pause 0.10
  (`score.WEIGHTS`). Loudness largest because it separated all five stressed
  words in stress1 and never got the sign wrong.
- **Marking levels**, light > 0.962 and strong > 1.408 (`markers.LEVEL_THRESHOLDS`),
  derived by reflecting the below-median half of the weight distribution about
  the median and finding where the real upper half exceeds it.
- **Sentence boundaries** as a union of a gap-plus-pitch-fall cue and
  punctuation (`prosody_core.assign_sentences`), because no gap threshold
  separates hesitations from boundaries in drag1.
- **Diarization thresholds**, `AHC_THRESHOLD` 0.60 and `CENTROID_MERGE_MAX`
  0.60 (`diarize.py`). Average linkage alone split one voice into three on
  turns1 with centroids 0.13 to 0.43 apart.
- **Recognition thresholds**, `MATCH_MAX` 0.70 and `CONFIDENT_MAX` 0.40
  (`recognize.py`), from 114 same-voice chunks: median 0.290, p95 0.436, worst
  0.513. The different-voice side was validated later on pod2 at 0.84.
- **Speech recogniser**: Apple SpeechAnalyzer, chosen over faster-whisper
  base.en and over Parakeet. Parakeet was disqualified on an 80 ms timestamp
  grid against Whisper's 20 ms. SpeechAnalyzer's own grid is 60 ms, accepted
  deliberately as a trade against 2-4x speed and Kannada.
- **Distance calibration, rejected.** Reverberation proxies separate three
  deliberate microphone distances monotonically but vary two to three times
  more with who is speaking than with where they stand; applying the correction
  would inject 39.7 dB of error to remove 12.87. Session-relative loudness
  stays the default. `baseline.calibrate_distance` keeps the machinery and its
  own reliability test.

## Lost

- **The six hand-labelled retrieval questions.** The basis for believing
  retrieval worked. Not in the repository, not reconstructable.
- **The BEAM assessment.** Conclusion survives as a summary, reasoning does not.
- **Prosody corpus measurements.** The runs behind the constants above: counts,
  distributions, rejected alternatives. The constants survive with their
  rationale; the data does not.
- **Anything before `store/spend.json`** on cost or timing.

## The two rules that came out of these rounds

In CLAUDE.md, each after more than one occurrence:

1. A check on the entry point protects the entry point — put the guard on the
   operation. The outward permission lived in one caller; the singleton lock
   lived in `main()`. Both left the thing they guarded open.
2. A test that cannot fail is not evidence. Three runs with no speech reported
   zero dropped frames; nineteen written examples reported no uninvited answers
   for a configuration that then spoke three times unprompted.

A third occurrence of pattern 1 has since been recorded: `session.t0` was set at
object construction rather than at stream open, so the self-hearing suppression
compared two clocks that had never agreed. The same mistake appeared three
times in one file — drift baseline, stall deadline, suppression anchor.
