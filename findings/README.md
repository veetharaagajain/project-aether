# findings

One dated note per round of work, holding what was measured and the number,
what was rejected and why, and any constant with where it came from.

Not a changelog. The code says what it does; git says when it changed. Neither
says why a threshold is 0.59 rather than 0.60, what was tried before it, or
which measurement would have to be redone to move it. That reasoning has twice
had to be reconstructed from a chat log, badly: six hand-labelled retrieval
questions are gone entirely, and the adversarial grounding cases had to be
rebuilt from one-line descriptions of themselves.

## What belongs in a note

- The measurement, the number, and the sample it rests on.
- What was rejected, and the number that rejected it. A rejected approach with
  its reason is worth as much as the accepted one, because it is what stops the
  next round re-deriving it.
- Every constant introduced or moved, with the evidence and the file it lives
  in.
- What was assumed. A number that rests on an assumption should say so where
  the number is, not only in the chat where it was agreed.

## What is lost

Recorded here rather than quietly absent, because a gap nobody wrote down is a
gap nobody remembers.

- **The six hand-labelled retrieval questions.** Referred to repeatedly as the
  spot check that said retrieval was adequate. Never in the repository. They
  are the reason the memory layer was believed to be working, and they cannot
  be re-run or checked.
- **The BEAM assessment.** An earlier round concluded BEAM was obtainable but
  not runnable without a reasoning model. The conclusion survives as a summary;
  the reasoning does not, so the LoCoMo-versus-BEAM comparison in
  `2026-08-31-locomo.md` rests on a second-hand account of it.
- **The original prosody rounds.** Constants like `SENTENCE_GAP_MIN_S`,
  `PITCH_FALL_MIN_RATIO`, `OCTAVE_FACTOR` and the emphasis `WEIGHTS` carry good
  docstrings, so the reasoning is recoverable from the code. The measurements
  behind them -- the corpus runs, the counts, the rejected alternatives -- are
  not. `notes/` entries reconstructed from docstrings are marked as such.
- **Per-round spend and timing before 2026-08-30.** `store/spend.json` starts
  at the day the accounting was built.
