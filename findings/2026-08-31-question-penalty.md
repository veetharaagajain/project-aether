# 2026-08-31 — the question penalty had nothing to act on

## The symptom

Asked "what did I have for breakfast", the top six hits were six copies of the
person's own past asking of that same question. `memory.QUESTION_PENALTY`
exists at 0.15 and was not firing.

## The hypothesis, and why it was wrong

The reasonable guess was that testing had put dozens of copies of the same
question in the store and a 15% penalty could not overcome that. It was worth
checking rather than assuming, and checking is what found it.

The query side was working: `looks_like_a_question` returned True for all eight
test questions. The store side was not. **1,903 observations ended in a
question mark and only 187 carried `is_question = 1`.**

## What it actually was

`memory.add_observation` never set the column. `is_question` was written only by
`backfill_questions`, so every observation stored since the column was added
carried the default 0 and the penalty in `search()` could never fire on any of
them. The 187 that had it were survivors of a backfill run once, long ago.

The penalty was not too small. It had nothing to act on.

## The fix and what it did

`add_observation` now sets `is_question` from `question_flags(body, text)` at
insert, and `backfill_questions` was run over the existing store: 187 to
**2,499 of 11,396**.

Top hits for "what did I have for breakfast", after:

1. "You had a sausage croissant for breakfast."
2. "Sausage croissant for breakfast."
3. "I had a sausage croissant for breakfast."

Answers, where before there were six copies of the question. Across the eight
test questions, zero of the 80 returned hits are now flagged `is_question` —
the penalty demotes every one of them out of the top ten.

## Note for whoever reads this next

The first hit is the being's own speech ("You had a sausage croissant for
breakfast"), which means self-hearing is still putting replies into the store
despite the suppression fix, or these predate it. Worth checking before relying
on that ranking.

## Assumed

That demoting questions entirely out of the top ten is right. A stored question
is occasionally the best answer — "what did I say about X" where the record is
someone asking it — and 0.15 now removes them wholesale at k=10. Untested.
