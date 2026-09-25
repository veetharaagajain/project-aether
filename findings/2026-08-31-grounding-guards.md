# 2026-08-31 — the grounding guards, and what they cannot do

## What was measured

`grounded()` and `answer_overlap()` exist to stop the model answering from what
it knows rather than from what was said. Swept `ANSWER_MIN_OVERLAP` from 0.60
to 0.20 against stem lengths 5, 4 and 3, against four adversarial cases that
must be refused and three legitimate ones that must be accepted.

Conditional accuracy on LoCoMo was **2.0% at every one of the fifteen
settings**. The guards were not the bottleneck; they were refusing wrong
answers, and loosening them mostly let wrong answers out.

## The finding that mattered

**A fabricated clause bolted onto a true one passed at the setting then
shipping.** "You went for a morning jog in the park and then drove to Bristol",
against lines mentioning only the jog, scores 0.71 — the true half carries the
false half over the line.

And it cannot be fixed by tuning: the adversarial maximum is 0.71 and the
weakest legitimate rephrasing is 0.67, so the populations overlap and any cut
that refuses the invention refuses honest paraphrase too. A longest-run-of-
unsupported-words metric fails the same way, at 2 against 2.

## What was built

`recall.clause_overlaps` and `weakest_clause`: the answer is split on
conjunctions and each clause scored on its own, and the weakest decides. "You
went for a morning jog in the park" scores 1.00, "then drove to Bristol" scores
0.00, and the sentence is refused on the second.

- `CLAUSE_MIN_OVERLAP = 0.34` — a clause may be about a third invented, no more.
- `CLAUSE_MIN_CONTENT = 2` — fewer content words than this is a joint, not a
  claim; scoring "and then" as zero would refuse every compound sentence.

## The cases are now in the repository

`tests/test_grounding.py`, runnable with plain python, exits non-zero on
failure. They were previously one-line descriptions in a chat, which is how a
check loses the reason it was built.

Twelve cases pass: six that must be refused, six that must be accepted. Two of
the legitimate ones exist to stop a fix breaking the thing it protects — a
genuinely true compound sentence, and answers drawn from the date and the
speaker on the line.

## Known gap, not fixed

**A relation invented across a released window.** Lines saying "I had toast",
"Sarah had the flu that week", "It rained all afternoon"; answer "Sarah had
toast." Every word was said and only the relation is invented, so overlap is
1.00.

No lexical rule separates it from legitimate rephrasing across two lines.
Scored against its own cited line the invention gets 0.50 and the honest case
gets 0.20 — the honest case scores *lower*, so a per-support threshold refuses
the good one and admits the bad one. Needs entailment, or a citation per claim.

Recorded in `tests/test_grounding.py` as `KNOWN_GAPS`: reported separately, not
counted as a failure so the suite stays green and gets read, with a note to
promote it if it ever starts being caught.
