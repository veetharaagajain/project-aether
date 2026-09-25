# 2026-08-31 — retrieval was releasing fragments

## What was measured

`memory.search` searches windows and answers with observations — "a centre is
never returned twice, and what comes back is the record, never the window",
which is right for identifying what matched and wrong for handing it to a
composer. On the real store the top hits for "what am I building" were "I.",
"I." and "I".

Eight questions, 80 retrieved lines, before the change:

- median 2 words, mean 4.3
- 51% two words or fewer
- 81% under eight words, so unable to stand alone

LoCoMo for comparison: median 19 words, no fragments. That is why the same
metadata change measured 12% to 40% there and nothing here.

After releasing the window for any line under the threshold:

- median 2 to **12** words, mean 4.3 to **18.4**
- two words or fewer 51% to **34%**
- under eight words 81% to **36%**
- 38 of 80 expanded, releasing 233 extra observations, 6.1 per expanded line

## Constant

`recall.STAND_ALONE_WORDS = 8`. The smallest number that clears the fragment
population without expanding lines that were already usable: on this store 51%
of retrieved lines are two words or fewer with a median of two, while LoCoMo
lines compose fine at a median of 19. At or above eight is already a clause
that can carry a claim, so expanding it would release speech for nothing.

No judgement is involved and none is needed — the test is word count, and the
window is the one `memory.search` already scored.

## Where it happens, and why there

In `gate.search_memory`, not in the caller. Releasing a window releases speech
the question did not match, which is what the gate exists to control. Doing it
in `live.py` would have meant the gate approving one line and the composer
receiving seven.

The approval text marks the matched line with `>>` via `memory.window_text` and
indents the neighbours; each hit carries `matched`, `released`
('observation' or 'window') and `n_released`; the access log records
`expanded` as N of M and `extra_observations` as the total.

## The residue, which is not a bug

36% still cannot stand alone. Those have windows containing only themselves:
`WINDOW_MAX_GAP_S` is 8 seconds and a fragment with more than eight seconds of
silence either side genuinely has no neighbourhood. 5% of windows on this store
are single-centre. Expanding those would mean reaching across a silence the
segmenter deliberately treated as a boundary.
