# 2026-08-31 — the composer was being starved

## What was measured

`gate.search_memory` returns observation rows carrying `at` and `person`. Every
caller passed `[h['text'] for h in hits]`, so the composer saw bare strings and
never the date or the speaker. Same 25 questions, same retrieved observations,
lines rewritten as `[15 May 2023] owner: what they said`:

- text only: **12%** correct (3/25)
- with date and speaker: **40%** correct (10/25)

3.3x from metadata the store already held. No model change, no threshold change.

## Why it was invisible

On room audio the being answers about "you", from one speaker, in recent
speech, where the date rarely decides anything. All 10,568 observations in the
real store carry `person = owner` and `speaker = 1`, so the speaker half of
every prefix is currently a constant and buys nothing until diarization puts a
second voice in the store. The date half is live across 13 days.

## What was built

- `recall.format_line` / `recall.format_lines`, `LINE_FORMAT` and
  `LINE_DATE = "%d %B %Y"`. A day, not a second: a question about when needs
  the day, and a wall-clock time would be noise in the haystack
  `answer_overlap` reads.
- Both prompts reworded to explain the format rather than leave it to
  inference — `RECALL_INSTRUCTIONS` in `bridge/relevance.swift` and the
  `compose_answer` system prompt in `capability.py`. They state the shape, say
  the date and speaker are part of the record rather than of the sentence, give
  the Lean Startup case as a worked example, and say explicitly that answering
  "he doesn't say" there is wrong.

## Bugs this found

- **`PaidAPI.body()` never implemented `compose_answer`.** It fell through to a
  default branch that sends the question and drops the transcript entirely, so
  Claude was asked "what are John's basketball goals" with no conversation
  attached and answered, correctly, "I don't have any information about John".
  Fifty-one benchmark questions were scored against that before it was noticed.
  Fixed with the same canAnswer/answer/support schema the Swift bridge
  produces, so both providers meet one contract.
- **`cap.ask` silently refused the first attempt** because the harness passed no
  `db`/`caller` for a non-local provider. That is the outward gate working; the
  harness swallowed the exception and reported zeros. Spend not moving is what
  gave it away.

## New hole this opened, and the fix

Dates and names in the lines enlarge the haystack `answer_overlap` reads, so an
invented date can now be lifted from a *different* line: "You started The Lean
Startup on 15 May 2023" citing the 22 June line scores 1.00, because every word
appears somewhere in view. Unreachable before this change.

Fixed by `recall.date_tokens` and a provenance check: every month and year in
the answer must appear on the line it cites. A date is the one thing that
cannot be borrowed between lines — a name recurs legitimately, a topic recurs,
but "15 May 2023" belongs to the line it is printed on.

## Assumed

That `person` is the right speaker field. It is the recognised person, and
`speaker` is the diarization label; `format_lines` prefers `person` and falls
back. On a multi-speaker recording with nobody enrolled that means "someone".
