# Working on Aether

Patterns that have each cost this project more than once. They are here
because in every case the second occurrence looked like a different bug, and
was not.

## A check on the entry point protects the entry point

Put the guard on the operation, not on the way in to it. There is usually more
than one way in, and the ones that skip the guard are exactly the ones nobody
counted: a test harness, an import, a second caller added later, a REPL.

Twice now:

**The outward permission.** `gate.release_outward` decides whether text may
leave the machine, and it was called from `gate.answer_or_search` — one caller,
the one it was written for. Every other caller reached the same providers with
no check at all. Consolidation would have sent a whole day of somebody's
speech to a hosted model with nothing asked and nothing logged, and that was
only noticed because a provider that actually worked got added. The check now
lives inside `capability.ask`, which is the single point where a request can
leave, and a non-local provider without a `db` and a `caller` is refused rather
than allowed. Forgetting is now a refusal instead of a leak.

**The singleton lock.** `live.take_lock()` was called from `live.main()`. So
anything that imported the module and called `run_stream` directly got a
complete live path — microphone open, transcribing, writing observations,
publishing to the viewer — with the flock never involved. That is how every
test in several rounds was run, and it put two capture paths on the same
socket. The lock now sits in `run_stream`, which every live path must call
whatever the entry point, with `singleton.held()` so a process can ask whether
it already has it.

The test for whether a guard is in the right place: can you reach the guarded
thing without passing it? If a plausible caller can, it is on the entry point.

## A test that cannot fail is not evidence

Before reporting a measurement, say what result would have shown the failure,
and whether the test could have produced it. If it could not, the number means
nothing and reporting it is worse than reporting nothing, because it closes the
question.

Three times now:

**Zero dropped frames.** Reported after three seventy-second microphone runs.
Every one of them contained no speech, so no segment ever reached the
pipeline, so nothing could stall the capture thread. The test could not have
dropped a frame under any circumstances. The fix it was supposedly validating
had in fact not been applied at all — an edit had silently matched nothing.
The real test was speech through the microphone with the pipeline deliberately
stalled by three seconds a segment.

**No uninvited answers.** Reported from nineteen written examples, for a
configuration that then spoke three times unprompted on real audio. Written
examples were the wrong instrument: they contained no half-heard fragments, no
television, and no cases where the transcript was already wrong.

**A gate approved on two samples.** Two were enough to agree with the proxy and
not enough to disagree with it; five reversed the decision.

The shape is always the same. The test exercised a path adjacent to the one
that fails, and the absence of a failure was read as evidence of its absence.

Two habits that catch it:

- State the falsifier before running. "This would have shown the problem by
  dropping frames" — then check the run actually created the conditions for
  dropping frames.
- Make the thing fail on purpose first. A suppression that has never suppressed
  anything, a retry that has never retried, a drop path that has never dropped:
  each of those was written and shipped here without once being made to fire.

## A corollary, learned the same way

An edit that silently matches nothing leaves the code unchanged and the report
confident. When patching by string replacement, assert the target exists before
replacing it, and verify the behaviour afterwards rather than grepping for the
line you just wrote.

## Check what is already being thrown away before building anything new

This has now happened six times, and it is more predictive than either rule
above. Every one of them looked like a missing capability and was in fact a
missing wire: the information already existed, had already been computed, and
was discarded a step before the place that needed it.

**The date and the speaker.** Retrieval knew both for every line it returned.
The composer was handed bare text. It answered 12% of LoCoMo's questions
correctly; given the two fields retrieval already had, 40%.

**The model's failure replies.** Every refusal and every error came back and
was discarded, so nothing could judge whether a provider was healthy. Claude
answered "I don't have any information about John" fifty-one times in a row
while the run reported success.

**The window that made a line findable.** A fragment was found by searching
its neighbourhood and then released on its own, without the neighbourhood.
Median retrieved line: two words.

**is_question.** question_flags() ran on every utterance and its answer was
never written at insert. 1,903 rows ended in "?" and 187 were flagged, so the
penalty built to use it had nothing to act on.

**Prosody and audio.** Both measured, both stored, neither ever read.

**The suppression count.** self_heard was counted and published to the viewer
and nowhere else, so a run's own summary could not answer the one question the
suppression exists to answer, and checking it needed a bespoke experiment.

The habit: before building anything new, check whether what you need is
already being produced and thrown away. Two forms this takes in practice —
follow the value from where it is computed to where it is consumed and find
the step that drops it; and when adding a new store or index, find every
existing path that writes, deletes or reads the old one and check each still
holds. The window index was added and forget() was never taught about it, so
deleting an observation removed the row and kept the words in twenty-nine
searchable windows.

It is worth stating why this class is so common here. Every one of these was
invisible from the failing end: a composer that cannot see a date does not
report a missing date, it reports a wrong answer. The symptom always appears
downstream of the drop, and always looks like the downstream component being
bad at its job.
