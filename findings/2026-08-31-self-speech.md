# The machine's own replies in the store

**Question.** The top hit for "what did I have for breakfast" was "You had a
sausage croissant for breakfast" — the machine's own reply, not anything the
owner said. Either self-hearing was still reaching the store despite the
suppression fix, or those rows predated it.

**Answer: they predate it. Suppression works now.**

## How it was established

Absence was not accepted as evidence. The store held no self-speech after
08-29 and the suppression fix landed on 08-30, which is consistent with the
fix working and equally consistent with the machine simply not having spoken
since. So the loop was run for real: the live service was stopped, `live.py
--seconds 60` was run against the microphone, and a question was played at it
through the speaker.

The machine heard the question, composed, and spoke "Nothing about breakfast
in what's been said." That reply came back through the microphone and was
transcribed at 28.80s as a normal utterance attributed to the owner — and was
not written to the store. Three segments produced words; two were stored. The
one that was dropped was its own sentence.

The run's summary could not say this. `session.self_heard` was counted and
published to the viewer channel and printed nowhere, so the only way to check
was to query the store afterwards. `live.py` now prints the count in its
summary.

## The sweep

Voice fingerprint over all 10,734 observations with archived audio, against a
centroid built from nine confirmed replies. The distribution is cleanly
bimodal: thirteen rows below 0.53, then nothing until 0.774.

Two signals were required, not one, and they disagreed in both directions:

- **Fingerprint said yes, content said no.** "You will be transported to the
  federal detention center in Davenport" scored 0.402, well inside the
  cluster. The surrounding forty seconds are courtroom drama, and the same
  line appears nine seconds earlier from the television. Not swept.
- **Content said yes, fingerprint said no.** Two rows reading "You had a
  sausage croissant" at 11:46:15 and 11:46:22 scored above the cut but are
  verbatim repeats of a confirmed reply seconds apart. Swept.

The 08-21 rows were each checked against what preceded them before sweeping,
because "Sausage croissant for breakfast" is also a sentence the owner could
have said, and sweeping the owner's own statement would have destroyed the
fact the query depends on. Every one of them directly follows a question.

Fourteen observations swept with tombstones, twelve audio blobs dropped, seven
access-log rows scrubbed.

## What the sweep uncovered

`incognito._apply` deleted observations, vectors, beliefs, edges, audio and
log references — and never touched `windows`. A window stores the joined text
of its span, not a pointer to it, so forgetting an observation removed the row
and left the words in every window containing it. Twenty-nine windows still
carried "you had a sausage croissant", all of them searchable, and the store
looked correct because the observation really was gone.

This is a leak in `forget()` generally, not just in this sweep. Fixed with
`memory.repair_windows`, called from `_apply`: windows spanning a deleted
observation are dropped and rebuilt around what survives. `tests/
test_forget_windows.py` covers it, and was confirmed to fail with the repair
disabled.

## The ranking, before and after

    before                                          after
    0.8992 No, but I had pancakes for breakfast.     0.9087 I had a sausage croissant for breakfast.
    0.8973 I had pancakes for breakfast.             0.9064 No, but I had pancakes for breakfast.
    0.8966 You had a sausage croissant for break...  0.9045 I had pancakes for breakfast.
    0.8965 Sausage croissant for breakfast.          0.8965 Actually, I had a pancake for breakfast...
    0.8965 Actually, I had a pancake for break...    0.8902 Had a burrito for breakfast.
    0.8956 I had a sausage croissant for breakfast.  0.8822 I was wrong.

The top hit is now the owner's own first-person statement, and every hit above
0.88 is something the owner said.

## Scope of the fix, honestly

Suppression is per-process: `voice.SPOKEN_WINDOWS` is module state inside
`live.py`. Speech played by any other process on the machine — the viewer, a
test script, a future scheduled job — is not suppressed and would be captured
as somebody talking. Nothing does that today. It is a real limit of the
design, not a defect that has fired.
