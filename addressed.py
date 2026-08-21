"""Was this said to me, or merely near me.

An always-listening system that answers everything it hears will answer the
television, one side of a phone call, and two people talking to each other.
Nothing else in this project asks the question, and without it the being cannot
be allowed to speak at all.

THE LADDER. The model call is the expensive part and the last resort, so
everything free runs first, cheapest and most decisive at the top. Each rung
can only say "no" -- none of them can say "yes" on its own, because every cheap
signal here is also true of speech that was not addressed to anything.

  1. capture paused        -- if it is not listening it does not answer
  2. too few words         -- 59 percent of observations are four words or
                              fewer, and "okay." is not a request
  3. not a question or a   -- the transcriber's punctuation and the pitch-rise
     request               detection are already computed and stored; a flat
                              declarative is almost never a request
  4. speaker not           -- diarization and recognition already ran; speech
     recognised               from a voice the system does not know is far more
                              likely to be a television or a stranger than an
                              instruction
  5. the model             -- the only rung that understands "did you take the
                              bins out" is for a person and not for a machine

Rungs 2 to 4 reject about nine tenths of segments for free. Rung 5 costs one
model call on what is left.

BIAS. Silence costs an answer, which can be asked for again. Speaking uninvited
happens out loud in a room in front of whoever is there and cannot be taken
back. Every rung, and the instructions in bridge/relevance.swift, err the same
way.
"""

import difflib
import json
import re
import time

MIN_WORDS = 4
CONTEXT_LINES = 3
REPLY_BUDGET_S = 25.0

# A request need not be a question: "tell me what she said" is an instruction.
# These are the openings that make a flat sentence a request rather than a
# remark. Deliberately short -- this rung only filters, and anything it lets
# through still has to convince the model.
REQUEST_OPENERS = {'tell', 'show', 'find', 'remind', 'play', 'read', 'give',
                   'look', 'search', 'check', 'list', 'summarise', 'summarize',
                   'explain', 'describe', 'repeat', 'stop', 'cancel', 'open'}


# --- the name ---------------------------------------------------------------
# Stored, not a constant, because the spoken name was always meant to be the
# person's to choose. It lives in meta under 'wake_word', is read on every
# check, and so changing it takes effect on the next utterance with nothing
# restarted.
#
# It is a rung and not a requirement. Saying the name settles the question --
# nothing after it in the ladder can overrule it. Not saying it changes nothing:
# the ladder runs exactly as before. A wake word that becomes mandatory is a
# command line with a microphone, and the point of this one is that it
# recognises something already happening rather than imposing ceremony.
DEFAULT_WAKE_WORD = "Jarvis"
WAKE_HEAD_TOKENS = 2        # "Jarvis, ..." and "Okay Jarvis, ..." and no further:
                            # at three, "I told Jarvis about it" matches, which is
                            # talk about the assistant rather than to it
WAKE_RATIO = 0.72           # see wake_matches()
ATTENTION_S = 45.0


def wake_word(db):
    r = db.execute("SELECT value FROM meta WHERE key='wake_word'").fetchone()
    return (r[0] if r and r[0].strip() else DEFAULT_WAKE_WORD).strip()


def set_wake_word(db, name):
    """Change the name. Takes effect on the next utterance; nothing restarts."""
    name = (name or '').strip()
    if not name:
        raise ValueError("the name cannot be empty")
    db.execute("INSERT INTO meta(key,value) VALUES('wake_word',?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (name,))
    db.commit()
    return wake_word(db)


def _soundex(w):
    w = re.sub(r'[^a-z]', '', (w or '').lower())
    if not w:
        return ''
    codes = {**{c: '1' for c in 'bfpv'}, **{c: '2' for c in 'cgjkqsxz'},
             **{c: '3' for c in 'dt'}, 'l': '4',
             **{c: '5' for c in 'mn'}, 'r': '6'}
    out, last = w[0].upper(), codes.get(w[0], '')
    for ch in w[1:]:
        c = codes.get(ch, '')
        if c and c != last:
            out += c
        if ch not in 'hw':
            last = c
    return (out + '000')[:4]


def wake_matches(token, name):
    """Did the transcriber probably write the name here.

    Exact matching fails on exactly the names people choose: an unusual name is
    what a recogniser mangles, and "Jarvis" comes back as "Java's", "Jervis" or
    "drivers". Two tests, either sufficient -- a soundex match, which catches
    substitutions that keep the consonant skeleton, and a character-similarity
    ratio, which catches the ones that do not.

    Tuned deliberately tight rather than generous, because this rung SETTLES
    the decision: a false match here speaks out loud with nothing left to stop
    it, while a miss costs one silence and the ladder still runs. At 0.72
    "java's" and "jervis" match and "travis" and "marvin" do not. "drivers" is
    not caught by either test and is a known miss.
    """
    t = re.sub(r"[^a-z]", "", (token or '').lower())
    n = re.sub(r"[^a-z]", "", (name or '').lower())
    if not t or not n:
        return False
    if t == n:
        return True
    if _soundex(t) == _soundex(n):
        return True
    return difflib.SequenceMatcher(None, t, n).ratio() >= WAKE_RATIO


def said_the_name(text, name):
    """Was the name used to get attention, rather than merely mentioned.

    Position is checked, and the rule is the first few words or the last one.
    Those are the two places a name is used to address someone -- "Jarvis, what
    did I say" and "what did I say, Jarvis" -- while a name in the middle is
    usually talk ABOUT the person: "I told Jarvis about it yesterday".

    Getting this wrong is asymmetric, which is why the rule is narrow. Missing a
    name that was there costs one silence and the ladder still runs. Matching a
    name that was only mentioned means answering a sentence that was about the
    assistant rather than to it, out loud, with the rest of the ladder skipped.
    """
    toks = re.findall(r"[A-Za-z']+", text or '')
    if not toks:
        return None
    for i, t in enumerate(toks[:WAKE_HEAD_TOKENS]):
        if wake_matches(t, name):
            return f'name at position {i + 1}'
    if wake_matches(toks[-1], name):
        return 'name at the end'
    return None


# --- who else is here --------------------------------------------------------
# A question said out loud when nobody else is present has nowhere else to go.
# With someone else in the room it usually does. Speaker recognition already
# knows which voices have been heard, so this costs nothing to ask.
#
# It SHIFTS the decision rather than settling it, unlike the name. Alone is not
# unaccompanied: the person could be on a call, or with someone silent, or with
# a television on. So it is passed to the model as evidence and the model still
# decides.
ALONE_WINDOW_S = 300.0

# ...and it is OFF, on the evidence. The reasoning for it is sound -- a question
# said out loud when nobody else is present has nowhere else to go -- but the
# model does not respond to being told so in the intended direction. Measured
# twice, with two different wordings, over the same nineteen labelled
# utterances: telling it the speaker is alone took genuine requests answered
# from 6 of 8 down to 4 of 8, and telling it someone else is present took them
# UP to 7 of 8. That is backwards both times, and the effect is large enough
# not to be noise on a deterministic model.
#
# who_is_here() is kept because it is correct and cheap and something else will
# want it. The lean is not applied unless this is turned on, because a rung that
# moves the answer the wrong way is worse than no rung.
PRESENCE_LEAN = False


def who_is_here(db, now=None, window=ALONE_WINDOW_S):
    """Distinct voices heard recently. Returns (state, detail).

    state is 'alone', 'accompanied' or 'no evidence'. Silence long enough to
    leave no evidence is reported as exactly that and shifts nothing -- absence
    of voices is not proof of absence of people, and a room that has been quiet
    for five minutes tells you nothing about who is sitting in it.
    """
    now = time.time() if now is None else now
    rows = [dict(r) for r in db.execute(
        "SELECT person_id, person_decision FROM observations "
        "WHERE kind='speech' AND started_at >= ?", (now - window,))]
    if not rows:
        return 'no evidence', 'nothing heard recently'
    known = {r['person_id'] for r in rows
             if r['person_decision'] in ('match', 'confident') and r['person_id']}
    unknown = sum(1 for r in rows if r['person_decision'] == 'unknown')
    if len(known) > 1:
        return 'accompanied', f'{len(known)} recognised voices'
    if unknown:
        return 'accompanied', f'{unknown} utterance(s) from an unrecognised voice'
    if len(known) == 1:
        return 'alone', 'one recognised voice and no others'
    return 'no evidence', 'nothing recognised'


class NotAddressed(Exception):
    """Not spoken to us. Carries which rung decided, for the log and the viewer."""

    def __init__(self, rung, detail=''):
        super().__init__(f"{rung}{': ' + detail if detail else ''}")
        self.rung = rung
        self.detail = detail


def looks_like_a_request(text, is_question):
    if is_question:
        return True
    words = re.findall(r"[a-z']+", (text or '').lower())
    return bool(words) and words[0] in REQUEST_OPENERS


def cheap_rungs(record, capturing=True):
    """Everything decidable without the model. Raises NotAddressed, or returns."""
    if not capturing:
        raise NotAddressed('paused', 'capture is paused')
    text = (record.get('plain') or '').strip()
    n = len(text.split())
    if n < MIN_WORDS:
        raise NotAddressed('too short', f'{n} word(s)')
    is_q = bool(record.get('question_by_punctuation')
                or record.get('question_by_pitch_rise')
                or text.endswith('?'))
    if not looks_like_a_request(text, is_q):
        raise NotAddressed('not a request', 'no question and no request opener')
    if record.get('person_decision') not in ('match', 'confident'):
        raise NotAddressed('speaker not recognised',
                           str(record.get('person_decision')))
    return {'words': n, 'question': is_q}


def ask_model(utterance, context_lines=(), presence='no evidence'):
    """The last rung. Returns (addressed, audience, seconds)."""
    import relevance as rel
    payload = json.dumps({'utterance': utterance,
                          'context': "\n".join(context_lines),
                          'presence': presence}).encode()
    r = rel.request(f"addressed {len(payload)}\n", payload, REPLY_BUDGET_S)
    if r.get('unavailable'):
        raise NotAddressed('model unavailable', r['unavailable'])
    if r.get('error'):
        raise NotAddressed('model declined', str(r['error'])[:80])
    return bool(r.get('addressed')), r.get('audience', ''), r.get('seconds', 0.0)


def decide(record, recent_lines=(), capturing=True, db=None, attention=None,
           now=None):
    """The whole ladder. Returns a dict when addressed; raises NotAddressed.

    Order, and what each rung can do:

      paused                 no
      the name               YES, and nothing after it runs
      still holding          YES, if the name was said moments ago
        attention
      too short              no
      not a request          no
      speaker not            no
        recognised
      who else is here       shifts the model, settles nothing
      the model              the last word

    Never returns 'probably'. Anything short of a clear yes raises, because the
    caller's only two options are speaking and not speaking.

    attention is a mutable dict the caller keeps across utterances; see
    hold_attention().
    """
    t0 = time.perf_counter()
    now = time.time() if now is None else now
    if not capturing:
        raise NotAddressed('paused', 'capture is paused')
    text = (record.get('plain') or '').strip()

    name = wake_word(db) if db is not None else DEFAULT_WAKE_WORD
    where = said_the_name(text, name)
    if where:
        if attention is not None:
            hold_attention(attention, now)
        return {'addressed': True, 'audience': 'assistant', 'rung': 'name',
                'detail': where, 'seconds': round(time.perf_counter() - t0, 3)}
    if attention and holding(attention, now):
        left = attention['until'] - now
        # the ladder's cheap filters still apply inside the window: holding
        # attention does not make "okay." a request
        cheap = cheap_rungs(record, capturing=capturing)
        hold_attention(attention, now)
        return {'addressed': True, 'audience': 'assistant',
                'rung': 'attention', 'detail': f'{left:.0f}s left of the window',
                'words': cheap['words'],
                'seconds': round(time.perf_counter() - t0, 3)}

    cheap = cheap_rungs(record, capturing=capturing)
    here, here_detail = who_is_here(db, now=now) if db is not None \
        else ('no evidence', 'no store')
    if not PRESENCE_LEAN:
        here_detail = f'{here} (not applied: PRESENCE_LEAN is off)'
        here = 'no evidence'
    addressed, audience, secs = ask_model(
        text, list(recent_lines)[-CONTEXT_LINES:], presence=here)
    if not addressed:
        raise NotAddressed('model says no', audience or 'unclear')
    # NOT opened here. Only the name opens a window; a model accept merely
    # extends one that is already open. Opening on any accept turned a single
    # mistaken accept into a 45-second licence: on ten minutes of real
    # conversation it spoke four times and reached the answer path nine more,
    # against zero before the window existed. A window is what a name buys.
    if attention is not None and holding(attention, now):
        hold_attention(attention, now)
    return {'addressed': True, 'audience': audience, 'rung': 'model',
            'presence': here, 'presence_detail': here_detail,
            'words': cheap['words'], 'question': cheap['question'],
            'model_seconds': secs,
            'seconds': round(time.perf_counter() - t0, 3)}


def new_attention():
    return {'until': 0.0}


def holding(attention, now=None):
    return (attention or {}).get('until', 0.0) > (time.time() if now is None else now)


def hold_attention(attention, now=None):
    """Start or extend the window in which follow-ups count as addressed.

    Called unconditionally by the name rung, and only-if-already-open by the
    others. See decide().

    ATTENTION_S is 45 seconds, and the floor on it is this system's own
    latency: an answer that had to search the store took 11.6 seconds to
    speak, so a window shorter than about 20 would expire while the person was
    still listening to the reply they asked for, and every follow-up would need
    the name again. The ceiling is the room: someone walking in and speaking a
    minute later must not inherit it.

    Extended by each addressed utterance rather than fixed from the name, which
    is how attention actually behaves -- a conversation that keeps going keeps
    it, and one that stops lets it lapse.
    """
    now = time.time() if now is None else now
    attention['until'] = now + ATTENTION_S
    return attention
