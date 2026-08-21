"""Answering without searching: the step before the store is consulted.

Every question currently goes to the store -- embedded, searched, five
candidates judged, something released. That is right when the answer is buried
in a year of speech and wasteful when it was said an hour ago. It is worse than
wasteful, because the search releases speech that never needed to leave.

So: given a question, decide whether it can be answered from what the being has
already heard, and if so answer it. Same on-device model, same process as the
relevance judge, through relevance.request().

WHAT IT MAY ANSWER FROM, in the order they are safe.

  RECENT SPEECH. The last RECENT_MINUTES of the current session. Cheapest and
  clearly right: a question about something said moments ago should not need a
  search, and the material is already the person's own words rather than
  anything inferred from them.

  BELIEFS. A belief is already a conclusion, which is usually what a question
  wants, so this is tempting. It is gated hard: only beliefs that are current
  rather than superseded, not faint, and above BELIEF_MIN_CERTAINTY. The reason
  for the gates is that a wrong belief would be answered from with exactly the
  same confidence as a right one, and unlike a fragment of speech there is no
  raw text behind it for the person to check against. This path is built and
  is close to untested -- see the report.

  GENERAL KNOWLEDGE. Never. See the grounding check below, and the report.

THE GROUNDING CHECK. Instructions are a request; this is a test. The model must
return the line it took the answer from, and grounded() rejects the answer if
that line does not actually occur in the text supplied. A model answering from
what it happens to know cannot produce supporting text that is in a transcript
which does not contain it. This is what makes "never general knowledge" a
property rather than a hope.

WHAT IT MUST NOT DO. It must not answer when unsure -- falling through costs a
few seconds, answering wrongly costs trust in everything else the system says --
and it must not release anything as a side effect. Whether the answer itself is
a release is dealt with in gate.answer_or_search(), and is not settled.
"""

import difflib
import json
import re
import time

import memory as mem
import relevance as rel

RECENT_MINUTES = 20.0
RECENT_MAX_OBS = 60
RECENT_MAX_WORDS = 400
BELIEF_MAX = 8
BELIEF_MIN_CERTAINTY = 0.6
REPLY_BUDGET_S = 25.0

# How closely the model's quoted support must match a line it was shown. Not an
# exact match, because the model reliably alters spacing and trailing
# punctuation; not loose, because the whole point is that it cannot invent one.
SUPPORT_MIN_RATIO = 0.80


class NotAnswerable(Exception):
    """No answer without searching. Carries why, for the log."""


def _norm(t):
    return re.sub(r'\s+', ' ', (t or '').strip().lower()).strip(' .,!?;:"\'')


def grounded(support, lines):
    """Did the model quote something it was actually shown.

    Returns the matching line, or None. Substring first, then a similarity
    ratio for the ordinary case of altered punctuation or a dropped filler.
    """
    s = _norm(support)
    if not s:
        return None
    for ln in lines:
        n = _norm(ln)
        if not n:
            continue
        if s in n or n in s:
            return ln
    best, score = None, 0.0
    for ln in lines:
        r = difflib.SequenceMatcher(None, s, _norm(ln)).ratio()
        if r > score:
            best, score = ln, r
    return best if score >= SUPPORT_MIN_RATIO else None


# The support check proves the model quoted something real. It does not prove
# the quote supports the answer: asked about interest, the model answered
# correctly and cited "Up." -- a line that is genuinely in the transcript and
# genuinely irrelevant. So a second, independent test on the answer itself.
#
# An answer drawn from the transcript reuses its words. An answer drawn from
# what the model knows does not: "Paris is the capital of France" shares almost
# no content words with a conversation about a fund. Requiring most of the
# answer's content words to appear somewhere in the supplied text catches that
# even when the quoted support is a real line.
ANSWER_MIN_OVERLAP = 0.6
STOPWORDS = {'the','a','an','is','was','were','are','of','to','in','on','for',
             'and','or','it','he','she','they','that','this','with','at','by',
             'from','as','his','her','their','about','said','says','have','had',
             'has','be','been','you','i','we','not','no','yes','there','here'}


def answer_overlap(answer, text):
    """What fraction of the answer's content words occur in the source text.

    Prefix-matched at five characters, because the answer is now composed
    rather than copied and composition inflects: "the audit was conducted by
    KKC" becomes "KKC audited it", and an exact-token test would score that as
    invention. Five characters is short enough to join audit/audited and long
    enough not to join unrelated words.
    """
    words = [w for w in re.findall(r"[a-z0-9']+", (answer or '').lower())
             if w not in STOPWORDS and len(w) > 1]
    if not words:
        return 0.0
    hay = set(re.findall(r"[a-z0-9']+", (text or '').lower()))
    stems = {h[:5] for h in hay}
    return sum(1 for w in words if w in hay or w[:5] in stems) / len(words)


def recent_context(db, session=None, now=None, minutes=RECENT_MINUTES):
    """What was said in the last few minutes, newest session first.

    Returns (lines, rows). rows keep the observation ids so an answer can say
    what it was drawn from without those observations being released.
    """
    now = time.time() if now is None else now
    cutoff = now - minutes * 60.0
    # Bounded at BOTH ends. Without the upper bound "the last twenty minutes"
    # means "anything after the cutoff", which in production is harmless because
    # now is the wall clock, and in any test or replay silently includes the
    # future -- a window asked for around one afternoon came back full of the
    # next day's speech.
    args = [cutoff, now]
    q = ("SELECT id,session,started_at,text FROM observations "
         "WHERE started_at >= ? AND started_at <= ? AND kind='speech'")
    if session:
        q += " AND session=?"
        args.append(session)
    q += " ORDER BY started_at DESC LIMIT ?"
    args.append(RECENT_MAX_OBS)
    rows = [dict(r) for r in db.execute(q, args)][::-1]
    out, words = [], 0
    for r in reversed(rows):
        w = len(r['text'].split())
        if words + w > RECENT_MAX_WORDS:
            break
        words += w
        out.insert(0, r)
    return [r['text'] for r in out], out


def belief_context(db, limit=BELIEF_MAX):
    """Current, non-faint, reasonably certain beliefs. Nothing superseded."""
    out = []
    for r in db.execute("SELECT id,statement,certainty,weight,last_touched "
                        "FROM beliefs ORDER BY formed_at DESC"):
        if r['certainty'] < BELIEF_MIN_CERTAINTY:
            continue
        if mem.replaced_by(db, r['id']):
            continue
        if mem.decayed_weight(r['weight'], r['last_touched']) < mem.FAINT:
            continue
        out.append(dict(r))
        if len(out) >= limit:
            break
    return out


def compose_from(question, texts):
    """One spoken sentence composed from the fragments the search released.

    The search path used to speak the highest-ranked fragment verbatim, which
    meant answering a question by reading the person's own sentence back at
    them. That is what made it unpleasant to hear: they said it, so it tells
    them nothing.

    The same model, the same warm process and the same two grounding checks as
    try_answer -- only the input differs. Composing is summarisation, which is
    what the on-device model is for, and it does not loosen where the answer
    comes from: the model must still quote a line that exists, and the words it
    composes must still be words that were said.

    Raises NotAnswerable, in which case nothing is spoken. Silence is better
    than an answer that could not be checked.
    """
    lines = [t for t in (texts or []) if (t or '').strip()]
    if not lines:
        raise NotAnswerable('nothing to compose from')
    payload = json.dumps({'question': question,
                          'transcript': "\n".join(lines)}).encode()
    t = time.perf_counter()
    r = rel.request(f"recall {len(payload)}\n", payload, REPLY_BUDGET_S)
    took = time.perf_counter() - t
    if r.get('unavailable'):
        raise rel.Unavailable(r['unavailable'])
    if r.get('error'):
        raise NotAnswerable(f"the model declined: {r['error']}")
    if not r.get('canAnswer') or not (r.get('answer') or '').strip():
        raise NotAnswerable('the fragments do not answer it')
    match = grounded(r.get('support', ''), lines)
    if match is None:
        raise NotAnswerable('the answer was not grounded in the fragments')
    joined = "\n".join(lines)
    overlap = answer_overlap(r['answer'], joined)
    if overlap < ANSWER_MIN_OVERLAP:
        raise NotAnswerable(
            f'the answer used words that were never said '
            f'({overlap:.0%} of it appears in the fragments)')
    return {'answer': r['answer'].strip(), 'support': match,
            'overlap': round(overlap, 3), 'seconds': round(took, 3),
            'from_fragments': len(lines)}


def try_answer(db, question, session=None, use_beliefs=True, now=None):
    """Answer from what is already known, or raise NotAnswerable.

    Never returns a guess. Every path that is not a grounded, confident answer
    raises, and the caller searches.
    """
    lines, rows = recent_context(db, session=session, now=now)
    beliefs = belief_context(db) if use_beliefs else []
    if not lines and not beliefs:
        raise NotAnswerable('nothing recent to answer from')

    block = []
    if beliefs:
        block += [f"[what is already concluded] {b['statement']}" for b in beliefs]
    block += list(lines)
    payload = json.dumps({'question': question,
                          'transcript': "\n".join(block)}).encode()
    t = time.perf_counter()
    r = rel.request(f"recall {len(payload)}\n", payload, REPLY_BUDGET_S)
    took = time.perf_counter() - t
    if r.get('unavailable'):
        raise rel.Unavailable(r['unavailable'])
    if r.get('error'):
        raise NotAnswerable(f"the model declined: {r['error']}")
    if not r.get('canAnswer') or not (r.get('answer') or '').strip():
        raise NotAnswerable('not answerable from what was recently said')

    match = grounded(r.get('support', ''), block)
    if match is None:
        # the model produced an answer it could not point at. This is the
        # general-knowledge case, and it is refused rather than logged and kept.
        raise NotAnswerable(
            'the answer was not grounded in anything that was said')
    joined = "\n".join(block)
    overlap = answer_overlap(r['answer'], joined)
    if overlap < ANSWER_MIN_OVERLAP:
        raise NotAnswerable(
            f'the answer used words that were never said '
            f'({overlap:.0%} of it appears in the transcript)')

    src_ids, src_kind = [], 'speech'
    if match.startswith('[what is already concluded] '):
        stmt = match[len('[what is already concluded] '):]
        src_kind = 'belief'
        src_ids = [b['id'] for b in beliefs if b['statement'] == stmt]
    else:
        src_ids = [r_['id'] for r_ in rows if r_['text'] == match]
    return {'answer': r['answer'].strip(), 'support': match,
            'overlap': round(overlap, 3),
            'source_kind': src_kind, 'source_ids': src_ids,
            'seconds': round(took, 3),
            'considered': {'observations': len(rows), 'beliefs': len(beliefs)}}
