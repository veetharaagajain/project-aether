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
# ...and the same test applied per clause, which is a different check and not a
# tighter one.
#
# answer_overlap scores the whole sentence, so a fabricated clause bolted onto
# a true one rides in on the true half's words: "You went for a morning jog in
# the park and then drove to Bristol", against lines that mention the jog and
# nothing else, scores 0.71 and is accepted at any setting this project has
# ever shipped. No threshold fixes it -- that case scores 0.71 and the weakest
# legitimate rephrasing scores 0.67, so the two populations overlap and any cut
# that refuses the invention refuses honest paraphrase too.
#
# The difference is not how much of the answer is supported but whether any
# whole limb of it is unsupported. So each clause is scored on its own and the
# weakest one decides. "You went for a morning jog in the park" scores 1.00 and
# "then drove to Bristol" scores 0.00; the sentence is refused on the second.
CLAUSE_MIN_OVERLAP = 0.34      # a clause may be a third invented, no more
CLAUSE_MIN_CONTENT = 2         # shorter than this is not a claim, it is a joint
CLAUSE_SPLIT = re.compile(
    r'\s*(?:,\s*(?:and|but|then|so|which|who|where|while)\b|'
    r'\b(?:and then|but then|and also|and|but|then|whereas|while)\b|[;:])\s*',
    re.I)


MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december',
          'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept',
          'oct', 'nov', 'dec')


def date_tokens(text):
    """Month names and years in a piece of text.

    Dates are the one thing that cannot be borrowed between lines. Every other
    word in the record is fair game across the whole retrieved set -- a name
    recurs, a topic recurs -- but "15 May 2023" belongs to the line it is
    printed on, and an answer that takes a date from one line to describe an
    event on another has invented something even though every word of it
    appears somewhere in the haystack.

    This only became reachable when lines started carrying dates. Before that
    a date in an answer had to have been spoken aloud to pass, and now it can
    be lifted from any prefix in view.
    """
    t = (text or '').lower()
    out = {w for w in re.findall(r"[a-z]+", t) if w in MONTHS}
    out |= set(re.findall(r"\b(?:19|20)\d{2}\b", t))
    return out


def clause_overlaps(answer, text):
    """Every clause of the answer, scored on its own.

    Returns a list of (clause, overlap). Clauses with fewer than
    CLAUSE_MIN_CONTENT content words are skipped rather than scored: "and then"
    carries no claim and scoring it as zero would refuse every compound
    sentence.
    """
    out = []
    for part in CLAUSE_SPLIT.split(answer or ''):
        part = (part or '').strip(' ,.;:!?')
        if not part:
            continue
        content = [w for w in re.findall(r"[a-z0-9']+", part.lower())
                   if w not in STOPWORDS and len(w) > 1]
        if len(content) < CLAUSE_MIN_CONTENT:
            continue
        out.append((part, answer_overlap(part, text)))
    return out


def weakest_clause(answer, text):
    """The least supported clause, or None when there is nothing to score."""
    scored = clause_overlaps(answer, text)
    return min(scored, key=lambda t: t[1]) if scored else None
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


# How a retrieved line is written for the model. The store holds when each
# observation happened and who said it, and both were being dropped on the way
# to the composer, which only ever saw bare text.
#
# That is invisible on room audio, where the being answers about "you" from one
# speaker in the last few minutes. It is fatal anywhere else. Asked "when did
# Jon start reading The Lean Startup", the retrieved line said "I'm currently
# reading The Lean Startup" and the answer -- May 2023 -- was the timestamp,
# not the words. Measured on the benchmark, attaching these took the same
# questions over the same retrieved observations from 12% to 40%.
#
# The date is a day, not a second: "15 May 2023" is what a question about when
# something happened needs, and a wall-clock time would be noise in the
# haystack the overlap check reads.
LINE_FORMAT = "[{when}] {who}: {text}"
LINE_DATE = "%d %B %Y"


def format_line(text, at=None, who=None):
    """One retrieved line, with what the store knows about it attached."""
    if at is None and not who:
        return text
    when = time.strftime(LINE_DATE, time.localtime(at)) if at else "undated"
    return LINE_FORMAT.format(when=when, who=(who or "someone"), text=text)


# Below this many words a released line cannot stand on its own, and the
# window that found it is released instead.
#
# memory.search deliberately searches windows and answers with observations --
# "a centre is never returned twice, and what comes back is the record, never
# the window" -- which is right for identifying WHAT matched and wrong for
# handing it to a composer. On this store the median observation is four words
# and 60 percent are four or fewer, so the top hits for "what am I building"
# were "I.", "I." and "I". The neighbourhood was already built and already
# scored; it was being discarded at the last step.
#
# Eight, because it is the smallest number that clears the fragment
# population: on this store 51 percent of retrieved lines are two words or
# fewer and the median is two, while LoCoMo lines -- which compose fine -- run
# to a median of 19. Anything at or above eight is already a clause that can
# carry a claim, and expanding it would release speech for nothing.
STAND_ALONE_WORDS = 8


def expand_short(db, hits, threshold=STAND_ALONE_WORDS):
    """Replace fragments with the neighbourhood that found them.

    No judgement is involved and none is needed: the test is word count, which
    is arithmetic, and the window is the one memory.search already scored. What
    changes is only how much of the record the composer is shown.

    Each hit gains 'released': 'observation' or 'window', 'matched' holding the
    fragment that actually scored, and for an expanded hit 'window_text' with
    the centre marked by memory.window_text so the caller can see which line
    matched and which came along with it.
    """
    import memory as mem
    out = []
    for h in hits:
        h = dict(h)
        h['matched'] = h.get('text', '')
        h['released'] = 'observation'
        if h.get('kind') == 'belief' or db is None:
            out.append(h)
            continue
        if len((h.get('text') or '').split()) >= threshold:
            out.append(h)
            continue
        w = db.execute("SELECT first_id, last_id, n_obs FROM windows "
                       "WHERE centre_id=?", (h.get('id'),)).fetchone()
        if w is None:
            out.append(h)
            continue
        rows = [dict(r) for r in db.execute(
            "SELECT id, text, started_at, person FROM observations "
            "WHERE id >= ? AND id <= ? ORDER BY id", (w['first_id'], w['last_id']))]
        if len(rows) <= 1:
            out.append(h)
            continue
        for r in rows:
            r['offset'] = 0 if r['id'] == h.get('id') else 1
        h['released'] = 'window'
        h['n_released'] = len(rows)
        h['window_text'] = mem.window_text(rows)
        h['text'] = " ".join(r['text'] for r in rows)
        h['at'] = rows[0]['started_at']
        out.append(h)
    return out


def format_lines(hits, db=None, expand=True):
    """The lines a composer should be shown, from search hits or observation
    rows. Accepts either shape: search returns 'at' and 'person', a row from
    the observations table has 'started_at' and 'speaker'.

    With a db and expand set, a line too short to stand alone is replaced by
    its window first.
    """
    if db is not None and expand:
        hits = expand_short(db, [h for h in hits if not isinstance(h, str)]) \
            + [h for h in hits if isinstance(h, str)]
    out = []
    for h in hits:
        if isinstance(h, str):
            out.append(h)
            continue
        at = h.get('at') if 'at' in h else h.get('started_at')
        who = h.get('person') or h.get('speaker')
        out.append(format_line(h.get('text', ''), at, who))
    return out


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
    return format_lines(out), out


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
    import capability as cap
    t = time.perf_counter()
    try:
        r, via = cap.ask('compose_answer',
                         {'question': question, 'transcript': "\n".join(lines)},
                         budget=REPLY_BUDGET_S)
    except cap.NoProvider as e:
        raise NotAnswerable(f'nothing can compose an answer: {e}')
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
    weak = weakest_clause(r['answer'], joined)
    if weak and weak[1] < CLAUSE_MIN_OVERLAP:
        raise NotAnswerable(
            f'one clause of the answer was never said '
            f'({weak[1]:.0%} of "{weak[0][:50]}" appears in the fragments)')
    stray = date_tokens(r['answer']) - date_tokens(match)
    if stray:
        raise NotAnswerable(
            f'the answer dates it {", ".join(sorted(stray))}, which is not on '
            f'the line it cites')
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
    import capability as cap
    t = time.perf_counter()
    try:
        r, via = cap.ask('compose_answer',
                         {'question': question, 'transcript': "\n".join(block)},
                         budget=REPLY_BUDGET_S)
    except cap.NoProvider as e:
        raise NotAnswerable(f'nothing can compose an answer: {e}')
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


# --- saying that it does not know -------------------------------------------
# Words that make a sentence a refusal rather than a claim. The composed reply
# has to contain one: the model is asked for a refusal, and this checks that it
# produced one, because the failure that matters here is a near miss arriving
# sounding like an answer.
NEGATIONS = ('nothing', 'not ', "n't", ' no ', 'never', 'nobody', 'none',
             'no one', 'anything about', 'haven', 'hasn')

# The nearby fact is held to the same standard as an answer would be, because
# it is the same kind of claim about what was heard. The framing around it is
# the model's own words and is not checked for overlap -- "nothing about
# breakfast" cannot appear in a transcript that contains nothing about
# breakfast.
NEARBY_MIN_OVERLAP = ANSWER_MIN_OVERLAP

# Topics that are not topics. A refusal built on one of these is a sentence
# about a function word rather than about anything the person asked.
DEGENERATE_TOPICS = {
    'nothing', 'anything', 'something', 'it', 'that', 'this', 'them', 'they',
    'you', 'me', 'i', 'we', 'us', 'thing', 'things', 'stuff', 'next', 'do',
    'the', 'a', 'an', 'what', 'which', 'who', 'why', 'how', 'when', 'where',
    'yes', 'no', 'ok', 'okay', 'hello', 'hi', 'hey', 'question', 'answer',
}

# Words the refusal is allowed to use that were never said and are not in the
# question: the framing itself. Anything outside this, the question and the
# retrieved lines is invention, and invention is what this whole path exists to
# avoid. This caught a real failure -- with no fragments at all the model
# produced "nothing about breakfast, you mentioned ramen for lunch", having
# copied the example out of its own instructions, while reporting nearby=false
# so the grounding check never ran on it.
FRAMING = {
    'nothing', 'about', 'anything', 'covers', 'covered', 'mention', 'mentioned',
    'said', 'say', 'says', 'saying', 'heard', 'hear', 'know', 'known', 'knows',
    'only', 'here', 'yet', 'still', 'but', 'though', 'while', 'else',
    'record', 'anyone', 'anybody', 'nobody', 'none', 'never', 'not', 'no',
    'been', 'was', 'were', 'is', 'are', 'am', 'do', 'does', 'did', 'doesn',
    'don', 'didn', 'haven', 'hasn', 'wasn', 'weren', 'isn', 'aren', 'can',
    'cannot', 'could', 'would', 'that', 'this', 'it', 'you', 'your', 'i',
    'me', 'my', 'we', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'and',
    'or', 'from', 'with', 'at', 'as', 'by', 'so', 'if', 'what', 'which',
    'there', 'their', 'they', 'them', 'anywhere', 'nothing', 'far', 'apart',
    # ...plus ordinary vocabulary a refusal reaches for while paraphrasing the
    # question back. These are not facts about anybody, so they are not the
    # invention this guards against, and refusing on them made it fire on
    # "any", "location" and "color".
    'any', 'anything', 'some', 'something', 'thing', 'things', 'one', 'other',
    'location', 'place', 'where', 'when', 'who', 'whom', 'whose', 'why', 'how',
    'time', 'times', 'date', 'day', 'today', 'tomorrow', 'yesterday', 'week',
    'colour', 'color', 'reason', 'reasons', 'answer', 'question', 'topic',
    'subject', 'detail', 'details', 'information', 'specific', 'specifics',
    'plan', 'plans', 'yet', 'so far', 'discussed', 'discuss', 'came', 'come',
    'up', 'talked', 'talk', 'told', 'tell', 'asked', 'ask', 'recall',
    'remember', 'been', 'having', 'have', 'has', 'had', 'get', 'got', 'going',
    'go', 'want', 'need', 'like', 'just', 'now', 'then', 'also', 'either',
    'nor', 'because', 'about', 'regarding', 'concerning', 'related', 'nearby',
    'else', 'more', 'less', 'much', 'many', 'all', 'both', 'each', 'every',
}


def invented_words(reply, question, lines):
    """Content words in the reply that came from nowhere.

    The framing of a refusal cannot be checked by overlap the way an answer can
    -- "nothing about breakfast" will never appear in a transcript containing
    nothing about breakfast. So the test is the other way round: every word the
    reply uses must come from the question, from the lines that were retrieved,
    or from the fixed vocabulary a refusal is made of. What is left over is
    invention.
    """
    allowed = set(FRAMING) | set(STOPWORDS)
    for src in [question or ''] + list(lines or []):
        allowed |= set(re.findall(r"[a-z0-9']+", src.lower()))
    stems = {w[:5] for w in allowed}
    out = []
    for w in re.findall(r"[a-z0-9']+", (reply or '').lower()):
        if len(w) <= 1 or w in allowed or w[:5] in stems:
            continue
        out.append(w)
    return out


class NotRecallable(Exception):
    """The question was never about what anybody said. Distinct from finding
    nothing: claiming to have looked is false when the question needed
    knowledge of the world."""

    def __init__(self, message, topic='', reply=''):
        super().__init__(message)
        self.topic = topic
        # composed, not fixed: the model wrote this sentence too, under
        # instructions that tell it to name the subject and not to claim the
        # record is empty.
        self.reply = reply


def compose_not_found(question, texts):
    """Compose what to say when nothing answers the question.

    Returns a dict with the sentence to speak and which case it was, or raises
    NotAnswerable, in which case the caller stays silent as before -- a refusal
    that cannot be composed is no better than the fixed string this exists to
    avoid. Raises NotRecallable when the question was never about speech.

    Three outcomes, because a person can tell them apart:

      'nothing'    nothing was retrieved at all
      'unanswered' fragments were retrieved and none of them answers
      NotRecallable  the question needed knowledge of the world or reasoning

    THE MODEL SUPPLIES THE PARTS AND THIS ASSEMBLES THEM, which is not where
    this started. Asked for the whole sentence, the on-device model kept
    producing refusals that then leaked a retrieved line: "There is nothing
    about what the dentist said. The meeting got moved to four." -- with
    nearby set false, so it was not even claiming the two were connected. It
    also, given no fragments at all, copied an example out of its own
    instructions and offered it as something the person had mentioned. Both
    are exactly the failure this path exists to prevent, and both survived a
    prompt that told it not to in three different ways.

    So the structure is guaranteed here instead of requested there. The
    refusal always comes first and the related thing, if any, is always
    attached as a separate clause that says it is a different fact. What
    varies between one of these and the next is the topic and the nearby
    fact, both of which come from the material -- which is the point, since
    the same sentence every time is what makes silence preferable.

    The grounding checks still run on the part that is a claim about what was
    heard: the supporting line must occur in the fragments, and the nearby
    fact must be made of words that were actually said.
    """
    lines = [t for t in (texts or []) if (t or '').strip()]
    import capability as cap
    # numbered, because picking a line out of a list is a much easier task for
    # a small model than judging a boolean about relatedness, and the boolean
    # version returned false on every pair tried, including lunch against
    # dinner.
    numbered = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    t = time.perf_counter()
    try:
        r, via = cap.ask('compose_not_found',
                         {'question': question, 'transcript': numbered},
                         budget=REPLY_BUDGET_S)
    except cap.NoProvider as e:
        raise NotAnswerable(f'nothing can compose a refusal: {e}')
    took = time.perf_counter() - t
    if r.get('unavailable'):
        raise rel.Unavailable(r['unavailable'])
    if r.get('error'):
        raise NotAnswerable(f"the model declined: {r['error']}")

    topic = _clean_topic(r.get('topic'), question)
    if not topic:
        raise NotAnswerable('could not name what was asked about')
    # "Nothing about nothing has been said here" was a real thing this said
    # out loud, to "what do you think I should do next". The topic extractor
    # correctly found no subject and reported it literally, and the sentence
    # was assembled around the word anyway. A degenerate topic means the
    # question had nothing in it to look up, which is silence, not a refusal
    # about a word.
    import addressed as _ad
    toks = [t for t in re.findall(r"[a-z0-9']+", topic) if t]
    degenerate = (not toks
                  or topic in DEGENERATE_TOPICS
                  or len(topic) < 2
                  or all(t in DEGENERATE_TOPICS or t in _ad.EMPTY_TOKENS
                         for t in toks))
    if degenerate:
        raise NotAnswerable(
            f'the question has no subject to be missing: topic came out '
            f'{topic!r}')

    # ...but only when the search came back with nothing at all. Asked "what
    # did the dentist say" against two unrelated lines, the model set this true
    # -- and that is a plainly recallable question. If retrieval matched
    # anything for it, it was a question about what was said, whatever the
    # model thinks, so the structure overrules it.
    if r.get('needsOutsideKnowledge') and not lines:
        reply = (r.get('reply') or '').strip()
        if not reply or invented_words(reply, question, []):
            reply = (f"I only know what's been said here, and nothing about "
                     f"{topic} has been.")
        raise NotRecallable('the question needs knowledge or reasoning, '
                            'not recall', topic=topic, reply=reply)

    picked = int(r.get('closestLine') or 0)
    fact = (r.get('nearbyFact') or '').strip().rstrip('.')
    nearby = bool(fact) and 1 <= picked <= len(lines)
    support = None
    if nearby:
        # the chosen line is the one the fact has to come from, not any line
        support = grounded(r.get('support', ''), [lines[picked - 1]]) \
            or grounded(r.get('support', ''), lines)
        if support is None:
            raise NotAnswerable('the nearby fact was not grounded in the fragments')
        overlap = answer_overlap(fact, "\n".join(lines))
        if overlap < NEARBY_MIN_OVERLAP:
            raise NotAnswerable(
                f'the nearby fact used words that were never said '
                f'({overlap:.0%} of it appears in the fragments)')
        made_up = invented_words(fact, question, lines)
        if made_up:
            raise NotAnswerable(
                f'the nearby fact used words from nowhere: '
                f'{", ".join(made_up[:6])}')

    if nearby:
        # the refusal first, then the other thing marked as a different fact.
        # "you did mention" rather than "you mentioned" because the did is
        # doing work: it concedes the near miss while denying the answer.
        answer = f"Nothing about {topic} -- though you did mention {fact}."
    elif lines:
        # deliberately not "and nothing close to it": that asserts a judgement
        # the model has proved unreliable at, declining to relate lunch to
        # dinner. Saying only what is certain -- that the answer is absent --
        # is the honest version of the same sentence.
        answer = f"Nothing about {topic} in what's been said."
    else:
        answer = f"Nothing about {topic} has been said here."

    return {'answer': answer, 'kind': 'unanswered' if lines else 'nothing',
            'topic': topic, 'nearby': nearby,
            'nearby_fact': fact if nearby else '',
            'support': support, 'from_fragments': len(lines),
            'seconds': round(took, 3)}


def _clean_topic(topic, question):
    """The topic, or a fallback taken from the question.

    It is spoken, so it has to read as a noun phrase: the model sometimes
    returns a whole clause or repeats the question back.
    """
    t = re.sub(r'\s+', ' ', (topic or '').strip().strip('."\'')).lower()
    t = re.sub(r'^(the question of|what|where|when|who|why|how)\s+', '', t)
    t = re.sub(r'^(is|was|are|were|did|do|does|i|you|your)\s+', '', t)
    # a trailing verb makes it a clause and it has to read after "nothing
    # about": "dentist said" and "breakfast location" both came back in
    # testing, and both are wrong in the mouth.
    t = re.sub(r'\s+(said|says|saying|told|is|was|were|are|do|did|does|'
               r'location|place|time|date)$', '', t)
    if not t or len(t.split()) > 6:
        q = re.sub(r'[?.!]+$', '', (question or '').strip()).lower()
        q = re.sub(r'^(what|where|when|who|why|how|do|did|does|is|are|was|were)\s+',
                   '', q)
        q = re.sub(r'^(i|you|we|my|your)\s+', '', q)
        t = ' '.join(q.split()[:5])
    return t.strip()
