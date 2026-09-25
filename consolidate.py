"""Noticing that something is worth concluding, and writing it down.

Nothing wrote beliefs on their own, so weeks of use produced thousands of
observations and almost no conclusions -- and the conclusion half of memory is
the half nothing has tested. Every constant in the fading rule is invented and
the evidence that would settle them cannot accumulate while nothing accumulates.

WHICH SHAPE. This is the overnight one, run over a whole day, and it is the
shape the original design called for: consolidation always runs from raw source
rather than from yesterday's summary, so a summary of a summary never compounds
drift. run_day() reads observations and nothing else.

The as-it-happens shape was the alternative and I did not build it, for a
reason that is the same as the hard part of the job. Deciding whether something
is worth a conclusion needs the rest of the day: "the same thing said three
ways" cannot be seen from the first way, and "an arrangement that changed ten
minutes later" cannot be seen until ten minutes later. A per-utterance version
is cheap precisely because it does not have that context, and would write from
one sentence lifted out of the conversation around it, which is the failure
mode being guarded against. It would suit a narrow case -- an explicit "remind
me that..." -- and that is a different job from this one.

EPISODES. A day is cut into runs of continuous talk by EPISODE_GAP_S, taken
from the gap distribution: within a session the median gap is 1.0 s, p95 is
24.9 s and p99 is 315.9 s. Two minutes sits between the last two, so an episode
holds a conversation and stops at the break between conversations.

CERTAINTY is not asked of the model. See certainty_for().
"""

import re
import time

import memory as mem

EPISODE_GAP_S = 120.0
EPISODE_MIN_WORDS = 25        # below this there is not enough to conclude from
EPISODE_MAX_WORDS = 400       # above it the model is reading a passage, not a remark
EPISODE_MAX_OBS = 80
BUDGET_S = 30.0

# Certainty, and why it is mechanical rather than asked for.
#
# A model's own confidence is not calibrated and this one has no way to become
# calibrated: nothing checks these conclusions afterwards, so a number it
# invented would be believed exactly as much as a number that meant something.
# So certainty is computed from things that are counted, not judged.
#
# It is also capped well below 1. Nothing in this system validates a belief
# after it is written, and there is currently no way to tell it a belief is
# wrong other than replacing it by hand, so a conclusion drawn by a small model
# from a transcript of room audio should never present as near-certain.
CERTAINTY_BASE = 0.45
CERTAINTY_PER_SUPPORTING = 0.05     # observations in the episode, beyond the first
CERTAINTY_RECOGNISED = 0.10         # the speaker was recognised, not unknown
CERTAINTY_CAP = 0.75
CERTAINTY_FLOOR = 0.35

# How closely the model's quoted support must match a line it was shown, and
# how much of the statement must be words that were actually said. Same rules
# and same thresholds as recall.py, for the same reason.
SUPPORT_MIN_RATIO = 0.80
# Much lower than recall's 0.6, and measured rather than guessed. A conclusion
# abstracts by nature: "He decided against Parakeet" drawn from "I'm staying on
# the Apple recogniser" shares one content word in three, because "decided" and
# "against" are the conclusion rather than the speech. A leaked or invented
# statement scores 0 -- when the model copied an instruction example into an
# answer about a flight, the overlap was zero. So the discriminating range is 0
# against 33 percent, and the threshold sits below the real case.
#
# The real guard here is the support line, which must be verbatim from the
# transcript and proves the model read it. This is the backstop for the case
# where the support is real and the statement is about something else.
STATEMENT_MIN_OVERLAP = 0.25


def episodes(db, day, gap=EPISODE_GAP_S):
    """One day's speech, cut into runs of continuous talk."""
    rows = [dict(r) for r in db.execute(
        "SELECT id,session,started_at,ended_at,text,person_decision "
        "FROM observations WHERE kind='speech' "
        "AND date(started_at,'unixepoch','localtime')=? "
        "ORDER BY started_at", (day,))]
    out, cur = [], []
    for r in rows:
        if cur and (r['session'] != cur[-1]['session']
                    or r['started_at'] - cur[-1]['ended_at'] > gap):
            out.append(cur)
            cur = []
        cur.append(r)
        if len(cur) >= EPISODE_MAX_OBS:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def usable(ep):
    """Is this episode even worth showing the model. Free, and rejects most."""
    words = sum(len(r['text'].split()) for r in ep)
    if words < EPISODE_MIN_WORDS:
        return False, f'{words} words'
    return True, f'{words} words, {len(ep)} utterances'


def episode_text(ep, max_words=EPISODE_MAX_WORDS):
    out, n = [], 0
    for r in ep:
        w = len(r['text'].split())
        if n + w > max_words:
            break
        n += w
        out.append(r['text'].strip())
    return "\n".join(out)


def certainty_for(ep, support_row):
    """How firmly to hold this, from what can be counted."""
    n = len(ep)
    recognised = sum(1 for r in ep
                     if r['person_decision'] in ('match', 'confident'))
    c = (CERTAINTY_BASE
         + CERTAINTY_PER_SUPPORTING * max(0, min(n - 1, 6))
         + (CERTAINTY_RECOGNISED if recognised > len(ep) / 2 else 0.0))
    return round(max(CERTAINTY_FLOOR, min(CERTAINTY_CAP, c)), 3)


def statement_overlap(statement, text):
    from recall import STOPWORDS
    words = [w for w in re.findall(r"[a-z0-9']+", (statement or '').lower())
             if w not in STOPWORDS and len(w) > 1]
    if not words:
        return 0.0
    hay = set(re.findall(r"[a-z0-9']+", (text or '').lower()))
    stems = {h[:5] for h in hay}
    return sum(1 for w in words if w in hay or w[:5] in stems) / len(words)


def consider(ep, prefer=None, db=None, caller='consolidation'):
    """One episode. Returns a proposal, or None with the reason it was rejected."""
    import capability as cap
    from recall import grounded
    ok, detail = usable(ep)
    if not ok:
        return None, f'too little to conclude from ({detail})'
    text = episode_text(ep)
    lines = text.split("\n")
    try:
        r, via = cap.ask('conclude_from', {'transcript': text},
                         budget=BUDGET_S, prefer=prefer, db=db, caller=caller)
    except cap.NoProvider as e:
        return None, f'no provider: {e}'
    except cap.OutwardRefused as e:
        return None, f'refused outward: {e}'
    except cap.OverBudget as e:
        return None, f'over budget: {e}'
    if r.get('error'):
        return None, f"model declined: {str(r['error'])[:60]}"
    if not r.get('worth'):
        return None, 'nothing worth concluding'
    stmt = (r.get('statement') or '').strip()
    # one subject, not a list: the model sometimes answers "travel, family",
    # and two subjects means the belief is findable under neither reliably
    about = (r.get('about') or '').strip().lower()
    about = re.split(r'[,;/]| and ', about)[0].strip()
    if not stmt:
        return None, 'said worth but wrote nothing'
    if not about:
        return None, 'no subject'
    match = grounded(r.get('support', ''), lines)
    if match is None:
        return None, 'the conclusion was not grounded in anything said'
    ov = statement_overlap(stmt, text)
    if ov < STATEMENT_MIN_OVERLAP:
        return None, f'used words that were never said ({ov:.0%} overlap)'
    return {'provider': via['provider'], 'forced': via.get('forced', False),
            'statement': stmt, 'about': about, 'support': match,
            'overlap': round(ov, 3),
            'certainty': certainty_for(ep, match),
            'sources': [r_['id'] for r_ in ep],
            'started_at': ep[0]['started_at'], 'n_obs': len(ep),
            'seconds': r.get('seconds', 0.0)}, None


def find_superseded(db, proposal, author):
    """An existing belief this one replaces, or None.

    Only current beliefs on the same subject are considered, so this cannot
    reach across topics, and the model is asked one pairwise question rather
    than being shown the whole store.
    """
    import capability as cap
    cands = [dict(r) for r in db.execute(
        "SELECT id,statement FROM beliefs WHERE about=? ORDER BY formed_at DESC",
        (proposal['about'],))]
    for c in cands:
        if mem.replaced_by(db, c['id']):
            continue
        try:
            r, _ = cap.ask('judge_supersedes',
                           {'old': c['statement'], 'new': proposal['statement']},
                           budget=BUDGET_S)
        except cap.NoProvider:
            return None
        if r.get('supersedes'):
            return c
    return None


def run_day(db, day, author='consolidation', dry_run=False, verbose=True,
            prefer=None):
    """Everything heard on one day, from raw observations. Returns a report."""
    eps = episodes(db, day)
    written, rejected = [], []
    t0 = time.perf_counter()
    for i, ep in enumerate(eps):
        proposal, why = consider(ep, prefer=prefer, db=db, caller=author)
        if proposal is None:
            rejected.append({'episode': i, 'n_obs': len(ep), 'why': why,
                             'first': ep[0]['text'][:60]})
            continue
        old = find_superseded(db, proposal, author)
        proposal['replaces'] = old['id'] if old else None
        proposal['replaces_statement'] = old['statement'] if old else None
        if not dry_run:
            bid = mem.add_belief(
                db, proposal['statement'], proposal['certainty'],
                proposal['sources'], author=author, about=proposal['about'],
                replaces=proposal['replaces'],
                formed_at=proposal['started_at'])
            proposal['id'] = bid
        written.append(proposal)
        if verbose:
            print(f"  [{i}] {proposal['statement']}", flush=True)
    return {'day': day, 'episodes': len(eps), 'written': written,
            'rejected': rejected, 'seconds': round(time.perf_counter() - t0, 1)}


# --- the nightly pass --------------------------------------------------------
# How far back a single run will reach. A laptop that slept for a fortnight
# should not wake up and spend a fortnight's worth of hosted-model calls in one
# unattended go; it catches up a few days at a time and says what it skipped.
MAX_BACKFILL_DAYS = 5
MARKER = 'consolidated_through'      # the last day fully consolidated


def marker(db):
    r = db.execute("SELECT value FROM meta WHERE key=?", (MARKER,)).fetchone()
    return r[0] if r else None


def set_marker(db, day):
    db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (MARKER, day))
    db.commit()
    return day


def pending_days(db, today=None, max_days=MAX_BACKFILL_DAYS):
    """Which days still need consolidating, oldest first.

    Everything from the day after the marker up to yesterday. Today is left
    alone because it is not over: consolidating a partial day would write
    beliefs from half an evening and then never revisit it.

    With no marker, this takes only yesterday rather than the whole store --
    a first run should not be a surprise bill, and older days can be asked for
    by hand.
    """
    import datetime as dt
    today = today or dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    m = marker(db)
    if not m:
        return [yesterday.isoformat()]
    last = dt.date.fromisoformat(m)
    days, d = [], last + dt.timedelta(days=1)
    while d <= yesterday:
        days.append(d.isoformat())
        d += dt.timedelta(days=1)
    return days[-max_days:], len(days) - len(days[-max_days:])


def preflight(prefer, verbose=True):
    """Is the judge this run needs actually there, before any day is touched.

    WHAT DEGRADING MEANS HERE, since "degrade rather than break" is the rule
    everywhere else and the obvious reading of it is wrong.

    The obvious reading is: if the hosted model is unavailable, fall back to
    the on-device one. That is what the router does for every other task and
    it is the wrong thing here, because the on-device model cannot do this
    job. It concluded nothing over 209 episodes -- not "fewer conclusions",
    none. So falling back would consume the day, advance the marker, write
    nothing, and leave a permanent hole that looks exactly like a quiet day.
    The failure would be invisible, which is the thing this project keeps
    getting wrong.

    So degrading, honestly, is to skip the day and say so. The marker does not
    advance, nothing is written, the log says which provider was missing and
    why, and the next run picks the day up as pending. A missed night costs a
    day of latency; a night that silently produced nothing costs the day
    itself, and nobody would ever know to go back for it.

    Returns (ok, why). A caller that wants the fallback anyway can pass
    prefer=None and take whatever the router gives it.
    """
    if not prefer:
        return True, 'default routing'
    import capability as cap
    p = next((x for x in cap.PROVIDERS if x.name == prefer), None)
    if p is None:
        return False, f'no provider named {prefer!r}'
    ok, why = p.available()
    if not ok:
        return False, f'{prefer} is not available: {why}'
    return True, f'{prefer} is ready'


def nightly(db, author='consolidation', prefer=None, today=None, verbose=True):
    """Consolidate every day that has not been done, oldest first.

    Idempotent by the marker rather than by inspecting beliefs: running twice
    in a night does nothing the second time, and a day is either done or it is
    not. The marker only advances on a day that completed, so a crash halfway
    through leaves that day to be retried rather than silently skipped.
    """
    out = pending_days(db, today=today)
    days, skipped = out if isinstance(out, tuple) else (out, 0)
    result = {'days': days, 'skipped_days': skipped, 'runs': [],
              'written': 0, 'rejected': 0, 'deferred': False}
    ok, why = preflight(prefer, verbose)
    if not ok:
        result.update(deferred=True, why=why, days=[])
        if verbose:
            print(f"DEFERRED: {why}.")
            print(f"  {len(days)} day(s) left pending: {', '.join(days) or 'none'}")
            print("  Nothing was written and the marker did not advance, so the "
                  "next run picks them up. Falling back to the on-device model "
                  "would have consumed the day and concluded nothing, which "
                  "reads as a quiet day rather than as a failure.")
        return result
    if verbose:
        print(f"judge: {why}")
    if skipped and verbose:
        print(f"skipping {skipped} day(s) older than the {MAX_BACKFILL_DAYS}-day "
              f"backfill limit; ask for them by hand if you want them")
    for day in days:
        r = run_day(db, day, author=author, prefer=prefer, verbose=verbose)
        result['runs'].append({'day': day, 'episodes': r['episodes'],
                               'written': len(r['written']),
                               'rejected': len(r['rejected'])})
        result['written'] += len(r['written'])
        result['rejected'] += len(r['rejected'])
        set_marker(db, day)
    return result


def main():
    """usage: consolidate.py nightly | <YYYY-MM-DD> [--dry-run] [--prefer NAME]"""
    import sys
    import time
    import memory as m
    args = sys.argv[1:]
    if not args:
        print(main.__doc__.strip())
        return 1
    prefer = args[args.index('--prefer') + 1] if '--prefer' in args else None
    db = m.open()
    started = time.time()
    print(f"=== consolidation {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"prefer={prefer or 'default routing'}")
    if args[0] == 'nightly':
        r = nightly(db, prefer=prefer, verbose=True)
        print(f"days {r['days']}, written {r['written']}, "
              f"rejected {r['rejected']}, {time.time()-started:.0f}s")
    else:
        r = run_day(db, args[0], prefer=prefer, dry_run=('--dry-run' in args))
        print(f"{args[0]}: written {len(r['written'])}, "
              f"rejected {len(r['rejected'])}, {time.time()-started:.0f}s")
    import capability as cap
    for p in ('claude-api', 'gemini'):
        d = cap.spend_today(p)
        if d['calls']:
            print(f"spend {p}: {d['calls']} calls, {d['in']}+{d['out']} tokens, "
                  f"${d['usd']:.4f}, {d['errors']} error(s)")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
