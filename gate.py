"""Aether: the one thing between a question and the store.

Memory is never exposed. Anything that wants to know something -- Claude,
another agent, eventually another person's being -- asks Aether, and Aether
decides what it gets. That is the role the being already plays for the person,
applied one layer down.

WHERE THE JUDGEMENT GOES. decide() is the whole of it, and right now it reads a
flat table: a caller may read everything or nothing, write or not. That is
deliberately too blunt to be the real answer. The real answer is per question,
not per caller -- this agent may know that he was in a bad mood on Tuesday but
not why, may have the conclusion but not the recording it came from -- and that
needs a model, which is not in this round. What is built here is the place that
decision is made and the log it has to leave, so that adding judgement later is
replacing the body of one function rather than threading a new concept through
everything.

WHAT THE CALLER SEES. Never the store. Tools return plain dictionaries built by
_shape_*, which means a caller cannot reach a column by asking for it, cannot
see another caller's requests, and gets ids it can only spend back through
these same four tools.

BINDING. localhost, and it refuses anything else. Anything that can reach this
can read a person's life.
"""

import hashlib
import hmac
import json
import secrets
import time

import memory as mem

import incognito as inc
import memory as mem

TOOLS = ('search_memory', 'fetch_observation', 'fetch_belief',
         'write_belief', 'record_answer')
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8787
LOCAL_ONLY = ('127.0.0.1', '::1', 'localhost')


class Denied(Exception):
    """The caller was not allowed. Carries no detail about what was withheld:
    a refusal that describes the thing refused is a disclosure."""


# --- who a caller is ---------------------------------------------------------
# A shared secret per caller. The threat is another process on this machine
# calling itself claude, not someone on the network: the server binds loopback
# and nothing else can reach it. So a secret both sides hold is enough, and
# certificates or OAuth would be answering a question nobody asked.
#
# Secrets are generated here and never chosen, 32 bytes from secrets.token_*,
# so there is nothing to guess. They are stored as a scrypt hash with a
# per-caller salt anyway, because the store also holds the person's speech and
# a file that leaks should not also hand over the keys to the interface. scrypt
# at n=2**14 costs about 18 ms per call, against the ~50 ms the embedding in
# search already costs, so the slower KDF is not what anyone will notice.
SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN = 2 ** 14, 8, 1, 32

AUTH_FAILED = "authentication failed"


def new_secret():
    return secrets.token_urlsafe(32)


def _hash_secret(secret, salt):
    return hashlib.scrypt(secret.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                          p=SCRYPT_P, dklen=SCRYPT_LEN)


def add_caller(db, caller, can_read=False, can_write=False, note=None,
               secret=None):
    """Register a caller and return its secret, which is shown exactly once.

    There is no way to create a caller without one: secret=None means generate,
    not omit. That is the whole point -- a caller that was never given a secret
    must be a caller that cannot connect, never a caller that is trusted
    because nobody got round to it.
    """
    secret = secret or new_secret()
    salt = secrets.token_bytes(16)
    now = time.time()
    db.execute("INSERT INTO callers(caller,can_read,can_write,note,added_at,"
               "secret_salt,secret_hash,secret_set_at) VALUES(?,?,?,?,?,?,?,?) "
               "ON CONFLICT(caller) DO UPDATE SET "
               "can_read=excluded.can_read, can_write=excluded.can_write, "
               "note=excluded.note, secret_salt=excluded.secret_salt, "
               "secret_hash=excluded.secret_hash, "
               "secret_set_at=excluded.secret_set_at",
               (caller, int(can_read), int(can_write), note, now,
                salt, _hash_secret(secret, salt), now))
    db.commit()
    return secret


def authenticate(db, caller, secret):
    """Is this caller who it says it is. Returns (ok, reason_for_the_log).

    The reason is for the person reading the log, not for the caller: an
    unknown name and a wrong secret are distinguished here and deliberately
    collapsed into one message on the way out, so a probe cannot use the
    difference to learn which callers exist.
    """
    r = db.execute("SELECT * FROM callers WHERE caller=?", (caller,)).fetchone()
    if r is None:
        return False, "unknown caller"
    if not r["secret_hash"] or not r["secret_salt"]:
        return False, "caller has no secret set"
    if not secret:
        return False, "no secret presented"
    ok = hmac.compare_digest(_hash_secret(secret, bytes(r["secret_salt"])),
                             bytes(r["secret_hash"]))
    return (True, "authenticated") if ok else (False, "wrong secret")


def callers(db):
    """Who is registered, for showing a person. Never the salt or the hash: a
    listing that includes the material is one nobody can safely paste."""
    return [{'caller': r['caller'], 'can_read': bool(r['can_read']),
             'can_write': bool(r['can_write']), 'note': r['note'],
             'added_at': r['added_at'], 'has_secret': bool(r['secret_hash'])}
            for r in db.execute("SELECT * FROM callers ORDER BY caller")]


def decide(db, caller, tool, arguments):
    """May this caller run this tool. Returns (allowed, reason).

    THIS IS THE PLACE. Everything above it is plumbing that will not change
    when the answer gets interesting. What will change is that arguments and
    the candidate results start mattering here, and that the reason returned
    stops being a restatement of a column.

    An unknown caller is refused. Defaulting an unrecognised name to read-only
    would mean the difference between a configured caller and a typo is
    invisible.
    """
    r = db.execute("SELECT * FROM callers WHERE caller=?", (caller,)).fetchone()
    if r is None:
        return False, 'unknown caller'
    if tool in ('write_belief', 'record_answer'):
        return (bool(r['can_write']),
                'may write' if r['can_write'] else 'not permitted to write')
    return (bool(r['can_read']),
            'may read' if r['can_read'] else 'not permitted to read')


# --- the log ----------------------------------------------------------------
def admit(db, caller, secret, tool, args):
    """Authenticate, then authorise, then log whichever failed.

    Two failures with deliberately different visibility. An authentication
    failure -- unknown name, no secret, wrong secret -- comes back to the
    caller as one flat message, because telling a prober that the name exists
    but the secret is wrong hands it the list of callers. An authorisation
    failure is specific, because the caller has already proved who it is and
    telling it what it may not do is not a disclosure. The log gets the exact
    cause either way; that is who the detail is for.
    """
    ok, why = authenticate(db, caller, secret)
    if not ok:
        log(db, caller, tool, args, 'denied', why)
        raise Denied(AUTH_FAILED)
    ok, why = decide(db, caller, tool, args)
    if not ok:
        log(db, caller, tool, args, 'denied', why)
        raise Denied(why)
    return why


def log(db, caller, tool, arguments, decision, reason, returned=()):
    """Who asked, what they asked, what left the store.

    Not for debugging. The person has to be able to see what left, so this is
    written for every call including the refused ones, and it is written before
    the caller is answered rather than after.

    Ids, never content. If this copied the text of what it returned then the
    log would become a second store holding the same speech under none of the
    same rules, and forget() would have to be able to erase it -- which would
    let forget() erase the evidence of what had already been handed out.
    """
    # the secret is never among the arguments -- the tool functions build
    # `args` without it -- but stripped here too, because the cost of being
    # wrong about that is a plaintext secret in a file that survives forget().
    arguments = {k: v for k, v in (arguments or {}).items()
                 if k not in ('secret', 'caller')}
    ids = [x if isinstance(x, str) else x.get('id') for x in returned]
    ids = [i for i in ids if i]
    db.execute("INSERT INTO access_log(id,at,caller,tool,arguments,decision,"
               "reason,n_returned,returned) VALUES(?,?,?,?,?,?,?,?,?)",
               (mem.ulid(), time.time(), caller, tool,
                json.dumps(arguments, separators=(',', ':'), default=str),
                decision, reason, len(ids),
                json.dumps(ids, separators=(',', ':'))))
    db.commit()


def recent_access(db, limit=50, caller=None):
    """What has left the store, for the person. Not exposed as a tool: the
    callers are what this watches, so it is not theirs to read."""
    q = "SELECT * FROM access_log"
    a = ()
    if caller:
        q += " WHERE caller=?"
        a = (caller,)
    return [dict(r) for r in db.execute(q + " ORDER BY at DESC LIMIT ?",
                                        a + (limit,))]


# --- manual approval --------------------------------------------------------
# Temporary, and off unless switched on. Every release pauses, shows the person
# what was asked and what would actually be handed over, and waits.
#
# TWO THINGS OR IT GETS SWITCHED OFF WITHIN A DAY.
#
# A TIMEOUT, because a request nobody answers must not hang forever. 45
# seconds, and the answer on expiry is DENY. Two reasons for that number:
# it has to be comfortably inside an MCP client's own tool timeout -- Claude
# Desktop gives up around a minute -- so that an unanswered request comes back
# as a clean refusal rather than the client dying on a dead connection; and it
# is long enough to read a few lines and press a key, which is what the person
# is actually being asked to do. Denying on expiry is the only defensible
# default: silence is not consent, and the alternative is that walking away
# from the laptop releases everything.
#
# STANDING APPROVAL, because being asked the same question forty times is how a
# control gets turned off. The grain is (caller, tool) with an expiry, granted
# from the same prompt that asks -- approve once, or approve this kind for the
# next N minutes. Deliberately not finer: a standing grant that could say
# "searches about work but not about health" is a policy language, and that is
# the judgement layer this round is not building. Every standing grant expires;
# there is no permanent one, because a control that can be permanently disabled
# from inside a single prompt is not a control.
#
# CROSS-PROCESS. The gate runs inside the MCP server and the person answers in
# the viewer, which is a different process, so an in-memory event will not do.
# The request goes into the approvals table and the gate polls it; the viewer
# writes the decision. The preview of what would be handed over does NOT go in
# that row -- it is published over the transient channel so the viewer can
# show it, and never written down. See the approvals table in memory.py.
APPROVAL_TIMEOUT_S = 45.0
APPROVAL_POLL_S = 0.2
STANDING_DEFAULT_MINUTES = 60.0


def approval_mode(db):
    r = db.execute("SELECT value FROM meta WHERE key='approval_mode'").fetchone()
    return (r[0] if r else 'off') == 'on'


def set_approval_mode(db, on):
    db.execute("INSERT INTO meta(key,value) VALUES('approval_mode',?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               ('on' if on else 'off',))
    db.commit()
    return approval_mode(db)


def standing(db, caller, tool, now=None):
    """Is this kind of request approved in advance and not yet expired."""
    now = time.time() if now is None else now
    r = db.execute("SELECT * FROM standing_approvals WHERE caller=? AND tool=?",
                   (caller, tool)).fetchone()
    if r is None:
        return None
    if r['expires_at'] is not None and r['expires_at'] <= now:
        db.execute("DELETE FROM standing_approvals WHERE caller=? AND tool=?",
                   (caller, tool))
        db.commit()
        return None
    return dict(r)


def grant_standing(db, caller, tool, minutes=STANDING_DEFAULT_MINUTES,
                   note=None):
    now = time.time()
    db.execute("INSERT INTO standing_approvals(caller,tool,granted_at,"
               "expires_at,note) VALUES(?,?,?,?,?) "
               "ON CONFLICT(caller,tool) DO UPDATE SET "
               "granted_at=excluded.granted_at, expires_at=excluded.expires_at,"
               " note=excluded.note",
               (caller, tool, now, now + minutes * 60.0, note))
    db.commit()
    return standing(db, caller, tool)


def revoke_standing(db, caller=None, tool=None):
    if caller and tool:
        db.execute("DELETE FROM standing_approvals WHERE caller=? AND tool=?",
                   (caller, tool))
    elif caller:
        db.execute("DELETE FROM standing_approvals WHERE caller=?", (caller,))
    else:
        db.execute("DELETE FROM standing_approvals")
    db.commit()


def pending(db):
    return [dict(r) for r in db.execute(
        "SELECT * FROM approvals WHERE decision='pending' ORDER BY at")]


def decide_approval(db, aid, decision, by='person', standing_minutes=None):
    r = db.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
    if r is None or r['decision'] != 'pending':
        return None
    db.execute("UPDATE approvals SET decision=?, decided_at=?, decided_by=?, "
               "standing_minutes=? WHERE id=?",
               (decision, time.time(), by, standing_minutes, aid))
    db.commit()
    if decision == 'approved' and standing_minutes:
        grant_standing(db, r['caller'], r['tool'], standing_minutes,
                       note=f'granted from request {aid[:12]}')
    return dict(db.execute("SELECT * FROM approvals WHERE id=?",
                           (aid,)).fetchone())


def ask(db, caller, tool, args, results, preview):
    """Hold the release until a person decides. Returns (ok, reason).

    results is what would be handed over; preview is the short human-readable
    form of it. Only ids and a count reach the table.
    """
    import channel as ch
    ids = [x.get('id') for x in results if isinstance(x, dict) and x.get('id')]
    aid = mem.ulid()
    now = time.time()
    db.execute("INSERT INTO approvals(id,at,caller,tool,arguments,"
               "n_would_return,ids) VALUES(?,?,?,?,?,?,?)",
               (aid, now, caller, tool,
                json.dumps({k: v for k, v in (args or {}).items()
                            if k not in ('secret', 'caller')},
                           separators=(',', ':'), default=str),
                len(ids), json.dumps(ids, separators=(',', ':'))))
    db.commit()
    # the content goes over the wire, not into the store
    ch.publish({'kind': 'approval', 'at': now, 'id': aid, 'caller': caller,
                'tool': tool, 'arguments': {k: v for k, v in (args or {}).items()
                                            if k not in ('secret', 'caller')},
                'n': len(ids), 'preview': preview,
                'timeout_s': APPROVAL_TIMEOUT_S})
    deadline = now + APPROVAL_TIMEOUT_S
    while time.time() < deadline:
        row = db.execute("SELECT decision FROM approvals WHERE id=?",
                         (aid,)).fetchone()
        if row and row['decision'] != 'pending':
            ok = row['decision'] == 'approved'
            ch.publish({'kind': 'approval-resolved', 'id': aid,
                        'decision': row['decision']})
            return ok, f"{row['decision']} by the person"
        time.sleep(APPROVAL_POLL_S)
    db.execute("UPDATE approvals SET decision='timeout', decided_at=? "
               "WHERE id=? AND decision='pending'", (time.time(), aid))
    db.commit()
    ch.publish({'kind': 'approval-resolved', 'id': aid, 'decision': 'timeout'})
    return False, (f"no answer within {APPROVAL_TIMEOUT_S:.0f}s; denied, "
                   f"because silence is not consent")


def release(db, caller, tool, args, results, preview):
    """The last gate before anything leaves. Returns (ok, reason).

    Order matters. Approval runs after the query, not before, because the
    person is being shown what would actually be handed over rather than what
    was asked for -- a search for "money" that returns nothing needs no
    decision, and one that returns a bank card does.

    There is no exemption. There was one, for a caller reading a belief it had
    written itself, and it was removed: it assumed a caller name identifies one
    continuing party, and a fresh process answering to the same name knows
    nothing of what that name concluded a month ago. Every release goes to the
    person or to a standing approval they granted.
    """
    if not approval_mode(db):
        return True, 'approval off'
    st = standing(db, caller, tool)
    if st:
        left = (st['expires_at'] - time.time()) / 60.0 if st['expires_at'] else None
        return True, ('standing approval'
                      + (f", {left:.0f} min left" if left is not None else ''))
    return ask(db, caller, tool, args, results, preview)


# --- what a caller is allowed to see of each thing --------------------------
def _shape_observation(o, with_body=False):
    d = {'id': o['id'], 'kind': o['kind'], 'text': o['text'],
         'started_at': o['started_at'], 'ended_at': o['ended_at'],
         'person': o['person'], 'person_decision': o['person_decision'],
         'has_audio': bool(o['audio_blob'])}
    if with_body:
        # the weights, per word. The audio digest is deliberately not here: a
        # caller may learn that a recording exists and may not learn how to
        # address it, because reaching the blob is not one of the four tools.
        d['words'] = (o['body'] or {}).get('words', [])
        d['signals'] = (o['body'] or {}).get('signals')
    return d


def _shape_belief(b):
    return {'id': b['id'], 'statement': b['statement'],
            'certainty': b['certainty'], 'weight': round(b['weight_now'], 4),
            'faint': b['faint'], 'formed_at': b['formed_at'],
            'about': b['about'], 'author': b['author'],
            'times_returned': b['times_returned'],
            'sources': b['sources'], 'superseded': b['superseded'],
            'replaced_by': b['replaced_by'], 'current': b['current'],
            'replaces': b['replaces'],
            'links': [{'type': l['type'], 'id': l['dst_id'],
                       'kind': l['dst_kind']} for l in b['links']]}


# --- the four tools, as plain functions -------------------------------------
def search_memory(db, caller, secret, query, limit=10, kinds=None,
                  include_superseded=False):
    args = {'query': query, 'limit': limit, 'kinds': kinds,
            'include_superseded': include_superseded}
    why = admit(db, caller, secret, 'search_memory', args)
    ks = tuple(kinds) if kinds else ('observation', 'belief')
    hits = mem.search(db, query, limit=limit, kinds=ks,
                      include_superseded=include_superseded)
    ok, why2 = release(db, caller, 'search_memory', args, hits,
                       [f"[{h['kind']}] {h['text']}" for h in hits])
    if not ok:
        log(db, caller, 'search_memory', args, 'denied', why2)
        raise Denied(why2)
    log(db, caller, 'search_memory', args, 'allowed', f"{why}; {why2}", hits)
    return hits


def fetch_observation(db, caller, secret, id, with_body=True):
    args = {'id': id, 'with_body': with_body}
    why = admit(db, caller, secret, 'fetch_observation', args)
    o = mem.get_observation(db, id)
    if o is None:
        t = db.execute("SELECT * FROM tombstones WHERE id=?", (id,)).fetchone()
        log(db, caller, 'fetch_observation', args, 'allowed',
            'forgotten' if t else 'no such observation')
        return {'id': id, 'gone': True,
                'reason': t['reason'] if t else 'no such observation'}
    ok, why2 = release(db, caller, 'fetch_observation', args, [o],
                       [f"[{o['kind']}] {o['text']}"])
    if not ok:
        log(db, caller, 'fetch_observation', args, 'denied', why2)
        raise Denied(why2)
    log(db, caller, 'fetch_observation', args, 'allowed', f"{why}; {why2}", [o])
    return _shape_observation(o, with_body=with_body)


def fetch_belief(db, caller, secret, id, follow=True):
    """Fetch a belief. follow=True answers with what the system thinks now.

    A superseded belief is still returned when asked for by id, because what
    the system used to think is a real answer to a real question. What it will
    not do is let a caller ask for a belief and receive a stale one without
    being told: the reply always carries superseded, replaced_by and current,
    and with follow=True the current version comes back alongside it.
    """
    args = {'id': id, 'follow': follow}
    why = admit(db, caller, secret, 'fetch_belief', args)
    b = mem.get_belief(db, id, reader=caller)
    if b is None:
        t = db.execute("SELECT * FROM tombstones WHERE id=?", (id,)).fetchone()
        log(db, caller, 'fetch_belief', args, 'allowed',
            'forgotten' if t else 'no such belief')
        return {'id': id, 'gone': True,
                'reason': t['reason'] if t else 'no such belief'}
    out = _shape_belief(b)
    returned = [b]
    cur = None
    if follow and b['superseded'] and b['current'] != b['id']:
        cur = mem.get_belief(db, b['current'], reader=caller)
        returned.append(cur)
    # No author exemption and no author-based withholding. Both were dropped
    # together: the exemption rested on a caller name being one continuing
    # party, which it is not, and the withholding existed only to keep the
    # exemption safe when following a replacement crossed into someone else's
    # belief. With every read going to the person anyway, the preview shows
    # both statements and they decide -- which is a better answer than the
    # gate silently splitting the reply by who wrote which half.
    ok, why2 = release(db, caller, 'fetch_belief', args, returned,
                       [x['statement'] for x in returned])
    if not ok:
        log(db, caller, 'fetch_belief', args, 'denied', why2)
        raise Denied(why2)
    if cur:
        out['current_version'] = _shape_belief(cur)
    log(db, caller, 'fetch_belief', args, 'allowed', f"{why}; {why2}", returned)
    return out


def record_answer(db, caller, secret, text, cites, model=None, about=None):
    """Record what a model said as an observation.

    A write, so it needs can_write, and it is not put to approval: nothing
    leaves the store. What it does need is citations, which memory.add_answer
    enforces, because forget() reaches model output by following them.
    """
    args = {'text_len': len(text or ''), 'cites': cites, 'model': model,
            'about': about}
    why = admit(db, caller, secret, 'record_answer', args)
    missing = [c for c in (cites or [])
               if db.execute("SELECT 1 FROM observations WHERE id=?",
                             (c,)).fetchone() is None]
    if missing or not cites:
        log(db, caller, 'record_answer', args, 'denied', 'bad citations')
        raise ValueError(
            "every citation must be an observation that exists; missing: "
            + (", ".join(missing) or "(none given)"))
    oid = mem.add_answer(db, text, cites, author=caller, model=model or caller,
                         about=about)
    log(db, caller, 'record_answer', args, 'allowed', why, [oid])
    o = mem.get_observation(db, oid)
    return {'id': oid, 'kind': o['kind'], 'text': o['text'],
            'cites': cites, 'started_at': o['started_at']}


def write_belief(db, caller, secret, statement, certainty, sources, about,
                 replaces=None, links=None):
    """Write a conclusion. sources is required; memory.add_belief enforces it."""
    args = {'statement': statement, 'certainty': certainty, 'sources': sources,
            'about': about, 'replaces': replaces, 'links': links}
    why = admit(db, caller, secret, 'write_belief', args)
    if not (about or '').strip():
        log(db, caller, 'write_belief', args, 'denied', 'no about')
        raise ValueError(
            "write_belief needs 'about': one or two words for what this belief "
            "concerns, so it can be found by subject and not only by resemblance")
    missing = [s for s in (sources or [])
               if db.execute("SELECT 1 FROM observations WHERE id=?",
                             (s,)).fetchone() is None]
    if missing or not sources:
        log(db, caller, 'write_belief', args, 'denied', 'bad provenance')
        raise ValueError(
            "every source must be an observation that exists; missing: "
            + (", ".join(missing) or "(none given)"))
    bid = mem.add_belief(db, statement, certainty, sources, author=caller,
                         about=about, replaces=replaces,
                         links=[(l['type'], l.get('kind', 'belief'), l['id'])
                                for l in (links or [])])
    log(db, caller, 'write_belief', args, 'allowed', why, [bid])
    return _shape_belief(mem.get_belief(db, bid))
