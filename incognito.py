"""Not capturing, and being able to see that you are not.

Three parts, and the third is what makes the other two worth anything.

PAUSE stops anything being written. live.py asks before it writes, so a paused
session still hears and still measures -- the gate, the recogniser and the
prosody all run -- but nothing reaches the store and the audio is not archived.
Measuring without storing is deliberate: the alternative is that resuming
starts from a cold session with no pitch range and no loudness reference, and
the first minute back would be measured badly.

FORGET removes the last N minutes, because people realise afterwards. What it
means is set out in forget() and is the one place in this design that deletes.

STATE is visible. state() is the single answer to "is this thing recording",
it is read from the store rather than from a variable in some process, and
capture_state keeps every transition so the answer to "was it recording at
four o'clock on Tuesday" exists too.
"""

import json
import time

import memory as mem


def state(db):
    """Capturing or paused, and since when. The latest row wins."""
    r = db.execute("SELECT * FROM capture_state ORDER BY at DESC, id DESC "
                   "LIMIT 1").fetchone()
    if r is None:
        return {'mode': 'capturing', 'since': None, 'note': 'default',
                'recording': True}
    return {'mode': r['mode'], 'since': r['at'], 'note': r['note'],
            'recording': r['mode'] == 'capturing'}


def capturing(db):
    return state(db)['recording']


def _set(db, mode, note):
    now = time.time()
    db.execute("INSERT INTO capture_state(id,at,mode,note,device) "
               "VALUES(?,?,?,?,?)", (mem.ulid(now * 1000), now, mode, note,
                                     db.device))
    db.commit()
    return state(db)


def pause(db, note=None):
    return _set(db, 'paused', note or 'paused by request')


def resume(db, note=None):
    return _set(db, 'capturing', note or 'resumed by request')


def banner(db):
    """One line, for anything with somewhere to print it.

    Worded so that the recording case is the loud one. A person should not have
    to notice the absence of a warning to know they are being recorded.
    """
    s = state(db)
    if s['recording']:
        return "RECORDING -- speech is being stored"
    since = time.strftime('%H:%M', time.localtime(s['since'])) if s['since'] else '?'
    return f"PAUSED since {since} -- nothing is being stored"


# --- forget -----------------------------------------------------------------
def _tomb(db, ids_kinds, reason):
    now = time.time()
    for i, kind in ids_kinds:
        db.execute("INSERT OR REPLACE INTO tombstones(id,kind,reason,at,device,"
                   "hlc) VALUES(?,?,?,?,?,?)",
                   (i, kind, reason, now, db.device, db.clock.tick()))


def _closure(db, seed):
    """Everything reachable from a set of doomed observations.

    The set is a closure, not a window. Three kinds of thing carry the person's
    words forward out of it:

      the observations themselves, whatever put them in the seed;
      model answers, which repeat those words back in their own sentences and
        are found by following the 'cites' edges memory.add_answer wrote;
      beliefs, which are conclusions drawn from either of those.

    A model answer can cite another model answer, so this iterates to a fixed
    point rather than sweeping once.
    """
    doomed_obs = set(seed)
    doomed_bel = set()
    rounds = 0
    while True:
        rounds += 1
        grew = False
        # snapshot the frontier: the set grows inside the round, and a
        # placeholder string built from a set that is still being added to no
        # longer matches the bindings handed with it
        frontier = tuple(doomed_obs)
        if frontier:
            q = ",".join("?" * len(frontier))
            for r in db.execute(
                    f"SELECT DISTINCT e.src_id FROM edges e "
                    f"WHERE e.type='cites' AND e.src_kind='observation' "
                    f"AND e.dst_id IN ({q})", frontier):
                if r['src_id'] not in doomed_obs:
                    doomed_obs.add(r['src_id'])
                    grew = True
            for r in db.execute(
                    f"SELECT DISTINCT e.src_id FROM edges e "
                    f"WHERE e.type='came-from' AND e.src_kind='belief' "
                    f"AND e.dst_id IN ({q})", frontier):
                if r['src_id'] not in doomed_bel:
                    doomed_bel.add(r['src_id'])
                    grew = True
        if not grew:
            break
    return doomed_obs, doomed_bel, rounds


def _plan(db, doomed_obs, doomed_bel, rounds, **extra):
    """Turn a closed set into the plan a person can look at before it runs."""
    obs = [dict(r) for r in db.execute(
        "SELECT id,kind,session,started_at,text,audio_blob FROM observations "
        "WHERE id IN (%s) ORDER BY started_at" % ",".join("?" * len(doomed_obs)),
        tuple(doomed_obs))] if doomed_obs else []
    bel = [dict(r) for r in db.execute(
        "SELECT id,statement,author FROM beliefs WHERE id IN (%s)"
        % ",".join("?" * len(doomed_bel)), tuple(doomed_bel))] if doomed_bel else []
    blobs = set()
    for o in obs:
        if not o['audio_blob']:
            continue
        others = db.execute(
            "SELECT count(*) FROM observations WHERE audio_blob=? AND id NOT IN "
            "(%s)" % ",".join("?" * len(doomed_obs)),
            (o['audio_blob'], *doomed_obs)).fetchone()[0]
        if not others:
            blobs.add(o['audio_blob'])
    plan = {'observations': obs, 'beliefs': bel, 'audio_blobs': sorted(blobs),
            'rounds': rounds,
            'answers': sum(1 for o in obs if o['kind'] == 'answer')}
    plan.update(extra)
    return plan


def plan_forget(db, minutes, now=None):
    """What forget(minutes) would remove, without removing it.

    Separate from doing it because a destructive operation the person cannot
    look at before running is not one they can trust.
    """
    now = time.time() if now is None else now
    cutoff = now - minutes * 60.0
    seed = {r['id'] for r in db.execute(
        "SELECT id FROM observations WHERE started_at >= ?", (cutoff,))}
    o, b, rounds = _closure(db, seed)
    p = _plan(db, o, b, rounds, cutoff=cutoff, minutes=minutes)
    p['in_window'] = sum(1 for x in p['observations'] if x['started_at'] >= cutoff)
    return p


def plan_forget_sessions(db, sessions):
    """The same removal, seeded by session instead of by clock.

    A window is the wrong instrument when what you want gone is interleaved in
    time with what you want kept -- test replays running alongside a real
    capture, which is exactly how this arose. Everything downstream is shared
    with plan_forget: same closure, same blob accounting, same deletion, so a
    session-scoped removal cannot quietly behave differently from a timed one.
    """
    sessions = list(sessions)
    if not sessions:
        return _plan(db, set(), set(), 0, sessions=[])
    q = ",".join("?" * len(sessions))
    seed = {r['id'] for r in db.execute(
        f"SELECT id FROM observations WHERE session IN ({q})", tuple(sessions))}
    o, b, rounds = _closure(db, seed)
    p = _plan(db, o, b, rounds, sessions=sessions)
    p['seeded'] = len(seed)
    p['pulled_in'] = len(o) - len(seed)
    return p


def forget(db, minutes, now=None, drop_audio=True):
    """Remove the last N minutes. The one place in this design that deletes.

    WHY DELETE, WHEN EVERYTHING ELSE DECAYS. Decay is the right model for
    attention: a memory nobody returns to should stop competing for the
    answer, and might still be the answer to a question nobody has asked yet.
    That reasoning does not survive contact with "I did not mean to say that in
    front of it". A decayed observation is still there, still searchable by
    someone who asks the right question, still restorable by a bug. Forget has
    to mean the bytes are gone or it means nothing, and a person who has to
    trust a weight is a person who will simply not talk near it.

    WHAT GOES.
      Observations that started inside the window, and their vectors.
      Model answers that cite any of them, transitively, because an answer
      repeats the person's words back in its own sentences -- the same reason
      beliefs go, and a worse one, since an answer is longer and quotes more.
      Audio blobs no surviving observation still points at.
      Beliefs drawing on ANY forgotten observation, and their vectors and
      edges. Not only the ones drawn entirely from the window: a belief formed
      partly from what is being forgotten still carries it forward in a form
      that is harder to see and impossible to search for. This is affordable
      precisely because the belief layer is rebuildable -- see
      memory.rebuild_check -- so deleting a belief costs a re-derivation, while
      keeping one costs the guarantee.

    WHAT STAYS, and why each.
      Tombstones, so a device that was offline learns of the deletion instead
      of resurrecting the row on the next union.
      The access log, minus content: rows keep who asked, when, and how many
      results, and any returned id that has just been deleted is replaced with
      'forgotten'. Deleting the log would mean forget() could erase the
      evidence of what had already left the store, which is the one thing the
      log exists to prevent.
      The edge_types census, which is counts of free-text words like
      'came-from' and holds nothing about anyone.
      capture_state, so the record of when it was and was not recording cannot
      be edited by the thing it is meant to hold to account.
    """
    now = time.time() if now is None else now
    plan = plan_forget(db, minutes, now)
    out = _apply(db, plan, f'forget {minutes}m', drop_audio)
    out.update(minutes=minutes, cutoff=plan['cutoff'])
    return out


def forget_sessions(db, sessions, drop_audio=True):
    """Remove whole sessions, through exactly the path forget() uses."""
    plan = plan_forget_sessions(db, sessions)
    out = _apply(db, plan, 'forget sessions ' + ",".join(sessions), drop_audio)
    out.update(sessions=list(sessions))
    return out


def _apply(db, plan, reason, drop_audio=True):
    oids = [o['id'] for o in plan['observations']]
    bids = [b['id'] for b in plan['beliefs']]

    if oids:
        q = ",".join("?" * len(oids))
        db.execute(f"DELETE FROM obs_vec WHERE id IN ({q})", oids)
        db.execute(f"DELETE FROM observations WHERE id IN ({q})", oids)
        _tomb(db, [(i, 'observation') for i in oids], reason)
    if bids:
        q = ",".join("?" * len(bids))
        db.execute(f"DELETE FROM bel_vec WHERE id IN ({q})", bids)
        db.execute(f"DELETE FROM beliefs WHERE id IN ({q})", bids)
        db.execute(f"DELETE FROM edges WHERE src_id IN ({q})", bids)
        db.execute(f"DELETE FROM edges WHERE dst_id IN ({q})", bids)
        _tomb(db, [(i, 'belief') for i in bids], reason)
    # edges that pointed at a forgotten observation from a surviving belief
    # cannot exist: any such belief was itself deleted above. This clears edges
    # left by observations that were their own source.
    if oids:
        q = ",".join("?" * len(oids))
        db.execute(f"DELETE FROM edges WHERE dst_id IN ({q})", oids)

    gone = set(oids) | set(bids)
    scrubbed = 0
    for r in db.execute("SELECT id,returned FROM access_log").fetchall():
        try:
            ids = json.loads(r['returned'])
        except ValueError:
            continue
        if not any(i in gone for i in ids):
            continue
        db.execute("UPDATE access_log SET returned=? WHERE id=?",
                   (json.dumps(['forgotten' if i in gone else i for i in ids]),
                    r['id']))
        scrubbed += 1

    dropped_blobs = []
    if drop_audio:
        import archive as ar
        for d in plan['audio_blobs']:
            p = ar.path_for(d)
            if p.exists():
                p.unlink()
                dropped_blobs.append(d)
        _tomb(db, [(d, 'audio') for d in dropped_blobs], reason)
    db.commit()
    return {'observations': len(oids), 'beliefs': len(bids),
            'audio_blobs': len(dropped_blobs), 'log_rows_scrubbed': scrubbed}
