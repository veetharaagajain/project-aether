"""What a running process is executing, against what is on disk now.

The fourth failure of one shape. provenance.py compares a stored value against
the code that reads it, which catches a baseline built under a different
pipeline. It cannot catch this: Python imports a module once and keeps it in
memory, so a process that started at 11:12 goes on executing the 11:12 version
of gate.py forever, and editing the file at 13:28 changes nothing about it. The
MCP server served two hours of unnarrowed private speech that way, and the only
symptom was a log line that looked plausible.

WHAT IS WATCHED. Every project module already in sys.modules when snapshot() is
called, plus the Swift bridge sources, which have the same problem one level
down: relevance.build() recompiles when the source is newer, but a bridge
subprocess already running holds the old binary.

CONTENT, NOT MTIME. A file is stale when its bytes differ, not when it was
touched. Comparing timestamps would make `touch` and a reverted edit look like
changes, and a check that fires on nothing is a check that gets ignored -- which
is the failure being fixed, not a new way to cause it.

WHERE IT SURFACES. Not a log line. The log did record the absence of narrowing
for two hours and nobody read it. Long-lived processes register in the
`processes` table and heartbeat, and viewer.py reads that table, compares each
process's recorded digests against disk, and puts anything stale in the header
where the recording indicator is -- the one place in this project a person
actually looks.
"""

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_ROOT_PREFIX = str(ROOT) + os.sep
HEARTBEAT_S = 20.0
DEAD_AFTER_S = 90.0        # no heartbeat for this long: assume it is gone

# Swift sources compiled into long-lived helper processes. A change here is as
# stale as a change to a .py module, and less visible, because the helper is a
# subprocess of a subprocess.
EXTRA_WATCHED = ('bridge/relevance.swift', 'bridge/daemon.swift')


def digest_file(p):
    try:
        return hashlib.blake2b(Path(p).read_bytes(), digest_size=8).hexdigest()
    except OSError:
        return None


def watched_modules():
    """Project modules this process has actually imported, by file path.

    Read from sys.modules rather than from a list, so a module added later is
    covered without anyone remembering to add it here.
    """
    out = {}
    for name, mod in list(sys.modules.items()):
        # vars(mod), not getattr. Lazy module proxies -- speechbrain uses them
        # -- implement __getattr__, so asking a proxy for __file__ performs the
        # import it was deferring, which pulled in optional dependencies that
        # are not installed and crashed the live path. Reading the module dict
        # observes without touching.
        try:
            f = vars(mod).get('__file__')
        except TypeError:
            continue
        if not f or not f.endswith('.py'):
            continue
        # A string prefix test, and nothing else. This runs on every release,
        # and the version that fell back to Path.resolve() for non-matching
        # paths spent 1.7 of its 1.8 ms proving that site-packages is not the
        # project -- a syscall per loaded module, to rescue a case that does
        # not occur: measured across every module this project loads, the
        # fallback found 0 that the prefix test missed.
        #
        # What it gives up is a project module imported through a path outside
        # ROOT, such as a symlink from elsewhere. sys.path holds the project
        # directory itself, so imports arrive already prefixed.
        if not f.startswith(_ROOT_PREFIX):
            continue
        rel = f[len(_ROOT_PREFIX):]
        if rel.startswith('.venv' + os.sep) or os.sep + '.venv' + os.sep in rel:
            continue
        out[rel] = None
    for extra in EXTRA_WATCHED:
        if (ROOT / extra).exists():
            out[extra] = None
    return out


def snapshot():
    """What this process is running, right now, as {path: digest}."""
    return {k: digest_file(ROOT / k) for k in watched_modules()}


def changed_since(recorded):
    """Which of the recorded files differ from disk now.

    Returns a list of (path, what) where what is 'edited', 'deleted' or 'new'.
    A file imported since the snapshot is not reported: it was loaded from the
    current disk contents, so it is not stale.
    """
    out = []
    for path, was in (recorded or {}).items():
        now = digest_file(ROOT / path)
        if now is None and was is not None:
            out.append((path, 'deleted'))
        elif now != was:
            out.append((path, 'edited'))
    return sorted(out)


# --- the register every long-lived process signs ----------------------------
_MINE = None


def register(db, kind, snap=None):
    """Record what this process is running, and keep saying it is alive.

    The heartbeat is what separates "stale" from "long gone": a row whose
    last_seen has stopped moving is a dead process, and warning about the code
    version of something that already exited would be noise.
    """
    global _MINE
    snap = snapshot() if snap is None else snap
    now = time.time()
    db.execute(
        "INSERT INTO processes(pid,kind,started_at,last_seen,argv,modules) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(pid) DO UPDATE SET "
        "kind=excluded.kind, started_at=excluded.started_at, "
        "last_seen=excluded.last_seen, argv=excluded.argv, "
        "modules=excluded.modules",
        (os.getpid(), kind, now, now, ' '.join(sys.argv[1:]) or '(no arguments)',
         json.dumps(snap, separators=(',', ':'))))
    db.commit()
    _MINE = snap
    return snap


def heartbeat(db_factory, interval=HEARTBEAT_S):
    """A daemon thread saying this process is still here.

    Takes a factory rather than a connection because sqlite3 connections are
    thread-bound and this runs on its own thread.
    """
    def beat():
        db = db_factory()
        while True:
            try:
                db.execute("UPDATE processes SET last_seen=? WHERE pid=?",
                           (time.time(), os.getpid()))
                db.commit()
            except Exception:                       # noqa: BLE001
                pass
            time.sleep(interval)
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return t


def unregister(db):
    try:
        db.execute("DELETE FROM processes WHERE pid=?", (os.getpid(),))
        db.commit()
    except Exception:                               # noqa: BLE001
        pass


def mine():
    """This process's own snapshot, or None if it never registered."""
    return _MINE


def refresh(db=None):
    """Adopt modules imported since the last look. Returns the new ones.

    THE HOLE THIS CLOSES. register() snapshots what sys.modules holds at
    startup, and gate.search_memory imports relevance lazily, on the first
    search. So the module most likely to be edited was the one module never
    watched: editing it changed nothing anyone could see, which is the exact
    shape of the failure this file exists for.

    WHAT A LATE ADOPTION MEANS. A module imported after startup was loaded from
    the disk contents at its import, so it is not stale at that moment -- it is
    the newest code by definition. Baselining it against disk when first seen
    is therefore correct, not an approximation, provided the look happens
    promptly after the import. It does: gate.release() checks on every release,
    and the import in search_memory is a few lines earlier in the same call.

    The residual gap is an edit landing between a module's import and the very
    next check, which for the release path is microseconds. It is not zero, and
    closing it properly would mean digesting each file as the import system
    reads it rather than afterwards.

    Modules already in the baseline keep their original digest. Adoption only
    ever adds; it never re-baselines something already watched, because that
    would quietly forgive the drift it is meant to report.
    """
    global _MINE
    if _MINE is None:
        return []
    fresh = [p for p in watched_modules() if p not in _MINE]
    if not fresh:
        return []
    for p in fresh:
        _MINE[p] = digest_file(ROOT / p)
    if db is not None:
        try:
            db.execute("UPDATE processes SET modules=? WHERE pid=?",
                       (json.dumps(_MINE, separators=(',', ':')), os.getpid()))
            db.commit()
        except Exception:                           # noqa: BLE001
            pass
    return fresh


def am_i_stale(paths=None, db=None):
    """Has anything I imported changed under me. Optionally only these paths.

    Refreshes first, so a module imported since startup is watched from the
    moment anything asks -- rather than never, which is what it was.
    """
    if _MINE is None:
        return []
    refresh(db)
    rec = _MINE if paths is None else {k: v for k, v in _MINE.items()
                                       if k in set(paths)}
    return changed_since(rec)


def pid_exists(pid):
    """Is that process still there. Cheap, exact, and local-only by nature."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def living(db, now=None):
    """Every registered process, with whether it is alive and whether it is
    running code that no longer exists on disk.

    Liveness is the pid existing AND a recent heartbeat, not either alone. The
    heartbeat alone leaves a crashed process looking alive for its whole
    interval, which is a false alarm, and a false alarm in the one banner a
    person is meant to trust is how the banner stops being read. The pid alone
    is not enough either, because pids are reused.
    """
    now = time.time() if now is None else now
    out = []
    for r in db.execute("SELECT * FROM processes ORDER BY started_at"):
        try:
            snap = json.loads(r['modules'])
        except ValueError:
            snap = {}
        alive = ((now - r['last_seen']) < DEAD_AFTER_S
                 and pid_exists(r['pid']))
        drift = changed_since(snap)
        out.append({'pid': r['pid'], 'kind': r['kind'],
                    'started_at': r['started_at'], 'last_seen': r['last_seen'],
                    'argv': r['argv'], 'alive': alive,
                    'stale': bool(drift) and alive,
                    'changed': [{'path': p, 'what': w} for p, w in drift]})
    return out


def reap(db, now=None):
    """Forget processes that are gone: no pid, or long since silent."""
    now = time.time() if now is None else now
    for r in db.execute("SELECT pid,last_seen FROM processes").fetchall():
        if not pid_exists(r['pid']) or r['last_seen'] < now - DEAD_AFTER_S * 10:
            db.execute("DELETE FROM processes WHERE pid=?", (r['pid'],))
    db.commit()
