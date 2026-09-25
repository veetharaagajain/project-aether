"""One of a thing at a time, and a refusal that says who has it.

Two processes of the same kind running at once has now cost this project
twice, both times silently. Two live paths captured the same speech into the
store against two different session references, and the only symptom was the
viewer showing everything twice. Two viewers were worse: channel.listener()
unlinks a stale socket before binding, which is right after a crash and wrong
while someone is still using it, so the second viewer took the live stream and
the first went deaf with no error at either end.

An advisory flock on a file, not a pid file. A pid file left behind by a crash
locks the path forever and a pid can be reused; a flock is released by the
kernel however the holder dies. The identity line inside is written for the
refusal to quote, not for the locking, so a stale line left by a killed process
is harmless -- the next holder overwrites it.

Advisory means it constrains processes that call this and nothing else. It
prevents recurrence; it cannot reach back and constrain something already
running from an older copy of the code.
"""

import os
import sys
import time
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parent / "store"
_HELD = {}          # path -> open file object, kept open on purpose


class AlreadyRunning(Exception):
    """Something of this kind is already running, and said so rather than
    quietly becoming a second one."""


def take(name, what, consequence, process='live.py'):
    """Hold the named lock for the life of this process, or explain who has it.

    name is the lock file's base name, what is a short noun for the thing being
    started, consequence says what running two would actually do -- a refusal
    that only says "already running" leaves the reader guessing whether it
    matters -- and process is what to tell them to grep for.
    """
    import fcntl
    path = LOCK_DIR / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open('a+')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        held = fh.read().strip() or '(the holder wrote nothing)'
        fh.close()
        raise AlreadyRunning(
            f"another {what} is already running and holding {path}:\n"
            f"    {held}\n"
            f"{consequence}\n"
            f"Stop that one first, or if you believe it is gone, check with:\n"
            f"    pgrep -fl {process}")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid {os.getpid()}  started {time.strftime('%Y-%m-%d %H:%M:%S')}  "
             f"argv {' '.join(sys.argv[1:]) or '(no arguments)'}\n")
    fh.flush()
    _HELD[str(path)] = fh          # closing it would drop the lock
    return fh


def held(name):
    """Does this process already hold that lock.

    Needed because the lock moved from the command-line entry point down to
    the function that actually captures: main() takes it, then run_stream
    checks whether it needs to. flock from a second descriptor in the same
    process would conflict with itself, so asking has to be possible.
    """
    return str(lock_path(name)) in _HELD


def lock_path(name):
    """Where a given lock lives. One definition, so nothing can name it twice
    and drift -- which it briefly did."""
    return LOCK_DIR / f"{name}.lock"
