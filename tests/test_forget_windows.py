"""Forgetting an observation must also take it out of the window index.

The window index stores the joined text of its span, not a pointer to it. So
deleting the observation row and leaving the windows alone removes the record
and keeps the words: the sentence stays in up to WINDOW_MAX_OBS windows, all
of them searchable, and search is exactly what a window is for. Nothing about
the store looks wrong afterwards -- the observation really is gone -- which is
why this needs a test rather than an inspection.

The case that produced it: the machine's own spoken replies were swept out of
the store, and twenty-nine windows still carried "you had a sausage croissant".

Built on a real temporary store rather than a stub, because the thing under
test is whether two tables agree, and a stub of one of them would agree by
construction.

run: python tests/test_forget_windows.py     (exits non-zero on any failure)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory as mem                                          # noqa: E402
import incognito as inc                                       # noqa: E402

SESSION = 'test-forget-windows'
LINES = [
    "the kettle is on",
    "secret pangolin sentence",          # the one that gets forgotten
    "and then we walked to the shop",
    "it rained the whole way there",
    "we bought bread and came back",
]
NEEDLE = 'pangolin'


def build():
    """A store with one short session in it, windows and all."""
    path = Path(tempfile.mkdtemp()) / 'forget-windows.db'
    db = mem.open(str(path))
    t = 1_700_000_000.0
    ids = []
    for line in LINES:
        ids.append(mem.add_observation(db, 'speech', t, t + 2.0, text=line,
                                       body={'plain': line}, session=SESSION))
        t += 3.0
    db.commit()
    mem.build_windows(db)
    return db, ids


def windows_with(db, needle):
    return db.execute("SELECT count(*) FROM windows WHERE lower(text) LIKE ?",
                      (f'%{needle}%',)).fetchone()[0]


def main():
    db, ids = build()
    failures = []

    before = windows_with(db, NEEDLE)
    if before == 0:
        # the test would pass trivially afterwards and prove nothing
        failures.append("setup: no window contains the line, so forgetting it "
                        "cannot be shown to remove it")

    doomed = ids[1:2]
    obs, bel, rounds = inc._closure(db, set(doomed))
    plan = inc._plan(db, sorted(obs), sorted(bel), rounds)
    res = inc._apply(db, plan, 'test', drop_audio=False)

    gone = db.execute("SELECT count(*) FROM observations WHERE id=?",
                      (doomed[0],)).fetchone()[0]
    if gone:
        failures.append("the observation itself survived the forget")

    after = windows_with(db, NEEDLE)
    if after:
        failures.append(f"{after} window(s) still carry the forgotten words "
                        f"(was {before} before the forget)")

    # and the repair must not be a scorched-earth drop: the neighbours are
    # still findable, which is the whole reason windows exist
    kept = db.execute("SELECT count(*) FROM windows WHERE session=?",
                      (SESSION,)).fetchone()[0]
    if kept != len(LINES) - 1:
        failures.append(f"{kept} window(s) left for {len(LINES)-1} surviving "
                        f"observations: the repair dropped more than it rebuilt")

    print(f"windows containing {NEEDLE!r}: {before} before, {after} after")
    print(f"windows for the session: {kept} for {len(LINES)-1} observations")
    print(f"forget reported: {res}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("ok: forgetting an observation removes it from the window index too")
    return 0


if __name__ == '__main__':
    sys.exit(main())
