"""Load LoCoMo into a store of its own, so it can be queried the real way.

Deliberately a separate database. The benchmark is 5,882 turns of two invented
people talking; mixing that into the store that holds actual speech would
corrupt every reference the live path has and could never be cleanly undone.
memory.open takes a path, so this costs one argument.

The mapping is direct because nothing downstream of retrieval needs prosody:
memory.add_observation takes text, times and a free-form body, and recall.py
and gate.py read text and ids. A LoCoMo turn becomes one observation, a
LoCoMo session becomes the session id, and the session's stated date becomes
the timestamp -- which matters, because the temporal category is questions
about when things happened.

usage: locomo_load.py [--db PATH] [--limit N]
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# the project is the parent directory; this lives in eval/ to keep the
# benchmark and its store out of the way of the real one
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path(__file__).resolve().parent / "locomo10.json"
DB = Path(__file__).resolve().parent / "locomo.db"

# "1:56 pm on 8 May, 2023"
DATE_RE = re.compile(r'(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})',
                     re.I)
MONTHS = {m: i + 1 for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july',
     'august', 'september', 'october', 'november', 'december'])}


def parse_when(s):
    """The session's stated time, as epoch seconds. Falls back to a fixed
    offset rather than to now(), so a load is reproducible."""
    m = DATE_RE.search(s or '')
    if not m:
        return None
    hh, mm, ap, d, mon, yy = m.groups()
    hh = int(hh) % 12 + (12 if ap.lower() == 'pm' else 0)
    mi = MONTHS.get(mon.lower())
    if not mi:
        return None
    return datetime(int(yy), mi, int(d), hh, int(mm)).timestamp()


def sessions(conv):
    """Session keys in order, with their parsed start times."""
    keys = sorted((k for k in conv
                   if k.startswith('session_') and isinstance(conv[k], list)),
                  key=lambda k: int(k.split('_')[1]))
    return [(k, parse_when(conv.get(k + '_date_time'))) for k in keys]


def load(path=None, src=None, limit=None, progress=True):
    import memory as mem
    db = mem.open(path or DB)
    data = json.loads((src or SRC).read_text())
    if limit:
        data = data[:limit]
    n_obs = 0
    ids = {}                      # dia_id -> observation id, for evidence checks
    for sample in data:
        sid = sample['sample_id']
        conv = sample['conversation']
        for skey, when in sessions(conv):
            base = when if when is not None else 0.0
            for i, turn in enumerate(conv[skey]):
                text = (turn.get('text') or '').strip()
                if not text:
                    continue
                # one second per turn inside a session: the order is real, the
                # spacing is not, and nothing in retrieval depends on the gap
                st = base + i
                oid = mem.add_observation(
                    db, 'speech', st, st + 1.0, text,
                    {'dia_id': turn.get('dia_id'), 'session': skey,
                     'sample': sid, 'source': 'locomo'},
                    session=f"{sid}:{skey}",
                    speaker=turn.get('speaker'),
                    person=turn.get('speaker'),
                    person_decision='given',
                    embed_now=True)
                ids[f"{sid}/{turn.get('dia_id')}"] = oid
                n_obs += 1
        if progress:
            print(f"  {sid}: {n_obs} observations so far", flush=True)
    return db, n_obs, ids


def main():
    args = sys.argv[1:]
    path = Path(args[args.index('--db') + 1]) if '--db' in args else DB
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else None
    if path.exists():
        print(f"{path} already exists; delete it to reload")
        return 1
    t0 = time.time()
    db, n, ids = load(path, limit=limit)
    print(f"loaded {n} observations in {time.time()-t0:.0f}s")
    import memory as mem
    t0 = time.time()
    built = mem.build_windows(db, progress=lambda d, t: None)
    print(f"built windows in {time.time()-t0:.0f}s: {built}")
    (path.parent / 'locomo_ids.json').write_text(json.dumps(ids))
    return 0


if __name__ == '__main__':
    sys.exit(main())
