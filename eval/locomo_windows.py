"""Is retrieval mis-tuned for paragraph-length turns, or is it just weak.

WINDOW_TARGET_WORDS is 30 and WINDOW_MAX_OBS is 7, chosen for speech where an
observation is one run between silences with a median of four words -- so a
window covers about seven utterances of neighbouring conversation. A LoCoMo
turn has a median of 20 words, so the same 30-word target is reached after one
or two turns, and the neighbourhood the window exists to provide is mostly
absent. That is a plausible mechanical explanation for a 32% recall and it is
cheap to test: rebuild the windows at settings scaled to this input and
re-measure.

Raw recall only, no relevance judging, because the earlier run showed narrow
costs 1.3 points and this is about the index.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
HERE = Path(__file__).resolve().parent

# target words, max observations. The first is what is there now; the rest
# scale the word target by roughly the ratio of turn lengths (20 words against
# four), holding or varying how many turns of context a window may span.
CONFIGS = [
    ('as shipped', 30, 7),
    ('same turn count', 140, 7),
    ('wider', 200, 9),
    ('narrower', 60, 3),
    ('single turn', 20, 1),
]


def gold_for(results, data):
    text_of = {}
    for s in data:
        for k, v in s['conversation'].items():
            if isinstance(v, list):
                for t in v:
                    text_of[(s['sample_id'], t.get('dia_id'))] = (t.get('text') or '')
    out = []
    for r in results:
        if r['category'] == 5:
            continue
        ev = r.get('evidence')
        if isinstance(ev, str):
            try:
                ev = eval(ev)
            except Exception:
                ev = []
        g = [text_of.get((r['sample'], e), '') for e in (ev or [])]
        g = [x for x in g if x.strip()]
        if g:
            out.append((r['question'], r['category'], g))
    return out


def recall(db, mem, items, k):
    hit = defaultdict(int)
    n = defaultdict(int)
    for q, c, gold in items:
        n[c] += 1
        n['all'] += 1
        hits = mem.search(db, q, limit=k, kinds=('observation',))
        texts = [h.get('text', '') for h in hits]
        if any(any(g[:60] in t or t[:60] in g for t in texts) for g in gold):
            hit[c] += 1
            hit['all'] += 1
    return {c: 100.0 * hit[c] / n[c] for c in n}


def main():
    import memory as mem
    db = mem.open(HERE / 'locomo.db')
    items = gold_for(json.loads((HERE / 'locomo_results.json').read_text()),
                     json.loads((HERE / 'locomo10.json').read_text()))
    names = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop'}
    print(f"{len(items)} questions with usable evidence\n")
    print(f"{'setting':<18}{'words':>6}{'obs':>5}{'win words':>11}"
          f"{'k=10':>8}{'k=30':>8}   by category at k=10")
    for label, words, maxobs in CONFIGS:
        mem.WINDOW_TARGET_WORDS = words
        mem.WINDOW_MAX_OBS = maxobs
        t0 = time.time()
        mem.build_windows(db, progress=lambda a, b: None)
        import numpy as np
        wl = [len((r['text'] or '').split())
              for r in db.execute("SELECT text FROM windows")]
        r10 = recall(db, mem, items, 10)
        r30 = recall(db, mem, items, 30)
        cats = "  ".join(f"{names[c][:9]} {r10.get(c, 0):.0f}%"
                         for c in sorted(names) if c in r10)
        print(f"{label:<18}{words:>6}{maxobs:>5}{int(np.median(wl)):>11}"
              f"{r10['all']:>7.1f}%{r30['all']:>7.1f}%   {cats}"
              f"   ({time.time()-t0:.0f}s)")
    # leave the store rebuilt at the shipped settings so nothing downstream
    # silently inherits a benchmark-tuned index
    mem.WINDOW_TARGET_WORDS, mem.WINDOW_MAX_OBS = 30, 7
    mem.build_windows(db, progress=lambda a, b: None)
    print("\nrebuilt at the shipped settings")


if __name__ == '__main__':
    main()
