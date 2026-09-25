"""Where the failure is: retrieval, the relevance judge, or composition.

The end-to-end score cannot tell those apart. LoCoMo gives the evidence turn
ids for every question, so this measures, for the same sampled questions:

  recall@k raw   did memory.search surface the evidence turn at all
  recall kept    did relevance.narrow keep it after judging
  answered       did anything come out the far end

If raw recall is high and the score is low, the memory layer's index is fine
and the fault is downstream of it. If raw recall is low, the index is the
problem.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
HERE = Path(__file__).resolve().parent


def main():
    import memory as mem
    import relevance as rl
    db = mem.open(HERE / 'locomo.db')
    data = json.loads((HERE / 'locomo10.json').read_text())
    # dia_id -> turn text, per sample
    text_of = {}
    for s in data:
        for k, v in s['conversation'].items():
            if isinstance(v, list):
                for t in v:
                    text_of[(s['sample_id'], t.get('dia_id'))] = (t.get('text') or '')
    results = json.loads((HERE / 'locomo_results.json').read_text())
    K = 10
    stats = defaultdict(lambda: {'n': 0, 'raw': 0, 'kept': 0, 'ev': 0})
    for r in results:
        if r['category'] == 5:
            continue
        ev = r.get('evidence')
        if isinstance(ev, str):
            try:
                ev = eval(ev)                       # the file stores "['D1:3']"
            except Exception:
                ev = []
        gold = [text_of.get((r['sample'], e), '') for e in (ev or [])]
        gold = [g for g in gold if g.strip()]
        if not gold:
            continue
        c = r['category']
        stats[c]['n'] += 1
        stats[c]['ev'] += len(gold)
        hits = mem.search(db, r['question'], limit=K, kinds=('observation',))
        texts = [h.get('text', '') for h in hits]
        if any(any(g[:60] in t or t[:60] in g for t in texts) for g in gold):
            stats[c]['raw'] += 1
        try:
            kept, _ = rl.narrow(db, r['question'], hits)
        except Exception:
            kept = hits
        ktexts = [h.get('text', '') for h in kept]
        if any(any(g[:60] in t or t[:60] in g for t in ktexts) for g in gold):
            stats[c]['kept'] += 1
    names = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop'}
    print(f"recall of the gold evidence turn, k={K}")
    print("category        n   raw recall   kept after narrow")
    tot = defaultdict(int)
    for c in sorted(stats):
        s = stats[c]
        for key in ('n', 'raw', 'kept'):
            tot[key] += s[key]
        print(f"{names[c]:<13} {s['n']:4d}   {100*s['raw']/s['n']:8.1f}%   "
              f"{100*s['kept']/s['n']:14.1f}%")
    print(f"{'all four':<13} {tot['n']:4d}   {100*tot['raw']/tot['n']:8.1f}%   "
          f"{100*tot['kept']/tot['n']:14.1f}%")


if __name__ == '__main__':
    main()
