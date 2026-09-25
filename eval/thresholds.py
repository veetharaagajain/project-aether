"""Sweep the composition guards, measuring what loosening them costs.

grounded() and answer_overlap() exist to stop the model answering from what it
happens to know rather than from what was said. On LoCoMo they are refusing
correct answers instead: 51 questions had the gold evidence retrieved and 3
produced a correct answer. Loosening them will raise accuracy and will also
let invention through, and a setting that does the second is worse than what
is there now however well it does the first.

THE TRICK THAT MAKES THIS AFFORDABLE: the model's reply does not depend on the
threshold. Only the accept/reject decision does. So the model is asked once
per question, the reply is captured, judged once, and every threshold is then
evaluated offline against the same captured replies. One pass of model calls
instead of one per setting.

usage: thresholds.py capture   ask the model once per question and judge it
       thresholds.py sweep     evaluate every setting against what was captured
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
HERE = Path(__file__).resolve().parent
CAPTURED = HERE / 'threshold_capture.json'

# --- the adversarial cases ---------------------------------------------------
# NOT found in the repository as runnable tests -- the reasoning for them is in
# recall.py's comments, the cases themselves were not written down. These are
# reconstructed from the four descriptions, and each is a triple of what the
# model returned (answer, support) against the lines it was actually shown.
# Every one of them MUST be refused at any shippable setting.
ADVERSARIAL = [
    {
        'name': 'invented answer, real supporting line',
        'lines': ["I've been going to the beach most weekends this summer.",
                  "The water was freezing but I went in anyway."],
        'answer': "You go to the beach every weekend with your brother Tom.",
        'support': "I've been going to the beach most weekends this summer.",
        'why': 'the brother is invented; the quoted line is real',
    },
    {
        'name': 'invented answer, invented supporting line',
        'lines': ["I've been going to the beach most weekends this summer."],
        'answer': "You bought a surfboard in April.",
        'support': "I bought a surfboard in April.",
        'why': 'neither the answer nor the line it cites was ever said',
    },
    {
        'name': 'half invented, attached to a true one',
        'lines': ["I went for a morning jog in the park yesterday.",
                  "Then I taught two yoga classes."],
        'answer': "You went for a morning jog in the park and then drove to Bristol.",
        'why': 'the jog is true, the drive to Bristol is not',
        'support': "I went for a morning jog in the park yesterday.",
    },
    {
        'name': 'general knowledge, real supporting line',
        'lines': ["I've been going to the beach most weekends this summer.",
                  "The water was freezing but I went in anyway."],
        'answer': "The sea is cold in summer because of upwelling and thermal inertia.",
        'support': "The water was freezing but I went in anyway.",
        'why': 'answered from what the model knows, citing a real line',
    },
]

# Cases that MUST be accepted at any usable setting: the composed, rephrased
# answers this system is designed to produce. Without these the sweep would
# just recommend the strictest setting.
LEGITIMATE = [
    {
        'name': 'rephrased, as compose_from is meant to',
        'lines': ["I had a sausage croissant for breakfast."],
        'answer': "You had a sausage croissant.",
        'support': "I had a sausage croissant for breakfast.",
    },
    {
        'name': 'rephrased across two lines',
        'lines': ["The beach is a great place for finding peace.",
                  "I also like sitting by the window in my Mom's house."],
        'answer': "The beach and the window seat at your Mom's house.",
        'support': "The beach is a great place for finding peace.",
    },
    {
        'name': 'inflected, which is why stemming exists',
        'lines': ["KKC conducted the audit last March."],
        'answer': "KKC audited it in March.",
        'support': "KKC conducted the audit last March.",
    },
]


# --- the two guards, parameterised so they can be swept ----------------------
def overlap_at(answer, text, stem):
    """recall.answer_overlap with the stem length as a parameter.

    Copied rather than imported because the shipped one closes over a constant
    of 5, and the point of this is to vary it.
    """
    import re
    from recall import STOPWORDS
    words = [w for w in re.findall(r"[a-z0-9']+", (answer or '').lower())
             if w not in STOPWORDS and len(w) > 1]
    if not words:
        return 0.0
    hay = set(re.findall(r"[a-z0-9']+", (text or '').lower()))
    stems = {h[:stem] for h in hay}
    hit = sum(1 for w in words if w in hay or w[:stem] in stems)
    return hit / len(words)


def accepts(case, min_overlap, stem, support_ratio=None):
    """Would compose_from emit this answer at these settings."""
    import recall as rc
    lines = case['lines']
    if support_ratio is not None:
        old, rc.SUPPORT_MIN_RATIO = rc.SUPPORT_MIN_RATIO, support_ratio
    try:
        if rc.grounded(case.get('support', ''), lines) is None:
            return False, 'support not grounded'
    finally:
        if support_ratio is not None:
            rc.SUPPORT_MIN_RATIO = old
    ov = overlap_at(case['answer'], "\n".join(lines), stem)
    if ov < min_overlap:
        return False, f'overlap {ov:.2f}'
    return True, f'overlap {ov:.2f}'


def capture():
    """Ask the model once per question, keep the reply, judge it once."""
    import anthropic
    import capability as cap
    import memory as mem
    from eval_judge import judge_one
    db = mem.open(HERE / 'locomo.db')
    res = json.loads((HERE / 'locomo_results.json').read_text())
    data = json.loads((HERE / 'locomo10.json').read_text())
    text_of = {}
    for s in data:
        for k, v in s['conversation'].items():
            if isinstance(v, list):
                for t in v:
                    text_of[(s['sample_id'], t.get('dia_id'))] = (t.get('text') or '')
    key, _ = cap.claude_key()
    client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2)
    out = []
    for i, r in enumerate(res, 1):
        if r['category'] == 5:
            continue
        ev = r.get('evidence')
        if isinstance(ev, str):
            try:
                ev = eval(ev)
            except Exception:
                ev = []
        gold_lines = [text_of.get((r['sample'], e), '') for e in (ev or [])]
        gold_lines = [g for g in gold_lines if g.strip()]
        hits = mem.search(db, r['question'], limit=10, kinds=('observation',))
        lines = [h.get('text', '') for h in hits]
        retrieved = any(any(g[:60] in t or t[:60] in g for t in lines)
                        for g in gold_lines) if gold_lines else None
        try:
            reply, _ = cap.ask('compose_answer',
                               {'question': r['question'],
                                'transcript': "\n".join(lines)}, budget=25.0)
        except Exception as e:                                # noqa: BLE001
            reply = {'error': str(e)}
        ans = (reply.get('answer') or '').strip()
        sup = (reply.get('support') or '').strip()
        rec = {'question': r['question'], 'category': r['category'],
               'gold': r['gold'], 'lines': lines, 'answer': ans,
               'support': sup, 'can_answer': bool(reply.get('canAnswer')),
               'retrieved': retrieved, 'judged': None}
        if ans:
            rec['judged'] = judge_one(client, r['question'], r['gold'], ans)
        out.append(rec)
        if i % 20 == 0:
            print(f"  {len(out)} captured", flush=True)
            CAPTURED.write_text(json.dumps(out))
    CAPTURED.write_text(json.dumps(out))
    print(f"captured {len(out)}")


def sweep():
    caps = json.loads(CAPTURED.read_text())
    answered = [c for c in caps if c['answer']]
    got = [c for c in caps if c['retrieved']]
    print(f"{len(caps)} questions captured; {len(answered)} produced a candidate "
          f"answer; {len(got)} had the gold evidence retrieved\n")

    OVERLAPS = [0.60, 0.50, 0.40, 0.30, 0.20]
    STEMS = [5, 4, 3]
    print("overlap stem | LoCoMo: emitted  correct  cond.acc | adversarial "
          "let through | legit refused")
    rows = []
    for stem in STEMS:
        for mo in OVERLAPS:
            emitted = [c for c in answered
                       if accepts({'lines': c['lines'], 'answer': c['answer'],
                                   'support': c['support']}, mo, stem)[0]]
            correct = [c for c in emitted if c['judged']]
            # conditional accuracy: of the questions where the memory layer
            # actually retrieved the evidence, how many end in a right answer
            cond = [c for c in got if c in emitted and c['judged']]
            bad = [a['name'] for a in ADVERSARIAL
                   if accepts(a, mo, stem)[0]]
            miss = [l['name'] for l in LEGITIMATE
                    if not accepts(l, mo, stem)[0]]
            rows.append((stem, mo, len(emitted), len(correct),
                         100.0 * len(cond) / max(len(got), 1), bad, miss))
            print(f"  {mo:.2f}   {stem}  |    {len(emitted):4d}     {len(correct):4d}"
                  f"    {100.0*len(cond)/max(len(got),1):6.1f}% | "
                  f"{len(bad)} of 4{' (' + ', '.join(b[:22] for b in bad) + ')' if bad else ''}"
                  f" | {len(miss)} of 3")
    print("\nshippable settings are those that let through 0 of 4 adversarial:")
    ok = [r for r in rows if not r[5]]
    if not ok:
        print("  none")
    else:
        best = max(ok, key=lambda r: (r[4], -r[0]))
        for stem, mo, em, cor, cond, bad, miss in sorted(ok, key=lambda r: -r[4]):
            mark = ' <-- best' if (stem, mo) == (best[0], best[1]) else ''
            print(f"  overlap {mo:.2f} stem {stem}: conditional accuracy "
                  f"{cond:.1f}%, {len(miss)} legitimate refused{mark}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'sweep'
    if cmd == 'capture':
        capture()
    else:
        sweep()
