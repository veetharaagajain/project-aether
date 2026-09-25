"""Ask LoCoMo's questions through the real retrieval path, and score them.

Through gate.answer_or_search with a registered caller, so admission, the
access log and relevance.narrow all run exactly as they do for a spoken
question. Measuring a bypass would measure nothing.

THE THREE CONSTANTS, and why each is set rather than accepted:

  DEFAULT_SEARCH_LIMIT, 5 in gate.py, set to 10 here. Five is tuned for
  speech, where an observation is one run between silences -- the median is
  four words -- so five hits is a couple of sentences. A LoCoMo turn is a
  whole paragraph, so five hits is already a lot of text, but multi-hop
  questions need evidence from at least two turns in different sessions and
  five gives the ranker very little room to place both. Ten is the smallest
  number that can hold two pieces of evidence plus distractors, and it is
  inside the range published RAG comparisons use.

  RECENT_MINUTES, 20 in recall.py, set to 0 here. It exists so a question
  about something said a minute ago is answered without searching. LoCoMo
  sessions are months apart and the whole corpus is loaded in bulk, so
  "recent" would mean whatever happened to be inserted last -- an artefact of
  load order, not of the conversation. At 0 the rung never fires and every
  question goes through retrieval, which is the thing being measured.

  EPISODE_GAP_S, 120 in consolidate.py, unused. It segments a day into
  episodes by silence, and there is no silence in written dialogue. Nothing
  here runs consolidation: the QA task is answered from turns, and beliefs
  are a separate mechanism that this does not test. If consolidation were
  ever run over LoCoMo the boundaries should come from the session keys the
  data already carries, not from a gap threshold.

usage: locomo_run.py [--db PATH] [--n PER_CATEGORY] [--limit K] [--out FILE]
"""

import json
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path(__file__).resolve().parent / "locomo10.json"
DB = Path(__file__).resolve().parent / "locomo.db"
OUT = Path(__file__).resolve().parent / "locomo_results.json"

SEARCH_LIMIT = 10
RECENT_MINUTES = 0.0
JUDGE_MODEL = 'claude-haiku-4-5'      # see judge(); a bounded classification

CATEGORIES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain',
              4: 'single-hop', 5: 'adversarial'}


# --- F1, the benchmark's own metric -----------------------------------------
def normalise(s):
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def f1(pred, gold):
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def exact(pred, gold):
    return float(normalise(pred) == normalise(gold))


# --- the judge ---------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are scoring a memory system's answer against a reference answer.\n\n"
    "Say correct only if the answer conveys the same fact as the reference. "
    "Wording, length and phrasing do not matter -- the system deliberately "
    "rephrases rather than quoting, so 'You had a sausage croissant' and "
    "'sausage croissant' are the same answer. Extra correct detail is fine. "
    "A different fact, a missing fact, or a refusal is not correct.\n\n"
    "Reply as JSON: {\"correct\": true|false, \"why\": \"a few words\"}")

JUDGE_SCHEMA = {'type': 'object',
                'properties': {'correct': {'type': 'boolean'},
                               'why': {'type': 'string'}},
                'required': ['correct', 'why'], 'additionalProperties': False}


def judge(client, question, gold, pred, adversarial=False):
    """Semantic equivalence, by a small model.

    Haiku rather than Opus deliberately: this is a bounded two-way judgement
    over two short strings, repeated two thousand times, which is what a small
    model is for. Using the same model that answers would also make the score
    partly a measure of that model agreeing with itself.
    """
    if adversarial:
        sysmsg = (
            "A question was asked that the conversation does NOT answer. The "
            "correct behaviour is to say so -- 'no information', 'nothing "
            "about that', 'I don't know', a refusal, or silence are all "
            "correct. Producing a confident factual answer is incorrect, "
            "because there was nothing to answer from.\n\n"
            "Reply as JSON: {\"correct\": true|false, \"why\": \"a few words\"}")
        user = f"Question: {question}\n\nThe system said: {pred or '(silence)'}"
    else:
        sysmsg = JUDGE_SYSTEM
        user = (f"Question: {question}\n\nReference answer: {gold}\n\n"
                f"The system said: {pred or '(silence)'}")
    r = client.messages.create(
        model=JUDGE_MODEL, max_tokens=200, system=sysmsg,
        messages=[{'role': 'user', 'content': user}],
        output_config={'format': {'type': 'json_schema', 'schema': JUDGE_SCHEMA}})
    txt = "".join(b.text for b in r.content if b.type == 'text')
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        d = {'correct': False, 'why': 'judge returned nothing parseable'}
    d['tokens'] = (r.usage.input_tokens, r.usage.output_tokens)
    return d


# --- one question ------------------------------------------------------------
def ask(db, secret, question, limit=SEARCH_LIMIT):
    """Through the gate, the way a spoken question goes."""
    import gate as g
    import recall as rc
    t0 = time.perf_counter()
    r = g.answer_or_search(db, 'eval', secret, question, limit=limit)
    answer = r.get('answer')
    hits = r.get('released') or []
    how = 'direct' if r.get('answered_directly') else 'search'
    if not answer and hits:
        try:
            answer = rc.compose_from(question, [h['text'] for h in hits])['answer']
            how = 'composed'
        except rc.NotAnswerable:
            try:
                nf = rc.compose_not_found(question, [h['text'] for h in hits])
                answer, how = nf['answer'], 'refused'
            except rc.NotRecallable:
                answer, how = None, 'not-recall'
            except Exception:
                answer, how = None, 'nothing'
        except Exception:
            answer, how = None, 'error'
    return {'answer': answer, 'how': how, 'n_hits': len(hits),
            'hit_texts': [h.get('text', '')[:200] for h in hits],
            'seconds': round(time.perf_counter() - t0, 2)}


# --- the run -----------------------------------------------------------------
def sample_questions(n_per_cat=None, seed=20260831):
    data = json.loads(SRC.read_text())
    by_cat = defaultdict(list)
    for s in data:
        for q in s.get('qa', []):
            c = int(q.get('category'))
            gold = q.get('answer', q.get('adversarial_answer'))
            by_cat[c].append({'sample': s['sample_id'], 'category': c,
                              'question': q['question'],
                              'gold': '' if c == 5 else str(gold),
                              'evidence': q.get('evidence')})
    rng = random.Random(seed)
    out = []
    for c in sorted(by_cat):
        items = by_cat[c]
        rng.shuffle(items)
        out += items[:n_per_cat] if n_per_cat else items
    return out, {c: len(v) for c, v in sorted(by_cat.items())}


def main():
    import anthropic
    import capability as cap
    import gate as g
    import memory as mem
    import recall as rc

    args = sys.argv[1:]
    path = Path(args[args.index('--db') + 1]) if '--db' in args else DB
    n_per = int(args[args.index('--n') + 1]) if '--n' in args else None
    limit = int(args[args.index('--limit') + 1]) if '--limit' in args else SEARCH_LIMIT
    out_path = Path(args[args.index('--out') + 1]) if '--out' in args else OUT

    # the two constants that are set rather than accepted; the third is unused
    rc.RECENT_MINUTES = RECENT_MINUTES
    # The daily ceilings are NOT raised here. They exist to stop an unattended
    # loop running up a bill, and a benchmark harness raising them by default
    # is how they stop meaning anything. A run large enough to hit them should
    # be an explicit decision each time: pass --raise-limits, which says so in
    # the log, or split the run across days.
    if '--raise-limits' in args:
        cap.DAILY_CALL_LIMIT = 5000
        cap.DAILY_TOKEN_LIMIT = 5_000_000
        print("WARNING: daily spend ceilings raised for this run "
              f"({cap.DAILY_CALL_LIMIT} calls, {cap.DAILY_TOKEN_LIMIT} tokens)")

    db = mem.open(path)
    if not any(c['caller'] == 'eval' for c in g.callers(db)):
        secret = g.add_caller(db, 'eval', can_read=True, can_write=False,
                              note='the LoCoMo benchmark harness')
        (path.parent / 'eval.secret').write_text(secret)
    secret = (path.parent / 'eval.secret').read_text().strip()
    g.set_outward_policy(db, 'off')
    g.set_caller_outward(db, 'eval', True)

    key, _ = cap.claude_key()
    client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2)

    qs, totals = sample_questions(n_per)
    print(f"corpus categories: " + ", ".join(
        f"{CATEGORIES[c]} {n}" for c, n in totals.items()))
    print(f"asking {len(qs)} question(s), search limit {limit}, "
          f"recent-speech rung {'off' if not RECENT_MINUTES else 'on'}")
    results = []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        adv = q['category'] == 5
        try:
            r = ask(db, secret, q['question'], limit=limit)
        except Exception as e:                                   # noqa: BLE001
            r = {'answer': None, 'how': f'error: {type(e).__name__}',
                 'n_hits': 0, 'hit_texts': [], 'seconds': 0.0}
        try:
            j = judge(client, q['question'], q['gold'], r['answer'], adv)
        except Exception as e:                                   # noqa: BLE001
            j = {'correct': False, 'why': f'judge failed: {e}', 'tokens': (0, 0)}
        rec = {**q, **r, 'judged': bool(j['correct']), 'judge_why': j['why'],
               'f1': 0.0 if adv else f1(r['answer'] or '', q['gold']),
               'em': 0.0 if adv else exact(r['answer'] or '', q['gold'])}
        results.append(rec)
        if i % 10 == 0 or i == len(qs):
            done = time.time() - t0
            print(f"  {i}/{len(qs)}  {done:.0f}s  "
                  f"({done/i:.1f}s each, ~{(len(qs)-i)*done/i/60:.0f} min left)",
                  flush=True)
            out_path.write_text(json.dumps(results))
    out_path.write_text(json.dumps(results))
    report(results)
    print()
    for name in ('claude-api', 'gemini'):
        d = cap.spend_today(name)
        if d['calls']:
            print(f"spend {name}: {d['calls']} calls, "
                  f"{d['in']}+{d['out']} tokens, {d['usd']:.3f} usd")
    print(f"judge: {len(results)} calls on {JUDGE_MODEL}")
    return 0


def report(results):
    by = defaultdict(list)
    for r in results:
        by[r['category']].append(r)
    print()
    print("category      n   judged      F1     EM   answered  refused")
    for c in sorted(by):
        rs = by[c]
        n = len(rs)
        jd = sum(r['judged'] for r in rs) / n
        f = sum(r['f1'] for r in rs) / n
        e = sum(r['em'] for r in rs) / n
        ans = sum(1 for r in rs if r['answer']) / n
        ref = sum(1 for r in rs if r['how'] in ('refused', 'not-recall')) / n
        tag = CATEGORIES[c] + (' *' if c == 5 else '')
        print(f"{tag:<13} {n:4d}  {100*jd:6.1f}%  {100*f:5.1f}% {100*e:5.1f}%  "
              f"{100*ans:6.1f}%  {100*ref:6.1f}%")
    core = [r for r in results if r['category'] != 5]
    if core:
        print(f"{'four-category':<13} {len(core):4d}  "
              f"{100*sum(r['judged'] for r in core)/len(core):6.1f}%  "
              f"{100*sum(r['f1'] for r in core)/len(core):5.1f}% "
              f"{100*sum(r['em'] for r in core)/len(core):5.1f}%")
    print("* adversarial: judged means it correctly declined to answer")


if __name__ == '__main__':
    sys.exit(main())
