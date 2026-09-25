"""Does a larger model fix composition, or is something else wrong.

The on-device model has now failed four judgement tasks: concluding from an
episode, judging relatedness, inferring whether a question needs reasoning,
and composing an answer from retrieved lines. Before building anything else on
that stage, the question is whether the stage is sound and the model is the
limit.

Same 51 questions where the gold evidence was retrieved, same retrieved lines
captured in threshold_capture.json, same judge. Only the composing model
changes. If accuracy jumps, composition is fine and the model is the limit. If
it does not, the fault is in the stage and a bigger model will not buy it.

usage: compose_model.py [--all]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent


def main():
    import anthropic
    import capability as cap
    import gate as g
    import memory as mem
    import recall as rc
    from eval_judge import judge_one

    # cap.ask refuses a non-local provider without a gate to ask, which is the
    # point of it -- and it caught this script trying to reach Claude without
    # one, silently, until the spend figure did not move.
    db = mem.open(HERE / 'locomo.db')
    if not any(c['caller'] == 'eval' for c in g.callers(db)):
        (HERE / 'eval.secret').write_text(
            g.add_caller(db, 'eval', can_read=True, can_write=False,
                         note='the LoCoMo benchmark harness'))
    g.set_caller_outward(db, 'eval', True)

    caps = json.loads((HERE / 'threshold_capture.json').read_text())
    subset = caps if '--all' in sys.argv else [c for c in caps if c['retrieved']]
    key, _ = cap.claude_key()
    client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2)

    print(f"{len(subset)} questions (gold evidence retrieved)")
    print(f"on-device baseline on this subset: "
          f"{sum(1 for c in subset if c['judged'])} correct "
          f"({100*sum(1 for c in subset if c['judged'])/len(subset):.1f}%)")
    print()
    out = []
    for i, c in enumerate(subset, 1):
        try:
            reply, via = cap.ask('compose_answer',
                                 {'question': c['question'],
                                  'transcript': "\n".join(c['lines'])},
                                 budget=40.0, prefer='claude-api',
                                 db=db, caller='eval')
        except Exception as e:                                # noqa: BLE001
            reply, via = {'error': str(e)}, {}
        ans = (reply.get('answer') or '').strip()
        sup = (reply.get('support') or '').strip()
        # the guards, applied exactly as compose_from applies them
        passed, why = True, 'accepted'
        if not ans:
            passed, why = False, f"no answer ({reply.get('error') or 'declined'})"
        elif rc.grounded(sup, c['lines']) is None:
            passed, why = False, 'support not grounded'
        else:
            ov = rc.answer_overlap(ans, "\n".join(c['lines']))
            if ov < rc.ANSWER_MIN_OVERLAP:
                passed, why = False, f'overlap {ov:.2f}'
        judged = judge_one(client, c['question'], c['gold'], ans) if ans else False
        out.append({**{k: c[k] for k in ('question', 'gold', 'category')},
                    'answer': ans, 'support': sup, 'passed_guards': passed,
                    'why': why, 'judged': judged})
        if i % 10 == 0:
            print(f"  {i}/{len(subset)}", flush=True)
            (HERE / 'compose_model.json').write_text(json.dumps(out))
    (HERE / 'compose_model.json').write_text(json.dumps(out))

    n = len(out)
    right = sum(1 for r in out if r['judged'])
    emitted = sum(1 for r in out if r['passed_guards'])
    both = sum(1 for r in out if r['judged'] and r['passed_guards'])
    print()
    print(f"claude-api composing, same 51 lines:")
    print(f"  produced a correct answer      : {right}/{n} ({100*right/n:.1f}%)")
    print(f"  passed the guards              : {emitted}/{n} ({100*emitted/n:.1f}%)")
    print(f"  correct AND emitted            : {both}/{n} ({100*both/n:.1f}%)")
    print(f"  correct but refused by a guard : {right-both}")
    from collections import Counter
    print(f"  refusal reasons: "
          f"{dict(Counter(r['why'].split()[0] for r in out if not r['passed_guards']))}")
    print(f"\nspend: {cap.spend_today('claude-api')}")


if __name__ == '__main__':
    main()
