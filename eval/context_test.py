"""Is the composer being starved of what it needs to answer.

The lines handed to recall.compose_from are bare text -- gate.search_memory
returns hits and the caller passes [h['text'] for h in hits]. The timestamp
and the speaker, both of which the store holds on every observation, are
dropped on the way.

That is invisible on room audio, where the being answers questions about "you"
from one speaker in recent speech. LoCoMo asks "when did Jon start reading The
Lean Startup" where the evidence line says only "I'm currently reading The Lean
Startup" and the answer, May 2023, is the session date. And it asks "which city
have both Jean and John visited", where the answer needs to know who said
which line.

So: same questions, same retrieved observations, but the lines carry their date
and speaker. If accuracy jumps, the memory layer is starving the composer.

usage: context_test.py [--n N]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent


def main():
    import anthropic
    import capability as cap
    import gate as g
    import memory as mem
    from eval_judge import judge_one

    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 25
    cap.DAILY_CALL_LIMIT = 5000          # attended, deliberate, reported
    cap.DAILY_TOKEN_LIMIT = 5_000_000
    print(f"WARNING: ceilings raised for this run ({cap.DAILY_CALL_LIMIT} calls)")

    db = mem.open(HERE / 'locomo.db')
    g.set_caller_outward(db, 'eval', True)
    key, _ = cap.claude_key()
    client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2)
    caps = [c for c in json.loads((HERE / 'threshold_capture.json').read_text())
            if c['retrieved']][:n]

    bare = enriched = 0
    for i, c in enumerate(caps, 1):
        # re-run the same search so the observation rows, not just their text,
        # are in hand
        hits = mem.search(db, c['question'], limit=10, kinds=('observation',))
        rich = []
        for h in hits:
            o = db.execute("SELECT started_at, speaker, text FROM observations "
                           "WHERE id=?", (h['id'],)).fetchone()
            if o is None:
                rich.append(h.get('text', ''))
                continue
            when = time.strftime('%d %B %Y', time.localtime(o['started_at']))
            who = o['speaker'] or 'someone'
            rich.append(f"[{when}] {who}: {o['text']}")
        for label, lines in (('bare', [h.get('text', '') for h in hits]),
                             ('rich', rich)):
            try:
                r, _ = cap.ask('compose_answer',
                               {'question': c['question'],
                                'transcript': "\n".join(lines)},
                               budget=40.0, prefer='claude-api',
                               db=db, caller='eval')
                a = (r.get('answer') or '').strip()
            except Exception:                                 # noqa: BLE001
                a = ''
            ok = judge_one(client, c['question'], c['gold'], a) if a else False
            if label == 'bare':
                bare += ok
            else:
                enriched += ok
        if i % 5 == 0:
            print(f"  {i}/{len(caps)}  bare {bare}  with date+speaker {enriched}",
                  flush=True)
    print()
    print(f"same {len(caps)} questions, same retrieved observations:")
    print(f"  lines as text only        : {bare}/{len(caps)} "
          f"({100*bare/len(caps):.0f}%)")
    print(f"  lines with date and speaker: {enriched}/{len(caps)} "
          f"({100*enriched/len(caps):.0f}%)")
    print(f"\nspend: {cap.spend_today('claude-api')}")


if __name__ == '__main__':
    main()
