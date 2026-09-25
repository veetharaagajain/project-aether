"""Deciding which fragments actually answer a question. Aether's own thinking.

Everything else in this project measures. This judges, and it is the smallest
useful thing the deciding layer can do: memory.search ranks by resemblance, and
resemblance is not an answer. A question about breakfast returned ten fragments
where one answered it and nine were merely near it in meaning, and the nine
were the person's speech handed over for no reason. Narrowing that is what
makes the privacy property real rather than aspirational.

THE MODEL. Apple's on-device model through FoundationModels, reached by the
long-lived Swift helper in bridge/relevance.swift, the same pattern speech.py
uses for transcription. On-device is not a preference here: a relevance
judgement reads the person's speech, so a judge that needed a network call
would undo the property the gate exists to provide. It is also free and
unmetered, which matters when it runs in front of every release.

WHAT IT DOES NOT DO. It does not decide whether a question was addressed to
Aether, which model should answer it, or whether to speak. One judgement,
narrowly: of these candidates, which answer this.

WHEN IT CANNOT RUN. See MODES. The default refuses to release rather than
falling back to releasing everything, because a fallback that quietly restores
the old behaviour is worse than an outage -- it looks like it is working.
"""

import json
import select
import subprocess
import sys
import threading
import time

import memory as mem
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent / "bridge" / "relevance"
SOURCE = Path(__file__).resolve().parent / "bridge" / "relevance.swift"
DEFAULT_MODE = "each"
# The bridge times out each candidate at PER_CALL_TIMEOUT_S and keeps going,
# so this only fires if the bridge itself stops answering. Generous enough
# not to race that mechanism, tight enough that a release cannot hang.
REPLY_BASE_S = 20.0
REPLY_PER_CANDIDATE_S = 15.0
STARTUP_TIMEOUT_S = 30.0

# How the gate behaves when the judge cannot run. Stored in meta, so it is the
# person's choice and visible rather than a constant someone edits.
#
#   required  no judge, no release. The narrowing IS the feature; releasing
#             unnarrowed is exactly the behaviour this exists to remove, and a
#             silent fallback to it would be indistinguishable from working.
#   off       no judging at all, release whatever search ranked. Honest, and
#             the only way to get the old behaviour is to ask for it by name.
MODES = ('required', 'off')
DEFAULT_POLICY = 'required'

_PROC = None
_LOCK = threading.Lock()
_UNAVAILABLE = None


class Unavailable(Exception):
    """The judge could not run. Deliberately distinct from 'nothing matched':
    a caller that gets an empty release must be able to tell the difference
    between the store having no answer and the judge being down."""


def build():
    """Compile the bridge if it is missing or older than its source.

    Every way this can fail becomes Unavailable, so a caller has exactly one
    exception to handle. A missing source file, a broken toolchain and a model
    that will not load are the same event as far as the gate is concerned: the
    judge cannot run, so nothing is released.
    """
    try:
        if BRIDGE.exists() and BRIDGE.stat().st_mtime >= SOURCE.stat().st_mtime:
            return
        if not SOURCE.exists():
            raise Unavailable(f"the bridge source {SOURCE} is missing")
        subprocess.run(["swiftc", "-O", "-parse-as-library", str(SOURCE),
                        "-o", str(BRIDGE)], check=True, cwd=str(SOURCE.parent),
                       capture_output=True)
    except Unavailable:
        raise
    except subprocess.CalledProcessError as e:
        raise Unavailable(
            "the relevance bridge would not compile: "
            + (e.stderr or b'').decode(errors='replace').strip()[:300]) from e
    except OSError as e:
        raise Unavailable(f"the relevance bridge could not be built: {e}") from e


def daemon():
    """The running helper, started once and kept warm.

    Warm matters more here than for transcription: the first call into
    FoundationModels pays model load, and bridge/relevance.swift spends that
    once at startup rather than on somebody's first question.
    """
    global _PROC, _UNAVAILABLE
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    build()
    try:
        _PROC = subprocess.Popen([str(BRIDGE)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 bufsize=0)
    except OSError as e:
        raise Unavailable(f"the relevance bridge would not start: {e}") from e
    _UNAVAILABLE = None
    while True:
        line = _PROC.stderr.readline()
        if not line:
            raise Unavailable("the relevance bridge exited before it was ready")
        s = line.decode(errors='replace').strip()
        if s.startswith('unavailable:'):
            _UNAVAILABLE = s.split(':', 1)[1].strip()
        if s == 'ready':
            return _PROC


# --- health --------------------------------------------------------------
# What available() used to be: "did the helper process start and print ready".
# That is a check on the wrapper, not on the thing. When Apple's on-device
# model went wrong at the OS level -- SensitiveContentAnalysisML error 15,
# ModelManagerError 1013 -- the bridge stayed up, kept answering, and failed
# every single call, while available() went on saying yes. In the router that
# is worse than a plain outage: a provider that lies about being available is
# first in order for classify, extract and compose, so nothing ever falls
# through to something that works.
#
# So health is decided by what actually happened, not by what is running.
# Every reply passes through note_outcome, and real traffic keeps the verdict
# fresh for free. A probe -- one real, tiny generation -- runs only when there
# is no recent evidence, which in normal use is never.
HEALTH_TTL_S = 60.0          # how long an observed outcome speaks for
FAILURES_TO_UNHEALTHY = 2    # consecutive, so one blip does not flap the router
PROBE_BUDGET_S = 12.0

# Errors that mean the model could not work, as opposed to the model working
# and saying no. The second is a normal outcome and must not mark anything
# unhealthy: "canAnswer false" is the on-device model doing its job.
INFRA_MARKERS = ('foundationmodels', 'modelmanager', 'sensitivecontent',
                 'languagemodelerror', 'bad json', 'timed out', 'unavailable',
                 'the bridge', 'inference')

_HEALTH = {'ok': None, 'why': 'never checked', 'at': 0.0, 'fails': 0,
           'probes': 0, 'probe_s': 0.0, 'observed': 0}


def health():
    """What is known about the model right now, for a header or the viewer."""
    return dict(_HEALTH)


def _infra_failure(reply):
    if not isinstance(reply, dict):
        return False
    if reply.get('unavailable'):
        return True
    err = str(reply.get('error') or '').lower()
    if err and any(m in err for m in INFRA_MARKERS):
        return True
    # ok is the bridge's own verdict on whether it could do the work at all
    return reply.get('ok') is False


def note_outcome(reply=None, failed_because=None):
    """Record what a real call did. This is the cheap half of the health check
    and the reason a probe almost never runs."""
    _HEALTH['observed'] += 1
    _HEALTH['at'] = time.time()
    if failed_because is not None or _infra_failure(reply):
        _HEALTH['fails'] += 1
        why = failed_because or str((reply or {}).get('error'))[:160]
        if _HEALTH['fails'] >= FAILURES_TO_UNHEALTHY:
            _HEALTH.update(ok=False, why=f'{_HEALTH["fails"]} calls in a row '
                                         f'failed: {why}')
        return
    _HEALTH.update(ok=True, why='a call succeeded', fails=0)


def probe():
    """One real generation, to find out rather than to assume.

    Deliberately the smallest thing the model can be asked that still exercises
    the whole path: the bridge, FoundationModels, guided generation, and the
    reply parse. A ready flag would exercise none of it, which is how this
    went wrong in the first place.
    """
    t0 = time.perf_counter()
    _HEALTH['probes'] += 1
    body = json.dumps({'utterance': 'hello', 'context': '',
                       'presence': ''}).encode()
    try:
        r = request(f"addressed {len(body)}\n", body, PROBE_BUDGET_S,
                    _note=False)
    except Unavailable as e:
        _HEALTH['probe_s'] += time.perf_counter() - t0
        _HEALTH.update(ok=False, why=f'probe failed: {e}', at=time.time())
        return False, _HEALTH['why']
    took = time.perf_counter() - t0
    _HEALTH['probe_s'] += took
    if _infra_failure(r):
        _HEALTH.update(ok=False, at=time.time(),
                       why=f'probe failed: {str(r.get("error"))[:160]}')
        return False, _HEALTH['why']
    _HEALTH.update(ok=True, why=f'probe answered in {took:.1f}s', at=time.time(),
                   fails=0)
    return True, _HEALTH['why']


def available(allow_probe=True):
    """Whether the model can actually do work.

    Three answers, cheapest first. If the bridge will not start, no. If a real
    call succeeded or failed recently, that is the answer and it costs nothing.
    Only with no recent evidence does this pay for a probe.
    """
    try:
        daemon()
    except Exception as e:                       # noqa: BLE001
        _HEALTH.update(ok=False, why=str(e), at=time.time())
        return False, str(e)
    if _UNAVAILABLE:
        _HEALTH.update(ok=False, why=_UNAVAILABLE, at=time.time())
        return False, _UNAVAILABLE
    fresh = (time.time() - _HEALTH['at']) < HEALTH_TTL_S
    if fresh and _HEALTH['ok'] is not None:
        return _HEALTH['ok'], _HEALTH['why']
    if not allow_probe:
        return True, 'not checked (probing disabled)'
    return probe()


def shutdown():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.stdin.write(b"quit\n")
            _PROC.stdin.flush()
            _PROC.wait(timeout=5)
        except Exception:                        # noqa: BLE001
            _PROC.kill()
    _PROC = None


def request(header, payload, budget, _note=True):
    """One request/response on the shared bridge. Used by judge and by recall.

    Exposed so recall.py runs in the same process as the judge: the model takes
    seconds to load and there is no reason to pay that twice, and two processes
    holding the same on-device model is a way to discover its concurrency
    limits by accident.
    """
    with _LOCK:
        try:
            p = daemon()
        except Unavailable:
            raise
        except Exception as e:                   # noqa: BLE001
            raise Unavailable(f"the relevance bridge is not usable: {e}") from e
        if _UNAVAILABLE:
            raise Unavailable(_UNAVAILABLE)
        try:
            p.stdin.write(header.encode())
            p.stdin.write(payload)
            p.stdin.flush()
            ready, _, _ = select.select([p.stdout], [], [], budget)
            if not ready:
                shutdown()
                if _note:
                    note_outcome(failed_because=f'timed out after {budget:.0f}s')
                raise Unavailable(
                    f"the bridge did not answer within {budget:.0f}s; it was "
                    f"restarted and nothing was returned")
            line = p.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            shutdown()
            if _note:
                note_outcome(failed_because=f'the bridge died: {e}')
            raise Unavailable(f"the relevance bridge died: {e}") from e
    if not line:
        shutdown()
        if _note:
            note_outcome(failed_because='the bridge closed the connection')
        raise Unavailable("the relevance bridge closed the connection")
    reply = json.loads(line)
    if _note:
        note_outcome(reply)
    return reply


def judge(question, candidates, mode=DEFAULT_MODE, concurrency=4):
    """Which candidates answer the question. Returns (keep_indices, meta).

    keep_indices are positions in `candidates`. Raises Unavailable when the
    model is not there or the bridge died; it never returns "all of them" as a
    way of coping.
    """
    if not candidates:
        return [], {'calls': 0, 'seconds': 0.0, 'mode': mode}
    payload = {'question': question, 'candidates': list(candidates),
               'mode': mode, 'concurrency': concurrency}
    # Through the capability boundary, not straight at the bridge. This asks
    # for a classification and does not care who provides it; capability.py is
    # the only file that knows the answer is currently Apple's on-device model.
    import capability as cap
    budget = REPLY_BASE_S + REPLY_PER_CANDIDATE_S * len(candidates)
    try:
        r, via = cap.ask('judge_relevance', payload, budget=budget)
    except cap.NoProvider as e:
        raise Unavailable(str(e)) from e
    if r.get('unavailable'):
        raise Unavailable(r['unavailable'])
    if not r.get('ok') and not r.get('keep'):
        raise Unavailable(r.get('error') or 'the judge failed')
    return r.get('keep', []), {
        'calls': r.get('calls', 0), 'seconds': r.get('seconds', 0.0),
        'per_call': r.get('perCall', []), 'mode': r.get('mode', mode),
        'needs_context': set(r.get('needsContext', [])),
        'partial_error': r.get('error')}


# --- the policy, stored beside everything else -------------------------------
def policy(db):
    r = db.execute("SELECT value FROM meta WHERE key='relevance_policy'").fetchone()
    v = r[0] if r else DEFAULT_POLICY
    return v if v in MODES else DEFAULT_POLICY


def set_policy(db, mode):
    if mode not in MODES:
        raise ValueError(f"policy must be one of {MODES}")
    db.execute("INSERT INTO meta(key,value) VALUES('relevance_policy',?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (mode,))
    db.commit()
    return policy(db)


def narrow(db, question, hits, text_of=lambda h: h.get('text', '')):
    """Narrow search hits to the ones that answer the question.

    Each observation is judged inside its neighbourhood rather than alone. The
    store is chunked by silence, so an observation is one run of speech: the
    median is four words and 59 percent are four or fewer. Judging those alone
    threw away answers that were split mid-sentence -- "Charitable trust and not
    a public authority under the RTI Act" was rejected because nothing in it
    says which fund, and the line naming the fund was 0.42 seconds earlier.

    Beliefs are judged as they are. A belief is already a whole statement
    written to stand alone, and it has no position in a session to reach from.

    Returns (kept_hits, note). Kept hits carry 'role': 'answer' for the fragment
    that was judged, 'context' for a neighbour released with it. note is written
    into the access log so a narrow release and a failed judge do not look alike.
    """
    mode = policy(db)
    if mode == 'off':
        return list(hits), {'judged': False, 'policy': 'off',
                            'reason': 'relevance judging is switched off'}
    if not hits:
        return [], {'judged': False, 'policy': mode, 'reason': 'nothing to judge'}

    windows, prompts = [], []
    for h in hits:
        if h.get('kind') == 'observation' and h.get('id'):
            rows = mem.neighbourhood(db, h['id'])
            windows.append(rows)
            prompts.append(mem.window_text(rows) if rows else text_of(h))
        else:
            windows.append(None)
            prompts.append(f">> {text_of(h)}")
    try:
        keep, meta = judge(question, prompts)
    except Unavailable as e:
        raise Unavailable(
            f"the relevance judge could not run ({e}), and the policy is "
            f"'required', so nothing was released. This is not an empty "
            f"result: the store was not consulted for an answer. Set the "
            f"policy to 'off' to release unnarrowed search results.") from e

    needs = meta.get('needs_context') or set()
    kept, seen, n_ctx = [], set(), 0
    for i in sorted(set(keep)):
        h = dict(hits[i], role='answer')
        if h.get('id') in seen:
            continue
        seen.add(h.get('id'))
        kept.append(h)
        if i not in needs or not windows[i]:
            continue
        # Only the neighbours tight enough to be the same run of speech, and
        # only the ones adjacent to the fragment. The judge saw two either side;
        # releasing all four because one was needed would hand over three
        # utterances nobody asked about.
        by_offset = {r['offset']: r for r in windows[i]}
        for off in (-1, 1):
            r = by_offset.get(off)
            if r is None or r['id'] in seen:
                continue
            # 'gap' on a row is the silence before that row, so the silence
            # separating a neighbour from the target is the target's own gap on
            # the left and the neighbour's own gap on the right.
            sep = by_offset[0]['gap'] if off == -1 else r['gap']
            if sep > mem.NEIGHBOUR_TIGHT_GAP_S:
                continue
            seen.add(r['id'])
            n_ctx += 1
            kept.append({'kind': 'observation', 'id': r['id'], 'text': r['text'],
                         'at': r['started_at'], 'score': hits[i].get('score'),
                         'role': 'context', 'context_for': hits[i]['id'],
                         'separation_s': round(float(sep), 2)})
    return kept, {'judged': True, 'policy': mode, 'considered': len(hits),
                  'kept': len(kept), 'answers': len(kept) - n_ctx,
                  'context': n_ctx, 'seconds': round(meta['seconds'], 3),
                  'calls': meta['calls'], 'partial_error': meta.get('partial_error')}


if __name__ == "__main__":
    ok, why = available()
    print(f"model available: {ok}" + (f" ({why})" if why else ""))
    if len(sys.argv) > 2:
        q, cands = sys.argv[1], sys.argv[2:]
        keep, meta = judge(q, cands)
        print(f"{meta['calls']} call(s), {meta['seconds']:.2f}s, mode {meta['mode']}")
        for i, c in enumerate(cands):
            print(f"  {'KEEP' if i in keep else '    '}  {c}")
    shutdown()
