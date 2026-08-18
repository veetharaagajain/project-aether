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
import subprocess
import sys
import threading
import time
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent / "bridge" / "relevance"
SOURCE = Path(__file__).resolve().parent / "bridge" / "relevance.swift"
DEFAULT_MODE = "each"
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


def available():
    """Whether the model is there, without judging anything."""
    try:
        daemon()
    except Exception as e:                       # noqa: BLE001
        return False, str(e)
    return (_UNAVAILABLE is None), _UNAVAILABLE


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


def judge(question, candidates, mode=DEFAULT_MODE, concurrency=4):
    """Which candidates answer the question. Returns (keep_indices, meta).

    keep_indices are positions in `candidates`. Raises Unavailable when the
    model is not there or the bridge died; it never returns "all of them" as a
    way of coping.
    """
    if not candidates:
        return [], {'calls': 0, 'seconds': 0.0, 'mode': mode}
    payload = json.dumps({'question': question,
                          'candidates': list(candidates),
                          'mode': mode,
                          'concurrency': concurrency}).encode()
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
            p.stdin.write(f"judge {len(payload)}\n".encode())
            p.stdin.write(payload)
            p.stdin.flush()
            line = p.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            shutdown()
            raise Unavailable(f"the relevance bridge died: {e}") from e
    if not line:
        shutdown()
        raise Unavailable("the relevance bridge closed the connection")
    r = json.loads(line)
    if r.get('unavailable'):
        raise Unavailable(r['unavailable'])
    if not r.get('ok') and not r.get('keep'):
        raise Unavailable(r.get('error') or 'the judge failed')
    return r.get('keep', []), {
        'calls': r.get('calls', 0), 'seconds': r.get('seconds', 0.0),
        'per_call': r.get('perCall', []), 'mode': r.get('mode', mode),
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

    Returns (kept_hits, note). note says what happened, and is written into the
    access log so a narrow release and a failed judge do not look alike.
    """
    mode = policy(db)
    if mode == 'off':
        return list(hits), {'judged': False, 'policy': 'off',
                            'reason': 'relevance judging is switched off'}
    if not hits:
        return [], {'judged': False, 'policy': mode, 'reason': 'nothing to judge'}
    try:
        keep, meta = judge(question, [text_of(h) for h in hits])
    except Unavailable as e:
        raise Unavailable(
            f"the relevance judge could not run ({e}), and the policy is "
            f"'required', so nothing was released. This is not an empty "
            f"result: the store was not consulted for an answer. Set the "
            f"policy to 'off' to release unnarrowed search results.") from e
    kept = [h for i, h in enumerate(hits) if i in set(keep)]
    return kept, {'judged': True, 'policy': mode, 'considered': len(hits),
                  'kept': len(kept), 'seconds': round(meta['seconds'], 3),
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
