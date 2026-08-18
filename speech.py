"""The one way this project turns audio into words.

Everything that transcribes goes through transcribe_file or transcribe_pcm:
prosody_core.measure, live.measure_segment, transcribe.py and plot_prosody.py.
There is no second recogniser. The batch and live paths already disagree about
how they cut speech into units, and having them also disagree about the words
would make neither of them checkable against the other.

The recogniser is Apple's SpeechAnalyzer, reached through the Swift helper in
bridge/daemon.swift, which is a long-lived process rather than one launch per
request: loading the Speech framework and building an analyzer costs far more
than transcribing a two-second segment, and the live path cannot pay that on
every utterance.

Words come back as objects with .word, .start and .end, which is the shape
faster-whisper produced and everything downstream expects.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BRIDGE = Path(__file__).resolve().parent / "bridge" / "daemon"
SOURCE = Path(__file__).resolve().parent / "bridge" / "daemon.swift"
DEFAULT_LOCALE = "en_US"
SAMPLE_RATE = 16000

_PROC = None
_LOCK = threading.Lock()


def build():
    """Compile the bridge if it is missing or older than its source."""
    if BRIDGE.exists() and BRIDGE.stat().st_mtime >= SOURCE.stat().st_mtime:
        return
    subprocess.run(["swiftc", "-O", "-parse-as-library", str(SOURCE),
                    "-o", str(BRIDGE)], check=True, cwd=str(SOURCE.parent))


def daemon():
    """The running helper, started once and kept."""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    build()
    _PROC = subprocess.Popen([str(BRIDGE)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=0)
    line = _PROC.stderr.readline()          # "ready"
    if b"ready" not in line:
        raise RuntimeError(f"bridge did not start: {line!r}")
    return _PROC


def shutdown():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.stdin.write(b"quit\n")
            _PROC.stdin.flush()
            _PROC.wait(timeout=5)
        except Exception:
            _PROC.kill()
    _PROC = None


def _request(header, payload=None):
    with _LOCK:
        p = daemon()
        p.stdin.write(header.encode())
        if payload is not None:
            p.stdin.write(payload)
        p.stdin.flush()
        line = p.stdout.readline()
    if not line:
        raise RuntimeError("bridge closed the connection")
    reply = json.loads(line)
    if not reply.get('ok'):
        raise RuntimeError(f"transcription failed: {reply.get('error')}")
    return reply


def _words(reply):
    """SpeechAnalyzer strips the leading space faster-whisper put on every
    token. Nothing downstream depended on it: prosody_core.split_punctuation
    calls .strip() before doing anything, which I checked rather than assumed.
    The space is added back anyway so the two are byte-identical in shape."""
    return [SimpleNamespace(word=' ' + w['text'], start=w['start'], end=w['end'])
            for w in reply['words']]


def transcribe_file(path, locale=DEFAULT_LOCALE):
    reply = _request(f"file {locale} {path}\n")
    return _words(reply), reply


def transcribe_pcm(audio, locale=DEFAULT_LOCALE):
    """Transcribe an array in memory, with no file involved.

    The live path holds each segment as a numpy array. A temp file per segment
    would add a write, a read and a delete to the one path whose entire point
    is latency, and would leave litter behind on every crash. The bridge takes
    raw float32 instead.
    """
    a = np.ascontiguousarray(np.asarray(audio, dtype='<f4'))
    raw = a.tobytes()
    reply = _request(f"pcm {locale} {len(raw)}\n", raw)
    return _words(reply), reply


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"
    locale = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LOCALE
    words, reply = transcribe_file(path, locale)
    print(f"{len(words)} words, {reply['audioSeconds']:.0f}s audio in "
          f"{reply['seconds']:.2f}s ({reply['audioSeconds']/reply['seconds']:.0f}x), "
          f"locale {reply['locale']}")
    for w in words:
        print(f"{w.start:7.2f} {w.end:7.2f}  {w.end-w.start:5.2f}  {w.word.strip()}")
    shutdown()
