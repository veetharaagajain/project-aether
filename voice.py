"""Speaking, with the system voice.

The trivial part, and the one that makes the loop a loop rather than a
recorder with a log. macOS `say` is the system voice; there is no reason to
carry a speech stack for this.

Speaking runs on its own thread so the live path is not blocked -- a segment
arriving while the being is mid-sentence must still be captured and measured.
Deliberately NOT here: interruption. If the person starts talking while it is
speaking, nothing stops it. That is the next piece and entangling it with this
one would make both harder to reason about.
"""

import subprocess
import threading
import time

# The voice is stored, not a constant, for the same reason the wake word is:
# it is the person's to choose and changing it should not need a restart. It
# lives in meta under 'voice' and is read on every utterance.
#
# Samantha is the default because it is the standard high-quality US English
# voice and the safest thing to pick on someone's behalf. 186 voices are
# installed; the ones worth choosing between are listed by voices(). Rishi
# (en_IN) and Soumya (kn_IN) exist and may suit this store better than a US
# default, but that is a choice I should not make for someone.
DEFAULT_VOICE = "Samantha"
DEFAULT_RATE = None            # None means the system default speaking rate

_LOCK = threading.Lock()
_SPEAKING = {'since': None, 'text': None, 'proc': None}


def voices(english_only=True, exclude_novelty=True):
    """What is actually installed, as (name, locale) pairs."""
    NOVELTY = {'Bad News', 'Bahh', 'Bells', 'Boing', 'Bubbles', 'Cellos',
               'Good News', 'Jester', 'Organ', 'Superstar', 'Trinoids',
               'Whisper', 'Wobble', 'Zarvox', 'Albert', 'Junior', 'Ralph',
               'Fred', 'Kathy', 'Grandma', 'Grandpa', 'Rocko', 'Shelley',
               'Sandy', 'Flo', 'Eddy', 'Reed'}
    out = []
    try:
        raw = subprocess.run(['say', '-v', '?'], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    seen = set()
    for line in raw.splitlines():
        if '#' not in line:
            continue
        head = line.split('#')[0].rstrip()
        parts = head.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, locale = parts[0].strip(), parts[1].strip()
        base = name.split(' (')[0]
        if english_only and not locale.startswith('en'):
            continue
        if exclude_novelty and base in NOVELTY:
            continue
        if (base, locale) in seen:
            continue
        seen.add((base, locale))
        out.append((base, locale))
    return sorted(out)


def voice_name(db=None):
    if db is None:
        return DEFAULT_VOICE
    r = db.execute("SELECT value FROM meta WHERE key='voice'").fetchone()
    return (r[0] if r and r[0].strip() else DEFAULT_VOICE).strip()


def set_voice(db, name):
    """Change the voice. Takes effect on the next thing spoken."""
    name = (name or '').strip()
    if not name:
        raise ValueError("the voice cannot be empty")
    installed = {n for n, _ in voices(english_only=False, exclude_novelty=False)}
    if installed and name.split(' (')[0] not in installed:
        raise ValueError(f"{name!r} is not installed; try voice.voices()")
    db.execute("INSERT INTO meta(key,value) VALUES('voice',?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (name,))
    db.commit()
    return voice_name(db)


def speaking():
    p = _SPEAKING['proc']
    return bool(p and p.poll() is None)


def say(text, voice=None, rate=None, block=False, db=None):
    """Speak one line. Returns immediately unless block is set.

    voice=None means look it up in the store, so a change takes effect on the
    next thing spoken with nothing restarted.
    """
    text = (text or '').strip()
    if not text:
        return None
    voice = voice or voice_name(db)
    rate = rate or DEFAULT_RATE
    cmd = ['say']
    if voice:
        cmd += ['-v', voice]
    if rate:
        cmd += ['-r', str(rate)]
    cmd.append(text)

    def run():
        with _LOCK:
            # queued rather than overlapped: two voices at once is worse than
            # a pause, and this path emits one short line at a time
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            except OSError:
                return
            _SPEAKING.update(since=time.time(), text=text, proc=p)
            p.wait()
            _SPEAKING.update(since=None, text=None, proc=None)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    if block:
        t.join()
    return t
