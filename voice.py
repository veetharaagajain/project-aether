"""Speaking, with Kokoro when it is loaded and `say` when it is not.

The trivial part, and the one that makes the loop a loop rather than a
recorder with a log.

Two backends behind one function. Kokoro-82M through mlx-audio runs on the GPU
and sounds like this decade; macOS `say` sounds like a Mac from ten years ago
and is always there. Which one speaks is decided by the stored voice name, and
if the model is missing, still warming, or throws, the same line goes out
through `say` instead. Degrade rather than break, as everywhere else.

The number that decided this was time to first audio, because the reply
already takes about a second to compose and a voice that adds two more makes
the thing worse however good it sounds. Warm, Kokoro reaches the speaker in
about 230 ms against about 60 for `say`, so it costs roughly 170 ms. Cold it
costs 33 seconds, which is why warm_up() exists and runs at service start
rather than on the first thing anybody says.

Speaking runs on its own thread so the live path is not blocked -- a segment
arriving while the being is mid-sentence must still be captured and measured.
Deliberately NOT here: interruption. If the person starts talking while it is
speaking, nothing stops it. That is the next piece and entangling it with this
one would make both harder to reason about.
"""

import subprocess
import threading
import time

# --- Kokoro ------------------------------------------------------------------
# 4-bit, because the difference from the full weights is inaudible in one short
# sentence and it is a third of the download. 609 MB on disk, about 708 MB
# resident once the pipeline is built -- far more than "82M parameters"
# suggests, because misaki drags in spacy and MLX keeps its own arena. It fits
# in the headroom this machine has, which was worth measuring rather than
# assuming.
KOKORO_REPO = 'mlx-community/Kokoro-82M-4bit'
KOKORO_RATE = 24000
KOKORO_LANG = 'a'              # 'a' is American English in Kokoro's scheme

# The voices the model ships, by the prefix convention Kokoro uses: first
# letter is the accent (a American, b British), second is the speaker's sex.
# Only the English ones are listed: the others exist and are for languages this
# store does not yet transcribe.
KOKORO_VOICES = {
    'af_alloy': 'American female, even', 'af_aoede': 'American female, warm',
    'af_bella': 'American female, bright', 'af_heart': 'American female, soft',
    'af_jessica': 'American female, dry', 'af_kore': 'American female, clear',
    'af_nicole': 'American female, close and quiet',
    'af_nova': 'American female, crisp', 'af_river': 'American female, level',
    'af_sarah': 'American female, gentle', 'af_sky': 'American female, light',
    'am_adam': 'American male, plain', 'am_echo': 'American male, soft',
    'am_eric': 'American male, firm', 'am_fenrir': 'American male, deep',
    'am_liam': 'American male, young', 'am_michael': 'American male, warm',
    'am_onyx': 'American male, low', 'am_puck': 'American male, lively',
    'bf_alice': 'British female, precise', 'bf_emma': 'British female, warm',
    'bf_isabella': 'British female, measured',
    'bf_lily': 'British female, light',
    'bm_daniel': 'British male, even', 'bm_fable': 'British male, storyteller',
    'bm_george': 'British male, older', 'bm_lewis': 'British male, low',
}

# Every window in which this machine was making noise, so the live path can
# refuse to store what it hears during one. Wall clock, bounded, in memory
# only: it is a fact about the last few minutes of a running process, not
# something the store needs.
#
# It exists because the being heard itself. Its own replies came back through
# the microphone, were transcribed, and were written as observations from an
# unknown voice with prosody measured on them -- which means a belief drawn
# from the record could cite something the being said rather than something
# anyone said. record_answer exists for storing a model's own words
# deliberately and labelled; this was the same content arriving accidentally
# and unlabelled, which is worse than the television.
from collections import deque
SPOKEN_WINDOWS = deque(maxlen=200)
_WINDOW_LOCK = threading.Lock()


def note_spoken(start, end, text, backend):
    with _WINDOW_LOCK:
        SPOKEN_WINDOWS.append({'start': start, 'end': end, 'text': text,
                               'backend': backend})


def spoke_during(start, end, lead=0.0, tail=0.0):
    """How much of [start, end] this machine was talking over, in seconds,
    and which of its own lines were playing."""
    total, which = 0.0, []
    with _WINDOW_LOCK:
        for w in SPOKEN_WINDOWS:
            a, b = w['start'] - lead, w['end'] + tail
            ov = min(end, b) - max(start, a)
            if ov > 0:
                total += ov
                which.append(w['text'])
    return total, which


_KOKORO = {'pipe': None, 'state': 'cold', 'why': '', 'seconds': None}
_WARMING = threading.Lock()


def kokoro_state():
    """What the model is doing, for a header line or the viewer."""
    return dict(_KOKORO)


def warm_up(block=False):
    """Load the model and build the pipeline, once, off the hot path.

    Building the phonemiser takes about 32 seconds the first time because
    misaki loads spacy, and paying that on the first thing anybody says would
    be worse than the voice it replaces. The service calls this at start-up;
    until it finishes, say() falls through to the system voice, so nothing
    waits and nothing breaks.
    """
    def run():
        if not _WARMING.acquire(blocking=False):
            return
        try:
            t0 = time.perf_counter()
            _KOKORO['state'] = 'warming'
            from mlx_audio.tts.utils import load_model
            from mlx_audio.tts.models.kokoro import KokoroPipeline
            m = load_model(KOKORO_REPO)
            pipe = KokoroPipeline(lang_code=KOKORO_LANG, model=m,
                                  repo_id=KOKORO_REPO)
            # one throwaway generation: the first is ten times slower than the
            # rest and it should not be the first thing the person hears
            list(pipe('Ready.', voice='af_heart'))
            _KOKORO.update(pipe=pipe, state='ready', why='',
                           seconds=round(time.perf_counter() - t0, 1))
        except Exception as e:                                # noqa: BLE001
            _KOKORO.update(pipe=None, state='unavailable',
                           why=f'{type(e).__name__}: {e}')
        finally:
            _WARMING.release()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    if block:
        t.join()
    return t


def synthesise(text, voice, speed=1.0):
    """Kokoro audio for one line, or None if the model cannot do it."""
    pipe = _KOKORO.get('pipe')
    if pipe is None:
        return None
    try:
        import numpy as np
        chunks = [np.asarray(r.audio).reshape(-1) for r in
                  pipe(text, voice=voice, speed=speed)]
        if not chunks:
            return None
        return np.concatenate(chunks).astype('float32')
    except Exception as e:                                    # noqa: BLE001
        _KOKORO['why'] = f'generate failed: {type(e).__name__}: {e}'
        return None

# The voice is stored, not a constant, for the same reason the wake word is:
# it is the person's to choose and changing it should not need a restart. It
# lives in meta under 'voice' and is read on every utterance.
#
# Samantha is the default because it is the standard high-quality US English
# voice and the safest thing to pick on someone's behalf. 186 voices are
# installed; the ones worth choosing between are listed by voices(). Rishi
# (en_IN) and Soumya (kn_IN) exist and may suit this store better than a US
# default, but that is a choice I should not make for someone.
# af_heart rather than Samantha, now that there is a choice. It is Kokoro's
# own default and the least mannered of the American voices, which is the
# safest thing to pick on somebody's behalf -- but the point of audition() is
# that this should not stay my choice.
DEFAULT_VOICE = "af_heart"
DEFAULT_RATE = None            # None means the system default speaking rate

_LOCK = threading.Lock()
_SPEAKING = {'since': None, 'text': None, 'proc': None, 'kokoro': False}


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


def is_kokoro(name):
    return (name or '').strip() in KOKORO_VOICES


def set_voice(db, name):
    """Change the voice. Takes effect on the next thing spoken.

    One setting for both backends, because from where the person sits there is
    one voice, not a backend and a voice. The name says which: a Kokoro id
    like af_heart uses the model, anything else is a system voice.
    """
    name = (name or '').strip()
    if not name:
        raise ValueError("the voice cannot be empty")
    if not is_kokoro(name):
        installed = {n for n, _ in voices(english_only=False,
                                          exclude_novelty=False)}
        if installed and name.split(' (')[0] not in installed:
            raise ValueError(
                f"{name!r} is neither a Kokoro voice nor an installed system "
                f"voice; try voice.KOKORO_VOICES or voice.voices()")
    db.execute("INSERT INTO meta(key,value) VALUES('voice',?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (name,))
    db.commit()
    return voice_name(db)


def speaking():
    if _SPEAKING.get('kokoro'):
        return True
    p = _SPEAKING['proc']
    return bool(p and p.poll() is None)


# The longest thing it may say in one go. A reply is spoken into a room, and
# a room has no scrollback: three sentences arrive as three separate things to
# keep track of, and heard back through the microphone they arrive as three
# separate utterances too. Asking the model for one sentence is a request;
# this is the guarantee.
ONE_LINE_MAX = 180


def one_line(text):
    """The first sentence, or a little more if the first is very short.

    A refusal often opens with a stub -- "I can't work that out." -- and
    cutting there loses the part that helps. So a short opener takes the next
    sentence with it, up to ONE_LINE_MAX, and everything after that is
    dropped rather than spoken.
    """
    import re
    t = re.sub(r'\s+', ' ', (text or '').strip())
    if not t:
        return t
    parts = re.split(r'(?<=[.!?])\s+', t)
    out = parts[0]
    for nxt in parts[1:]:
        if len(out) >= 60 or len(out) + 1 + len(nxt) > ONE_LINE_MAX:
            break
        out = f"{out} {nxt}"
    return out[:ONE_LINE_MAX].strip()


def say(text, voice=None, rate=None, block=False, db=None, speed=1.0,
        whole=False):
    """Speak one line. Returns immediately unless block is set.

    voice=None means look it up in the store, so a change takes effect on the
    next thing spoken with nothing restarted. If the stored voice is a Kokoro
    one and the model is ready, it speaks; otherwise this falls through to
    `say` with a system voice, and the reason is left in kokoro_state()['why']
    rather than being raised at somebody mid-conversation.
    """
    text = (text or '').strip()
    if not text:
        return None
    if not whole:
        text = one_line(text)
    voice = voice or voice_name(db)
    rate = rate or DEFAULT_RATE
    want_kokoro = is_kokoro(voice)
    if want_kokoro and _KOKORO['state'] == 'cold':
        warm_up()                       # start it, do not wait for it

    def run_system(name):
        cmd = ['say']
        if name:
            cmd += ['-v', name]
        if rate:
            cmd += ['-r', str(rate)]
        cmd.append(text)
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError:
            return
        t_start = time.time()
        _SPEAKING.update(since=t_start, text=text, proc=p)
        p.wait()
        note_spoken(t_start, time.time(), text, 'say')
        _SPEAKING.update(since=None, text=None, proc=None)

    def run():
        with _LOCK:
            # queued rather than overlapped: two voices at once is worse than
            # a pause, and this path emits one short line at a time
            audio = synthesise(text, voice, speed) if want_kokoro else None
            if audio is None:
                # the fallback, and the only place the two backends meet. A
                # Kokoro voice name means nothing to `say`, so this hands it
                # the system default rather than a name it would reject.
                run_system(None if want_kokoro else voice)
                return
            try:
                import sounddevice as sd
                t_start = time.time()
                _SPEAKING.update(since=t_start, text=text, kokoro=True)
                sd.play(audio, KOKORO_RATE)
                sd.wait()
                note_spoken(t_start, time.time(), text, 'kokoro')
            except Exception as e:                            # noqa: BLE001
                _KOKORO['why'] = f'playback failed: {type(e).__name__}: {e}'
                _SPEAKING.update(kokoro=False)
                run_system(None)
                return
            finally:
                _SPEAKING.update(since=None, text=None, kokoro=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    if block:
        t.join()
    return t


def audition(names=None, text=None, db=None, pause=0.4):
    """Play a line in each voice, announcing which is which.

    Choosing a voice from a list of names is choosing blind, so this says the
    name and then speaks in it. Defaults to one line the being would actually
    produce, because a voice that reads a demo sentence well can still be
    wrong for the thing it will spend its life saying.
    """
    text = text or "Nothing about breakfast has been said here."
    names = names or list(KOKORO_VOICES)
    if _KOKORO['state'] != 'ready':
        warm_up(block=True)
    out = []
    for n in names:
        label = KOKORO_VOICES.get(n, 'system voice')
        print(f"  {n:<12} {label}")
        say(f"{n.replace('_', ' ')}.", voice=n, block=True)
        say(text, voice=n, block=True, db=db)
        out.append(n)
        time.sleep(pause)
    return out
