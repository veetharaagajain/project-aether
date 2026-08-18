"""Cutting continuous audio into runs of speech, before anything transcribes it.

This was the live path's gate and nothing else used it. It is here because the
batch path now uses it too: prosody_core.measure segments a file with the same
detector and the same hysteresis the microphone path uses, and transcribes each
run separately, instead of handing the recogniser a whole file and asking a
gap-and-punctuation rule to find the sentences afterwards.

Two things came out of that. The boundaries are better, because a boundary is
then an acoustic fact the detector observed rather than an inference from a
gap the transcript reported. And the last word of a run can no longer be given
an end time somewhere out in the following silence, because there is no
following silence inside the run: SpeechAnalyzer given a whole file stretched
"topics" in natural1 across 2.82 s, of which 2.3 s was nobody speaking, and
every measurement defined over that span was wrong for it.

live.py imports its gate from here, so the two paths cannot drift apart.
"""

from collections import deque

import numpy as np

SR = 16000
FRAME = 512                  # silero-vad wants exactly this at 16 kHz, 32 ms

VAD_THRESHOLD = 0.5          # silero's own speech probability
OPEN_FRAMES = 2              # consecutive speech frames to open a segment
CLOSE_S = 0.30               # quiet needed to close one
PREROLL_S = 0.25             # audio kept from before the gate opened, so the
                             # first consonant is not clipped off
MAX_SEGMENT_S = 12.0         # hard cap, so one long turn is not one long wait
MIN_SEGMENT_S = 0.20         # shorter than this is a click, not speech


# All of these change where the audio is cut and therefore what the recogniser
# sees, so every one of them changes the measured numbers. Watched by
# provenance.py.
PROVENANCE = ('SR', 'FRAME', 'VAD_THRESHOLD', 'OPEN_FRAMES', 'CLOSE_S',
              'PREROLL_S', 'MAX_SEGMENT_S', 'MIN_SEGMENT_S')


class Gate:
    """silero-vad, plus the hysteresis that turns per-frame probabilities into
    segments. Nothing expensive runs unless this says someone is talking."""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self.torch = torch
        self.model = load_silero_vad()
        self.model.reset_states()
        self.open = False
        self.speech_run = 0
        self.quiet_run = 0
        self.buf = []
        self.pre = deque(maxlen=int(PREROLL_S * SR / FRAME))
        self.start_t = 0.0
        self.just_opened = False
        self.dropped_short = False

    def push(self, frame, t):
        """One 512-sample frame in; (closed segment or None, speech probability).

        The probability comes back out so the caller can report what the
        detector actually thought, which is the difference between "nobody
        spoke" and "the detector never fired".
        """
        frame = np.ascontiguousarray(
            (frame[:, 0] if frame.ndim > 1 else frame), dtype='float32')
        if len(frame) != FRAME:
            raise ValueError(
                f"the gate needs exactly {FRAME} samples at {SR} Hz, got "
                f"{len(frame)}. silero rejects anything else.")
        p = float(self.model(self.torch.from_numpy(frame), SR).item())
        speech = p >= VAD_THRESHOLD
        self.just_opened = False

        if not self.open:
            self.pre.append(frame)
            self.speech_run = self.speech_run + 1 if speech else 0
            if self.speech_run >= OPEN_FRAMES:
                self.open = True
                self.just_opened = True
                self.buf = list(self.pre)
                self.start_t = t - len(self.buf) * FRAME / SR
                self.quiet_run = 0
            return None, p

        self.buf.append(frame)
        self.quiet_run = 0 if speech else self.quiet_run + 1
        dur = len(self.buf) * FRAME / SR

        if self.quiet_run * FRAME / SR >= CLOSE_S:
            return self._close('silence'), p
        if dur >= MAX_SEGMENT_S:
            return self._close('length cap'), p
        return None, p

    def _close(self, reason):
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, dtype='float32')
        start = self.start_t
        self.open = False
        self.buf = []
        self.pre.clear()
        self.speech_run = self.quiet_run = 0
        self.model.reset_states()
        if len(audio) / SR < MIN_SEGMENT_S:
            self.dropped_short = True
            return None
        return audio, start, reason


def load_16k(path):
    """The samples the gate wants: mono float32 at 16 kHz."""
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(x, dtype='float32')


def speech_runs(samples):
    """Every run of speech in an array, as (audio, start_s, reason).

    The same gate the microphone drives, fed from an array instead. A run
    still open when the samples run out is closed as 'end of audio', which is
    the offline equivalent of the caller stopping the stream.
    """
    gate = Gate()
    runs = []
    n = len(samples) // FRAME
    for i in range(n):
        seg, _ = gate.push(samples[i * FRAME:(i + 1) * FRAME], i * FRAME / SR)
        if seg:
            runs.append(seg)
    if gate.open:
        seg = gate._close('end of audio')
        if seg:
            runs.append(seg)
    return runs


def transcribe_runs(path, locale):
    """Segment a file on silence, then transcribe each run on its own.

    Returns the same (words, meta) shape speech.transcribe_file returns, with
    absolute times, so nothing downstream has to know which way the audio was
    cut. Each word carries .seg, the index of the run it came out of, which is
    what lets assign_sentences put a boundary there without inferring one.

    Every run still goes through speech.transcribe_pcm, so this adds a way of
    cutting the audio and not a second recogniser.
    """
    import speech
    samples = load_16k(path)
    runs = speech_runs(samples)
    words, per_run = [], []
    for i, (audio, t0, reason) in enumerate(runs):
        ws, reply = speech.transcribe_pcm(audio, locale)
        for w in ws:
            w.start += t0
            w.end += t0
            w.seg = i
        words += ws
        per_run.append({'index': i, 'start': round(t0, 3),
                        'seconds': round(len(audio) / SR, 3),
                        'closed_by': reason, 'words': len(ws)})
    return words, {'runs': per_run, 'n_runs': len(runs),
                   'speech_seconds': round(sum(r['seconds'] for r in per_run), 3),
                   'audio_seconds': round(len(samples) / SR, 3)}
