"""The live path: capture, gate, transcribe, measure and attribute as speech
arrives, instead of after a recording has been saved.

The batch path in prosody_core.measure sees a whole file and can take its
references from it: the pitch range from every voiced frame, the silence
threshold from the file's own energy histogram, the suspect cut-off from the
median word loudness. None of that exists two seconds into a conversation. So
the live path keeps a SessionState that carries those references forward
across segments, and each segment is measured against what the session has
heard so far rather than against itself.

The processing unit is one run of speech, opened and closed by silero-vad and
capped in length. See SEGMENT NOTES below for why that and not a rolling
window.

This does capture and measurement only. It does not decide when a person has
finished talking, it does not respond and it does not speak. The VAD closing a
segment after 300 ms of quiet is an acoustic fact about the waveform, not a
claim that anyone has finished a thought; that judgement is a separate piece
of work and nothing here should be mistaken for it.

usage:
  live.py                       capture from the microphone until interrupted
  live.py --file <wav>          replay a file through the same path
  live.py --file <wav> --realtime   ...at the speed it was recorded
  live.py --locale kn_IN        transcribe this session as Kannada
"""

import json
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import baseline as bl
import markers as mk
import normalize as nz
import prosody_core as pc
import score as sc

SR = 16000
FRAME = 512                  # silero-vad wants exactly this at 16 kHz, 32 ms

# --- the gate ---------------------------------------------------------------
VAD_THRESHOLD = 0.5          # silero's own speech probability
OPEN_FRAMES = 2              # consecutive speech frames to open a segment
CLOSE_S = 0.30               # quiet needed to close one
PREROLL_S = 0.25             # audio kept from before the gate opened, so the
                             # first consonant is not clipped off
MAX_SEGMENT_S = 12.0         # hard cap, so one long turn is not one long wait
MIN_SEGMENT_S = 0.20         # shorter than this is a click, not speech

# --- what the session carries forward ---------------------------------------
PITCH_CLOUD_MAX = 40000      # probe frames kept for the derived pitch range
WORD_HISTORY_MAX = 2000      # recent words, for the rolling reference
LEVEL_EWMA = 0.05            # how fast the noise and speech levels track

RECORDS_PATH = Path(__file__).resolve().parent / "live_records.jsonl"

# SEGMENT NOTES
# -------------
# The unit is a run of speech, not a rolling window that gets revised.
#
# A rolling window gives constant latency and would let a word appear sooner,
# but every measurement here is defined over a complete word and most are
# defined against the utterance around it: a pitch median over half a vowel, a
# duration measured to a boundary that has not happened yet, a sentence
# position in a sentence still being spoken. Revising those means emitting a
# weight and then retracting it, and the record is the primary stored form, so
# a retraction is a write. Emitting once, late, is cheaper than emitting early
# and correcting.
#
# The cost is that latency is bounded by segment length rather than by
# processing. MAX_SEGMENT_S caps it: a segment that reaches the cap is closed
# at the quietest frame in its last second, so a monologue arrives in pieces
# rather than at the end.


class SessionState:
    """Everything a segment needs that a segment cannot supply itself."""

    def __init__(self, speaker='owner', locale=None):
        self.t0 = time.time()
        self.audio_s = 0.0
        self.noise_db = None
        self.speech_db = None
        self.pitch_cloud = deque(maxlen=PITCH_CLOUD_MAX)
        self.p90 = deque(maxlen=WORD_HISTORY_MAX)
        self.words = deque(maxlen=WORD_HISTORY_MAX)
        self.n_segments = 0
        self.n_words = 0
        self.store = bl.load()
        self.stored_ok = bool(self.store.get(speaker)) and bool(
            self.store[speaker]['decl']['n_sentences']
            or self.store[speaker]['pitch']['n'])
        self.ref = bl.summary(self.store[speaker]) if self.stored_ok else None
        self.speaker = speaker
        # Fixed for the session. Nothing switches locale mid-run: the
        # recogniser takes one per request and the measurement layer's
        # references are per language, so changing it halfway would silently
        # mix two of them.
        self.locale = locale or pc.TRANSCRIBE_LOCALE
        self.people = None       # loaded lazily by recognition
        self.timings = []

    # -- levels ------------------------------------------------------------
    def observe_levels(self, db, speaking):
        """Track the noise floor and the speech level from frames the gate has
        already classified, instead of running a two-means split on a segment
        that may contain no silence at all."""
        if not len(db):
            return
        m = float(np.median(db))
        if speaking:
            self.speech_db = m if self.speech_db is None else \
                (1 - LEVEL_EWMA) * self.speech_db + LEVEL_EWMA * m
        else:
            self.noise_db = m if self.noise_db is None else \
                (1 - LEVEL_EWMA) * self.noise_db + LEVEL_EWMA * m

    def silence_threshold(self):
        if self.speech_db is None and self.noise_db is None:
            return None
        sp = self.speech_db if self.speech_db is not None else self.noise_db + 20
        no = self.noise_db if self.noise_db is not None else sp - 25
        return max(sp - pc.SILENCE_DROP_DB, no + pc.NOISE_MARGIN_DB)

    def pitch_range(self):
        """The session's derived range, falling back to the stored baseline's
        own spread, and only then to the fixed range."""
        v = np.array(self.pitch_cloud)
        lo, hi, src = pc.derive_pitch_range(v)
        if src == 'derived':
            return lo, hi, 'session'
        if self.ref is not None and not np.isnan(self.ref.get('pitch_sd', np.nan)):
            mu, sd = self.ref['pitch_mu'], self.ref['pitch_sd']
            return (max(mu - 4 * sd, pc.RANGE_ABS_FLOOR_HZ),
                    min(mu + 4 * sd, pc.RANGE_ABS_CEILING_HZ), 'baseline')
        return lo, hi, 'fallback'

    def p90_median(self):
        return float(np.median(self.p90)) if len(self.p90) >= 8 else None

    def session_loudness(self):
        v = [r['int_p90'] for r in self.words
             if not r['suspect'] and not np.isnan(r['int_p90'])]
        return float(np.median(v)) if len(v) >= 8 else float('nan')


# --- the gate ---------------------------------------------------------------
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


# --- measurement on a segment ----------------------------------------------
def asr():
    """The transcription bridge, started once and kept warm.

    The live path holds each segment as an array, so it goes through
    speech.transcribe_pcm rather than writing a temp file: a file per segment
    would add a write, a read and a delete to the one path whose whole point
    is latency, and would leave litter behind on every crash.
    """
    import speech
    speech.daemon()
    return speech


def measure_segment(audio, t0, session, gap_before):
    """prosody_core.measure, but over a few seconds and against the session.

    Every reference the batch path takes from the whole file is taken from
    SessionState here instead. What each substitution is and why is in
    question 4 of the report.
    """
    import parselmouth
    stamps = {}
    m0 = time.perf_counter()

    import speech
    words, _ = speech.transcribe_pcm(audio, session.locale)
    stamps['transcribe'] = time.perf_counter() - m0
    if not words:
        return [], stamps, None

    m1 = time.perf_counter()
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    total = snd.get_total_duration()
    env_t, env_db = pc.energy_envelope(snd)
    thr = session.silence_threshold()
    if thr is None:
        sp, no = pc.split_speech_noise_db(env_db)
        thr, _, _ = pc.silence_threshold_db(sp, no)
    silences = pc.find_silences(env_t, env_db, thr)

    intensity = snd.to_intensity()
    int_t, int_db = intensity.xs(), intensity.values[0]

    p_floor, p_ceiling, range_src = session.pitch_range()
    pitch = snd.to_pitch(pitch_floor=p_floor, pitch_ceiling=p_ceiling)
    pitch_t, pitch_hz = pitch.xs(), pitch.selected_array['frequency']

    rows = pc.build_rows(words, silences, pitch_t, pitch_hz, int_t, int_db, total)
    pc.flag_suspect(rows, p90_median=session.p90_median())

    # feed the session before the segment is scored, so the next segment is
    # measured against a range and a median that include this one
    probe = snd.to_pitch(pitch_floor=pc.PROBE_FLOOR_HZ,
                         pitch_ceiling=pc.PROBE_CEILING_HZ)
    ph = probe.selected_array['frequency']
    session.pitch_cloud.extend(ph[(ph > 0) & ~np.isnan(ph)].tolist())
    session.p90.extend(r['int_p90'] for r in rows if not np.isnan(r['int_p90']))

    # absolute time, and the real gap before the first word: the segment's own
    # gap_before would be zero because the segment starts at its first sound
    for i, r in enumerate(rows):
        for k in ('start', 'end', 'orig_start', 'orig_end'):
            r[k] += t0
    rows[0]['gap_before'] = gap_before
    for r in rows:
        r['speaker'] = 1
        r['speaker_source'] = 'live, single stream'

    pc.assign_sentences(rows)
    for s in set(r['sentence'] for r in rows):
        members = [r for r in rows if r['sentence'] == s]
        n = len(members)
        for j, r in enumerate(members):
            r['sent_pos'] = j / (n - 1) if n > 1 else 0.0

    stamps['measure'] = time.perf_counter() - m1
    return rows, stamps, {'pitch_range': (p_floor, p_ceiling, range_src),
                          'silence_threshold': thr, 'n_silences': len(silences),
                          'total': total}


def score_segment(rows, session, info):
    """Normalise, score and mark, against the session rather than the file."""
    t = time.perf_counter()
    ref = session.ref if session.stored_ok else nz.per_file_reference(
        list(session.words) + rows)
    stored = session.stored_ok

    # the rolling reference needs the words before this segment, so it is run
    # over recent history with the new rows appended. Only the history the
    # rolling window can actually see is included: apply_rolling is linear in
    # the list it is handed and is called once per word, so handing it the
    # whole session made scoring quadratic in session length, 3 ms at the
    # start of a ten-minute session and 63 ms at the end. Cutting at the
    # window length bounds it and changes no result, since everything older
    # was outside the window anyway.
    horizon = rows[0]['start'] - bl.SHORT_TERM_WINDOW_S
    context = [r for r in session.words if r['end'] >= horizon] + rows
    nz.apply_baselines(context, ref)
    nz.apply_rolling(context, ref, stored)
    nz.apply_defaults(context)
    nz.apply_stored_decline(context, ref)
    nz.flag_questions(rows, ref, stored)
    sc.apply_duration_position(context, ref)
    sc.apply_pause(context, ref, rows[-1]['end'],
                   only={i for i in range(len(context) - len(rows), len(context))})
    sc.apply_weight(rows)
    for r in rows:
        r['scored'] = True
    return time.perf_counter() - t, ref


def recognise_segment(audio, rows, session):
    """One fingerprint per utterance, matched against people already known."""
    import recognize as rc
    import diarize as dz
    t = time.perf_counter()
    if session.people is None:
        session.people = rc.load()
    spans = [(0.0, len(audio) / SR)]
    t_fp = time.perf_counter()
    emb = dz.embed(audio, SR, spans)
    t_fp = time.perf_counter() - t_fp
    res = rc.match(session.people, emb[0]) if len(emb) else {
        'decision': 'unknown', 'reason': 'no audio', 'name': None,
        'pid': None, 'distance': None, 'confidence': 0.0}
    for r in rows:
        r['person'] = res.get('name')
        r['person_id'] = res.get('pid')
        r['person_decision'] = res['decision']
        r['person_distance'] = res.get('distance')
        r['person_confidence'] = res.get('confidence', 0.0)
    return time.perf_counter() - t, t_fp, res


def handle_segment(audio, t0, session, gap_before, out=sys.stdout):
    """One segment, all the way from audio to a rendered line and a record."""
    wall = time.perf_counter()
    rows, stamps, info = measure_segment(audio, t0, session, gap_before)
    if not rows:
        return None
    t_score, ref = score_segment(rows, session, info)
    t_rec, t_fp, ident = recognise_segment(audio, rows, session)

    t_render = time.perf_counter()
    us = mk.utterances(rows, {'total': rows[-1]['end'], 'n_speakers': 1,
                              'distance': {}}, ref, rows[-1]['end'])
    lines = [mk.render_record(u) for u in us]
    t_render = time.perf_counter() - t_render

    session.words.extend(rows)
    session.n_segments += 1
    session.n_words += len(rows)
    total = time.perf_counter() - wall
    timing = {'transcribe': stamps['transcribe'], 'measure': stamps['measure'],
              'score': t_score, 'fingerprint': t_fp,
              'match': t_rec - t_fp, 'render': t_render, 'total': total,
              'audio_s': len(audio) / SR, 'words': len(rows)}
    session.timings.append(timing)

    with RECORDS_PATH.open('a') as f:
        for u in us:
            u['segment'] = session.n_segments
            u['segment_start'] = round(t0, 3)
            u['timing'] = {k: round(v, 4) for k, v in timing.items()}
            f.write(json.dumps(u, separators=(',', ':')) + "\n")

    who = ident.get('name') or 'unknown'
    print(f"[{t0:7.2f}s +{total*1000:5.0f}ms  {len(audio)/SR:4.1f}s audio, "
          f"{len(rows):3d} words, {who}]", file=out)
    for line in lines:
        print(line, file=out)
    print(file=out)
    return timing


# --- the loops --------------------------------------------------------------
STATUS_EVERY_S = 2.0


class Meters:
    """What each stage of the live path has actually seen. A run that produces
    nothing has to be able to say which stage saw nothing."""

    def __init__(self):
        self.frames = 0
        self.level_sum = 0.0
        self.level_max = -200.0
        self.level_min = 200.0
        self.vad_max = 0.0
        self.vad_over = 0          # frames the detector called speech
        self.opens = 0
        self.closes = 0
        self.too_short = 0
        self.segments_no_words = 0
        self.segments_with_words = 0
        self.gate_time = 0.0

    def diagnose(self, capture=None):
        """One line naming the first stage that saw nothing."""
        if self.frames == 0:
            return ("no audio ever reached the gate. The device callback "
                    + (f"fired {capture.blocks} time(s)" if capture else "")
                    + ". Check the input device with --devices and pass "
                      "--device N.")
        mean = self.level_sum / self.frames
        if self.level_max < -70.0:
            return (f"audio arrived ({self.frames} frames) but it was silent: "
                    f"peak level {self.level_max:.0f} dB. The device is "
                    f"delivering zeros, which on macOS usually means "
                    f"microphone permission or a Bluetooth input that is not "
                    f"really open.")
        if self.vad_over == 0:
            return (f"audio arrived at {mean:.0f} dB mean, {self.level_max:.0f} dB "
                    f"peak, but the voice detector never called any of it "
                    f"speech: highest probability {self.vad_max:.3f} against a "
                    f"threshold of {VAD_THRESHOLD}. Either nothing was said or "
                    f"the level is too low for the detector.")
        if self.opens == 0:
            return (f"the detector saw speech in {self.vad_over} frame(s) but "
                    f"never {OPEN_FRAMES} in a row, so no segment opened.")
        if self.closes == 0:
            return (f"{self.opens} segment(s) opened but none closed. Speech "
                    f"never stopped for {CLOSE_S}s and the {MAX_SEGMENT_S}s cap "
                    f"was not reached before the run ended.")
        if self.segments_with_words == 0:
            return (f"{self.closes} segment(s) captured but the recogniser "
                    f"found no words in any of them "
                    f"({self.too_short} rejected as too short).")
        return None


def run_stream(session, source, realtime=False, out=sys.stdout, meters=None,
               capture=None, quiet=False):
    """Drive the gate from a frame source and process what it closes."""
    gate = Gate()
    m = meters or Meters()
    last_end = 0.0
    next_status = STATUS_EVERY_S
    for frame, t in source:
        f0 = time.perf_counter()
        closed, prob = gate.push(frame, t)
        m.gate_time += time.perf_counter() - f0
        m.frames += 1
        session.audio_s = t + FRAME / SR

        db = 20.0 * np.log10(max(float(np.sqrt(
            (frame.astype(np.float64) ** 2).mean())), 1e-12))
        m.level_sum += db
        m.level_max = max(m.level_max, db)
        m.level_min = min(m.level_min, db)
        m.vad_max = max(m.vad_max, prob)
        if prob >= VAD_THRESHOLD:
            m.vad_over += 1
        if gate.open and gate.just_opened:
            m.opens += 1
        session.observe_levels(np.array([db]), gate.open)

        if not quiet and t >= next_status:
            next_status = t + STATUS_EVERY_S
            extra = ''
            if capture is not None:
                extra = (f", {capture.blocks} device block(s)"
                         + (f", {capture.dropped} frame(s) dropped"
                            if capture.dropped else "")
                         + (f", status {list(capture.status)}"
                            if capture.status else ""))
            print(f"  [{t:6.1f}s  level {db:6.1f} dB  peak {m.level_max:6.1f}  "
                  f"vad {prob:.3f} (max {m.vad_max:.3f})  "
                  f"speech frames {m.vad_over}  segments {m.closes}{extra}]",
                  file=out, flush=True)

        if closed is not None:
            m.closes += 1
            audio, start, reason = closed
            r = handle_segment(audio, start, session,
                               max(0.0, start - last_end), out)
            if r is None:
                m.segments_no_words += 1
            else:
                m.segments_with_words += 1
            last_end = start + len(audio) / SR
        elif gate.dropped_short:
            m.too_short += 1
            gate.dropped_short = False
    return m


def frames_from_file(path, realtime=False):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    n = len(x) // FRAME
    for i in range(n):
        if realtime:
            time.sleep(FRAME / SR)
        yield x[i * FRAME:(i + 1) * FRAME], i * FRAME / SR


QUEUE_MAX = 200              # ~6.4 s of audio at 32 ms a frame


class Capture:
    """The microphone side, with everything about it observable.

    The previous version put frames on an unbounded queue in the callback and
    told nobody anything. When it produced nothing there was no way to tell
    whether audio had arrived, whether it was silent, or whether the detector
    had simply never fired, because none of those three was counted.
    """

    def __init__(self, device=None):
        self.device = device
        self.q = queue.Queue(maxsize=QUEUE_MAX)
        self.blocks = 0          # callback invocations
        self.samples = 0
        self.dropped = 0         # frames thrown away because the queue was full
        self.status = {}         # PortAudio status flags, counted
        self.peak = 0.0
        self.carry = np.zeros(0, dtype='float32')

    def _callback(self, indata, frames, tinfo, status):
        if status:
            self.status[str(status)] = self.status.get(str(status), 0) + 1
        self.blocks += 1
        self.samples += len(indata)
        x = indata[:, 0] if indata.ndim > 1 else indata
        p = float(np.abs(x).max()) if len(x) else 0.0
        if p > self.peak:
            self.peak = p
        # re-block to exactly FRAME samples. PortAudio is allowed to hand over
        # a different block size than the one asked for, and silero refuses
        # anything that is not 512 samples at 16 kHz, so this cannot be left
        # to chance.
        self.carry = np.concatenate([self.carry, x.astype('float32')])
        while len(self.carry) >= FRAME:
            frame, self.carry = self.carry[:FRAME], self.carry[FRAME:]
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                # Drop the oldest rather than blocking. Blocking here blocks
                # the audio device callback, which on CoreAudio means the
                # stream glitches and can be torn down entirely, losing
                # everything after it. Dropping loses a frame and keeps the
                # stream alive, and the count is reported.
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    self.dropped += 1

    def frames(self):
        import sounddevice as sd
        info = sd.query_devices(self.device if self.device is not None else None,
                                kind='input')
        print(f"input device: [{sd.default.device[0] if self.device is None else self.device}] "
              f"{info['name']}, device default {info['default_samplerate']:.0f} Hz, "
              f"opening at {SR} Hz mono float32, {FRAME}-sample blocks")
        try:
            sd.check_input_settings(device=self.device, samplerate=SR,
                                    channels=1, dtype='float32')
        except Exception as e:
            print(f"WARNING: the device rejected these settings: {e}")
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=FRAME, callback=self._callback,
                            device=self.device):
            i = 0
            while True:
                try:
                    frame = self.q.get(timeout=1.0)
                except queue.Empty:
                    if self.blocks == 0:
                        print("WARNING: no audio has arrived from the device yet")
                    continue
                yield frame, i * FRAME / SR
                i += 1


def frames_from_mic(device=None):
    """Kept for callers that only want frames; Capture is the observable one."""
    return Capture(device).frames()


def bounded(src, seconds):
    """Stop a frame source after a fixed amount of audio, so a check run ends
    on its own instead of needing ctrl-c."""
    for frame, t in src:
        yield frame, t
        if t >= seconds:
            return


def list_devices():
    import sounddevice as sd
    print("input devices (pass the number to --device):")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] < 1:
            continue
        mark = ' <- system default' if i == default_in else ''
        ok = 'ok'
        try:
            sd.check_input_settings(device=i, samplerate=SR, channels=1,
                                    dtype='float32')
        except Exception as e:
            ok = f'rejects {SR} Hz mono float32: {type(e).__name__}'
        print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch, default "
              f"{d['default_samplerate']:.0f} Hz)  {ok}{mark}")
    print("Bluetooth headsets are the usual cause of a silent capture: macOS "
          "will list them as an input and then hand over zeros or fail inside "
          "CoreAudio without raising. If in doubt use the built-in microphone.")


def main():
    args = sys.argv[1:]
    if '--devices' in args:
        list_devices()
        return
    realtime = '--realtime' in args
    locale = args[args.index('--locale') + 1] if '--locale' in args else None
    session = SessionState(locale=locale)

    # load and warm every model before anything is timed. In a real session
    # this happens once at start-up; folding it into the first utterance's
    # latency would report a number no later utterance ever pays.
    import diarize as dz
    t_warm = time.perf_counter()
    warm = np.zeros(SR, dtype='float32')
    asr().transcribe_pcm(warm, session.locale)
    dz.embed(warm, SR, [(0.0, 1.0)])
    Gate()
    print(f"models loaded and warmed in {time.perf_counter()-t_warm:.1f}s")
    capture = None
    if '--file' in args:
        path = args[args.index('--file') + 1]
        src = frames_from_file(path, realtime)
        print(f"replaying {path} through the live path"
              + (" at recording speed" if realtime else " as fast as it runs"))
    else:
        dev = None
        if '--device' in args:
            d = args[args.index('--device') + 1]
            dev = int(d) if d.isdigit() else d
        capture = Capture(dev)
        src = capture.frames()
        if '--seconds' in args:
            src = bounded(src, float(args[args.index('--seconds') + 1]))
            print(f"listening for {args[args.index('--seconds')+1]}s. "
                  f"Talk. A status line prints every {STATUS_EVERY_S:.0f}s.")
        else:
            print(f"listening. ctrl-c to stop. A status line prints every "
                  f"{STATUS_EVERY_S:.0f}s so you can see audio arriving.")
    print(f"locale: {session.locale}")

    meters = Meters()
    t_wall = time.perf_counter()
    try:
        run_stream(session, src, realtime, meters=meters, capture=capture,
                   quiet=('--file' in args and '--verbose' not in args))
    except KeyboardInterrupt:
        print("\nstopped.")
    wall = time.perf_counter() - t_wall

    print()
    print(f"frames into the gate: {meters.frames}"
          + (f"   device blocks: {capture.blocks}, samples {capture.samples}, "
             f"peak {capture.peak:.4f}, dropped {capture.dropped}"
             if capture else ""))
    if capture and capture.status:
        print(f"PortAudio status flags: {capture.status}")
    if meters.frames:
        print(f"level: mean {meters.level_sum/meters.frames:.1f} dB, "
              f"min {meters.level_min:.1f}, max {meters.level_max:.1f}")
        print(f"voice detector: highest probability {meters.vad_max:.3f}, "
              f"{meters.vad_over} frame(s) at or over {VAD_THRESHOLD}")
        print(f"segments: {meters.opens} opened, {meters.closes} closed, "
              f"{meters.too_short} discarded as too short, "
              f"{meters.segments_with_words} produced words, "
              f"{meters.segments_no_words} produced none")

    problem = meters.diagnose(capture)
    if problem:
        print(f"\nNOTHING WAS PRODUCED, and this is why: {problem}")
        return

    T = session.timings
    print(f"\n{session.n_segments} segment(s), {session.n_words} word(s) over "
          f"{session.audio_s:.1f}s of audio in {wall:.1f}s wall")
    if meters.frames:
        print(f"gate: {meters.frames} frames, {meters.gate_time:.2f}s total, "
              f"{1000*meters.gate_time/meters.frames:.3f} ms per 32 ms frame "
              f"= {100*meters.gate_time/(meters.frames*FRAME/SR):.2f}% of one core")
    for k in ('transcribe', 'measure', 'score', 'fingerprint', 'match',
              'render', 'total'):
        v = np.array([t[k] for t in T])
        print(f"  {k:<12} mean {1000*v.mean():7.1f} ms  median "
              f"{1000*np.median(v):7.1f}  p90 {1000*np.percentile(v,90):7.1f}  "
              f"max {1000*v.max():7.1f}")
    a = np.array([t['audio_s'] for t in T])
    tot = np.array([t['total'] for t in T])
    print(f"  segment audio: mean {a.mean():.2f}s median {np.median(a):.2f}s "
          f"max {a.max():.2f}s")
    print(f"  processing per second of speech: {tot.sum()/a.sum():.3f}x "
          f"({100*tot.sum()/a.sum():.1f}% of one core while talking)")
    print(f"  records appended to {RECORDS_PATH.name}")


if __name__ == "__main__":
    main()
