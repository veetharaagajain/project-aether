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
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import archive as ar
import baseline as bl
import channel as ch
import incognito as inc
import markers as mk
import normalize as nz
import prosody_core as pc
import score as sc
import singleton

# The gate and its constants live in segment.py, because the batch path now
# cuts speech the same way and the two must not drift apart. They are bound
# here under their original names so this file reads as it always did.
from segment import (SR, FRAME, VAD_THRESHOLD, OPEN_FRAMES, CLOSE_S,
                     PREROLL_S, MAX_SEGMENT_S, MIN_SEGMENT_S, Gate)

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
        # When the microphone actually started, which is not when this object
        # was built: the models warm up in between, and on this machine that is
        # several seconds. Utterance times are counted in frames from the first
        # block, so converting them with t0 puts every one of them early by the
        # whole warm-up -- which is why the self-hearing suppression stopped
        # firing and its own replies went back into the store as an unknown
        # voice. Set by run_stream from the capture. Third time this project has
        # measured from construction instead of from when the thing started;
        # see CLAUDE.md.
        self.audio_epoch = None
        self.audio_s = 0.0
        self.noise_db = None
        self.speech_db = None
        self.pitch_cloud = deque(maxlen=PITCH_CLOUD_MAX)
        self.p90 = deque(maxlen=WORD_HISTORY_MAX)
        self.words = deque(maxlen=WORD_HISTORY_MAX)
        self.n_segments = 0
        self.n_words = 0
        # bl.load refuses a baseline built under another pipeline; this adds
        # the threshold half, so the live path cannot mark words with levels
        # cut against a baseline that has since been rebuilt either.
        self.store = bl.load()
        if self.store.get(speaker):
            import provenance as pv
            pv.check_levels(mk.LEVEL_PROVENANCE, self.store[speaker], speaker,
                            str(bl.BASELINE_PATH))
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
        import provenance as pv
        self.config_digest = pv.digest(pv.current())
        self.id = time.strftime('%Y%m%dT%H%M%S')
        self._db = None
        # read once at session start and cached: a pause taking effect only at
        # the next session would be useless, so live_loop rechecks it, but a
        # per-utterance database read is not what this is for.
        self.capturing = True
        self.silent = deque(maxlen=200)     # what it heard and did not answer
        self.self_heard = 0                 # utterances that were its own voice
        self.write_failures = 0             # observations the store refused
        # kept current by run_stream from the capture, so the conversion from
        # frame-counted utterance time to wall time can correct for it
        self.clock_drift = 0.0
        self.clock_drift_rate = 0.0
        self.clock_untrusted = 0            # utterances judged under bad drift
        self._recent_plain = deque(maxlen=20)
        self._being_secret = None
        # the window in which follow-ups still count as addressed, opened by
        # the name and extended by each answer
        self.attention = None

    @property
    def attention(self):
        if self._attention is None:
            import addressed as ad
            self._attention = ad.new_attention()
        return self._attention

    @attention.setter
    def attention(self, v):
        self._attention = v

    def recent_lines(self, n=6):
        """The last few utterances as plain text, for the addressed check.

        Kept in the session rather than read back from the store, because the
        question of who someone was talking to is about the conversation, not
        about what happened to be written down -- and it must still work while
        capture is paused.
        """
        return list(self._recent_plain)[-n:]

    def being_secret(self):
        """The being's own credential, created on first use.

        It authenticates like any other caller rather than being waved through:
        the gate's rules are the only thing standing between a question and the
        store, and a caller that skips them because it happens to live in the
        same process is a hole with a comment on it.
        """
        if self._being_secret is None:
            import gate as g
            from pathlib import Path
            p = Path(__file__).resolve().parent / "store" / "being.secret"
            db = self.memory()
            if p.exists() and any(c['caller'] == 'being' for c in g.callers(db)):
                self._being_secret = p.read_text().strip()
            else:
                self._being_secret = g.add_caller(
                    db, 'being', can_read=True, can_write=False,
                    note='the being answering out loud in the room')
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self._being_secret)
                p.chmod(0o600)
        return self._being_secret

    def memory(self):
        if self._db is None:
            import memory as mem
            self._db = mem.open()
        return self._db

    def check_capture(self):
        import incognito as inc
        was = self.capturing
        self.capturing = inc.capturing(self.memory())
        return was, self.capturing

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
    # measured against a range and a median that include this one.
    #
    # The cloud takes only the frames prosody_core.derive_pitch_range says it
    # takes: inside a word that was not flagged suspect, and surviving octave
    # rejection within that word. Handing it every voiced frame in the segment
    # instead put breath, room noise and the gaps between words into the
    # percentiles it brackets the speaker with, and its docstring already
    # named the consequence -- a 572.9 Hz ceiling on turns1 for a voice that
    # lives between 90 and 140. Replayed through this path, turns1 derived a
    # median ceiling of 567.0 Hz and read 21 words above 250 Hz, one of them
    # at 558.4 Hz on a speaker whose baseline is 102.
    probe = snd.to_pitch(pitch_floor=pc.PROBE_FLOOR_HZ,
                         pitch_ceiling=pc.PROBE_CEILING_HZ)
    probe_t = probe.xs()
    ph = probe.selected_array['frequency']
    for r in rows:
        if r['suspect']:
            continue
        v = ph[(probe_t >= r['start']) & (probe_t <= r['end'])]
        v = v[(v > 0) & ~np.isnan(v)]
        if not len(v):
            continue
        v, _, _ = pc.reject_octave_errors(v)
        session.pitch_cloud.extend(v.tolist())
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
    # The gate opened this segment after CLOSE_S of quiet, so the silence
    # before its first word is an utterance boundary by construction.
    # assign_sentences cannot know that: it only ever looks at word i against
    # word i-1, so the first word of a segment comes out with no cue, and
    # baseline.eligible_gaps then reads the whole inter-segment silence as an
    # emphatic pause around that word. That put the first word of 83 of the
    # 186 stored live utterances at the pause cap, against 0 of 109 in the
    # batch corpus, where assign_sentences sees the boundary and excludes it.
    rows[0]['boundary_cue'] = rows[0]['boundary_cue'] or 'segment'
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


def maybe_answer(session, u, out=sys.stdout):
    """Decide whether this was said to us, and if so answer it out loud.

    Runs when a segment closes, like everything else. It does not decide when
    somebody has finished talking -- that is endpointing and a separate problem
    -- and it does not stop speaking if the person starts, which is
    interruption and the next piece.

    The answer goes through the gate rather than round it, using the 'being'
    caller. Nothing about being in the room makes this not a release: the answer
    is audible to whoever else is present, and routing it through gate means it
    is refused if the caller cannot read, and appears in the access log and the
    viewer like every other disclosure.
    """
    import addressed as ad
    import channel as ch
    try:
        decision = ad.decide(u, recent_lines=session.recent_lines(),
                             capturing=session.capturing,
                             db=session.memory(), attention=session.attention)
    except ad.NotAddressed as e:
        session.silent.append({'at': time.time(), 'text': u['plain'],
                               'rung': e.rung, 'detail': e.detail})
        ch.publish({'kind': 'addressed', 'at': time.time(), 'spoke': False,
                    'text': u['plain'], 'rung': e.rung, 'detail': e.detail})
        return None
    except Exception as e:                                   # noqa: BLE001
        # anything unexpected in the deciding layer means silence, not speech
        print(f"  [addressed check failed, staying quiet: {e}]", file=out)
        return None

    # What is left once the name is taken off decides what this even is. The
    # name settles whether it was addressed; it is not the thing being asked.
    # Passing the whole utterance on made "Jarvis?" a question about Jarvis.
    query, kind = ad.strip_wake(u['plain'], ad.wake_word(session.memory()))
    if kind in ('bare', 'topicless'):
        spoken = ACKNOWLEDGE if kind == 'bare' else None
        if spoken:
            import voice
            voice.say(spoken, db=session.memory())
        print(f"  [addressed with no question ({kind}): "
              f"{'acknowledged' if spoken else 'stayed quiet'}]", file=out)
        ch.publish({'kind': 'addressed', 'at': time.time(),
                    'spoke': bool(spoken), 'text': u['plain'],
                    'answer': spoken, 'rung': f'no question, {kind}',
                    'detail': f'remainder {query!r}'})
        return spoken

    t = time.perf_counter()
    try:
        import gate as g
        r = g.answer_or_search(session.memory(), 'being', session.being_secret(),
                               query, session=session.id)
    except Exception as e:                                   # noqa: BLE001
        print(f"  [could not answer: {e}]", file=out)
        return None
    spoken = r.get('answer')
    if not spoken:
        # Compose from what the search released rather than speaking the
        # top-ranked fragment. Reading the person's own sentence back at them
        # answers nothing -- they said it.
        import recall as rc
        hits = [h for h in (r.get('released') or []) if h.get('role') == 'answer'] \
            or (r.get('released') or [])
        try:
            spoken = rc.compose_from(query, rc.format_lines(hits, expand=False))['answer']
        except Exception as e:                               # noqa: BLE001
            # Not silence. Being spoken to and saying nothing is
            # indistinguishable from being ignored, and it happened within
            # minutes of this running. The refusal is composed from the same
            # fragments the answer failed to come from, so it can say what is
            # missing and what was near it, and it goes through the same
            # grounding checks because it is still a claim about what was
            # heard. See recall.compose_not_found.
            why = str(e)
            spoken = None
            kind = None
            # ASK THE LOCAL JUDGEMENT FIRST, then reason only if it says this
            # was never a recall question.
            #
            # The previous order reasoned before refusing, on the grounds that
            # retrieved-and-unhelpful is not the same as absent. That is true
            # and it was still the wrong order: "what did I have for breakfast"
            # is a recall question with an honest local answer -- "Nothing
            # about breakfast in what's been said" -- and reasoning about it
            # sent the retrieved fragments to a paid hosted model and got back
            # "There's no information here about what you ate", which is worse,
            # slower, costs money, and puts speech over the network on every
            # ordinary miss.
            #
            # compose_not_found already distinguishes the two: it raises
            # NotRecallable exactly when the question needed knowledge or
            # working out rather than recall. That is the signal to reason on,
            # and it is free.
            try:
                nf = rc.compose_not_found(query, rc.format_lines(hits, expand=False))
                spoken, kind = nf['answer'], nf['kind']
                print(f"  [nothing answered it ({why}); saying so: "
                      f"{kind}, nearby={nf['nearby']}]", file=out)
            except rc.NotRecallable as nr:
                # never about what anybody said, so this is what reasoning is
                # for. The composed line is the fallback if reasoning cannot
                # happen -- no provider, gate refused, over budget.
                try:
                    rr = g.reason_about(session.memory(), 'being', query,
                                        context=rc.format_lines(hits, expand=False))
                    if rr.get('answer'):
                        spoken, kind = rr['answer'], 'reasoned'
                        print(f"  [not a recall question ({nr.topic}); "
                              f"reasoned by {rr['from']} over {len(hits)} "
                              f"fragment(s)]", file=out)
                except Exception as e_r:                     # noqa: BLE001
                    print(f"  [could not reason: {e_r}]", file=out)
                if not spoken:
                    spoken, kind = nr.reply, 'not_recall'
                    print(f"  [not a recall question ({nr.topic}); saying so]",
                          file=out)
            except Exception as e2:                          # noqa: BLE001
                print(f"  [found {len(hits)} fragment(s), could not compose "
                      f"an answer ({why}) or a refusal ({e2}), "
                      f"staying quiet]", file=out)
    if not spoken:
        print(f"  [addressed, but nothing to say]", file=out)
        ch.publish({'kind': 'addressed', 'at': time.time(), 'spoke': False,
                    'text': u['plain'], 'rung': 'nothing to say', 'detail': ''})
        return None

    import voice
    voice.say(spoken, db=session.memory())
    took = time.perf_counter() - t
    print(f"  [ANSWERED aloud in {decision['seconds']+took:.1f}s] {spoken}", file=out)
    ch.publish({'kind': 'addressed', 'at': time.time(), 'spoke': True,
                'text': u['plain'], 'answer': spoken,
                'rung': decision.get('rung'), 'detail': decision.get('detail'),
                'presence': decision.get('presence'),
                'direct': r.get('answered_directly'),
                'seconds': round(decision['seconds'] + took, 2)})
    return spoken


# How far around its own speech to distrust the microphone. Both are the cost
# of the fix, in opposite directions: too wide and a person talking over it is
# lost, too narrow and the tail of its own sentence gets written down.
#
# SELF_LEAD covers the gate's own PREROLL_S, since a segment that opens just
# after it starts talking already contains a quarter second from before.
# SELF_TAIL covers room reverberation and the fact that playback ends at the
# last sample, not at the last thing audible in the room.
#
# SELF_OVERLAP_DROP is why this is a fraction and not a boolean. Dropping any
# segment that touches a speaking window would throw away a whole utterance
# because its first 50 ms clipped the end of a reply. A segment is treated as
# the machine's own only when most of it lies inside a window.
# Said when it is called by name and nothing else. A fixed string, and
# deliberately: this is not a claim about the record, it is the sound of
# turning your head, and varying it would be affectation. A topicless
# question gets silence instead, because "yes?" in answer to "what do you
# think I should do next" is worse than nothing -- it would read as evasion
# rather than as attention.
ACKNOWLEDGE = "Yes?"

SELF_LEAD_S = 0.30
SELF_TAIL_S = 0.60
SELF_OVERLAP_DROP = 0.5


def self_heard(session, us, t0):
    """Which of these utterances are the machine hearing itself.

    Returns {index: (overlap fraction, what it was saying)} for the ones that
    should not be stored.
    """
    import voice
    # The wall clock the frame counter is anchored to, plus the measured
    # offset between the audio clock and the wall clock. Falls back to t0 only
    # for a file replay, which has no microphone and no speaker to hear.
    base = session.audio_epoch or session.t0
    drift = session.clock_drift
    # the residual after correcting by the current offset: how far the offset
    # itself moves across this segment. Absolute drift is corrected and does
    # not matter; this is what is left.
    span = max((us[-1]['end'] - us[0]['start']) if us else 0.0, 0.0)
    trusted = abs(session.clock_drift_rate) * span <= SELF_TAIL_S
    out = {}
    for i, u in enumerate(us):
        a, b = base + u['start'] + drift, base + u['end'] + drift
        if b <= a:
            continue
        ov, which = voice.spoke_during(a, b, lead=SELF_LEAD_S, tail=SELF_TAIL_S)
        frac = ov / (b - a)
        if frac >= SELF_OVERLAP_DROP:
            out[i] = (round(frac, 3), which)
        elif not trusted and ov > 0:
            # it overlapped but not enough to drop, and the clocks are too far
            # apart for that judgement to mean anything. Said out loud rather
            # than stored quietly as somebody's speech.
            session.clock_untrusted += 1
            print(f"  [clock drift {drift:+.2f}s: cannot tell whether "
                  f"{u['plain'][:40]!r} was its own voice]", file=sys.stderr,
                  flush=True)
    return out


def store_utterances(session, us, t0, blob):
    """Every utterance of this segment, into the memory store.

    One observation per utterance rather than per segment, because an utterance
    is the unit everything upstream already produces and the unit a question
    will be answered with. kind is 'speech' so that sight does not need a
    second table when it arrives.
    """
    import memory as mem
    db = session.memory()
    # session-relative seconds out, epoch seconds in. Everything upstream
    # counts from the start of the session because that is what the measurement
    # needs; the store needs wall time, because forget() works on a window of
    # real minutes and a ULID carries the moment it was minted.
    #
    # Anchored to when the microphone opened, not to when this object was
    # built. Using t0 put every stored timestamp early by the model warm-up --
    # a few seconds, invisible in the viewer, and wrong.
    base = session.audio_epoch or session.t0
    mine = self_heard(session, us, t0)
    for i, u in enumerate(us):
        if i in mine:
            # not stored, but not silent either: the count is published so the
            # viewer can show it and a run can be checked afterwards
            frac, which = mine[i]
            session.self_heard += 1
            ch.publish({'kind': 'self_heard', 'at': time.time(),
                        'text': u['plain'], 'overlap': frac,
                        'while_saying': (which or [''])[0]})
            continue
        try:
            mem.add_observation(
                db, 'speech', base + u['start'], base + u['end'],
                text=u['plain'], body=u,
                session=session.id,
                person_id=u.get('person_id'), person=u.get('person'),
                person_decision=u.get('person_decision'),
                speaker=u.get('speaker'),
                audio=(dict(blob, offset=round(u['start'] - t0, 3))
                       if blob else None),
                config_digest=session.config_digest)
        except mem.Busy as e:
            # a dropped observation is bad; a dead capture is worse, and it
            # was the second that actually happened. Counted and published
            # rather than raised, so it cannot be silent either.
            session.write_failures += 1
            print(f"  [STORE BUSY, observation dropped: {e}]", file=sys.stderr,
                  flush=True)
            ch.publish({'kind': 'store_busy', 'at': time.time(),
                        'text': u['plain'], 'detail': str(e)})
    return len(mine)


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

    # Keep the audio. It is stored once per run, whole and before anything has
    # decided who was speaking, and every utterance in the run points at that
    # one blob with its own offset. Measuring a segment and dropping it left
    # nothing to re-transcribe when a better recogniser arrives and nothing
    # for the bottom rung of the memory ladder to read.
    #
    # Unless capture is paused. Everything above this line still ran: the
    # segment was gated, transcribed and measured, so the session keeps its
    # pitch range and its loudness reference and the first minute after
    # resuming is not measured against nothing. What stops here is storage.
    t_arch = time.perf_counter()
    blob = ar.put(audio, SR) if session.capturing else None
    t_arch = time.perf_counter() - t_arch

    session.words.extend(rows)
    session.n_segments += 1
    session.n_words += len(rows)
    total = time.perf_counter() - wall
    timing = {'transcribe': stamps['transcribe'], 'measure': stamps['measure'],
              'score': t_score, 'fingerprint': t_fp,
              'match': t_rec - t_fp, 'render': t_render, 'archive': t_arch,
              'total': total,
              'audio_s': len(audio) / SR, 'words': len(rows)}
    session.timings.append(timing)

    if session.capturing:
        store_utterances(session, us, t0, blob)

    # The loop closes here: the thing that heard the question answers it.
    for u in us:
        maybe_answer(session, u, out)
        session._recent_plain.append(u['plain'])

    for u in us:
        u['segment'] = session.n_segments
        u['segment_start'] = round(t0, 3)
        # how this record finds its own audio: the blob is the whole run,
        # offset is where this utterance starts inside it
        u['audio'] = (dict(blob, offset=round(u['start'] - t0, 3))
                      if blob else None)
        # what the pitch was measured against. 'fallback' means the fixed
        # 60-300 range, which a session only ever uses for its first
        # segment, before the cloud has enough clean frames to derive one
        # and when the speaker has no stored baseline to borrow from. Those
        # words are the ones a voice near either rail is read worst at, and
        # recording the source is what makes them findable later: the audio
        # is now archived, so they can be re-measured rather than trusted.
        u['pitch_range'] = {'floor': round(info['pitch_range'][0], 1),
                            'ceiling': round(info['pitch_range'][1], 1),
                            'source': info['pitch_range'][2]}
        # what this record was measured under. The weights in it are
        # computed once and stored, never recomputed, so a record outlives
        # the configuration that produced it and nothing else on the line
        # would say so. Pooling records across a configuration change is
        # the same mistake the baseline made, one level up.
        u['config_digest'] = session.config_digest
        u['timing'] = {k: round(v, 4) for k, v in timing.items()}

    # The record file is storage, so pause switches it off with everything
    # else. It did not, until the viewer was built: a paused session was still
    # leaving a transcript on disk, which is exactly what pause promises it
    # will not do.
    if session.capturing:
        with RECORDS_PATH.open('a') as f:
            for u in us:
                f.write(json.dumps(u, separators=(',', ':')) + "\n")

    # Published whether or not anything is being stored, and whether or not
    # anything is listening. Watching and recording are independent: a person
    # needs to be able to see that a paused session really is producing nothing.
    #
    # A view payload, not a copy of the record. The page renders words, marks
    # and who was speaking, so that is what goes over the wire; anything wanting
    # the z-scores and the audio pointer reads the stored observation through
    # the viewer's own API. Keeping it small is also what keeps it inside one
    # datagram on a long utterance.
    for u, line in zip(us, lines):
        ch.publish({
            'kind': 'utterance', 'at': time.time(),
            'capturing': session.capturing, 'session': session.id,
            'segment': session.n_segments, 'rendered': line,
            'record': {
                'person': u.get('person'),
                'person_decision': u.get('person_decision'),
                'plain': u['plain'],
                'pause_cap_s': u.get('pause_cap_s'),
                'pitch_range': u.get('pitch_range'),
                'start': u['start'], 'end': u['end'],
                'words': [{'word': w['word'], 'punct': w['punct'],
                           'weight': w['weight'],
                           'gap_before': w['gap_before'],
                           'suspect': w['suspect']} for w in u['words']],
            }})

    who = ident.get('name') or 'unknown'
    print(f"[{t0:7.2f}s +{total*1000:5.0f}ms  {len(audio)/SR:4.1f}s audio, "
          f"{len(rows):3d} words, {who}]", file=out)
    for line in lines:
        print(line, file=out)
    print(file=out)
    return timing


# --- the loops --------------------------------------------------------------
# Every two seconds is right for a person watching a terminal and wrong for a
# file that grows forever: at ~120 bytes a line that is 5 MB a day of "still
# nothing". Under a supervisor the line still has a job -- it is how you tell a
# living service from a wedged one -- so it drops to once a minute rather than
# off, and launchd does not rotate anything.
STATUS_EVERY_S = 2.0 if sys.stdout.isatty() else 60.0


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
        self.segments_dropped = 0
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


# Closed segments waiting to be transcribed, measured and answered. Bounded,
# because an unbounded one would hide exactly the backlog it is there to
# absorb.
SEGMENT_QUEUE_MAX = 12


def run_stream(session, source, realtime=False, out=sys.stdout, meters=None,
               capture=None, quiet=False):
    """Drive the gate from a frame source and process what it closes.

    THE LOCK IS TAKEN HERE, not in main(). It used to be taken by the
    command-line entry point, which meant anything that imported this module
    and called run_stream directly was a second live path with no lock at all
    -- and that is not hypothetical, it is how I ran every test in the last
    several rounds. Two of them captured, measured and published alongside the
    service, which is what put seg 1 and seg 8 on the viewer at the same
    moment, one on a fresh session reference and one on the baseline.

    A lock on the entry point protects the entry point. A lock here protects
    the thing that actually captures, which is what needed protecting.
    """
    if not singleton.held('live'):
        take_lock()
    gate = Gate()
    m = meters or Meters()
    last_end = 0.0
    next_status = STATUS_EVERY_S

    # THE PIPELINE RUNS OFF THIS THREAD. It used to run on it: a closed segment
    # was transcribed, measured, scored, fingerprinted and answered inline,
    # while frames piled up in a queue holding 6.4 seconds. Anything slower
    # than that dropped audio, and dropped audio is missed speech. The first
    # time it fired was 116 frames during start-up, with the Kokoro warm-up
    # competing for the machine -- but the warm-up only made it visible. A long
    # segment, a slow transcription or a reply that has to reason would all do
    # the same, and none of those is a bug to be fixed once.
    #
    # So this thread now does the gate and nothing else, which is 0.1 ms a
    # frame. One worker, not a pool, because segments must be processed in the
    # order they were spoken.
    segq = queue.Queue(maxsize=SEGMENT_QUEUE_MAX)

    def worker():
        while True:
            item = segq.get()
            if item is None:
                segq.task_done()
                return
            audio, start, gap = item
            try:
                r = handle_segment(audio, start, session, gap, out)
                if r is None:
                    m.segments_no_words += 1
                else:
                    m.segments_with_words += 1
            except Exception as e:                            # noqa: BLE001
                print(f"  [segment failed: {type(e).__name__}: {e}]",
                      file=sys.stderr, flush=True)
            finally:
                segq.task_done()

    pump = threading.Thread(target=worker, daemon=True)
    pump.start()
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
        if capture is not None:
            session.clock_drift = capture.drift
            session.clock_drift_rate = capture.drift_rate
            if session.audio_epoch is None and capture.opened_wall is not None:
                session.audio_epoch = capture.opened_wall
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
                         + f", clock drift {capture.drift:+.2f}s "
                           f"({100*capture.drift_rate:+.1f}%)"
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
            # checked per segment, not per session: pausing has to take effect
            # in the conversation it was asked for, and a segment is the
            # smallest thing that can be wholly kept or wholly dropped.
            was, now_on = session.check_capture()
            if was != now_on:
                print(f"  [{inc.banner(session.memory())}]", file=out, flush=True)
                ch.publish({'kind': 'capture', 'at': time.time(),
                            'capturing': now_on,
                            'banner': inc.banner(session.memory())})
            try:
                segq.put_nowait((audio, start, max(0.0, start - last_end)))
            except queue.Full:
                # the worker is more than SEGMENT_QUEUE_MAX segments behind,
                # which is a real backlog and not a blip. Said out loud rather
                # than blocking the gate, which is the thing this exists to
                # keep running.
                m.segments_dropped += 1
                print(f"  [PIPELINE BEHIND: dropped a {len(audio)/SR:.1f}s "
                      f"segment, {m.segments_dropped} so far]",
                      file=sys.stderr, flush=True)
            last_end = start + len(audio) / SR
        elif gate.dropped_short:
            m.too_short += 1
            gate.dropped_short = False
    # let the worker finish what it has before the summary is printed
    segq.join()
    segq.put(None)
    pump.join(timeout=30)
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

# How long the device may deliver nothing before we call it dead. Audio arrives
# every 32 ms, so seconds of nothing is not a slow moment, it is a stopped
# stream. Sleep is the case that matters: CoreAudio tears the input device down
# and PortAudio does not raise, so the old loop sat in `continue` forever,
# awake and deaf, with the lock still held. Exiting is what lets a supervisor
# restart us into a working device; see service.py.
STALL_S = 6.0
STALL_START_S = 20.0         # ...and this long for the first block ever
# Blocks arriving that are all exactly zero is a different failure with the
# same result: the stream is open, the callback fires on time, and every
# sample is digital silence. A Bluetooth input in the wrong profile does this,
# and so does a denied microphone permission and a hardware mute. It is not a
# quiet room -- a real microphone in a silent room still has a noise floor
# around -60 dB, never -240.
DEAD_SILENCE_S = 30.0

# Utterance times are counted in frames; speaking windows are stamped from the
# wall clock. Those are two different clocks and nothing was comparing them.
# If the audio device runs even slightly fast or slow, or a frame is dropped,
# the two drift apart, and the self-hearing suppression quietly stops lining up
# with the thing it is suppressing -- with nothing noticing, which is the shape
# of the four silent failures this project has already had.
#
# So the drift is measured every callback and reported. It is the difference
# between how much wall time has passed since the stream opened and how much
# audio the device has handed over. A constant small positive offset is normal
# and is the buffer; growth is the failure.
CLOCK_DRIFT_WARN_S = 0.50
# ...and the rate, which is the one that actually decides anything. The offset
# is corrected for: self_heard adds the measured drift before comparing, so a
# large but steady offset is harmless. What is not correctable is how far the
# offset moves DURING a segment, which is the rate times the segment length.
# This machine sits around 3 percent, which over a 12-second segment is 0.36s
# -- inside the tail margin, so the suppression still lines up.
CLOCK_DRIFT_RATE_WARN = 0.06


# Exit codes the supervisor reads. 0 is a clean stop and must not be restarted
# into; the rest are conditions a restart might actually fix.
EXIT_STALLED = 3             # the device died under us
EXIT_LOCKED = 4              # somebody else holds the lock


class AudioStalled(Exception):
    """The input device stopped delivering. Not recoverable in place: the
    stream has to be torn down and reopened, which means exiting."""


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
        # set when the stream actually opens, not here. frames() is a
        # generator, so the InputStream is not created until something starts
        # iterating it, and everything between construction and that first
        # next() -- model warm-up, in practice -- would otherwise count
        # against the no-audio deadline. It did: a run with a working
        # microphone reported "the device delivered nothing in 20s after
        # opening" having never opened it. Exactly the mistake the drift
        # baseline made, in the same class.
        self.opened_t = None
        self.opened_wall = None
        self.last_block_t = None
        self.last_nonzero_t = None
        # Measured from the FIRST block, not from construction. The first
        # version measured from __init__ and reported +0.75s of drift "after
        # 0s of audio", because everything between building the object and the
        # device handing over its first block -- opening the stream, and
        # whatever else the process was doing -- was being counted as clock
        # divergence. Drift is a difference in RATE between two clocks; a
        # constant offset at the start is not drift, and reporting it as drift
        # is a false alarm, which is no better than the silent failure this
        # was built to prevent.
        self.first_block_t = None
        self.first_block_samples = 0
        self.drift = 0.0          # wall seconds minus audio seconds, since then
        self.drift_t = None
        self.drift_rate = 0.0     # seconds of divergence per second
        self.drift_max = 0.0
        self.drift_warned = False

    def _callback(self, indata, frames, tinfo, status):
        if status:
            self.status[str(status)] = self.status.get(str(status), 0) + 1
        self.blocks += 1
        now = time.monotonic()
        self.last_block_t = now
        self.samples += len(indata)
        # measured here rather than at the consumer: a frame waiting in the
        # queue is behind by the queue depth, which is not drift
        if self.first_block_t is None:
            self.first_block_t = now
            self.first_block_samples = self.samples
        elapsed = now - self.first_block_t
        heard = (self.samples - self.first_block_samples) / SR
        prev, prev_t = self.drift, self.drift_t
        self.drift = elapsed - heard
        self.drift_t = elapsed
        # the RATE of divergence, which is what decides whether correcting by
        # the current offset is good enough inside one segment. This machine
        # delivers audio about 3 percent slow, so the offset grows without
        # bound and an absolute threshold on it would simply latch on forever;
        # what matters is how far it moves during a segment.
        if prev_t is not None and elapsed - prev_t > 1.0:
            self.drift_rate = (self.drift - prev) / (elapsed - prev_t)
        if abs(self.drift) > abs(self.drift_max):
            self.drift_max = self.drift
        if abs(self.drift_rate) > CLOCK_DRIFT_RATE_WARN and not self.drift_warned:
            self.drift_warned = True
            print(f"WARNING: the audio clock runs {100*self.drift_rate:+.1f}% "
                  f"against the wall clock ({self.drift:+.2f}s apart over "
                  f"{heard:.0f}s of audio). Self-hearing suppression compares the two and is "
                  f"unreliable beyond this point.", file=sys.stderr, flush=True)
        x = indata[:, 0] if indata.ndim > 1 else indata
        p = float(np.abs(x).max()) if len(x) else 0.0
        if p > self.peak:
            self.peak = p
        if p > 0.0:
            self.last_nonzero_t = self.last_block_t
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
        self.opened_t = time.monotonic()
        self.opened_wall = time.time()
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=FRAME, callback=self._callback,
                            device=self.device):
            i = 0
            while True:
                try:
                    frame = self.q.get(timeout=1.0)
                except queue.Empty:
                    now = time.monotonic()
                    if self.last_block_t is None:
                        if now - self.opened_t > STALL_START_S:
                            raise AudioStalled(
                                f"the device delivered nothing in "
                                f"{STALL_START_S:.0f}s after opening")
                        print("WARNING: no audio has arrived from the device yet",
                              flush=True)
                    elif (self.last_nonzero_t is None
                          and now - self.opened_t > DEAD_SILENCE_S):
                        raise AudioStalled(
                            f"{self.blocks} block(s) arrived and every sample "
                            f"in all of them was exactly zero. The device is "
                            f"open but deaf: a Bluetooth input in the wrong "
                            f"profile, a denied microphone permission, or a "
                            f"hardware mute. Pin a known-good device with "
                            f"--device.")
                    elif (self.last_nonzero_t is not None
                          and now - self.last_nonzero_t > DEAD_SILENCE_S):
                        raise AudioStalled(
                            f"the device has delivered nothing but digital "
                            f"silence for {now - self.last_nonzero_t:.0f}s "
                            f"after {self.blocks} block(s); it went deaf "
                            f"without stopping")
                    elif now - self.last_block_t > STALL_S:
                        raise AudioStalled(
                            f"the device stopped delivering "
                            f"{now - self.last_block_t:.0f}s ago after "
                            f"{self.blocks} block(s); it was probably torn "
                            f"down by sleep or unplugged")
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


# --- one live path at a time ------------------------------------------------
# Two of these ran at once and nothing noticed. Both captured, both stored,
# both published to the same socket, and the only visible symptom was the
# viewer showing every utterance twice -- by which point 44 duplicate
# observations were in the store, measured against two different session
# references, with no way to say which copy was authoritative. The lock itself
# is in singleton.py, shared with the viewer, which had the same failure.
LOCK_PATH = singleton.lock_path('live')
AlreadyRunning = singleton.AlreadyRunning


def take_lock():
    return singleton.take(
        'live', 'live path',
        "Two at once means both capture, both store, and every utterance is\n"
        "written twice against two different session references.",
        process='live.py')


def main():
    args = sys.argv[1:]
    if '--devices' in args:
        list_devices()
        return
    # before anything is opened, loaded or captured: a refusal after thirty
    # seconds of model loading is a refusal nobody reads
    try:
        take_lock()
    except AlreadyRunning as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return EXIT_LOCKED
    realtime = '--realtime' in args
    locale = args[args.index('--locale') + 1] if '--locale' in args else None
    session = SessionState(locale=locale)

    # load and warm every model before anything is timed. In a real session
    # this happens once at start-up; folding it into the first utterance's
    # latency would report a number no later utterance ever pays.
    import diarize as dz
    import voice
    # off the hot path: building Kokoro's phonemiser takes seconds, and paying
    # that on the first thing anybody says would be worse than the voice it
    # replaces. Until it finishes, voice.say falls through to the system voice.
    voice.warm_up()
    t_warm = time.perf_counter()
    warm = np.zeros(SR, dtype='float32')
    asr().transcribe_pcm(warm, session.locale)
    dz.embed(warm, SR, [(0.0, 1.0)])
    Gate()
    print(f"models loaded and warmed in {time.perf_counter()-t_warm:.1f}s")
    # said once at the top, and again whenever it changes. The recording state
    # is not something a person should have to go and ask for.
    import staleness as st
    st.reap(session.memory())
    st.register(session.memory(), 'live')
    st.heartbeat(lambda: __import__('memory').open())
    session.check_capture()
    print(f"[{inc.banner(session.memory())}]")
    ch.publish({'kind': 'session', 'at': time.time(), 'session': session.id,
                'capturing': session.capturing,
                'banner': inc.banner(session.memory())})
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
    stalled = None
    t_wall = time.perf_counter()
    try:
        run_stream(session, src, realtime, meters=meters, capture=capture,
                   quiet=('--file' in args and '--verbose' not in args))
    except KeyboardInterrupt:
        print("\nstopped.")
    except AudioStalled as e:
        # Not an error to swallow. Exiting with a non-zero status is the whole
        # recovery mechanism: the supervisor in service.py reopens us against
        # whatever device exists now. Staying alive would hold the lock and
        # capture nothing.
        stalled = e
        print(f"\nAUDIO STALLED: {e}", file=sys.stderr, flush=True)
    wall = time.perf_counter() - t_wall

    print()
    print(f"frames into the gate: {meters.frames}"
          + (f"   device blocks: {capture.blocks}, samples {capture.samples}, "
             f"peak {capture.peak:.4f}, dropped {capture.dropped}, "
             f"clock drift {capture.drift:+.3f}s at "
             f"{100*capture.drift_rate:+.1f}% (worst offset "
             f"{capture.drift_max:+.3f}s)"
             + (", RATE OVER THE LIMIT" if abs(capture.drift_rate)
                > CLOCK_DRIFT_RATE_WARN else "")
             if capture else ""))
    if capture and capture.status:
        print(f"PortAudio status flags: {capture.status}")
    if meters.frames:
        print(f"level: mean {meters.level_sum/meters.frames:.1f} dB, "
              f"min {meters.level_min:.1f}, max {meters.level_max:.1f}")
        print(f"voice detector: highest probability {meters.vad_max:.3f}, "
              f"{meters.vad_over} frame(s) at or over {VAD_THRESHOLD}")
        print(f"segments: {meters.opens} opened, {meters.closes} closed, "
              + (f"{meters.segments_dropped} DROPPED by a busy pipeline, "
                 if meters.segments_dropped else "")
              + f"{meters.too_short} discarded as too short, "
              + f"{meters.segments_with_words} produced words, "
              + f"{meters.segments_no_words} produced none")
        # the suppression count, said out loud rather than only published to
        # the viewer. It was computed all along and reached one consumer, so a
        # run's own summary could not answer whether self-hearing had been
        # caught -- which is the only question the fix exists to answer.
        print(f"its own voice: {session.self_heard} utterance(s) recognised and "
              f"not stored"
              + (f", {session.clock_untrusted} undecidable under clock drift"
                 if session.clock_untrusted else ""))

    problem = meters.diagnose(capture)
    if problem:
        print(f"\nNOTHING WAS PRODUCED, and this is why: {problem}")
        return EXIT_STALLED if stalled else None

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
    return EXIT_STALLED if stalled else None


if __name__ == "__main__":
    sys.exit(main() or 0)
