"""The notation the input layer emits: a weighted transcript a model can read.

Everything upstream produces continuous numbers. A model does not read
decimals well and does read ordinary text natively, so at this boundary the
numbers are turned into marks and, where a mark cannot carry the meaning,
into a short line of English. Nothing is invented; every mark already means
what it needs to mean in ordinary writing:

  **word**   strong emphasis, the level a listener would call stressed
  *word*     light emphasis, audible but not pointed
  word       unmarked
  ...        a beat of silence between two words inside one utterance
  ?          a question. Kept when the transcript supplied it, and added when
             pitch rose like a question and the transcript did not notice
  em dash    at the end of an utterance that did not sound finished
  . , ! ;    otherwise kept exactly as the transcript produced them

Above that, one line of English, and only when there is something to say. An
ordinary finished statement delivered normally gets no line at all: announcing
the default case is pure cost. What can only be said in words is the graded
stuff, how fast and how loud and how varied the delivery was, and those are
said with a grading vocabulary rather than named boxes, so that mildly lively
and shouting do not render identically.

The record is the primary form. It holds every measurement, continuous and
unquantised, and a schema version. The rendered string is generated from the
record at send time by render_record(), so a stored transcript can be
re-rendered under a different scheme without going back to the audio.

usage:
  markers.py <wav> [speaker_label] [max_utterances]
  markers.py levels <wav> [<wav> ...]     recompute the word-level thresholds
"""

import json
import math
import sys

import numpy as np

import baseline as bl
import normalize as nz
import score as sc
from prosody_core import measure

SCHEMA_VERSION = 3

# --- the one named place for the notation ------------------------------------
MARK = {2: ('**', '**'), 1: ('*', '*'), 0: ('', '')}

# An utterance shorter than this gets no emphasis marks. A mark is a claim
# that one word stood out from the others, and a lone word has no others: the
# rendered "**Stop.**" says only that the utterance was loud, which is what the
# signal line is for. The threshold is two rather than the three that
# baseline.UTT_MIN_WORDS uses for the signal line, because the two are gating
# different claims. rate and the spreads genuinely need three points. A mark
# needs a contrast, and two words supply one: across the corpus every marked
# word in a two-word utterance is carried by loudness against the speaker's
# own baseline, loud_z 2.9 to 6.1 on the four "stop" tokens in drag1, with
# pitch in range and no pause contribution. Those are real and a minimum of
# three would discard them.
#
# What this excludes is one-word utterances, which are 36 of 186 stored live
# utterances and, once measure() segments first, 65 of 576 corpus ones. It
# silences four marked words across the corpus, of which tone1's shouted
# "Stop." is a real measurement: loud_z 6.5, pitch in range. A mark cannot
# express it, having nothing to contrast it against, and the signal line
# cannot either, needing three words for a rate. The weight is still on the
# record; what is lost is only the ability to render it. It is applied here at render time and
# not in score.apply_weight, so the continuous weight stays in the record and
# a stored transcript can be recut under a different minimum.
MARK_MIN_WORDS = 2
PAUSE_MARK = '...'
UNFINISHED_MARK = '—'          # em dash
SENTENCE_FINAL = '.!?…'

# Level thresholds on the continuous weight, derived by derive_levels() over
# the whole corpus: stress1, neutral1, tone1, drag1, turns1, 674 non-suspect
# words. They are not round numbers and were not chosen to make any particular
# word come out marked.
#
# The derivation, in full in derive_levels(): a word cannot be less than
# unmarked, so the half of the weight distribution below its median contains no
# emphasis at all. Reflecting that half upward about the median gives the shape
# an entirely unmarked distribution would have. 'light' is where more than half
# the words above it have no unmarked counterpart below it, and 'strong' is the
# median of that unaccounted-for population.
#
# Rederived twice, and both times because something under them moved.
#
# The pair before these, 0.962 and 1.408, was derived under faster-whisper.
# SpeechAnalyzer reports word spans on a 60 ms grid and cuts words in
# different places, which moves the duration cue and the weight under it;
# rederiving against the whole-file Apple transcript gave 1.042 and 1.518.
#
# Then prosody_core.measure began segmenting on silence before transcribing,
# which stopped the recogniser running a phrase-final word out into the
# following silence: across the corpus that cut words with a reported span
# over LONG_WORD_S from 96 to 26.
#
# That last change also invalidated the stored baseline, which had been
# accumulated over the whole-file path and still carried the stretched spans
# in its duration statistics. Rebuilding it halved dur_resid_sd, 0.387 s to
# 0.200 s, which is the denominator of the duration z and therefore roughly
# doubled that cue everywhere. These are derived against a rebuilt baseline
# and the path that actually runs, over 692 non-suspect words. Both have to
# move together: rederiving these against a whole-file baseline is what
# produced the intermediate 0.973 and 1.491, which fitted neither.
LEVEL_THRESHOLDS = {'light': 1.143, 'strong': 1.659}

# What these were cut against. Not decoration: provenance.check_levels compares
# every entry of 'config' against the configuration this run measures under,
# and 'baseline_digest' against the content of the baseline actually loaded, so
# thresholds cut against a superseded baseline cannot quietly go on marking
# words. Both halves are needed. The config catches a changed pipeline; the
# digest catches the case that actually happened, where the pipeline was
# unchanged but the baseline underneath had been rebuilt.
#
# `markers.py levels <corpus>` prints a replacement for this block. It is
# pasted rather than written by the program because a constant the program can
# rewrite is a constant nobody reviews.
LEVEL_PROVENANCE = {   'baseline_digest': 'c8f25c83f065204d092dad55d8a74b8e',
    'baseline_speaker': 'owner',
    'config': {   'baseline.DISTANCE_MIN_SAME_FILES': 4,
                  'baseline.DISTANCE_MIN_STEP_RATIO': 0.25,
                  'baseline.EMPHATIC_CAP_PCT': 80.0,
                  'baseline.MIN_SENTENCE_WORDS': 4,
                  'baseline.POSITION_CLASSES': ['initial', 'middle', 'final'],
                  'baseline.SHORT_TERM_MIN_WORDS': 20,
                  'baseline.SHORT_TERM_WINDOW_S': 180.0,
                  'baseline.UTT_KEEP_RAW': True,
                  'baseline.UTT_MIN_WORDS': 3,
                  'baseline.UTT_STATS': [   'rate',
                                            'loud_mean',
                                            'loud_rel',
                                            'loud_sd',
                                            'pitch_sd'],
                  'config_version': 1,
                  'measure.segmented': True,
                  'prosody_core.ENVELOPE_HOP_S': 0.01,
                  'prosody_core.ENVELOPE_WINDOW_S': 0.025,
                  'prosody_core.LONG_WORD_MIN_VOICED': 5,
                  'prosody_core.LONG_WORD_S': 1.2,
                  'prosody_core.MIN_SILENCE_S': 0.15,
                  'prosody_core.MIN_VOICED_FRAMES': 3,
                  'prosody_core.NOISE_MARGIN_DB': 6.0,
                  'prosody_core.OCTAVE_FACTOR': 2.0,
                  'prosody_core.OCTAVE_MAX_PASSES': 5,
                  'prosody_core.PITCH_CEILING_HZ': 300.0,
                  'prosody_core.PITCH_FALL_LOOKBACK': 0,
                  'prosody_core.PITCH_FALL_MIN_RATIO': 0.05,
                  'prosody_core.PITCH_FLOOR_HZ': 60.0,
                  'prosody_core.PROBE_CEILING_HZ': 500.0,
                  'prosody_core.PROBE_FLOOR_HZ': 50.0,
                  'prosody_core.QUIET_MARGIN_DB': 12.0,
                  'prosody_core.RANGE_ABS_CEILING_HZ': 600.0,
                  'prosody_core.RANGE_ABS_FLOOR_HZ': 40.0,
                  'prosody_core.RANGE_CEILING_FACTOR': 1.5,
                  'prosody_core.RANGE_FLOOR_FACTOR': 0.75,
                  'prosody_core.RANGE_HIGH_PCT': 95.0,
                  'prosody_core.RANGE_LOW_PCT': 5.0,
                  'prosody_core.RANGE_MIN_FRAMES': 100,
                  'prosody_core.SENTENCE_FINAL_PUNCT': '.!?…',
                  'prosody_core.SENTENCE_GAP_ALWAYS_S': 2.0,
                  'prosody_core.SENTENCE_GAP_MIN_S': 0.6,
                  'prosody_core.SILENCE_DROP_DB': 15.0,
                  'prosody_core.TRAILING_PUNCT': '.,!?;:…"\')]}',
                  'prosody_core.TRANSCRIBE_LOCALE': 'en_US',
                  'score.CUE_COLUMN': {   'dur': 'dur_resid_z',
                                          'loud': 'loud_z',
                                          'pause': 'pause_z',
                                          'pitch': 'pitch_resid_z'},
                  'score.WEIGHTS': {   'dur': 0.25,
                                       'loud': 0.45,
                                       'pause': 0.1,
                                       'pitch': 0.2},
                  'segment.CLOSE_S': 0.3,
                  'segment.FRAME': 512,
                  'segment.MAX_SEGMENT_S': 12.0,
                  'segment.MIN_SEGMENT_S': 0.2,
                  'segment.OPEN_FRAMES': 2,
                  'segment.PREROLL_S': 0.25,
                  'segment.SR': 16000,
                  'segment.VAD_THRESHOLD': 0.5,
                  'transcriber.bridge_source': 'b29729ceecf12bd1',
                  'transcriber.engine': 'SpeechAnalyzer'},
    'corpus': [   'corpus/stress1.wav',
                  'corpus/neutral1.wav',
                  'corpus/tone1.wav',
                  'corpus/drag1.wav',
                  'corpus/turns1.wav'],
    'n_words': 692}

# A pause gets its own mark once it reaches the stored emphatic cap, which is
# the point at which the pause cue saturates in score.py. Below the cap a
# pause still moves the weight of the words around it continuously; it just
# does not earn a glyph of its own.

# --- grading the continuous utterance signals --------------------------------
# Four words, each a step up. The step sizes are not chosen: each band is four
# times rarer than the one below it, and the ladder starts where a remark stops
# being worth its cost. See grade_thresholds().
GRADE_WORDS = ('slightly', 'noticeably', 'markedly', 'unusually')
GRADE_BASE_P = 0.10        # the most ordinary nine utterances in ten say nothing
GRADE_RARITY_STEP = 4.0    # each band up is this many times rarer

# Each utterance signal, with the plain adjective for each direction and
# whether it reaches the rendered line at all. The adjectives take an adverb
# directly, so "markedly fast" and "unusually flat in pitch" come out of the
# same template, and nothing needs a comparative.
#
# loud_mean is measured, stored and never rendered. It is the honest answer to
# "quieter than usual for this person", but it cannot distinguish a raised
# voice from a closer microphone, and on drag1 it produced eighteen identical
# "unusually loud" lines that described the recording rather than any
# utterance in it. loud_rel is the same measurement with the recording's own
# level subtracted, so a shift affecting the whole file cancels and what
# survives is how this utterance sat against the rest of the same session.
UTT_SIGNALS = (
    ('rate', 'fast', 'slow', True),
    ('loud_rel', 'loud', 'quiet', True),
    ('loud_sd', 'uneven in loudness', 'level in loudness', True),
    ('pitch_sd', 'varied in pitch', 'flat in pitch', True),
    ('loud_mean', 'loud', 'quiet', False),
)

# Stacked marginal clauses cost more than they carry: three "slightly" phrases
# over a five-word question is 94 characters saying almost nothing. The line is
# capped at its strongest clause rather than dropping the weakest rung
# outright, because in turns1 the single-clause "slightly" lines read well and
# it was only the stacking that was waste. See signal_line().
LINE_MAX_CLAUSES = 1


GRADE_TAIL_P = [GRADE_BASE_P / GRADE_RARITY_STEP ** i
                for i in range(len(GRADE_WORDS))]

# Below this many stored utterances beyond a boundary, the boundary cannot be
# read off the data and is extrapolated from an exponential fit to the tail.
TAIL_MIN_POINTS = 2.0
TAIL_THRESHOLD_P = 0.15     # exceedances above this one-sided tail feed the fit


def two_sided_z(p):
    """The z beyond which |Z| falls with probability p, by bisection.

    Kept for the fallback ladder, used when a speaker has no stored utterance
    distribution to grade against.
    """
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if math.erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


GRADE_Z = [two_sided_z(p) for p in GRADE_TAIL_P]


def tail_quantile(values, p, upper):
    """The value at one-sided tail probability p, from the data where the data
    reach and from an exponential tail fit where they do not.

    Grading against a normal ladder assumed these statistics were normal and
    they are not: with the normal ladder the corpus counts ran 13, 2, 4, 5,
    roughly flat above the bottom rung and with the top rung more common than
    the two beneath it, which is what a heavy tail looks like when it is
    forced through a light-tailed ruler.

    Where at least TAIL_MIN_POINTS stored utterances lie beyond the boundary
    it is an order statistic and nothing is assumed. Beyond that the top rungs
    would be pure invention from 54 samples, so the exceedances above
    TAIL_THRESHOLD_P are fitted with an exponential, which is the
    peaks-over-threshold model with no shape parameter, and the boundary comes
    off the fit. Both cases are honest about which they are.
    """
    v = np.asarray(sorted(values), dtype=float)
    n = len(v)
    if n < 4:
        return float('nan'), 'too few'
    if p * n >= TAIL_MIN_POINTS:
        q = float(np.quantile(v, 1.0 - p if upper else p))
        return q, 'empirical'
    u = float(np.quantile(v, 1.0 - TAIL_THRESHOLD_P if upper else TAIL_THRESHOLD_P))
    exc = (v[v > u] - u) if upper else (u - v[v < u])
    if len(exc) < 2 or exc.mean() <= 0:
        return float('nan'), 'no tail'
    beta = float(exc.mean())
    step = beta * math.log(TAIL_THRESHOLD_P / p)
    return (u + step if upper else u - step), 'extrapolated'


def empirical_boundaries(values):
    """The four rung boundaries on each side, in the signal's own raw units."""
    v = [x for x in (values or []) if x is not None and not np.isnan(x)]
    if len(v) < 4:
        return None
    out = {'median': float(np.median(v)), 'n': len(v), 'high': [], 'low': [],
           'high_source': [], 'low_source': []}
    for p in GRADE_TAIL_P:
        hi, hs = tail_quantile(v, p / 2.0, True)
        lo, ls = tail_quantile(v, p / 2.0, False)
        out['high'].append(hi)
        out['low'].append(lo)
        out['high_source'].append(hs)
        out['low_source'].append(ls)
    return out


def grade_value(x, bounds):
    """The grading word for a raw value, or None when it is too ordinary.

    Returns (word, side, rung index, how far past its boundary).
    """
    if x is None or bounds is None or np.isnan(x):
        return None, None, -1, 0.0
    upper = x > bounds['median']
    edges = bounds['high'] if upper else bounds['low']
    rung = -1
    for i, e in enumerate(edges):
        if np.isnan(e):
            continue
        if (x >= e) if upper else (x <= e):
            rung = i
    if rung < 0:
        return None, None, -1, 0.0
    span = abs(edges[rung] - bounds['median']) or 1.0
    return (GRADE_WORDS[rung], 'high' if upper else 'low', rung,
            abs(x - edges[rung]) / span)


def grade(z):
    """Fallback ladder in standard deviations, for a speaker with no stored
    utterance distribution to grade against."""
    if z is None or np.isnan(z) or abs(z) < GRADE_Z[0]:
        return None
    word = GRADE_WORDS[0]
    for w, t in zip(GRADE_WORDS, GRADE_Z):
        if abs(z) >= t:
            word = w
    return word


def derive_levels(weights):
    """Where to cut the continuous weight, from the distribution itself.

    Reflect the below-median half about the median. That reflection is what
    the upper half would look like if no word in the corpus were emphasised,
    so the difference between the real upper half and the reflection is the
    marked population.

    light  the lowest weight above which the marked population is the
           majority: past this point most words have no unmarked counterpart
    strong the median of that marked population: among the words that carry
           marking at all, the upper half get the strong mark
    """
    w = np.array([x for x in weights if not np.isnan(x)], dtype=float)
    med = float(np.median(w))
    mirror = med + (med - w[w <= med])
    right = w[w > med]
    grid = np.arange(med, float(w.max()) + 1e-9, 0.001)
    act = np.array([(right > t).sum() for t in grid])
    exc = act - np.array([(mirror > t).sum() for t in grid])

    light = next((float(t) for t, a, e in zip(grid, act, exc)
                  if a and e >= a / 2.0), float('nan'))
    at_light = int(exc[int(np.argmin(np.abs(grid - light)))])
    strong = next((float(t) for t, e in zip(grid, exc)
                   if t >= light and e <= at_light / 2.0), float('nan'))
    return {'light': round(light, 3), 'strong': round(strong, 3),
            'median': med, 'n': len(w), 'marked_est': at_light}


def level(weight):
    if weight is None or np.isnan(weight):
        return 0
    if weight > LEVEL_THRESHOLDS['strong']:
        return 2
    if weight > LEVEL_THRESHOLDS['light']:
        return 1
    return 0


# --- measurement -------------------------------------------------------------
def loudness_reference(ref, info):
    """Which loudness comparison this recording gets, and why.

    'distance-corrected' subtracts only the gain a calibrated distance proxy
    attributes to microphone placement, leaving a genuinely raised voice
    intact. 'session-relative' subtracts the recording's own median level,
    which removes the placement artifact and any genuine whole-file loudness
    with it. The first is better and is used whenever a stored calibration
    passes its own reliability test; the second is the fallback and is what
    every recording currently gets, because the calibration does not pass.
    """
    cal = ((ref.get('distance') or {}).get('calibration')) or {}
    if not cal.get('usable'):
        return 'session-relative', ('no usable distance calibration'
                                    if cal else 'no distance calibration stored')
    key = cal['chosen']
    c = cal['candidates'][key]
    v = (info.get('distance') or {}).get(key, float('nan'))
    lo, hi = min(c['values']), max(c['values'])
    if np.isnan(v) or not (lo <= v <= hi):
        return 'session-relative', f'{key} {v:.2f} outside the calibrated range'
    return 'distance-corrected', f'{key} {v:.2f}, {c["slope_db_per_unit"]:.2f} dB per unit'


def distance_offset_db(ref, info):
    """How much of this recording's level the calibration blames on placement."""
    cal = ((ref.get('distance') or {}).get('calibration')) or {}
    if not cal.get('usable'):
        return float('nan')
    key = cal['chosen']
    c = cal['candidates'][key]
    v = (info.get('distance') or {}).get(key, float('nan'))
    mid = float(np.median(c['values']))
    return c['slope_db_per_unit'] * (v - mid)


def utterance_signals(members, ref, session_db):
    """The four continuous utterance measurements and how far each sits from
    the speaker's stored reference for that same statistic.

    All four come out of numbers prosody_core already produced. Rate is words
    over the utterance's span. Loudness level and loudness spread are the mean
    and the spread of the per-word 90th-percentile intensity. Pitch spread is
    the spread of the per-word median pitch, which is what "animated" was
    measuring before the loudness half of it became a signal of its own.

    Each is compared against the mean and spread of that statistic across the
    utterances in the stored baseline, so an utterance is unusual relative to
    how this person usually speaks rather than relative to this file.
    """
    raw = bl.utterance_stats(members, session_db)
    uref = ref.get('utt') or {}
    out = {'measurable': raw is not None,
           'session_db': None if np.isnan(session_db) else round(session_db, 4)}
    for key, _, _, _ in UTT_SIGNALS:
        r = (uref.get(key) or {})
        v = raw[key] if raw else float('nan')
        z = nz.z(v, r.get('mean', float('nan')), r.get('sd', float('nan'))) \
            if raw else float('nan')
        out[key] = None if raw is None else round(float(v), 4)
        out[f'{key}_z'] = None if np.isnan(z) else round(float(z), 4)
    out['n_words'] = raw['n_words'] if raw else len(members)
    out['span_s'] = round(raw['span_s'], 3) if raw else None
    return out


def is_question(members):
    """Two cues. normalize.flag_questions sets a flag when pitch rose across
    the utterance by more than the stored declination expects it to fall. The
    transcript also emits question marks, which cost the speaker no pitch at
    all and catch the questions asked without raising the voice."""
    by_pitch = bool(members[0].get('question'))
    by_punct = any('?' in r['punct'] for r in members)
    return by_pitch, by_punct


def sounded_finished(rows, last_index):
    """Did the speaker sound like they had come to the end of it?

    The pitch half is the same fall test prosody_core uses for sentence
    boundaries. The punctuation half is handled by the caller, because a
    question mark this layer adds itself also ends the utterance.
    """
    from prosody_core import pitch_fell
    last = rows[last_index]
    punct = any(c in SENTENCE_FINAL for c in last['punct'])
    fell, _, _, _ = pitch_fell(rows, last_index, last['sentence'])
    return punct, bool(fell)


def word_record(r):
    """Everything measured about the word, continuous and unquantised.

    No level and no marks: those are render-time decisions and live in
    render_record(). gap_before is kept raw so the pause mark can be recut
    against a different cap later.
    """
    def n(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) \
            else round(float(x), 4)
    r = {'loud_z_lt': float('nan'), 'loud_z_st': float('nan'),
         'dur_expected': float('nan'), 'dur_resid': float('nan'),
         'pitch_resid': float('nan'), 'pause_before': float('nan'),
         'pause_after': float('nan'), 'pause_raw': float('nan'), **r}
    return {
        'word': r['word'], 'punct': r['punct'],
        'speaker': r.get('speaker', 1),
        'speaker_source': r.get('speaker_source'),
        'person': r.get('person'),
        'person_id': r.get('person_id'),
        'person_decision': r.get('person_decision', 'unknown'),
        'person_distance': n(r.get('person_distance')),
        'person_confidence': r.get('person_confidence', 0.0),
        'scored': r.get('scored', True),
        'weight': n(r['weight']), 'cues_used': r['cues_used'],
        'start': n(r['start']), 'end': n(r['end']), 'dur': n(r['dur']),
        'orig_start': n(r['orig_start']), 'orig_end': n(r['orig_end']),
        'trimmed': n(r['trimmed']),
        'pitch_med': n(r['pitch_med']), 'pitch_mean': n(r['pitch_mean']),
        'n_voiced': r['n_voiced'], 'n_octave_removed': r['n_octave_removed'],
        'int_p90': n(r['int_p90']), 'int_mean': n(r['int_mean']),
        'gap_before': n(r['gap_before']), 'gap_after': n(r['gap_after']),
        'sent_pos': n(r['sent_pos']), 'position': r['dur_pos_class'],
        'loud_z': n(r['loud_z']), 'loud_z_lt': n(r['loud_z_lt']),
        'loud_z_st': n(r['loud_z_st']),
        'dur_expected': n(r['dur_expected']), 'dur_resid': n(r['dur_resid']),
        'dur_resid_z': n(r['dur_resid_z']),
        'pitch_resid': n(r['pitch_resid']), 'pitch_resid_z': n(r['pitch_resid_z']),
        'pause_before': n(r['pause_before']), 'pause_after': n(r['pause_after']),
        'pause_raw': n(r['pause_raw']), 'pause_z': n(r['pause_z']),
        'suspect': r['suspect'], 'suspect_reasons': r['suspect_reasons'],
    }


def utterances(rows, info, ref, total):
    """One record per utterance. Records only; nothing here is rendered."""
    cap = ref.get('pause_cap_s', float('nan'))
    mode, why = loudness_reference(ref, info)
    off = distance_offset_db(ref, info)
    bounds = {key: empirical_boundaries((ref.get('utt') or {}).get(key, {})
                                        .get('values'))
              for key, _, _, _ in UTT_SIGNALS}
    out = []
    for u, s in enumerate(sorted(set(r['sentence'] for r in rows))):
        idx = [i for i, r in enumerate(rows) if r['sentence'] == s]
        members = [rows[i] for i in idx]
        sid = members[0].get('speaker', 1)
        if mode == 'distance-corrected':
            session_db = ref.get('loud_mu', float('nan')) + off
        else:
            # the session level is this voice's own level in this recording,
            # not the file's: two people at one microphone have two levels
            same = [i for i, r in enumerate(rows)
                    if r.get('speaker', 1) == sid]
            session_db = bl.file_loudness([rows[i] for i in same],
                                          [same.index(i) for i in idx])
        q_pitch, q_punct = is_question(members)
        fin_punct, fin_pitch = sounded_finished(rows, idx[-1])
        before = members[0]['gap_before'] if idx[0] else members[0]['start']
        after = (members[-1]['gap_after'] if idx[-1] < len(rows) - 1
                 else total - members[-1]['end'])
        out.append({
            'schema_version': SCHEMA_VERSION,
            'utterance': u,
            'first_word_index': idx[0],
            'start': round(members[0]['start'], 3),
            'end': round(members[-1]['end'], 3),
            'silence_before': round(float(before), 3),
            'silence_after': round(float(after), 3),
            'boundary_cue': members[0]['boundary_cue'] or 'start of file',
            'speaker': sid,
            'person': members[0].get('person'),
            'person_id': members[0].get('person_id'),
            'person_decision': members[0].get('person_decision', 'unknown'),
            'person_distance': members[0].get('person_distance'),
            'person_confidence': members[0].get('person_confidence', 0.0),
            'speaker_changed': bool(u and out[-1]['speaker'] != sid),
            'n_speakers': info.get('n_speakers', 1),
            'scored': bool(members[0].get('scored', True)),
            'question_by_pitch_rise': q_pitch,
            'question_by_punctuation': q_punct,
            'finished_by_punctuation': fin_punct,
            'finished_by_pitch_fall': fin_pitch,
            'signals': utterance_signals(members, ref, session_db),
            'loudness_reference': mode,
            'loudness_reference_why': why,
            'distance': info.get('distance'),
            'grading': 'empirical' if all(bounds.get(k) for k, _, _, r in
                                          UTT_SIGNALS if r) else 'normal ladder',
            'boundaries': {k: b for k, b in bounds.items() if b},
            'reference': {'files': ref.get('files', []),
                          'n_words': ref.get('n_words', 0),
                          'n_utterances': (ref.get('utt') or {}).get('n', 0)},
            'pause_cap_s': None if np.isnan(cap) else round(float(cap), 4),
            'plain': ' '.join(r['word'] + r['punct'] for r in members),
            'words': [word_record(r) for r in members],
        })
    return out


# --- rendering ---------------------------------------------------------------
def signal_line(rec):
    """The line above the utterance, or None when there is nothing to say.

    Only the graded signals reach this, because the binary ones are carried by
    punctuation in the string itself. Phrases come out strongest first, so a
    model reading only the opening words still gets the main thing.
    """
    sig = rec['signals']
    bounds = rec.get('boundaries') or {}
    found = []
    for key, high, low, rendered in UTT_SIGNALS:
        if not rendered:
            continue
        b = bounds.get(key)
        if b:
            word, side, rung, past = grade_value(sig.get(key), b)
            if word:
                found.append((rung + min(past, 0.99),
                              f"{word} {high if side == 'high' else low}"))
        else:                                   # no stored distribution
            z = sig.get(f'{key}_z')
            g = grade(z)
            if g:
                found.append((abs(z), f"{g} {high if z > 0 else low}"))
    if not found:
        return None
    phrases = [p for _, p in sorted(found, key=lambda t: -t[0])][:LINE_MAX_CLAUSES]
    if len(phrases) == 1:
        body = phrases[0]
    else:
        body = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"Spoken {body} for this speaker."


def render_words(rec):
    """The words with their marks, from the stored continuous weights.

    An utterance below MARK_MIN_WORDS renders unmarked. The weights are still
    there in the record; this decides only what the string shows.
    """
    cap = rec.get('pause_cap_s')
    markable = len(rec['words']) >= MARK_MIN_WORDS
    out = []
    for k, w in enumerate(rec['words']):
        if k and cap is not None and (w['gap_before'] or 0.0) >= cap:
            out.append(PAUSE_MARK)
        open_, close = MARK[level(w['weight'])] if markable else ('', '')
        out.append(f"{open_}{w['word']}{close}{w['punct']}")
    return ' '.join(out)


def apply_terminal(text, rec):
    """Punctuation for the two binary signals, in place of a written line.

    A question the transcript already marked needs nothing. A question only
    pitch heard gets a question mark, replacing a full stop if the transcript
    put one there, because that is the notation every model reads without
    being told. An utterance that did not sound finished ends on an em dash,
    which is what trailing off looks like in print; a trailing comma is
    replaced by it rather than kept beside it, since both say the same thing
    and the dash says it more strongly.

    Returns (text, added_question, added_dash).
    """
    add_q = rec['question_by_pitch_rise'] and not rec['question_by_punctuation']
    if add_q:
        text = text[:-1] + '?' if text.endswith(('.', ',', '…')) else text + '?'
    finished = rec['finished_by_punctuation'] or add_q or rec['finished_by_pitch_fall']
    add_dash = not finished
    if add_dash:
        text = (text[:-1] if text.endswith(',') else text) + UNFINISHED_MARK
    return text, add_q, add_dash


def speaker_prefix(rec):
    """How a transcript marks who is talking: a label at the head of the turn.

    This is the ordinary interview and hansard convention, "Speaker 1:", and a
    model reads it without being told what it means. It appears only when the
    recording holds more than one voice, and only on the first utterance of
    each turn, because repeating it on every utterance of one turn is what a
    badly made transcript looks like. The numbers are local to the recording:
    diarization says these are different voices, not whose they are.
    """
    if rec.get('n_speakers', 1) < 2:
        return ''
    if rec.get('utterance', 0) and not rec.get('speaker_changed'):
        return ''
    who = rec.get('person')
    if who and rec.get('person_decision') in ('confident', 'match'):
        # a recognised person is named; an unrecognised voice keeps its
        # diarization number, because a wrong name is worse than a number
        return f"{who}: "
    return f"Speaker {rec.get('speaker', 1)}: "


def render_record(rec):
    """The whole rendered form for one utterance, from the record alone."""
    text, add_q, add_dash = apply_terminal(render_words(rec), rec)
    text = speaker_prefix(rec) + text
    line = signal_line(rec)
    return (line + "\n" + text) if line else text


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'levels':
        import provenance as pv
        speaker = LEVEL_PROVENANCE.get('baseline_speaker', 'owner')
        pool = []
        paths = sys.argv[2:]
        for p in paths:
            rows, _, _, _, _ = score_rows(p, speaker, check_levels=False)
            pool += [r['weight'] for r in rows if not r['suspect']]
        d = derive_levels(pool)
        print(f"over {d['n']} non-suspect words, median {d['median']:.3f}, "
              f"about {d['marked_est']} of them marked")
        print(f"light {d['light']}   strong {d['strong']}")
        print(f"currently in LEVEL_THRESHOLDS: {LEVEL_THRESHOLDS}")
        print()
        print("paste both of these into markers.py:")
        print()
        print(f"LEVEL_THRESHOLDS = {{'light': {d['light']}, "
              f"'strong': {d['strong']}}}")
        stats = bl.load()[speaker]
        block = {'baseline_speaker': speaker,
                 'baseline_digest': pv.content_digest(stats),
                 'corpus': list(paths), 'n_words': d['n'],
                 'config': pv.current()}
        # pprint and not json.dumps: this is pasted into a .py file, and JSON
        # writes true, false and null, none of which Python will import.
        import pprint
        print("LEVEL_PROVENANCE = " + pprint.pformat(block, indent=4,
                                                     sort_dicts=True))
        return

    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"
    speaker = sys.argv[2] if len(sys.argv) > 2 else "owner"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    rows, info, ref, stored, refs = score_rows(path, speaker)
    us = utterances(rows, info, ref, info['total'])
    shown = us[:limit] if limit else us

    print(f"file: {path}   speaker: {speaker}   "
          f"{len(rows)} words, {info['n_sentences']} utterance(s)")
    print(f"levels: light > {LEVEL_THRESHOLDS['light']}, "
          f"strong > {LEVEL_THRESHOLDS['strong']}; "
          f"pause mark at {ref.get('pause_cap_s', float('nan')):.2f} s")
    d = info.get('diarization') or {}
    print(f"speakers: {info.get('n_speakers', 1)} voice(s), "
          f"{d.get('n_windows', 0)} window(s), "
          f"{info.get('speaker_only_boundaries', 0)} utterance boundary(ies) "
          f"forced by a change of voice")
    for sid, entry in sorted(refs.items()):
        share = (d.get('speakers') or {}).get(sid, {}).get('share', float('nan'))
        print(f"  speaker {sid}: {entry['n_words']} usable word(s), "
              f"{100*share:.0f}% of speech, reference {entry['kind']} "
              f"-- {entry['why']}")
    live = sc.live_cues(rows)
    if len(live) < len(sc.CUE_COLUMN):
        print("WARNING: only these emphasis cues are live: "
              + ", ".join(live)
              + ". Declination, the positional duration model and the "
                "emphatic pause cap all come from a stored baseline, and a "
                "within-file reference has none of them. The level "
                "thresholds were derived against the full four-cue weight, "
                "so marking on this file is not comparable.")
    ident = info.get('identification') or {}
    named = sorted({v['name'] for v in ident.values() if v.get('name')})
    print(f"recognition: {info.get('n_identified', 0)} utterance(s) matched a "
          f"known person, {info.get('n_unknown', 0)} unknown"
          + (f"; recognised: {', '.join(named)}" if named else "")
          + (f"  ({next(iter(ident.values()))['reason']})" if ident and not named
             else ""))
    head = us[0] if us else {}
    print(f"loudness reference: {head.get('loudness_reference', 'n/a')} "
          f"({head.get('loudness_reference_why', '')})")
    d = info.get('distance') or {}
    print(f"distance proxy: mod_ratio {d.get('mod_ratio', float('nan')):.2f} dB, "
          f"dyn_range {d.get('dyn_range', float('nan')):.2f} dB, "
          f"file level {d.get('level_db', float('nan')):.2f} dB")
    print(f"grading: {head.get('grading', 'n/a')}, rungs at two-sided tail "
          + ", ".join(f"{w} {p:.4g}" for w, p in zip(GRADE_WORDS, GRADE_TAIL_P))
          + ", silent below")
    print()
    print("=== RENDERED, generated from the records below at send time")
    for u in shown:
        print(render_record(u))
        print()
    print("=== RECORD, the stored primary form")
    for u in shown:
        print(json.dumps(u, separators=(',', ':')))


def score_rows(path, speaker='owner', check_levels=True):
    """The same pipeline score.py runs, plus the question flag it does not
    need and this does, run once per voice in the recording.

    Everything downstream of measurement is a comparison against a reference,
    and a reference belongs to one voice. So the rows are split by speaker and
    each subset is scored against its own reference. A voice with no reference
    is scored not at all: its words keep their raw measurements and get no
    z-scores, no weight and therefore no marks.
    """
    rows, info = measure(path)
    store = bl.load()
    # bl.load has already refused a baseline built under another pipeline. This
    # is the other half: thresholds cut against another baseline. check_levels
    # is off only for `markers.py levels`, which exists to rederive them and
    # would otherwise be unable to run whenever it was most needed.
    if check_levels and speaker in store:
        import provenance as pv
        pv.check_levels(LEVEL_PROVENANCE, store[speaker], speaker,
                        str(bl.BASELINE_PATH))
    refs = nz.per_speaker_references(rows, store, speaker)

    for sid, entry in refs.items():
        idx = {i for i, r in enumerate(rows) if r.get('speaker', 1) == sid}
        mine = [rows[i] for i in sorted(idx)]
        ref = entry['ref']
        if ref is None:
            for r in mine:
                for k in ('pitch_z', 'loud_z', 'dur_z', 'pitch_resid_z',
                          'dur_resid_z', 'pause_z', 'weight'):
                    r[k] = float('nan')
                r['cues_used'] = ''
                r['dur_pos_class'] = bl.position_class(r)
                r['question'] = False
                r['scored'] = False
            continue
        nz.apply_baselines(mine, ref)
        nz.apply_rolling(mine, ref, entry['stored'])
        nz.apply_defaults(mine)
        nz.apply_stored_decline(mine, ref)
        nz.flag_questions(mine, ref, entry['stored'])
        sc.apply_duration_position(mine, ref)
        sc.apply_pause(rows, ref, info['total'], only=idx)
        sc.apply_weight(mine)
        for r in mine:
            r['scored'] = True

    main_id = max(refs, key=lambda s: refs[s]['n_words'])
    return rows, info, refs[main_id]['ref'] or nz.per_file_reference(rows), \
        refs[main_id]['stored'], refs


if __name__ == "__main__":
    main()
