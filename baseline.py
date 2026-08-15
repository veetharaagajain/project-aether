"""Persistent per-speaker prosody baseline.

Everything is stored as running sums, so adding another recording later
updates the baseline without reprocessing the recordings already in it.
Suspect rows never contribute to anything here.

usage:
  baseline.py add <speaker> <wav> [<wav> ...]
  baseline.py calibrate <speaker> <near.wav> <mid.wav> <far.wav>
  baseline.py show <speaker>
  baseline.py reset <speaker>
"""

import json
import sys
from pathlib import Path

import numpy as np

from prosody_core import measure

BASELINE_PATH = Path(__file__).resolve().parent / "baselines.json"

MIN_SENTENCE_WORDS = 4      # sentences shorter than this do not get a decline fit

# Two references, not one. The long-term one answers "loud for this person",
# accumulated over every recording ever added. The short-term one answers
# "loud for right now", over a rolling window of recent speech, so that a
# session recorded quietly at night is scored against how this person sounds
# tonight rather than against how they sounded in daylight. Mic distance and
# room move absolute levels as much as mood does, and neither is emphasis.
SHORT_TERM_WINDOW_S = 180.0   # three minutes of recent speech
SHORT_TERM_MIN_WORDS = 20     # below this the window is too thin to trust

# The emphatic pause is a short beat around a word. Its cap is taken from the
# distribution of within-sentence gaps in the baseline recordings rather than
# picked as a round number: those gaps are 85 percent exactly zero, and the
# non-zero ones are bimodal with an empty band between 0.80 and 1.00 s. The
# percentile below sits inside the lower population, under that empty band.
EMPHATIC_CAP_PCT = 80.0

# Duration by position in the sentence. A linear ramp against sent_pos
# explains nothing (correlation 0.013), but the final word is a step above the
# rest, so the model is three levels rather than a slope.
POSITION_CLASSES = ('initial', 'middle', 'final')

# Utterance-level references. Everything else here describes a word; these
# describe a whole utterance, because "spoken fast" and "spoken flatly" are
# properties of a stretch of speech and have no per-word meaning. All four are
# arithmetic over measurements prosody_core already produces, and all four are
# stored in their own raw units rather than as z-scores, so accumulating them
# does not depend on a baseline that is still being built.
#
# An utterance shorter than this has too few words for a spread to mean
# anything, and its words-per-second is one word divided by one duration.
UTT_MIN_WORDS = 3
UTT_STATS = ('rate', 'loud_mean', 'loud_rel', 'loud_sd', 'pitch_sd')

# The raw per-utterance values are kept alongside the running sums, because a
# mean and a spread cannot answer "how rare is this" for a statistic whose
# tails are not normal, and all four of these have heavier tails than normal.
# 54 utterances times 5 numbers is nothing next to what the pause block
# already stores raw.
UTT_KEEP_RAW = True

# A distance calibration is trustworthy only if the proxy actually moves with
# distance and moves in one direction. Both are checked in calibrate(): the
# steps between adjacent distances must all have the same sign, and the
# smallest must be at least this fraction of the largest, or the proxy is
# reporting one boundary rather than a scale.
DISTANCE_MIN_STEP_RATIO = 0.25

# ...and the span test needs enough same-distance recordings to mean anything.
# With two it passed a proxy whose spread across five same-distance recordings
# turned out to be three times its span across the three distances.
DISTANCE_MIN_SAME_FILES = 4


# --- storage -----------------------------------------------------------------
def load():
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def save(store):
    BASELINE_PATH.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")


def empty_stats():
    return {
        'files': [],
        'pitch': {'n': 0, 'sum': 0.0, 'sumsq': 0.0},
        'loud': {'n': 0, 'sum': 0.0, 'sumsq': 0.0},
        'dur': {'n': 0, 'sum': 0.0, 'sumsq': 0.0},
        'decl': {
            'n_sentences': 0,      # sentences whose fit was accumulated
            'n_rising': 0,         # excluded, positive slope, i.e. questions
            'n_short': 0,          # excluded, fewer than MIN_SENTENCE_WORDS
            'slope_sum': 0.0,
            'intercept_sum': 0.0,
            # sufficient statistics over the words of the accumulated
            # sentences, enough to get the residual sd against any straight
            # line without keeping the words themselves
            'resid': {'n': 0, 'Sx': 0.0, 'Sy': 0.0,
                      'Sxx': 0.0, 'Sxy': 0.0, 'Syy': 0.0},
        },
        # tail of the most recently added recording: the last
        # SHORT_TERM_WINDOW_S of non-suspect words, kept as raw values so the
        # short-term reference can be recomputed if the window length changes
        'recent': {'source': None, 'window_s': SHORT_TERM_WINDOW_S, 'values': []},
        # emphatic pause. The non-zero within-sentence gaps are kept raw
        # because the cap is a percentile of them, and the per-word component
        # pairs are kept raw because capping has to happen after the cap is
        # known. Both lists only hold non-zero cases, so they stay small.
        'pause': {'within_nonzero': [], 'word_components': [], 'word_zeros': 0},
        # duration by position class, as running sums
        'dur_pos': {c: {'n': 0, 'sum': 0.0, 'sumsq': 0.0} for c in POSITION_CLASSES},
        # utterance-level references, running sums over utterances rather than
        # over words, so an utterance statistic is compared against the spread
        # of that same statistic across this speaker's utterances
        'utt': {'n': 0, 'n_short': 0,
                **{k: {'n': 0, 'sum': 0.0, 'sumsq': 0.0} for k in UTT_STATS},
                'raw': {k: [] for k in UTT_STATS}},
        # what the microphone was doing, per recording added, and the
        # calibration from the deliberate-distance recordings if there is one
        'distance': {'per_file': {}, 'calibration': None},
    }


def file_loudness(rows, exclude=()):
    """This recording's own overall level: the median 90th-percentile
    intensity of its non-suspect words, in dB.

    Median rather than mean because a handful of shouted or swallowed words
    should not move it. exclude holds the indices of the utterance being
    measured, so an utterance is never part of the level it is compared
    against; without that a five-utterance file would have each utterance
    supplying a fifth of its own reference.
    """
    ex = set(exclude)
    v = [r['int_p90'] for i, r in enumerate(rows)
         if i not in ex and not r['suspect'] and not np.isnan(r['int_p90'])]
    return float(np.median(v)) if v else float('nan')


def utterance_stats(members, session_db=float('nan')):
    """The utterance-level measurements, in their raw units.

    rate       words per second over the utterance's whole span, gaps and all
    loud_mean  mean 90th-percentile intensity, dB, absolute
    loud_rel   the same, minus this recording's own level: how loud this
               utterance was for this session rather than for this person
    loud_sd    spread of that intensity across the words, dB
    pitch_sd   spread of median pitch across the words, Hz

    loud_mean and loud_rel are both kept. The absolute one answers "quieter
    than usual for this person", which is a real signal about tiredness or
    mood, but it cannot tell a raised voice from a closer microphone, because
    both move every word in the file together. Subtracting the file's own
    level cancels anything that moved the whole recording, which is why
    loud_rel is the one that gets rendered.

    Suspect words are excluded from the summaries but counted in the rate,
    since a mismeasured word still took time to say. Returns None when the
    utterance is too short to support any of it.
    """
    if len(members) < UTT_MIN_WORDS:
        return None
    span = members[-1]['end'] - members[0]['start']
    clean = [r for r in members if not r['suspect']]
    loud = [r['int_p90'] for r in clean if not np.isnan(r['int_p90'])]
    pitch = [r['pitch_med'] for r in clean if not np.isnan(r['pitch_med'])]
    if span <= 0 or len(loud) < UTT_MIN_WORDS or len(pitch) < UTT_MIN_WORDS:
        return None
    mean_loud = float(np.mean(loud))
    return {
        'rate': len(members) / span,
        'loud_mean': mean_loud,
        'loud_rel': mean_loud - session_db,
        'session_db': session_db,
        'loud_sd': float(np.std(loud, ddof=1)),
        'pitch_sd': float(np.std(pitch, ddof=1)),
        'n_words': len(members), 'span_s': span,
    }


def position_class(r):
    if r['sent_pos'] >= 0.999:
        return 'final'
    if r['sent_pos'] <= 0.001:
        return 'initial'
    return 'middle'


def eligible_gaps(rows, i, total):
    """The two gaps around word i that may count as an emphatic pause.

    The gap before the first word and after the last are the edges of the
    recording, not pauses in speech. A gap the boundary detector marked as a
    sentence boundary is utterance structure, not a beat around a word.
    """
    before = 0.0
    if i > 0 and not rows[i]['boundary_cue']:
        before = rows[i]['gap_before']
    after = 0.0
    if i + 1 < len(rows) and not rows[i + 1]['boundary_cue']:
        after = rows[i + 1]['gap_before']
    return before, after


# --- running statistics ------------------------------------------------------
def add_values(acc, values):
    for v in values:
        if v is None or np.isnan(v):
            continue
        v = float(v)
        acc['n'] += 1
        acc['sum'] += v
        acc['sumsq'] += v * v


def _acc(values):
    """Wrap a plain list of values in the running-sum shape mean_sd_from wants."""
    a = {'n': 0, 'sum': 0.0, 'sumsq': 0.0}
    add_values(a, values)
    return a


def mean_sd_from(acc):
    n = acc['n']
    if n == 0:
        return float('nan'), float('nan')
    mean = acc['sum'] / n
    if n < 2:
        return mean, float('nan')
    var = (acc['sumsq'] - n * mean * mean) / (n - 1)
    return mean, float(np.sqrt(max(var, 0.0)))


def fit_sentence(members, key):
    """Least squares of key against sent_pos over non-suspect members.

    Returns (slope, intercept, n_used). Falls back to a flat line at the
    mean when there are fewer than two usable points or no spread in
    sent_pos, so residuals are still defined.
    """
    pts = [(r['sent_pos'], r[key]) for r in members
           if not r['suspect'] and not np.isnan(r[key])]
    if len(pts) < 2:
        return 0.0, (pts[0][1] if pts else float('nan')), len(pts)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    if np.ptp(x) == 0.0:
        return 0.0, float(y.mean()), len(pts)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), len(pts)


def accumulate(stats, path):
    """Fold one recording into stats. Returns (used, rising, short)."""
    rows, info = measure(path)
    clean = [r for r in rows if not r['suspect']]

    add_values(stats['pitch'], [r['pitch_med'] for r in clean])
    add_values(stats['loud'], [r['int_p90'] for r in clean])
    add_values(stats['dur'], [r['dur'] for r in clean])

    # duration by position in the sentence
    for r in clean:
        add_values(stats['dur_pos'][position_class(r)], [r['dur']])

    # emphatic pause material
    pz = stats.setdefault('pause', {'within_nonzero': [], 'word_components': [],
                                    'word_zeros': 0})
    total = info['total']
    for i, r in enumerate(rows):
        if i and not r['boundary_cue'] and r['gap_before'] > 0.001:
            pz['within_nonzero'].append(float(r['gap_before']))
        if r['suspect']:
            continue
        b, a = eligible_gaps(rows, i, total)
        if b > 0.001 or a > 0.001:
            pz['word_components'].append([float(b), float(a)])
        else:
            pz['word_zeros'] += 1

    # utterance-level material
    ut = stats.setdefault('utt', {'n': 0, 'n_short': 0,
                                  **{k: {'n': 0, 'sum': 0.0, 'sumsq': 0.0}
                                     for k in UTT_STATS}})
    raw = ut.setdefault('raw', {k: [] for k in UTT_STATS})
    for s in sorted(set(r['sentence'] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r['sentence'] == s]
        u = utterance_stats([rows[i] for i in idx], file_loudness(rows, idx))
        if u is None:
            ut['n_short'] += 1
            continue
        ut['n'] += 1
        for k in UTT_STATS:
            add_values(ut[k], [u[k]])
            if UTT_KEEP_RAW and not np.isnan(u[k]):
                raw.setdefault(k, []).append(float(u[k]))

    dst = stats.setdefault('distance', {'per_file': {}, 'calibration': None})
    dst['per_file'][path] = info['distance']

    d = stats['decl']
    res = d['resid']
    used = rising = short = 0
    for s in sorted(set(r['sentence'] for r in rows)):
        members = [r for r in rows if r['sentence'] == s]
        pts = [(r['sent_pos'], r['pitch_med']) for r in members
               if not r['suspect'] and not np.isnan(r['pitch_med'])]
        if len(pts) < MIN_SENTENCE_WORDS:
            short += 1
            continue
        slope, intercept, _ = fit_sentence(members, 'pitch_med')
        if slope > 0.0:
            rising += 1     # question with a rising terminal, different shape
            continue
        used += 1
        d['slope_sum'] += slope
        d['intercept_sum'] += intercept
        for x, y in pts:
            res['n'] += 1
            res['Sx'] += x
            res['Sy'] += y
            res['Sxx'] += x * x
            res['Sxy'] += x * y
            res['Syy'] += y * y

    d['n_sentences'] += used
    d['n_rising'] += rising
    d['n_short'] += short

    # short-term reference: the tail of this recording replaces whatever the
    # previous one left, since "recently" means the most recent session
    if clean:
        last_end = max(r['end'] for r in clean)
        tail = [r for r in clean if r['end'] >= last_end - SHORT_TERM_WINDOW_S]
        stats['recent'] = {
            'source': path,
            'window_s': SHORT_TERM_WINDOW_S,
            'values': [[float(r['end']), float(r['pitch_med']),
                        float(r['int_p90']), float(r['dur'])]
                       for r in tail if not np.isnan(r['pitch_med'])],
        }

    stats['files'].append(path)
    return used, rising, short


def calibrate_distance(paths, same_distance_paths=()):
    """Fit level against a distance proxy over recordings made at deliberate
    distances, and decide whether the fit is usable.

    paths are ordered nearest first. Each is measured with
    prosody_core.distance_proxy. For each candidate proxy the steps between
    adjacent distances must all share a sign and the smallest must be at least
    DISTANCE_MIN_STEP_RATIO of the largest, or the proxy is detecting one
    boundary rather than measuring a scale. On top of that, the span the proxy
    covers across the distances is compared against its spread across
    recordings all made at one distance: if a proxy varies more with who is
    talking than with where they are standing, using it to estimate a gain
    offset injects more error than it removes.
    """
    from prosody_core import distance_proxy
    import parselmouth

    obs = {p: distance_proxy(parselmouth.Sound(p)) for p in paths}
    same = {p: distance_proxy(parselmouth.Sound(p)) for p in same_distance_paths}
    levels = np.array([obs[p]['level_db'] for p in paths])

    out = {'files': list(paths), 'observations': obs,
           'same_distance': same, 'candidates': {}, 'usable': None,
           'chosen': None}
    for key in ('mod_ratio', 'dyn_range'):
        v = np.array([obs[p][key] for p in paths])
        steps = np.diff(v)
        signs_agree = bool(np.all(steps > 0) or np.all(steps < 0))
        ratio = (float(np.abs(steps).min() / np.abs(steps).max())
                 if np.abs(steps).max() > 0 else 0.0)
        span = float(np.abs(v[-1] - v[0]))
        sv = [s[key] for s in same.values() if not np.isnan(s[key])]
        enough = len(sv) >= DISTANCE_MIN_SAME_FILES
        same_range = float(max(sv) - min(sv)) if len(sv) > 1 else float('nan')
        slope, icept = (np.polyfit(v, levels, 1) if len(set(v)) > 1 else (0.0, 0.0))
        resid = levels - (slope * v + icept)
        ok = (signs_agree and ratio >= DISTANCE_MIN_STEP_RATIO
              and enough and span > 2.0 * same_range)
        out['candidates'][key] = {
            'values': v.tolist(), 'steps': steps.tolist(),
            'monotonic': signs_agree, 'min_step_ratio': ratio,
            'span': span, 'same_distance_range': same_range,
            'n_same_distance': len(sv), 'enough_same_distance': bool(enough),
            'slope_db_per_unit': float(slope), 'intercept': float(icept),
            'residual_db': resid.tolist(),
            'implied_db_spread_at_one_distance': (
                float(abs(slope) * same_range) if not np.isnan(same_range)
                else float('nan')),
            'usable': bool(ok),
        }
    usable = [k for k, c in out['candidates'].items() if c['usable']]
    out['usable'] = bool(usable)
    out['chosen'] = usable[0] if usable else None
    out['levels_db'] = levels.tolist()
    return out


# --- reading the baseline back out -------------------------------------------
def summary(stats):
    """Turn the running sums into the numbers normalize.py actually uses."""
    pitch_mu, pitch_sd = mean_sd_from(stats['pitch'])
    loud_mu, loud_sd = mean_sd_from(stats['loud'])
    dur_mu, dur_sd = mean_sd_from(stats['dur'])

    d = stats['decl']
    n_sent = d['n_sentences']
    if n_sent:
        slope = d['slope_sum'] / n_sent
        intercept = d['intercept_sum'] / n_sent
        offset = intercept - pitch_mu
    else:
        slope = intercept = offset = float('nan')

    # residual sd against the averaged line, from the sufficient statistics
    res = d['resid']
    n = res['n']
    if n_sent and n > 1:
        c, b = intercept, slope
        sum_r = res['Sy'] - n * c - b * res['Sx']
        sum_r2 = (res['Syy'] + n * c * c + b * b * res['Sxx']
                  - 2 * c * res['Sy'] - 2 * b * res['Sxy'] + 2 * b * c * res['Sx'])
        var = (sum_r2 - sum_r * sum_r / n) / (n - 1)
        resid_sd = float(np.sqrt(max(var, 0.0)))
    else:
        resid_sd = float('nan')

    rec = stats.get('recent') or {'source': None, 'values': [],
                                  'window_s': SHORT_TERM_WINDOW_S}
    vals = rec.get('values') or []
    st_pitch_mu, st_pitch_sd = mean_sd_from(_acc([v[1] for v in vals]))
    st_loud_mu, st_loud_sd = mean_sd_from(_acc([v[2] for v in vals]))
    st_dur_mu, st_dur_sd = mean_sd_from(_acc([v[3] for v in vals]))

    # emphatic pause: cap from the within-sentence gap distribution, then the
    # mean and spread of the per-word capped sum
    pz = stats.get('pause') or {'within_nonzero': [], 'word_components': [],
                                'word_zeros': 0}
    wn = pz.get('within_nonzero') or []
    cap = float(np.percentile(wn, EMPHATIC_CAP_PCT)) if wn else float('nan')
    if np.isnan(cap):
        pause_mu = pause_sd = float('nan')
        pause_n = 0
    else:
        pause_vals = [min(b, cap) + min(a, cap)
                      for b, a in pz.get('word_components', [])]
        pause_vals += [0.0] * int(pz.get('word_zeros', 0))
        pause_mu, pause_sd = mean_sd_from(_acc(pause_vals))
        pause_n = len(pause_vals)

    # duration by position class, as offsets from the overall long-term mean
    dp = stats.get('dur_pos') or {}
    dur_pos = {}
    for c in POSITION_CLASSES:
        m, s = mean_sd_from(dp.get(c) or {'n': 0, 'sum': 0.0, 'sumsq': 0.0})
        dur_pos[c] = {'n': (dp.get(c) or {}).get('n', 0), 'mean': m, 'sd': s,
                      'offset': m - dur_mu if not np.isnan(m) else float('nan')}
    # residual spread after the three-level model, pooled
    num = sum((dur_pos[c]['n'] - 1) * dur_pos[c]['sd'] ** 2
              for c in POSITION_CLASSES
              if dur_pos[c]['n'] > 1 and not np.isnan(dur_pos[c]['sd']))
    den = sum(dur_pos[c]['n'] - 1 for c in POSITION_CLASSES if dur_pos[c]['n'] > 1)
    dur_resid_sd = float(np.sqrt(num / den)) if den else float('nan')

    ut = stats.get('utt') or {'n': 0, 'n_short': 0}
    utt = {'n': ut.get('n', 0), 'n_short': ut.get('n_short', 0),
           'min_words': UTT_MIN_WORDS}
    raw = ut.get('raw') or {}
    for k in UTT_STATS:
        m, s = mean_sd_from(ut.get(k) or {'n': 0, 'sum': 0.0, 'sumsq': 0.0})
        v = sorted(raw.get(k) or [])
        utt[k] = {'mean': m, 'sd': s, 'values': v,
                  'percentiles': {str(p): float(np.percentile(v, p)) for p in
                                  (0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100)}
                  if len(v) > 1 else {}}

    return {
        'files': stats['files'],
        'n_words': stats['pitch']['n'],
        'utt': utt,
        'distance': stats.get('distance') or {'per_file': {}, 'calibration': None},
        'pause_cap_s': cap,
        'pause_mu': pause_mu, 'pause_sd': pause_sd, 'pause_n': pause_n,
        'pause_within_n': len(wn),
        'dur_pos': dur_pos, 'dur_resid_sd': dur_resid_sd,
        'pitch_mu': pitch_mu, 'pitch_sd': pitch_sd,
        'loud_mu': loud_mu, 'loud_sd': loud_sd,
        'dur_mu': dur_mu, 'dur_sd': dur_sd,
        'st_source': rec.get('source'),
        'st_window_s': rec.get('window_s', SHORT_TERM_WINDOW_S),
        'st_n': len(vals),
        'st_pitch_mu': st_pitch_mu, 'st_pitch_sd': st_pitch_sd,
        'st_loud_mu': st_loud_mu, 'st_loud_sd': st_loud_sd,
        'st_dur_mu': st_dur_mu, 'st_dur_sd': st_dur_sd,
        'decl_slope': slope,
        'decl_intercept': intercept,
        'decl_offset': offset,
        'decl_n_sentences': n_sent,
        'decl_n_rising': d['n_rising'],
        'decl_n_short': d['n_short'],
        'decl_resid_sd': resid_sd,
        'decl_resid_n': n,
    }


def expected_pitch(b, sent_pos):
    """Stored expected pitch at a position in a sentence."""
    return b['pitch_mu'] + b['decl_offset'] + b['decl_slope'] * sent_pos


def print_summary(speaker, b):
    print(f"speaker: {speaker}")
    print(f"LONG-TERM reference, {len(b['files'])} file(s), "
          f"{b['n_words']} non-suspect words:")
    for f in b['files']:
        print(f"  {f}")
    print(f"  pitch_med   mean {b['pitch_mu']:7.1f} Hz   sd {b['pitch_sd']:6.1f} Hz")
    print(f"  int_p90     mean {b['loud_mu']:7.1f} dB   sd {b['loud_sd']:6.1f} dB")
    print(f"  dur         mean {b['dur_mu']:7.2f} s    sd {b['dur_sd']:6.2f} s")
    print(f"  emphatic pause cap {b['pause_cap_s']:.2f} s "
          f"(p{EMPHATIC_CAP_PCT:.0f} of {b['pause_within_n']} non-zero "
          f"within-sentence gaps); capped per-word pause mean "
          f"{b['pause_mu']:.3f} s sd {b['pause_sd']:.3f} s over {b['pause_n']} words")
    dp = b['dur_pos']
    print("  duration by position: " + "   ".join(
        f"{c} n={dp[c]['n']} mean {dp[c]['mean']:.3f} s "
        f"({dp[c]['offset']:+.3f})" for c in POSITION_CLASSES)
        + f"   residual sd {b['dur_resid_sd']:.3f} s")
    u = b['utt']
    print(f"UTTERANCE-LEVEL reference, {u['n']} utterance(s) of at least "
          f"{u['min_words']} measurable words, {u['n_short']} too short to use:")
    print(f"  rate        mean {u['rate']['mean']:7.2f} w/s  sd {u['rate']['sd']:6.2f}")
    print(f"  loudness    mean {u['loud_mean']['mean']:7.1f} dB   "
          f"sd {u['loud_mean']['sd']:6.1f}   (absolute, record only)")
    print(f"  loud vs session mean {u['loud_rel']['mean']:+5.2f} dB   "
          f"sd {u['loud_rel']['sd']:6.2f}   (rendered)")
    print(f"  loud spread mean {u['loud_sd']['mean']:7.2f} dB   "
          f"sd {u['loud_sd']['sd']:6.2f}")
    print(f"  pitch spread mean {u['pitch_sd']['mean']:6.2f} Hz   "
          f"sd {u['pitch_sd']['sd']:6.2f}")
    cal = (b.get('distance') or {}).get('calibration')
    if cal:
        print(f"DISTANCE calibration from {len(cal['files'])} recording(s): "
              f"{'USABLE, ' + cal['chosen'] if cal['usable'] else 'NOT USABLE'}")
        print(f"  levels {', '.join(f'{v:.2f}' for v in cal['levels_db'])} dB "
              f"over {', '.join(cal['files'])}")
        for k, c in cal['candidates'].items():
            print(f"  {k}: {', '.join(f'{v:.2f}' for v in c['values'])}   "
                  f"{'monotonic' if c['monotonic'] else 'NOT monotonic'}, "
                  f"smallest step {c['min_step_ratio']*100:.0f}% of largest, "
                  f"span {c['span']:.2f} against {c['same_distance_range']:.2f} "
                  f"across {c.get('n_same_distance', 0)} same-distance "
                  f"recording(s)"
                  + ("" if c.get('enough_same_distance', True) else
                     f", fewer than {DISTANCE_MIN_SAME_FILES} so the span test "
                     f"cannot be trusted")
                  + f"   -> {'usable' if c['usable'] else 'unusable'}"
                  + (f" (would imply {c['implied_db_spread_at_one_distance']:.1f} dB "
                     f"of gain spread among recordings at one distance)"
                     if not np.isnan(c['implied_db_spread_at_one_distance'])
                     else ""))
    else:
        print("DISTANCE calibration: none stored, "
              "loudness falls back to the session-relative comparison")
    print(f"SHORT-TERM reference, last {b['st_window_s']:.0f} s of "
          f"{b['st_source']}, {b['st_n']} non-suspect words:")
    print(f"  pitch_med   mean {b['st_pitch_mu']:7.1f} Hz   sd {b['st_pitch_sd']:6.1f} Hz")
    print(f"  int_p90     mean {b['st_loud_mu']:7.1f} dB   sd {b['st_loud_sd']:6.1f} dB")
    print(f"  dur         mean {b['st_dur_mu']:7.2f} s    sd {b['st_dur_sd']:6.2f} s")
    print("expected pitch decline across a sentence:")
    print(f"  slope {b['decl_slope']:+.1f} Hz from sent_pos 0 to 1"
          f"   intercept offset {b['decl_offset']:+.1f} Hz"
          f"   (absolute intercept {b['decl_intercept']:.1f} Hz)")
    print(f"  residual sd {b['decl_resid_sd']:.1f} Hz over {b['decl_resid_n']} words")
    print(f"  sentences used {b['decl_n_sentences']}"
          f"   excluded as rising {b['decl_n_rising']}"
          f"   excluded as shorter than {MIN_SENTENCE_WORDS} words {b['decl_n_short']}")


# --- commands ----------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 1
    cmd, speaker = sys.argv[1], sys.argv[2]
    store = load()

    if cmd == "add":
        paths = sys.argv[3:]
        if not paths:
            print("add needs at least one wav path")
            return 1
        stats = store.get(speaker) or empty_stats()
        for p in paths:
            used, rising, short = accumulate(stats, p)
            print(f"{p}: {used} sentence(s) used, {rising} excluded as rising, "
                  f"{short} excluded as shorter than {MIN_SENTENCE_WORDS} words")
        store[speaker] = stats
        save(store)
        print(f"written to {BASELINE_PATH}")
        print()
        print_summary(speaker, summary(stats))
        return 0

    if cmd == "calibrate":
        args = sys.argv[3:]
        cut = args.index('--same') if '--same' in args else len(args)
        paths, extra = args[:cut], args[cut + 1:]
        if len(paths) < 3:
            print("calibrate needs the distance recordings, nearest first, "
                  "optionally followed by --same and recordings known to have "
                  "been made at one ordinary distance")
            return 1
        stats = store.get(speaker) or empty_stats()
        same = list((stats.get('distance') or {}).get('per_file') or {})
        same += [p for p in extra if p not in same]
        cal = calibrate_distance(paths, same)
        stats.setdefault('distance', {'per_file': {}, 'calibration': None})
        stats['distance']['calibration'] = cal
        store[speaker] = stats
        save(store)
        print(f"written to {BASELINE_PATH}")
        print()
        print_summary(speaker, summary(stats))
        return 0

    if cmd == "show":
        if speaker not in store:
            print(f"no baseline for speaker '{speaker}' in {BASELINE_PATH}")
            return 1
        print_summary(speaker, summary(store[speaker]))
        return 0

    if cmd == "reset":
        if speaker in store:
            del store[speaker]
            save(store)
            print(f"baseline for speaker '{speaker}' reset")
        else:
            print(f"no baseline for speaker '{speaker}' to reset")
        return 0

    print(__doc__.strip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
