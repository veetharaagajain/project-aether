"""Persistent per-speaker prosody baseline.

Everything is stored as running sums, so adding another recording later
updates the baseline without reprocessing the recordings already in it.
Suspect rows never contribute to anything here.

usage:
  baseline.py add <speaker> <wav> [<wav> ...]
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

    return {
        'files': stats['files'],
        'n_words': stats['pitch']['n'],
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
