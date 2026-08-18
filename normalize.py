"""Normalized views of the per-word measurements in prosody_core.

Loudness and duration are expressed against a persistent speaker baseline
from baseline.py. Pitch is expressed three ways side by side, because no one
of them is settled:

  pitch_resid_fit  per-sentence least squares, removes the sentence's own
                   slope and level, but a stressed word at either end pulls
                   the line toward itself
  pitch_resid      the stored expected decline, nothing this file shaped,
                   but anchored to the speaker's global mean so the residual
                   is dominated by how high the sentence started
  pitch_resid_lvl  the stored slope for the shape, the sentence's own mean
                   for the level

No scoring, no weighting, no combined score.

usage: normalize.py <wav> [speaker_label]
"""

import sys
from collections import Counter

import numpy as np

import baseline as bl
from prosody_core import measure


def mean_sd(values):
    """Sample mean and sd (ddof=1) over finite values only."""
    v = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float('nan')), float('nan')
    return float(v.mean()), float(v.std(ddof=1))


def z(value, mean, sd):
    if np.isnan(value) or np.isnan(sd) or sd == 0.0:
        return float('nan')
    return (value - mean) / sd


# A voice needs at least this many usable words in a recording before a
# within-file reference means anything. It is the same figure the short-term
# rolling window already refuses to work below.
MIN_REF_WORDS = 20


def per_speaker_references(rows, store, speaker_key='owner'):
    """One reference per voice in the recording, and how each was arrived at.

    Every baseline, z-score and comparison in this layer was written assuming
    one voice per file. Mixing two people into one reference makes all of them
    meaningless: the quieter speaker reads as permanently unemphatic and the
    louder one as permanently emphatic, and neither statement is about
    emphasis. So a word is scored against its own speaker or against nothing.

    Which reference a voice gets depends on what can honestly be claimed:

      stored      the recording holds one voice, so it is the person whose
                  baseline is stored. This is the assumption the whole layer
                  has always made, now stated rather than implied.
      within-file the recording holds more than one voice. Diarization numbers
                  them; it does not say which is the owner, and recognising
                  them is a separate piece of work that does not exist yet. So
                  no stored baseline may be attributed to any of them, and
                  each is scored against its own words in this recording.
      none        the voice has fewer than MIN_REF_WORDS usable words here.
                  Nothing is computed for it: no z-scores, no weight, no
                  marks. Its words render as plain text and the record says
                  why. It does not quietly borrow anyone else's reference.
    """
    ids = sorted({r.get('speaker', 1) for r in rows})
    stored_ok = bool(store.get(speaker_key)) and bool(
        store[speaker_key]['decl']['n_sentences'] or store[speaker_key]['pitch']['n'])
    single = len(ids) == 1
    out = {}
    for sid in ids:
        mine = [r for r in rows if r.get('speaker', 1) == sid]
        usable = [r for r in mine if not r['suspect']]
        if single and stored_ok:
            out[sid] = {'ref': __import__('baseline').summary(store[speaker_key]),
                        'kind': 'stored', 'stored': True, 'n_words': len(usable),
                        'why': f"one voice in the recording, scored against the "
                               f"stored baseline for '{speaker_key}'"}
        elif len(usable) >= MIN_REF_WORDS:
            out[sid] = {'ref': per_file_reference(mine), 'kind': 'within-file',
                        'stored': False, 'n_words': len(usable),
                        'why': f"{len(ids)} voices in the recording and no way "
                               f"to say which is the stored speaker, so this "
                               f"voice is scored against its own {len(usable)} "
                               f"words here"}
        else:
            out[sid] = {'ref': None, 'kind': 'none', 'stored': False,
                        'n_words': len(usable),
                        'why': f"only {len(usable)} usable word(s), fewer than "
                               f"the {MIN_REF_WORDS} a reference needs; nothing "
                               f"is scored for this voice"}
    return out


def per_file_reference(rows):
    """Fallback reference when the speaker has no stored baseline.

    The declination entries are NaN, so the two stored-curve residual
    columns come out NaN rather than silently borrowing this file's shape.
    """
    clean = [r for r in rows if not r['suspect']]
    pitch_mu, pitch_sd = mean_sd([r['pitch_med'] for r in clean])
    loud_mu, loud_sd = mean_sd([r['int_p90'] for r in clean])
    dur_mu, dur_sd = mean_sd([r['dur'] for r in clean])
    return {
        'pitch_mu': pitch_mu, 'pitch_sd': pitch_sd,
        'loud_mu': loud_mu, 'loud_sd': loud_sd,
        'dur_mu': dur_mu, 'dur_sd': dur_sd,
        'decl_slope': float('nan'), 'decl_offset': float('nan'),
        'decl_resid_sd': float('nan'),
        'st_window_s': bl.SHORT_TERM_WINDOW_S, 'st_n': 0, 'st_source': None,
        'st_pitch_mu': float('nan'), 'st_pitch_sd': float('nan'),
        'st_loud_mu': float('nan'), 'st_loud_sd': float('nan'),
        'st_dur_mu': float('nan'), 'st_dur_sd': float('nan'),
    }


# Which reference each measurement defaults to. Both columns are always
# computed and printed; these say which one anything downstream should read.
# Duration is short-term because its rolling mean swings 36 percent of its own
# value inside a single session and individual words disagree between the two
# references by up to 2.43 z. Loudness is long-term because inside a session
# the two agree to a correlation of 0.987 over a 1.59 dB spread, so the
# rolling version buys nothing until there is a recording at a different
# microphone distance. Pitch is long-term because the declination curve it
# feeds is meant to describe the voice, not the session.
DEFAULT_REFERENCE = {'pitch': 'lt', 'loud': 'lt', 'dur': 'st'}

# Which reference each cue is expressed against, and how many words a
# reference needs before it is one. Watched by provenance.py.
PROVENANCE = ('MIN_REF_WORDS', 'DEFAULT_REFERENCE')


def apply_defaults(rows):
    """Expose the chosen reference under a plain name, both still present."""
    for r in rows:
        for key in ('pitch', 'loud', 'dur'):
            r[f'{key}_z'] = r[f'{key}_z_{DEFAULT_REFERENCE[key]}']


def apply_baselines(rows, ref):
    """Long-term z-scores: this word against how this person always sounds."""
    for r in rows:
        r['pitch_z_lt'] = z(r['pitch_med'], ref['pitch_mu'], ref['pitch_sd'])
        r['loud_z_lt'] = z(r['int_p90'], ref['loud_mu'], ref['loud_sd'])
        r['dur_z_lt'] = z(r['dur'], ref['dur_mu'], ref['dur_sd'])


def apply_rolling(rows, ref, stored):
    """Short-term z-scores: this word against the last few minutes of speech.

    The reference for each word is the non-suspect words ending inside the
    trailing window, the word itself excluded so it cannot shift the thing it
    is measured against. Speaking quietly at night moves every word in the
    window together, so it cancels here and survives in the long-term column.

    Early words have no window behind them. Those fall back to the stored
    short-term reference from the last session if there is one, and otherwise
    to the nearest SHORT_TERM_MIN_WORDS non-suspect words in either direction.
    """
    w = ref.get('st_window_s') or bl.SHORT_TERM_WINDOW_S
    ends = np.array([r['end'] for r in rows])
    ok = np.array([not r['suspect'] for r in rows])
    cols = {'pitch': np.array([r['pitch_med'] for r in rows]),
            'loud': np.array([r['int_p90'] for r in rows]),
            'dur': np.array([r['dur'] for r in rows])}
    order = np.argsort(ends)

    # the rolling window only ever looks at the same voice; a reference made
    # of two people's recent speech describes neither
    spk = np.array([r.get('speaker', 1) for r in rows])

    for i, r in enumerate(rows):
        m = ok & (ends >= ends[i] - w) & (ends <= ends[i]) & (spk == spk[i])
        m[i] = False
        source = 'window'
        if int(m.sum()) < bl.SHORT_TERM_MIN_WORDS:
            if stored and ref.get('st_n', 0) >= bl.SHORT_TERM_MIN_WORDS:
                source = 'stored'
            else:
                near = [j for j in order[np.argsort(np.abs(ends[order] - ends[i]))]
                        if ok[j] and j != i
                        and spk[j] == spk[i]][:bl.SHORT_TERM_MIN_WORDS]
                m = np.zeros(len(rows), dtype=bool)
                m[near] = True
                source = 'nearest'
        r['st_source'] = source
        r['st_n'] = int(m.sum()) if source != 'stored' else ref['st_n']
        for key, colname, val in (('pitch', 'pitch_z_st', r['pitch_med']),
                                  ('loud', 'loud_z_st', r['int_p90']),
                                  ('dur', 'dur_z_st', r['dur'])):
            if source == 'stored':
                mu, sd = ref[f'st_{key}_mu'], ref[f'st_{key}_sd']
            else:
                mu, sd = mean_sd(cols[key][m])
            r[colname] = z(val, mu, sd)
            r[f'{key}_st_mu'] = mu


def apply_fitted_decline(rows, resid_sd):
    """Approach one: a line fitted to the sentence being measured.

    Returns the residual sd actually used, which is the passed-in stored one
    when there is a baseline and this file's own otherwise.
    """
    for s in sorted(set(r['sentence'] for r in rows)):
        members = [r for r in rows if r['sentence'] == s]
        slope, intercept, _ = bl.fit_sentence(members, 'pitch_med')
        for r in members:
            r['pitch_resid_fit'] = r['pitch_med'] - (slope * r['sent_pos'] + intercept)
    if np.isnan(resid_sd):
        clean = [r for r in rows if not r['suspect']]
        _, resid_sd = mean_sd([r['pitch_resid_fit'] for r in clean])
    for r in rows:
        r['pitch_resid_fit_z'] = z(r['pitch_resid_fit'], 0.0, resid_sd)
    return resid_sd


def apply_stored_decline(rows, ref):
    """Approach two: the stored curve, level taken from the speaker mean."""
    for r in rows:
        r['pitch_resid'] = r['pitch_med'] - bl.expected_pitch(ref, r['sent_pos'])
        r['pitch_resid_z'] = z(r['pitch_resid'], 0.0, ref['decl_resid_sd'])


def apply_level_decline(rows, ref):
    """Approach three: stored slope for the shape, this sentence's own mean
    pitch at its own mean sent_pos for the level.

    The mean over a sentence's non-suspect words is a low-leverage statistic,
    so one emphasised word cannot tilt it the way it can tilt a slope.
    """
    slope = ref['decl_slope']
    for s in sorted(set(r['sentence'] for r in rows)):
        members = [r for r in rows if r['sentence'] == s]
        clean = [r for r in members
                 if not r['suspect'] and not np.isnan(r['pitch_med'])]
        if clean:
            mx = float(np.mean([r['sent_pos'] for r in clean]))
            my = float(np.mean([r['pitch_med'] for r in clean]))
        else:
            mx = my = float('nan')
        for r in members:
            r['pitch_resid_lvl'] = r['pitch_med'] - (my + slope * (r['sent_pos'] - mx))
            r['pitch_resid_lvl_z'] = z(r['pitch_resid_lvl'], 0.0, ref['decl_resid_sd'])


def question_threshold(ref, span, stored):
    """How far the last word must sit above the first before the sentence
    looks like a question: the size of the decline expected over the span
    actually measured, plus one stored residual sd."""
    if not stored or np.isnan(ref['decl_slope']):
        return ref['pitch_sd']          # fallback, the old looser rule
    return -ref['decl_slope'] * span + ref['decl_resid_sd']


def flag_questions(rows, ref, stored):
    """Pitch is supposed to fall across a sentence. Flag the sentences where
    it rises by more than the expected fall plus a margin."""
    for s in sorted(set(r['sentence'] for r in rows)):
        members = [r for r in rows if r['sentence'] == s]
        clean = [r for r in members
                 if not r['suspect'] and not np.isnan(r['pitch_med'])]
        rising = False
        if len(clean) >= 2:
            span = clean[-1]['sent_pos'] - clean[0]['sent_pos']
            thr = question_threshold(ref, span, stored)
            rise = clean[-1]['pitch_med'] - clean[0]['pitch_med']
            rising = not np.isnan(thr) and rise > thr
        for r in members:
            r['question'] = rising


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"
    speaker = sys.argv[2] if len(sys.argv) > 2 else "owner"

    rows, _ = measure(path)

    store = bl.load()
    stored = bool(store.get(speaker)) and bool(
        store[speaker]['decl']['n_sentences'] or store[speaker]['pitch']['n'])
    ref = bl.summary(store[speaker]) if stored else per_file_reference(rows)

    apply_baselines(rows, ref)
    apply_rolling(rows, ref, stored)
    apply_defaults(rows)
    fit_sd = apply_fitted_decline(rows, ref['decl_resid_sd'])
    apply_stored_decline(rows, ref)
    apply_level_decline(rows, ref)
    flag_questions(rows, ref, stored)

    print(f"speaker: {speaker}    file: {path}")
    if stored:
        print(f"baseline: STORED, from {bl.BASELINE_PATH.name}, "
              f"{len(ref['files'])} file(s), {ref['n_words']} non-suspect words: "
              f"{', '.join(ref['files'])}")
    else:
        print(f"WARNING: no stored baseline for speaker '{speaker}' in "
              f"{bl.BASELINE_PATH.name}.")
        print("WARNING: falling back to a baseline computed from THIS file. "
              "pitch_resid and pitch_resid_lvl need a stored decline and print nan.")
        print("WARNING: the numbers below are self-referential. "
              "Run baseline.py add to fix this.")
    print("LONG-TERM reference, columns ending _lt:")
    print(f"  pitch_med   mean {ref['pitch_mu']:7.1f} Hz   sd {ref['pitch_sd']:6.1f} Hz")
    print(f"  int_p90     mean {ref['loud_mu']:7.1f} dB   sd {ref['loud_sd']:6.1f} dB")
    print(f"  dur         mean {ref['dur_mu']:7.2f} s    sd {ref['dur_sd']:6.2f} s")
    st_kinds = Counter(r['st_source'] for r in rows)
    print(f"SHORT-TERM reference, columns ending _st: rolling "
          f"{ref.get('st_window_s', bl.SHORT_TERM_WINDOW_S):.0f} s window over this "
          f"file, per word, excluding the word itself")
    print("defaults, for anything downstream reading one value per measurement: "
          f"pitch {DEFAULT_REFERENCE['pitch']}, loudness {DEFAULT_REFERENCE['loud']}, "
          f"duration {DEFAULT_REFERENCE['dur']}  (exposed as pitch_z, loud_z, dur_z; "
          "both columns still printed)")
    print(f"  window source per word: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(st_kinds.items()))}"
          f"   (stored fallback: last {ref.get('st_window_s', 0):.0f} s of "
          f"{ref.get('st_source')}, {ref.get('st_n', 0)} words)")
    if stored and not np.isnan(ref['decl_slope']):
        print(f"declination: STORED, slope {ref['decl_slope']:+.1f} Hz across a "
              f"sentence, intercept offset {ref['decl_offset']:+.1f} Hz, "
              f"residual sd {ref['decl_resid_sd']:.1f} Hz")
        print(f"  fitted over {ref['decl_n_sentences']} sentence(s), "
              f"{ref['decl_n_rising']} excluded as rising, "
              f"{ref['decl_n_short']} excluded as too short")
    else:
        print("declination: none stored")
    print(f"pitch_resid_fit_z uses residual sd {fit_sd:.1f} Hz")
    thr_full = question_threshold(ref, 1.0, stored)
    print(f"question flag: last non-suspect word more than {thr_full:.1f} Hz above "
          f"the first, over a full sentence")
    if stored and not np.isnan(ref['decl_slope']):
        print(f"  = expected drop {-ref['decl_slope']:.1f} Hz + residual sd "
              f"{ref['decl_resid_sd']:.1f} Hz, scaled to the span actually measured")

    print("residual columns abbreviated to keep the table pasteable: "
          "r_fit = pitch_resid_fit (per-sentence fit), r_cur = pitch_resid "
          "(stored curve), r_lvl = pitch_resid_lvl (stored slope, sentence level)")
    print(f"{'word':<12}{'start':>7}{'dur':>6}{'pitch_med':>10}{'int_p90':>8}"
          f"{'sent_pos':>9}"
          f"{'pitch_z_lt':>11}{'pitch_z_st':>11}{'loud_z_lt':>10}{'loud_z_st':>10}"
          f"{'dur_z_lt':>9}{'dur_z_st':>9}"
          f"{'r_fit':>8}{'r_fit_z':>9}{'r_cur':>8}{'r_cur_z':>9}"
          f"{'r_lvl':>8}{'r_lvl_z':>9}"
          f"{'question':>9}{'suspect':>8}")
    for r in rows:
        print(f"{r['word']:<12}{r['start']:7.2f}{r['dur']:6.2f}"
              f"{r['pitch_med']:10.1f}{r['int_p90']:8.1f}{r['sent_pos']:9.2f}"
              f"{r['pitch_z_lt']:11.2f}{r['pitch_z_st']:11.2f}"
              f"{r['loud_z_lt']:10.2f}{r['loud_z_st']:10.2f}"
              f"{r['dur_z_lt']:9.2f}{r['dur_z_st']:9.2f}"
              f"{r['pitch_resid_fit']:8.1f}{r['pitch_resid_fit_z']:9.2f}"
              f"{r['pitch_resid']:8.1f}{r['pitch_resid_z']:9.2f}"
              f"{r['pitch_resid_lvl']:8.1f}{r['pitch_resid_lvl_z']:9.2f}"
              f"{'Q' if r['question'] else '':>9}{'YES' if r['suspect'] else '':>8}")


if __name__ == "__main__":
    main()
