"""Emphasis weight per word: the thing the input layer exists to produce.

Four measured cues, each already expressed against a reference that the word
itself did not help define, combined into one number per word.

  loudness  90th-percentile intensity against the LONG-TERM speaker baseline.
            Separated all five deliberately stressed words in stress1 from
            their flat counterparts, by 4.3 to 12.3 dB, and never got the sign
            wrong. The strongest cue by a clear margin, so the largest weight.
  duration  trimmed span against the SHORT-TERM rolling reference, because a
            speaker's rate drifts within a session by 36 percent of its own
            mean. Correct on four of the five stressed words; the one failure
            was a tokenisation mismatch, not a measurement error.
  pitch     median pitch against the long-term baseline with the stored
            declination removed, so a word late in a sentence is not penalised
            for the fall every sentence has. Correct on three of five, but
            strong on different words than loudness rather than being a
            noisier copy of it, so it earns a small weight rather than none.
  pause     silence either side of the word. Never tested as an emphasis cue,
            so the smallest weight, and score.py reports separately whether it
            changed anything.

usage: score.py <wav> [speaker_label]
"""

import sys

import numpy as np

import baseline as bl
import normalize as nz
from prosody_core import measure

# The one place to edit. Weights are relative; they are renormalised over
# whichever cues a given word actually has, so a word with no pitch is scored
# on the other three rather than penalised for the gap.
WEIGHTS = {
    'loud': 0.45,
    'dur': 0.25,
    'pitch': 0.20,
    'pause': 0.10,
}

CUE_COLUMN = {
    'loud': 'loud_z',            # default reference, currently long-term
    'dur': 'dur_resid_z',        # short-term level, stored positional shape
    'pitch': 'pitch_resid_z',
    'pause': 'pause_z',
}


def apply_pause(rows, ref, total, only=None):
    """The emphatic pause: a short beat around a word, and nothing else.

    Three things are excluded before anything is measured. The gap before the
    first word and after the last are the edges of the recording. A gap the
    boundary detector marked as a sentence boundary is utterance structure.
    What survives is capped at the stored cap, which is a percentile of the
    within-sentence gaps in the baseline recordings rather than a round
    number, because everything longer is doing the other job that
    silence_log records.

    The reference is the stored one, so this is no longer scored against the
    file's own distribution.
    """
    cap = ref.get('pause_cap_s', float('nan'))
    mu, sd = ref.get('pause_mu', float('nan')), ref.get('pause_sd', float('nan'))
    for i, r in enumerate(rows):
        # the gaps always come from the whole recording, because the silence
        # around a word is real silence whoever else was or was not talking;
        # only which words get scored here is restricted to one voice
        if only is not None and i not in only:
            continue
        b, a = bl.eligible_gaps(rows, i, total)
        b_cap = min(b, cap) if not np.isnan(cap) else b
        a_cap = min(a, cap) if not np.isnan(cap) else a
        r['pause_before'] = b
        r['pause_after'] = a
        r['pause_raw'] = b_cap + a_cap
        r['pause_z'] = nz.z(r['pause_raw'], mu, sd)
    return cap, mu, sd


def apply_duration_position(rows, ref):
    """Duration against what this word's position in the sentence predicts.

    Phrase-final lengthening is a property of English, not of emphasis: the
    last word of a sentence runs about 20 to 50 percent longer than a middle
    one whether or not it was stressed. Without this correction the duration
    cue puts a sentence-final word near the top of a flat reading, which is
    what it did in neutral1. The level comes from the short-term reference,
    since rate drifts within a session; the positional offsets and the
    residual spread come from the long-term baseline.
    """
    dp = ref.get('dur_pos') or {}
    resid_sd = ref.get('dur_resid_sd', float('nan'))
    for r in rows:
        cls = bl.position_class(r)
        off = (dp.get(cls) or {}).get('offset', float('nan'))
        level = r.get('dur_st_mu', float('nan'))
        expected = level + off
        r['dur_pos_class'] = cls
        r['dur_expected'] = expected
        r['dur_resid'] = r['dur'] - expected
        r['dur_resid_z'] = nz.z(r['dur_resid'], 0.0, resid_sd)


def apply_weight(rows):
    """Weighted sum of whichever cues the word has, in z units."""
    for r in rows:
        num = den = 0.0
        present = []
        for cue, col in CUE_COLUMN.items():
            v = r.get(col, float('nan'))
            if v is None or np.isnan(v):
                continue
            num += WEIGHTS[cue] * v
            den += WEIGHTS[cue]
            present.append(cue)
        r['weight'] = num / den if den else float('nan')
        r['cues_used'] = ','.join(present)


def live_cues(rows):
    """Which cues actually contributed anywhere in this file.

    Not from cues_used initials: 'pitch' and 'pause' share a letter, and a
    check written on the initials reported three of four cues on every file
    including ones where all four were live.
    """
    return [k for k, col in CUE_COLUMN.items()
            if any(not np.isnan(r.get(col, float('nan'))) for r in rows)]


def score_file(path, speaker='owner'):
    """Score every word against its own speaker's reference.

    This used to take one reference for the whole file, which is right for a
    recording of one person and wrong for any other. On a two-host podcast it
    scored both hosts against the stored baseline of whoever owns the machine.
    """
    rows, info = measure(path)
    store = bl.load()
    refs = nz.per_speaker_references(rows, store, speaker)
    pause = (float('nan'),) * 3

    for sid, entry in refs.items():
        idx = {i for i, r in enumerate(rows) if r.get('speaker', 1) == sid}
        mine = [rows[i] for i in sorted(idx)]
        ref = entry['ref']
        if ref is None:
            for r in mine:
                r['weight'] = float('nan')
                r['cues_used'] = ''
                r['dur_pos_class'] = bl.position_class(r)
                r['scored'] = False
            continue
        nz.apply_baselines(mine, ref)
        nz.apply_rolling(mine, ref, entry['stored'])
        nz.apply_defaults(mine)
        nz.apply_stored_decline(mine, ref)
        apply_duration_position(mine, ref)
        pause = apply_pause(rows, ref, info['total'], only=idx)
        apply_weight(mine)
        for r in mine:
            r['scored'] = True

    main_id = max(refs, key=lambda s: refs[s]['n_words'])
    ref = refs[main_id]['ref'] or nz.per_file_reference(rows)
    return rows, info, ref, refs[main_id]['stored'], pause, refs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"
    speaker = sys.argv[2] if len(sys.argv) > 2 else "owner"
    rows, info, ref, stored, (pause_cap, pause_mu, pause_sd), refs = \
        score_file(path, speaker)

    print(f"speaker: {speaker}    file: {path}")
    for sid, entry in sorted(refs.items()):
        print(f"  speaker {sid}: {entry['n_words']} usable word(s), "
              f"reference {entry['kind']} -- {entry['why']}")
    live = live_cues(rows)
    if len(live) < len(CUE_COLUMN):
        print("WARNING: only these cues are live on this file: "
              + ", ".join(live)
              + ". The rest need a stored baseline (declination, positional "
                "duration offsets, the emphatic pause cap) and a within-file "
                "reference has none of them, so the weight is a weighted sum "
                "of fewer things than it looks like.")
    if stored:
        print(f"baseline: STORED, {len(ref['files'])} file(s), "
              f"{ref['n_words']} non-suspect words")
    else:
        print("WARNING: no stored baseline, every reference is this file's own")
    print("weights: " + ", ".join(f"{k} {v}" for k, v in WEIGHTS.items())
          + "  (renormalised per word over the cues present)")
    dp = ref.get('dur_pos') or {}
    print(f"cue columns: loud_z = loud_z_{nz.DEFAULT_REFERENCE['loud']}; "
          f"dur_z = duration against what its sentence position predicts "
          f"(offsets initial {dp.get('initial', {}).get('offset', float('nan')):+.3f}, "
          f"middle {dp.get('middle', {}).get('offset', float('nan')):+.3f}, "
          f"final {dp.get('final', {}).get('offset', float('nan')):+.3f} s); "
          f"pitch_z = pitch_resid_z (declination removed)")
    print(f"emphatic pause: capped at {pause_cap:.2f} s, sentence-boundary and "
          f"file-edge gaps excluded, scored against the stored reference "
          f"(mean {pause_mu:.3f} s, sd {pause_sd:.3f} s)")
    print(f"{len(rows)} words in {info['n_sentences']} sentence(s); "
          f"{len(info['silence_log'])} silence(s) logged separately")

    print(f"{'word':<14}{'start':>7}{'sent':>5}{'pos':>6}{'cls':>8}"
          f"{'loud_z':>8}{'dur_z':>8}{'pitch_z':>8}{'pause_s':>9}{'pause_z':>8}"
          f"{'WEIGHT':>8}{'cues':>6}  suspect")
    for r in rows:
        print(f"{r['word'] + r['punct']:<14}{r['start']:7.2f}"
              f"{r['sentence']:5d}{r['sent_pos']:6.2f}{r['dur_pos_class']:>8}"
              f"{r['loud_z']:8.2f}{r['dur_resid_z']:8.2f}{r['pitch_resid_z']:8.2f}"
              f"{r['pause_raw']:9.2f}{r['pause_z']:8.2f}"
              f"{r['weight']:8.2f}{r['cues_used']:>6}"
              f"  {'+'.join(r['suspect_reasons'])}")

    print()
    print("SILENCE LOG, not an emphasis cue and not fed into the weight above")
    print(f"{'start':>8}{'length':>8}{'after':>14}{'filler':>8}{'pitch':>8}"
          f"{'fell':>6}{'boundary':>10}  next")
    for s in info['silence_log']:
        fell = '-' if s['pitch_fell'] is None else ('yes' if s['pitch_fell'] else 'no')
        print(f"{s['start']:8.2f}{s['length']:8.2f}"
              f"{s['after_word'] + s['after_punct']:>14}"
              f"{'yes' if s['after_filler'] else '':>8}"
              f"{s['pitch_before']:8.1f}{fell:>6}"
              f"{(s['boundary_cue'] or ''):>10}  {s['next_word']}")


if __name__ == "__main__":
    main()
