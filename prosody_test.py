import sys

from prosody_core import (LONG_WORD_S, MIN_SILENCE_S, QUIET_MARGIN_DB,
                          SENTENCE_GAP_ALWAYS_S, SENTENCE_GAP_MIN_S,
                          SILENCE_DROP_DB, measure)

path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"

rows, info = measure(path)
p90_median = info['p90_median']
n_sentences = len(set(r['sentence'] for r in rows))

print(f"median speech energy {info['median_speech_db']:.1f} dB, "
      f"noise floor {info['median_noise_db']:.1f} dB")
print(f"silence threshold {info['silence_threshold_db']:.1f} dB, set by the "
      f"{info['threshold_set_by']} "
      f"(speech drop would give {info['threshold_from_speech_db']:.1f}, "
      f"noise margin {info['threshold_from_noise_db']:.1f}), "
      f"minimum silence {MIN_SILENCE_S * 1000:.0f} ms")
print(f"pitch range {info['pitch_floor']:.1f} to {info['pitch_ceiling']:.1f} Hz, "
      f"{info['range_source'].upper()} from {info['range_frames']} probe frames: "
      f"{info['probe_voiced_all']} voiced, less {info['probe_outside_words']} "
      f"outside non-suspect words, less {info['probe_octave_removed']} rejected "
      f"as octave errors")
print(f"octave rejection on the measurement pass: {info['octave_removed']} of "
      f"{info['voiced_raw']} voiced frames dropped across {info['octave_words']} "
      f"word(s), {info['octave_multipass_words']} needed more than one pass")
print(f"{len(info['silences'])} silence(s) found, "
      f"{len(rows)} words in {n_sentences} sentence(s)")
print(f"gaps at or over {SENTENCE_GAP_MIN_S:.2f} s: {info['gap_candidates']} "
      f"candidate(s), {info['gap_boundaries']} passed the pitch-fall test, "
      f"{info['gap_rejected']} rejected as hesitations "
      f"(over {SENTENCE_GAP_ALWAYS_S:.1f} s is a boundary regardless of pitch)")
print(f"boundary cues: {info['both']} both agreed, {info['gap_only']} gap only, "
      f"{info['punct_only']} punctuation only")
print(f"median p90 intensity across words: {p90_median:.1f} dB "
      f"(suspect below {p90_median - QUIET_MARGIN_DB:.1f} dB, "
      f"or long: over {LONG_WORD_S:.1f} s barely voiced)")
print(f"{'word':<12}{'pn':>4}{'o_start':>8}{'o_end':>8}{'start':>7}{'end':>7}{'dur':>6}"
      f"{'trim_s':>8}{'pitch_med':>10}{'pitch_mean':>11}{'int_p90':>8}{'int_mean':>9}"
      f"{'gap_before':>11}{'gap_after':>10}{'sent_pos':>9}{'cue':>7}  {'suspect'}")
for r in rows:
    print(f"{r['word']:<12}{r['punct']:>4}{r['orig_start']:8.2f}{r['orig_end']:8.2f}"
          f"{r['start']:7.2f}{r['end']:7.2f}{r['dur']:6.2f}{r['trimmed']:8.2f}"
          f"{r['pitch_med']:10.1f}{r['pitch_mean']:11.1f}"
          f"{r['int_p90']:8.1f}{r['int_mean']:9.1f}"
          f"{r['gap_before']:11.2f}{r['gap_after']:10.2f}{r['sent_pos']:9.2f}"
          f"{r['boundary_cue']:>7}  {'+'.join(r['suspect_reasons'])}")
