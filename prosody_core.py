"""Per-word prosody measurement, shared by prosody_test.py and normalize.py.

measure(path) returns (rows, info). Each row is a dict with the word, the
span the transcript reported, the span after silence has been trimmed off
its edges, pitch and intensity summaries over the trimmed span, gaps
measured from the trimmed spans, sentence membership, position within the
sentence, and the suspect flag with the reasons that fired.

Silences come from the waveform, not from the transcript. The transcript
reports every within-sentence gap as 0.00, so gaps, sentence boundaries and
the sentence-position column were all meaningless without this.

Nothing here scores, weights or normalizes anything.
"""

import numpy as np
import parselmouth
from faster_whisper import WhisperModel

PITCH_FLOOR_HZ = 60.0        # fallback range, used only when a file has too
PITCH_CEILING_HZ = 300.0     # little voiced speech to derive one from

PROBE_FLOOR_HZ = 50.0        # first pass, deliberately permissive
PROBE_CEILING_HZ = 500.0
RANGE_LOW_PCT = 5.0          # robust spread of the first pass's voiced frames
RANGE_HIGH_PCT = 95.0
RANGE_FLOOR_FACTOR = 0.75    # widened by this much before the second pass
RANGE_CEILING_FACTOR = 1.5
RANGE_MIN_FRAMES = 100       # below this, fall back to the fixed range
RANGE_ABS_FLOOR_HZ = 40.0    # sanity rails, not voice-specific tuning
RANGE_ABS_CEILING_HZ = 600.0

MIN_VOICED_FRAMES = 3        # fewer voiced frames than this -> suspect
QUIET_MARGIN_DB = 12.0       # p90 this far under the file median -> suspect
LONG_WORD_S = 1.2            # trimmed span longer than this, and barely
LONG_WORD_MIN_VOICED = 5     # voiced, is still an artifact -> suspect

SENTENCE_GAP_MIN_S = 0.60    # shorter gaps are never boundaries
SENTENCE_GAP_ALWAYS_S = 2.0  # longer gaps are boundaries whatever pitch did
PITCH_FALL_LOOKBACK = 0      # words of context for the fall; 0 = the whole run
PITCH_FALL_MIN_RATIO = 0.05  # pre-gap word must sit this far below that context

TRAILING_PUNCT = '.,!?;:…"\')]}'   # peeled off the word token into its own column
SENTENCE_FINAL_PUNCT = '.!?…'      # ...of which these claim a sentence ended

OCTAVE_FACTOR = 2.0          # frames above this multiple of the word's running
OCTAVE_MAX_PASSES = 5        # median are doubling errors, not the voice

# Words that commonly precede a silence without ending a thought. Used only to
# annotate the silence log; nothing keys off it and nothing is classified.
FILLER_WORDS = {'um', 'uh', 'erm', 'er', 'ah', 'hmm', 'mm', 'mmm', 'like',
                'so', 'well', 'and', 'but', 'i', 'the', 'a', 'you', 'know'}
SILENCE_LOG_MIN_S = 0.30     # gaps shorter than this are not worth logging

ENVELOPE_HOP_S = 0.010       # short-time energy envelope, hop and window
ENVELOPE_WINDOW_S = 0.025
SILENCE_DROP_DB = 15.0       # silence is this far below median speech energy
NOISE_MARGIN_DB = 6.0        # ...but never nearer than this to the noise floor
MIN_SILENCE_S = 0.150        # shorter dips than this are not pauses

WHISPER_MODEL = "small.en"


def energy_envelope(snd):
    """Short-time RMS of the whole file, in dB, one value per hop.

    Returns (frame centre times, level in dB). The dB reference is the
    sample scale, so only differences within a file are meaningful, which
    is all the silence threshold uses.
    """
    x = np.asarray(snd.values[0], dtype=np.float64)
    fs = snd.sampling_frequency
    hop = max(1, int(round(ENVELOPE_HOP_S * fs)))
    win = max(hop, int(round(ENVELOPE_WINDOW_S * fs)))
    n = 1 + max(0, (len(x) - win) // hop)
    if n < 1:
        return np.array([]), np.array([])
    cumsq = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.arange(n) * hop
    ends = starts + win
    rms = np.sqrt((cumsq[ends] - cumsq[starts]) / win)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    times = (starts + win / 2.0) / fs
    return times, db


def split_speech_noise_db(db):
    """Median energy of the speech frames and of the quiet frames.

    A plain median of the file will not do: these recordings are mostly
    silence, so the file median sits in the noise floor. This is a 1-D
    two-means on the dB values, which converges in a handful of passes and
    is derived entirely from the file.
    """
    if not len(db):
        return float('nan'), float('nan')
    t = float(db.mean())
    for _ in range(50):
        hi, lo = db[db > t], db[db <= t]
        if not len(hi) or not len(lo):
            m = float(np.median(db))
            return m, m
        nt = (float(np.median(hi)) + float(np.median(lo))) / 2.0
        if abs(nt - t) < 1e-6:
            break
        t = nt
    return float(np.median(db[db > t])), float(np.median(db[db <= t]))


def silence_threshold_db(speech_db, noise_db):
    """Where to put the silence line.

    A fixed drop below median speech energy is the intent, but the distance
    from speech to noise floor is not constant across recordings: it is 17 dB
    in one of the corpus files and 27 dB in another. A fixed drop therefore
    lands at very different heights above the noise, and when it lands too
    close, noise frames poke above the line, chop the silences into pieces
    shorter than MIN_SILENCE_S, and no silence is found at all. So the drop
    from speech applies unless it would come nearer than NOISE_MARGIN_DB to
    the noise floor, in which case the noise floor sets the line.
    """
    from_speech = speech_db - SILENCE_DROP_DB
    from_noise = noise_db + NOISE_MARGIN_DB
    return max(from_speech, from_noise), from_speech, from_noise


def find_silences(times, db, threshold_db):
    """Every stretch below threshold_db lasting at least MIN_SILENCE_S.

    Boundaries are taken at frame centres, which errs toward a slightly
    shorter silence and so toward trimming slightly less than the true
    amount. Under-trimming is the safer error here.
    """
    if not len(db):
        return []
    quiet = db < threshold_db
    silences = []
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(quiet) and quiet[j + 1]:
            j += 1
        if times[j] - times[i] >= MIN_SILENCE_S:
            silences.append((float(times[i]), float(times[j])))
        i = j + 1
    return silences


def trim_span(start, end, silences):
    """Shrink a reported span to its speech part.

    Only silence overlapping the leading or trailing edge is removed; a
    pause in the middle of a span is left alone, since that is a word the
    transcript merged rather than padding on the edges.
    """
    s, e = start, end
    for a, b in silences:
        if b <= s or a >= e:
            continue
        if a <= s < b:          # silence covers the start
            s = min(b, e)
        if a < e <= b:          # silence covers the end
            e = max(a, s)
    if s >= e:                  # the whole span was silence
        return start, start
    return s, e


def split_punctuation(token):
    """Separate a Whisper word token into its text and its trailing marks.

    Whisper attaches punctuation to the word, so "mean." and "say?" were
    being carried around as word text. The marks are a sentence-end claim
    that costs no silence to make, which matters for speakers who do not
    pause at their own boundaries.
    """
    t = token.strip()
    i = len(t)
    while i > 0 and t[i - 1] in TRAILING_PUNCT:
        i -= 1
    return (t[:i], t[i:]) if i else (t, '')


def ends_sentence(punct):
    return any(c in SENTENCE_FINAL_PUNCT for c in punct)


def reject_octave_errors(v):
    """Drop frames more than an octave above the running median of the word.

    Breathy and creaky speech is read at double its true pitch, and inside
    turns1 that put whole percentiles of the frame cloud at roughly twice the
    speaker: p50 96.2, p75 112.8, p90 144.3, then p95 371.4. Two populations,
    the upper one at about double the lower. Taking the median of a word's
    voiced frames and discarding anything above twice it removes the doubled
    ones, and repeating catches cases where the first median was itself
    dragged upward.

    Returns (kept frames, number removed, passes that removed something).
    """
    kept = np.asarray(v, dtype=float)
    removed = passes = 0
    for _ in range(OCTAVE_MAX_PASSES):
        if len(kept) < 2:
            break
        keep = kept <= float(np.median(kept)) * OCTAVE_FACTOR
        if keep.all():
            break
        removed += int((~keep).sum())
        passes += 1
        kept = kept[keep]
    return kept, removed, passes


def derive_pitch_range(v):
    """Two-pass pitch range, taken from the file rather than from a voice.

    A first pass at a deliberately permissive range gives a cloud of voiced
    frames. The fifth and ninety-fifth percentiles of that cloud bracket the
    speaker, and widening them gives the range for the real pass. Fixed
    limits chosen for one speaker press against both ends on another: on the
    video file the old 60 to 300 read one word at 277.5 Hz and the next at
    61.7 Hz in a passage that never leaves 90 to 140.

    v is the pooled cloud of probe frames that survived two filters: they lie
    inside a word the probe pass did not flag suspect, and they survived
    octave rejection within that word. The first keeps breath and smoke
    between words out; the second keeps doubled frames inside words out. With
    neither, turns1 derived a ceiling of 572.9 Hz for a speaker who lives
    between 90 and 140.

    Returns (floor, ceiling, source).
    """
    if len(v) < RANGE_MIN_FRAMES:
        return PITCH_FLOOR_HZ, PITCH_CEILING_HZ, 'fallback'
    lo = float(np.percentile(v, RANGE_LOW_PCT)) * RANGE_FLOOR_FACTOR
    hi = float(np.percentile(v, RANGE_HIGH_PCT)) * RANGE_CEILING_FACTOR
    lo = max(lo, RANGE_ABS_FLOOR_HZ)
    hi = min(hi, RANGE_ABS_CEILING_HZ)
    if hi <= lo:
        return PITCH_FLOOR_HZ, PITCH_CEILING_HZ, 'fallback'
    return lo, hi, 'derived'


def pitch_fell(rows, j, run_start):
    """Did the word at j end on a fall, relative to its own neighbours?

    The reference is the median pitch of the non-suspect words before it,
    by default the whole sentence so far rather than the last few. A short
    window fails: declination means the last words of a sentence are all low
    together, so a falling final word looks unremarkable beside its immediate
    neighbours and the boundary is missed. Measured against the run's own
    level the fall is obvious.

    When the word is the first of its run, which is exactly the
    discourse-marker case ("So," followed by a long pause), the reference
    comes from the words before it in the file instead. Nothing stored is
    consulted, so this works on a speaker with no baseline.
    """
    p = rows[j]['pitch_med']
    if np.isnan(p):
        return True, float('nan'), float('nan'), 'no pitch'

    def context(lo_index):
        out = []
        for k in range(j - 1, lo_index - 1, -1):
            if not rows[k]['suspect'] and not np.isnan(rows[k]['pitch_med']):
                out.append(rows[k]['pitch_med'])
                if PITCH_FALL_LOOKBACK and len(out) == PITCH_FALL_LOOKBACK:
                    break
        return out

    ref_vals, scope = context(run_start), 'run'
    if not ref_vals:
        ref_vals, scope = context(0), 'file'
    if not ref_vals:
        return True, p, float('nan'), 'no context'

    ref = float(np.median(ref_vals))
    return (ref - p) / ref > PITCH_FALL_MIN_RATIO, p, ref, scope


def assign_sentences(rows):
    """A boundary is the union of two independent cues.

    The gap cue: a gap long enough, after a word that fell in pitch. Gap
    length alone cannot do this, because in drag1 the gaps interleave: 1.29 s
    is a hesitation, 1.33, 1.41 and 1.53 are boundaries, 1.58 is a hesitation
    and 1.62 is a boundary. No threshold separates those classes.

    The punctuation cue: a full stop, question mark or exclamation mark on
    the preceding word. This costs no silence, so it reaches boundaries the
    gap cue structurally cannot - in the video file "leap." is followed by
    "This" across a 0.04 s gap.

    Either cue alone is enough. Neither vetoes the other, and which fired is
    recorded so their contributions stay separable.
    """
    candidates = gap_boundaries = rejected = 0
    punct_only = gap_only = both = 0
    decisions = []
    run_start = 0
    for i, r in enumerate(rows):
        r['boundary_cue'] = ''
        if i > 0:
            gap = r['gap_before']
            gap_fired, p, ref, scope = False, float('nan'), float('nan'), ''
            if gap >= SENTENCE_GAP_MIN_S:
                candidates += 1
                if gap >= SENTENCE_GAP_ALWAYS_S:
                    gap_fired, p, ref, scope = True, rows[i - 1]['pitch_med'], \
                        float('nan'), 'long gap'
                else:
                    gap_fired, p, ref, scope = pitch_fell(rows, i - 1, run_start)
                if gap_fired:
                    gap_boundaries += 1
                else:
                    rejected += 1

            punct_fired = ends_sentence(rows[i - 1]['punct'])
            if gap_fired or punct_fired:
                cue = 'both' if (gap_fired and punct_fired) else \
                      ('punct' if punct_fired else 'gap')
                if cue == 'both':
                    both += 1
                elif cue == 'punct':
                    punct_only += 1
                else:
                    gap_only += 1
                decisions.append({
                    'index': i, 'gap': gap, 'pre_word': rows[i - 1]['word'],
                    'pre_punct': rows[i - 1]['punct'],
                    'pre_end': rows[i - 1]['end'], 'pitch': p, 'ref': ref,
                    'scope': scope, 'cue': cue,
                })
                r['boundary_cue'] = cue
                run_start = i
        r['sentence'] = run_start
    return {
        'gap_candidates': candidates,
        'gap_boundaries': gap_boundaries,
        'gap_rejected': rejected,
        'punct_only': punct_only,
        'gap_only': gap_only,
        'both': both,
        'decisions': decisions,
    }


def build_rows(words, silences, pitch_t, pitch_hz, int_t, int_db, total):
    """Per-word measurements over silence-trimmed spans, plus gaps."""

    def voiced_pitch(start, end):
        m = (pitch_t >= start) & (pitch_t <= end)
        v = pitch_hz[m]
        return v[(v > 0) & ~np.isnan(v)]

    def valid_intensity(start, end):
        m = (int_t >= start) & (int_t <= end)
        v = int_db[m]
        return v[~np.isnan(v) & (v > -200.0)]

    rows = []
    for w in words:
        s, e = trim_span(w.start, w.end, silences)
        v_raw = voiced_pitch(s, e)
        v, n_oct, oct_passes = reject_octave_errors(v_raw)
        d = valid_intensity(s, e)
        text, punct = split_punctuation(w.word)
        rows.append({
            'word': text,
            'punct': punct,
            'pitch_frames': v,
            'n_voiced_raw': len(v_raw),
            'n_octave_removed': n_oct,
            'octave_passes': oct_passes,
            'orig_start': w.start,
            'orig_end': w.end,
            'orig_dur': w.end - w.start,
            'start': s,
            'end': e,
            'dur': e - s,
            'trimmed': (w.end - w.start) - (e - s),
            'n_voiced': len(v),
            'pitch_med': float(np.median(v)) if len(v) else float('nan'),
            'pitch_mean': float(v.mean()) if len(v) else float('nan'),
            'int_p90': float(np.percentile(d, 90)) if len(d) else float('nan'),
            'int_mean': float(d.mean()) if len(d) else float('nan'),
        })

    # gaps from the trimmed spans, so silence the transcript swallowed shows up
    for i, r in enumerate(rows):
        r['gap_before'] = r['start'] - (rows[i - 1]['end'] if i > 0 else 0.0)
        r['gap_after'] = (rows[i + 1]['start'] if i + 1 < len(rows) else total) - r['end']
    return rows


def flag_suspect(rows):
    """Mark rows that are not trustworthy measurements, with the reasons."""
    p90_all = [r['int_p90'] for r in rows if not np.isnan(r['int_p90'])]
    p90_median = float(np.median(p90_all)) if p90_all else float('nan')
    for r in rows:
        reasons = []
        if r['n_voiced'] < MIN_VOICED_FRAMES:
            reasons.append('few_voiced')
        if not np.isnan(r['int_p90']) and r['int_p90'] < p90_median - QUIET_MARGIN_DB:
            reasons.append('quiet')
        if r['dur'] > LONG_WORD_S and r['n_voiced'] < LONG_WORD_MIN_VOICED:
            reasons.append('long_unvoiced')
        r['suspect_reasons'] = reasons
        r['suspect'] = bool(reasons)
    return p90_median


def silence_log(rows, total):
    """Every silence of consequence, with the context that makes it readable.

    This is not an emphasis cue and nothing here feeds the weight. It is the
    other half of what a gap means: how long the speaker was not talking, and
    what the moment before the silence looked like. Searching for a word,
    trailing off, waiting, and having left the room all produce silence, and
    telling them apart is a later decision that needs this raw material first.

    Nothing is classified into a state. Each record carries the length, the
    word before it, whether that word is a common filler, whether pitch fell
    before the silence or stayed level, and whether a sentence boundary was
    detected at that point.
    """
    log = []
    for i, r in enumerate(rows):
        if i == 0:
            continue
        gap = r['gap_before']
        if gap < SILENCE_LOG_MIN_S:
            continue
        prev = rows[i - 1]
        run_start = prev['sentence']
        fell, p, ref, scope = pitch_fell(rows, i - 1, run_start)
        log.append({
            'start': prev['end'],
            'end': r['start'],
            'length': gap,
            'after_word': prev['word'],
            'after_punct': prev['punct'],
            'after_filler': prev['word'].lower() in FILLER_WORDS,
            'pitch_before': p,
            'pitch_ref': ref,
            'pitch_fell': bool(fell) if not np.isnan(p) else None,
            'at_boundary': bool(r['boundary_cue']),
            'boundary_cue': r['boundary_cue'],
            'next_word': r['word'],
        })
    # the tail of the recording, which is silence but not a pause in speech
    if rows and total - rows[-1]['end'] >= SILENCE_LOG_MIN_S:
        prev = rows[-1]
        log.append({
            'start': prev['end'], 'end': total, 'length': total - prev['end'],
            'after_word': prev['word'], 'after_punct': prev['punct'],
            'after_filler': prev['word'].lower() in FILLER_WORDS,
            'pitch_before': prev['pitch_med'], 'pitch_ref': float('nan'),
            'pitch_fell': None, 'at_boundary': False, 'boundary_cue': '',
            'next_word': '<end of recording>',
        })
    return log


def measure(path):
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, word_timestamps=True, vad_filter=True)

    words = [w for seg in segments for w in seg.words]

    snd = parselmouth.Sound(path)
    total = snd.get_total_duration()

    env_t, env_db = energy_envelope(snd)
    speech_db, noise_db = split_speech_noise_db(env_db)
    silence_db, from_speech, from_noise = silence_threshold_db(speech_db, noise_db)
    silences = find_silences(env_t, env_db, silence_db)

    intensity = snd.to_intensity()
    int_t = intensity.xs()
    int_db = intensity.values[0]

    # probe pass, then a provisional set of rows from it, so the range can be
    # derived from frames the transcript places inside real words only
    probe = snd.to_pitch(pitch_floor=PROBE_FLOOR_HZ, pitch_ceiling=PROBE_CEILING_HZ)
    probe_t = probe.xs()
    probe_hz = probe.selected_array['frequency']
    prov = build_rows(words, silences, probe_t, probe_hz, int_t, int_db, total)
    flag_suspect(prov)
    probe_voiced_all = int(((probe_hz > 0) & ~np.isnan(probe_hz)).sum())
    probe_in_words = sum(r['n_voiced_raw'] for r in prov if not r['suspect'])
    probe_octave = sum(r['n_octave_removed'] for r in prov if not r['suspect'])
    probe_multipass = sum(1 for r in prov if r['octave_passes'] > 1)
    pool = [r['pitch_frames'] for r in prov
            if not r['suspect'] and len(r['pitch_frames'])]
    pool = np.concatenate(pool) if pool else np.array([])
    p_floor, p_ceiling, range_source = derive_pitch_range(pool)

    pitch = snd.to_pitch(pitch_floor=p_floor, pitch_ceiling=p_ceiling)
    pitch_t = pitch.xs()
    pitch_hz = pitch.selected_array['frequency']  # 0.0 marks unvoiced frames

    rows = build_rows(words, silences, pitch_t, pitch_hz, int_t, int_db, total)
    p90_median = flag_suspect(rows)

    sent = assign_sentences(rows)
    for s in set(r['sentence'] for r in rows):
        members = [r for r in rows if r['sentence'] == s]
        n = len(members)
        for j, r in enumerate(members):
            r['sent_pos'] = j / (n - 1) if n > 1 else 0.0

    return rows, {
        'pitch_floor': p_floor,
        'pitch_ceiling': p_ceiling,
        'range_frames': len(pool),
        'range_source': range_source,
        'probe_voiced_all': probe_voiced_all,
        'probe_outside_words': probe_voiced_all - probe_in_words,
        'probe_octave_removed': probe_octave,
        'probe_octave_multipass_words': probe_multipass,
        'octave_removed': sum(r['n_octave_removed'] for r in rows),
        'octave_words': sum(1 for r in rows if r['n_octave_removed']),
        'octave_multipass_words': sum(1 for r in rows if r['octave_passes'] > 1),
        'voiced_raw': sum(r['n_voiced_raw'] for r in rows),
        'gap_candidates': sent['gap_candidates'],
        'gap_boundaries': sent['gap_boundaries'],
        'gap_rejected': sent['gap_rejected'],
        'punct_only': sent['punct_only'],
        'gap_only': sent['gap_only'],
        'both': sent['both'],
        'gap_decisions': sent['decisions'],
        'silence_log': silence_log(rows, total),
        'n_sentences': len(set(r['sentence'] for r in rows)),
        'p90_median': p90_median,
        'median_speech_db': speech_db,
        'median_noise_db': noise_db,
        'silence_threshold_db': silence_db,
        'threshold_from_speech_db': from_speech,
        'threshold_from_noise_db': from_noise,
        'threshold_set_by': 'noise floor' if from_noise > from_speech else 'speech drop',
        'silences': silences,
        'total': total,
    }
