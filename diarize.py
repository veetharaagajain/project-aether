"""Who spoke when. Diarization only: voices are numbered, never named.

The output is that a recording contains some number of distinct voices and
which stretches of it belong to each, labelled 1, 2 and so on. The numbers are
local to the recording and mean nothing across recordings; putting names to
them is a separate problem this file does not touch and does not assume.

The pipeline is the classic one, embedding plus clustering, on the parts of
the waveform prosody_core has already decided are speech:

  1. speech regions, from the silence detector already in prosody_core, so
     diarization and the rest of the layer agree about where speech is
  2. overlapping windows across each region
  3. one ECAPA-TDNN speaker embedding per window
  4. agglomerative clustering of those embeddings with a cosine-distance
     threshold, so the number of speakers falls out rather than being given
  5. a median smoothing pass over time, because a real speaker change lasts
     and a spurious one does not
  6. a label per word, by majority over the windows that overlap it

usage: diarize.py <wav> [<wav> ...]
"""

import sys
import time

import numpy as np
import soundfile as sf

import prosody_core as pc

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = "models/ecapa"
MODEL_RATE = 16000

WINDOW_S = 1.50          # long enough for a stable embedding, short enough to
HOP_S = 0.75             # catch a turn change without straddling it badly
MIN_REGION_S = 0.40      # speech shorter than this gives no usable embedding
SMOOTH_WINDOWS = 3       # median filter width over the window sequence

# Cosine distance between length-normalised ECAPA embeddings. Same-speaker
# pairs sit well below this and different-speaker pairs well above it, but the
# upper half of that claim is untested here: the corpus contains no recording
# with two people in it. See the report from diarize.py run on a file, which
# prints the within-file distance distribution so the margin is visible.
AHC_THRESHOLD = 0.60

# ...and then a second pass on the cluster centroids. The first threshold is
# applied by average linkage, which compares every pair of windows across two
# clusters, so a voice with a wide spread of windows gets split against itself:
# turns1, which is one person, came back as three clusters whose centroids were
# only 0.13, 0.27 and 0.43 apart, all of them well inside one voice. Averaging
# a cluster's windows cancels most of that per-window noise, so the centroids
# are a much better estimate of the voice than any single window is. Clusters
# whose centroids are closer than this are the same person.
CENTROID_MERGE_MAX = 0.60

# A window must contain at least this much real speech. Without it, windows
# that are mostly silence or a single breath get an embedding of noise, and
# noise is what widened turns1's within-speaker spread in the first place.
MIN_WINDOW_SPEECH_S = 0.75

MIN_SPEAKER_SHARE = 0.06   # a cluster holding less of the speech than this is
MIN_SPEAKER_WINDOWS = 3    # absorbed into its nearest neighbour, not a speaker

_MODEL = None


def model():
    """Loaded once. The import is here rather than at module scope so the rest
    of the layer does not pay for torch when diarization is switched off."""
    global _MODEL
    if _MODEL is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _MODEL = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE, savedir=MODEL_DIR, run_opts={'device': 'cpu'})
    return _MODEL


def load_audio(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != MODEL_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=MODEL_RATE)
        sr = MODEL_RATE
    return x, sr


def speech_regions(total, silences):
    """The complement of the silences, which is where prosody_core already
    believes there is speech. Reusing it rather than running a second voice
    detector keeps diarization and the word measurements on the same map."""
    out, t = [], 0.0
    for a, b in silences:
        if a - t >= MIN_REGION_S:
            out.append((t, a))
        t = b
    if total - t >= MIN_REGION_S:
        out.append((t, total))
    return out


def speech_seconds(a, b, silences):
    """How much of a window is speech rather than silence."""
    quiet = 0.0
    for s, e in silences:
        quiet += max(0.0, min(b, e) - max(a, s))
    return (b - a) - quiet


def windows(regions):
    """Overlapping windows inside each speech region, never crossing one.

    A window that straddled a silence could straddle a turn change, which is
    exactly the boundary the clustering is supposed to find.
    """
    out = []
    for a, b in regions:
        if b - a <= WINDOW_S:
            out.append((a, b))
            continue
        t = a
        while t + WINDOW_S <= b + 1e-9:
            out.append((t, t + WINDOW_S))
            t += HOP_S
        if out and out[-1][1] < b - HOP_S / 2:
            out.append((max(a, b - WINDOW_S), b))
    return out


def embed(x, sr, spans):
    """One length-normalised embedding per window."""
    import torch
    m = model()
    vecs = []
    for a, b in spans:
        seg = x[int(a * sr):int(b * sr)]
        if len(seg) < int(0.3 * sr):
            seg = np.pad(seg, (0, int(0.3 * sr) - len(seg)))
        with torch.no_grad():
            e = m.encode_batch(torch.from_numpy(seg[None, :])).squeeze().numpy()
        n = np.linalg.norm(e)
        vecs.append(e / n if n else e)
    return np.array(vecs, dtype=np.float64)


def centroids(emb, labels):
    out = {}
    for i in sorted(set(labels.tolist())):
        c = emb[labels == i].mean(axis=0)
        n = np.linalg.norm(c)
        out[i] = c / n if n else c
    return out


def merge_centroids(emb, labels, limit=CENTROID_MERGE_MAX):
    """Repeatedly merge the two closest cluster centroids while they are
    closer than limit. Returns (labels, trace of the merges)."""
    labels = labels.copy()
    trace = []
    while True:
        cent = centroids(emb, labels)
        ids = sorted(cent)
        if len(ids) < 2:
            break
        best, pair = None, None
        for k, i in enumerate(ids):
            for j in ids[k + 1:]:
                d = 1.0 - float(np.dot(cent[i], cent[j]))
                if best is None or d < best:
                    best, pair = d, (i, j)
        if best is None or best >= limit:
            if best is not None:
                trace.append({'closest_pair': pair, 'distance': best,
                              'merged': False})
            break
        trace.append({'closest_pair': pair, 'distance': best, 'merged': True})
        labels[labels == pair[1]] = pair[0]
    return labels, trace


def cluster(emb):
    """Agglomerative clustering with a distance threshold, not a fixed count.

    A threshold rather than a number of speakers is what lets a single-speaker
    recording come back as one speaker: with a fixed k the algorithm is
    obliged to split a voice against itself. Average linkage on cosine
    distance is the standard first pass, because single linkage chains through
    the borderline windows a long recording always has, and complete linkage
    splits on the worst window in each cluster.

    Average linkage on its own is still too eager to split, because it judges
    two clusters by every pair of windows across them and a single voice
    produces windows 0.5 to 1.0 apart routinely. The second pass compares
    centroids instead, which is the same comparison made on an estimate of
    the voice rather than on individual windows.
    """
    from sklearn.cluster import AgglomerativeClustering
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int), []
    a = AgglomerativeClustering(n_clusters=None, metric='cosine',
                                linkage='average',
                                distance_threshold=AHC_THRESHOLD)
    return a.fit_predict(emb), []


def smooth(labels, k=SMOOTH_WINDOWS):
    """Median filter over the window sequence.

    A genuine speaker change persists for many windows; a misassigned window
    in the middle of someone's turn does not. Nothing here can create a label
    that the clustering did not already produce.
    """
    if len(labels) < k or k < 3:
        return labels.copy()
    out = labels.copy()
    h = k // 2
    for i in range(h, len(labels) - h):
        vals, counts = np.unique(labels[i - h:i + h + 1], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def absorb_small(labels, emb):
    """Fold clusters too small to be a speaker into their nearest neighbour."""
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        if len(ids) < 2:
            break
        share = counts / counts.sum()
        small = [i for i, c, s in zip(ids, counts, share)
                 if c < MIN_SPEAKER_WINDOWS or s < MIN_SPEAKER_SHARE]
        if not small:
            break
        victim = small[int(np.argmin([counts[list(ids).index(i)] for i in small]))]
        cent = {i: emb[labels == i].mean(axis=0) for i in ids}
        others = [i for i in ids if i != victim]
        d = {i: 1.0 - float(np.dot(cent[victim], cent[i])
                            / (np.linalg.norm(cent[victim])
                               * np.linalg.norm(cent[i]))) for i in others}
        labels[labels == victim] = min(d, key=d.get)
    return labels


def renumber(labels, spans):
    """Speaker 1 is whoever speaks first, which is the only ordering available
    without recognising anyone."""
    order, seen = {}, 0
    out = np.zeros(len(labels), dtype=int)
    for i in np.argsort([s[0] for s in spans]):
        if labels[i] not in order:
            seen += 1
            order[labels[i]] = seen
        out[i] = order[labels[i]]
    return out


def label_words(rows, spans, labels):
    """A speaker per word, by which windows overlap its trimmed span.

    Three cases, because a word does not always have a span to overlap with.

    The ordinary case is a word with real duration: the windows it overlaps
    vote, weighted by how much of the word each covers.

    A word with a zero-length span has no duration to weight by, and weighting
    by it is what divided by zero on pod2. Mostly they come from Whisper's
    word aligner, which emits a word with its start and end already identical
    (15 of the 17 on pod2); the rest are words the aligner gave a duration to
    and the silence detector then trimmed away entirely (2 of 17). Either way
    the word occupies a single instant of the transcript, and the instant is
    still evidence: a window containing it is as good a witness as one
    overlapping a longer word. So the containing windows vote equally instead
    of by duration, which uses the same evidence measured a way that does not
    vanish.

    A word with no window at all takes the label of the nearest one. That is
    almost always a word in a burst of speech too short to carry a window,
    not an error.

    Which case fired is recorded in speaker_source, so none of this is silent.
    """
    starts = np.array([s[0] for s in spans])
    ends = np.array([s[1] for s in spans])

    def nearest(r):
        if not len(spans):
            return 0
        mid = (r['start'] + r['end']) / 2.0
        return int(labels[int(np.argmin(np.abs((starts + ends) / 2.0 - mid)))])

    for r in rows:
        m = (ends > r['start']) & (starts < r['end'])
        if not m.any():
            r['speaker'] = nearest(r)
            r['speaker_source'] = 'nearest window'
            r['speaker_share'] = float('nan')
            continue
        overlap = np.minimum(ends[m], r['end']) - np.maximum(starts[m], r['start'])
        if overlap.sum() > 0:
            weights, source = overlap, 'overlap'
        else:                       # a zero-length span: one vote per window
            weights, source = np.ones(len(overlap)), 'instant in window'
        won = {}
        for lab, w in zip(labels[m], weights):
            won[int(lab)] = won.get(int(lab), 0.0) + float(w)
        r['speaker'] = max(won, key=won.get)
        r['speaker_source'] = source
        r['speaker_share'] = won[r['speaker']] / sum(won.values())
    return rows


def segments(spans, labels):
    """Contiguous stretches of one speaker, merged across adjacent windows."""
    out = []
    for (a, b), lab in zip(spans, labels):
        if out and out[-1]['speaker'] == int(lab) and a <= out[-1]['end'] + 1e-6:
            out[-1]['end'] = max(out[-1]['end'], b)
        else:
            out.append({'start': float(a), 'end': float(b), 'speaker': int(lab)})
    return out


def distance_report(emb, labels):
    """The within-file distance distribution, so the margin is inspectable.

    On a single-speaker recording every pair here is a same-speaker pair, so
    this is a direct measurement of where same-speaker distances actually sit
    for this microphone and this voice, which is the only half of the
    threshold the corpus can currently validate.
    """
    if len(emb) < 2:
        return {}
    d = 1.0 - emb @ emb.T
    iu = np.triu_indices(len(emb), 1)
    v = d[iu]
    out = {'n_pairs': int(len(v)),
           'pct': {str(p): float(np.percentile(v, p))
                   for p in (5, 25, 50, 75, 90, 95, 99, 100)},
           'threshold': AHC_THRESHOLD,
           'frac_over_threshold': float((v > AHC_THRESHOLD).mean())}
    ids = sorted(set(labels.tolist()))
    if len(ids) > 1:
        cent = {i: emb[labels == i].mean(axis=0) for i in ids}
        out['between_centroids'] = {
            f"{i}-{j}": float(1.0 - np.dot(cent[i], cent[j])
                              / (np.linalg.norm(cent[i]) * np.linalg.norm(cent[j])))
            for k, i in enumerate(ids) for j in ids[k + 1:]}
    return out


def diarize(path, rows=None, silences=None, total=None):
    """Everything above, in order. Returns (rows, info)."""
    t0 = time.time()
    x, sr = load_audio(path)
    if silences is None or total is None:
        import parselmouth
        snd = parselmouth.Sound(path)
        total = snd.get_total_duration()
        et, edb = pc.energy_envelope(snd)
        sp, no = pc.split_speech_noise_db(edb)
        thr, _, _ = pc.silence_threshold_db(sp, no)
        silences = pc.find_silences(et, edb, thr)

    regions = speech_regions(total, silences)
    all_spans = windows(regions)
    spans = [(a, b) for a, b in all_spans
             if speech_seconds(a, b, silences) >= MIN_WINDOW_SPEECH_S]
    dropped = len(all_spans) - len(spans)
    t_seg = time.time() - t0

    t1 = time.time()
    emb = embed(x, sr, spans) if spans else np.zeros((0, 192))
    t_emb = time.time() - t1

    t2 = time.time()
    raw, _ = cluster(emb) if len(emb) else (np.zeros(0, dtype=int), [])
    sm = smooth(raw)
    merged, trace = merge_centroids(emb, sm) if len(emb) else (sm, [])
    ab = absorb_small(merged, emb) if len(emb) else merged
    labels = renumber(ab, spans) if len(spans) else ab
    t_clu = time.time() - t2

    if rows is not None:
        label_words(rows, spans, labels)

    ids = sorted(set(labels.tolist()))
    speech_s = sum(b - a for a, b in regions)
    per = {}
    for i in ids:
        sel = [s for s, l in zip(spans, labels) if l == i]
        per[i] = {'windows': len(sel),
                  'seconds': float(sum(b - a for a, b in sel)),
                  'share': len(sel) / max(len(labels), 1)}
    return rows, {
        'window_spans': spans,
        'window_emb': emb,
        'n_speakers': len(ids),
        'speakers': per,
        'segments': segments(spans, labels),
        'n_windows': len(spans),
        'n_regions': len(regions),
        'speech_s': speech_s,
        'total_s': total,
        'distances': distance_report(emb, labels),
        'raw_clusters': int(len(set(raw.tolist()))) if len(raw) else 0,
        'after_smoothing': int(len(set(sm.tolist()))) if len(sm) else 0,
        'after_centroid_merge': int(len(set(merged.tolist()))) if len(merged) else 0,
        'merge_trace': trace,
        'windows_dropped_quiet': dropped,
        'seconds': {'segment': t_seg, 'embed': t_emb, 'cluster': t_clu,
                    'total': time.time() - t0},
    }


def main():
    for path in sys.argv[1:] or ["corpus/test.wav"]:
        _, info = diarize(path)
        d = info['distances']
        print(f"{path}: {info['n_speakers']} speaker(s) over "
              f"{info['n_windows']} windows in {info['n_regions']} speech "
              f"region(s), {info['speech_s']:.1f} s of speech in "
              f"{info['total_s']:.1f} s")
        print(f"  clusters: {info['raw_clusters']} raw, "
              f"{info['after_smoothing']} after smoothing, "
              f"{info['after_centroid_merge']} after centroid merge, "
              f"{info['n_speakers']} after absorbing small ones "
              f"({info['windows_dropped_quiet']} windows dropped as too quiet)")
        for m in info['merge_trace']:
            print(f"    centroids {m['closest_pair']} at {m['distance']:.3f}: "
                  f"{'merged' if m['merged'] else 'left apart, above the limit'}")
        for i, s in info['speakers'].items():
            print(f"  speaker {i}: {s['windows']} windows, {s['seconds']:.1f} s, "
                  f"{100*s['share']:.0f}% of speech")
        if d:
            print("  within-file embedding distances, percentiles: "
                  + ", ".join(f"p{k} {v:.3f}" for k, v in d['pct'].items()))
            print(f"  threshold {d['threshold']}, "
                  f"{100*d['frac_over_threshold']:.1f}% of pairs above it")
            if 'between_centroids' in d:
                print("  between cluster centroids: "
                      + ", ".join(f"{k} {v:.3f}"
                                  for k, v in d['between_centroids'].items()))
        t = info['seconds']
        print(f"  time: segment {t['segment']:.1f}s, embed {t['embed']:.1f}s, "
              f"cluster {t['cluster']:.1f}s, total {t['total']:.1f}s "
              f"({info['total_s']/max(t['total'],1e-9):.1f}x realtime)")


if __name__ == "__main__":
    main()
