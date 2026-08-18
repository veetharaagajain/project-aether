"""Who this is, matched against people already known.

Diarization answers "how many voices and which stretches", and it can only do
that once it has the whole recording, because clustering is global. That makes
it an overnight job. Recognition is the opposite shape: a voice fingerprint is
compared against a handful of stored ones, the comparison is a few hundred dot
products, and it needs no view of anything but the utterance in hand. So it can
run while someone is still talking, which is what responding correctly needs.

The two are not alternatives. diarize.py stays as the batch pass that finds
voices nobody has enrolled; this file decides, live, whether a voice is one of
the people already known.

A person here is a profile that grows rather than a one-shot enrolment:

  anchor    the fingerprints from the original enrolment recording. Never
            merged, never decayed, never removed by anything automatic. If
            learning goes wrong the anchor is what the profile can be reset to.
  learned   fingerprints added from confidently matched utterances, each with
            a strength that rises when it is matched and decays when it is not.
            Bounded and periodically consolidated into a small number of modes.

The set exists because one fingerprint per person is a bad model of a person.
A shouted voice, an exhausted voice and a calm enrolment are measurably
different, and a single-template system fails on exactly the utterances that
matter most.

usage:
  recognize.py enroll <name> <wav> [<wav> ...]
  recognize.py identify <wav> [<wav> ...]
  recognize.py list [name]
  recognize.py consolidate [name]
  recognize.py forget <name>
  recognize.py reset <name>          drop everything learned, keep the anchor
"""

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import diarize as dz

PEOPLE_PATH = Path(__file__).resolve().parent / "people.json"
SCHEMA_VERSION = 1

# --- the two numbers that decide everything ---------------------------------
# Cosine distance between length-normalised ECAPA embeddings, taken as the
# minimum over a person's fingerprint set.
#
# BOTH ARE UNTESTED AGAINST A DIFFERENT SPEAKER. Nothing in the corpus contains
# two real people, so the only half of this that has been measured is the
# same-voice half: see same_voice_report() and the report in the repository.
# Treat them as provisional. What would settle them is in the docstring of
# same_voice_report().
#
# The same-voice half, measured by same_voice_report() over 114 utterance-sized
# chunks from turns1, video1_5min, stress1 and drag1, each held against the
# first 30 seconds of its own recording as a stand-in anchor: median 0.290,
# p90 0.392, p95 0.436, worst case 0.513. That is a far tighter distribution
# than the diarization centroid distances suggested, because an utterance
# compared against an averaged anchor is much more stable than two noisy
# sub-cluster centroids compared against each other.
#
# MATCH_MAX sits about a third above the worst same-voice case observed, so
# essentially no genuine match is refused. CONFIDENT_MAX sits at roughly the
# 90th percentile of that distribution, so learning takes the bulk of genuine
# matches and refuses the tail. The asymmetry is deliberate: failing to learn
# from a real utterance costs a slower profile, learning from a wrong one
# compounds.
MATCH_MAX = 0.70        # further than this from everyone: unknown
CONFIDENT_MAX = 0.40    # closer than this, and clear of the runner-up: learn
MARGIN_MIN = 0.10       # ...where "clear of the runner-up" means this much
                        # further to the second-best person than to the best

# --- how the set is bounded -------------------------------------------------
MAX_LEARNED = 24        # per person, excluding the anchor
DECAY = 0.92            # applied to every learned strength at consolidation
MIN_STRENGTH = 0.25     # below this a learned fingerprint is dropped
MATCH_BONUS = 1.0       # added to the strength of whichever one matched
CONSOLIDATE_EVERY = 16  # confident matches between consolidation passes


# --- storage ----------------------------------------------------------------
def load(check=True):
    """The enrolled people, refusing fingerprints from a different model.

    An empty store is never stale: there is nothing in it to be incomparable.
    """
    if not PEOPLE_PATH.exists():
        return {'version': SCHEMA_VERSION, 'people': {}}
    store = json.loads(PEOPLE_PATH.read_text())
    if check and store.get('people'):
        want = embedding_config()
        got = store.get('embedding')
        if got != want:
            raise StaleFingerprints(
                "The enrolled fingerprints came out of a different embedding "
                "setup.\n"
                f"  stored in: {PEOPLE_PATH}\n"
                f"  was: {got}\n"
                f"  now: {want}\n\n"
                "Cosine distances across two models are still numbers and will "
                "still\nname people, wrongly. Nothing will run until this is "
                "resolved.\n\n"
                "To resolve, re-enrol everyone against the current model.")
    return store


# What makes a stored fingerprint comparable to a freshly computed one: the
# weights it came out of, and the preprocessing diarize.embed applied on the
# way in. The match thresholds are deliberately not here -- they change what is
# decided with the vectors, not whether the vectors mean the same thing, and
# folding them in would force a re-enrolment every time one was tuned.
def embedding_config():
    import diarize as dz
    return {'model': 'speechbrain/spkrec-ecapa-voxceleb',
            'revision': dz.model_revision(),
            'rate': dz.MODEL_RATE,
            'length_normalised': True,
            'min_span_s': 0.3}


class StaleFingerprints(Exception):
    """Stored fingerprints came out of a different embedding model.

    Cosine distances between vectors from two different models are still
    numbers, and still land inside MATCH_MAX often enough to name the wrong
    person, so this cannot be a warning either.
    """


def save(store):
    store['embedding'] = embedding_config()
    PEOPLE_PATH.write_text(json.dumps(store, indent=1, sort_keys=True) + "\n")


def new_person(name):
    return {
        'id': uuid.uuid4().hex,
        'name': name,
        'created': time.time(),
        'anchor': [],
        'learned': [],
        'stats': {'matched': 0, 'confident': 0, 'borderline': 0,
                  'since_consolidate': 0, 'last_seen': None,
                  'consolidations': 0},
    }


def find(store, name):
    for pid, p in store['people'].items():
        if p['name'].lower() == name.lower() or pid == name:
            return pid, p
    return None, None


# --- fingerprints -----------------------------------------------------------
def fingerprint(x, sr, spans):
    """One length-normalised ECAPA embedding per span. This is the expensive
    half; everything below it is arithmetic."""
    return dz.embed(x, sr, spans)


def enrolment_fingerprints(path):
    """Fingerprints for an enrolment recording: one per usable speech window.

    The same windowing diarization uses, so an enrolment and a live utterance
    are described the same way and are comparable.
    """
    import parselmouth
    import prosody_core as pc
    snd = parselmouth.Sound(path)
    total = snd.get_total_duration()
    et, edb = pc.energy_envelope(snd)
    sp, no = pc.split_speech_noise_db(edb)
    thr, _, _ = pc.silence_threshold_db(sp, no)
    sil = pc.find_silences(et, edb, thr)
    regions = dz.speech_regions(total, sil)
    spans = [(a, b) for a, b in dz.windows(regions)
             if dz.speech_seconds(a, b, sil) >= dz.MIN_WINDOW_SPEECH_S]
    x, sr = dz.load_audio(path)
    return fingerprint(x, sr, spans), spans


def vectors(person):
    """The whole comparison set, anchor first, with where each came from."""
    v, kind, idx = [], [], []
    for i, f in enumerate(person.get('anchor', [])):
        v.append(f['vec'])
        kind.append('anchor')
        idx.append(i)
    for i, f in enumerate(person.get('learned', [])):
        v.append(f['vec'])
        kind.append('learned')
        idx.append(i)
    return (np.array(v, dtype=np.float64) if v else np.zeros((0, 192))), kind, idx


# --- matching ---------------------------------------------------------------
def match(store, emb):
    """Compare one fingerprint against every known person.

    Returns a dict with the best person, the distance to them, the distance to
    the runner-up, a confidence, and a decision of 'confident', 'match' or
    'unknown'. Unknown is a correct answer; a wrong name is not, so nothing
    within MATCH_MAX of nobody is ever given a name.

    The cost is one matrix product against a few hundred stored vectors, which
    is why this can run per utterance and clustering cannot.
    """
    e = np.asarray(emb, dtype=np.float64)
    n = np.linalg.norm(e)
    e = e / n if n else e
    best = []
    for pid, p in store.get('people', {}).items():
        V, kind, idx = vectors(p)
        if not len(V):
            continue
        d = 1.0 - V @ e
        j = int(np.argmin(d))
        best.append({'pid': pid, 'name': p['name'], 'distance': float(d[j]),
                     'which': kind[j], 'index': idx[j]})
    if not best:
        return {'decision': 'unknown', 'reason': 'nobody enrolled',
                'pid': None, 'name': None, 'distance': None,
                'runner_up': None, 'margin': None, 'confidence': 0.0}
    best.sort(key=lambda b: b['distance'])
    top = best[0]
    second = best[1]['distance'] if len(best) > 1 else None
    margin = (second - top['distance']) if second is not None else float('inf')

    if top['distance'] > MATCH_MAX:
        decision, reason = 'unknown', (
            f"nearest is {top['name']} at {top['distance']:.3f}, "
            f"beyond the {MATCH_MAX} limit")
    elif top['distance'] <= CONFIDENT_MAX and margin >= MARGIN_MIN:
        decision, reason = 'confident', 'inside the confident limit and clear'
    else:
        decision, reason = 'match', (
            'within the match limit but not confident'
            + ('' if margin >= MARGIN_MIN else
               f", runner-up only {margin:.3f} away"))

    # confidence: how far inside the match limit, tightened by how close the
    # runner-up is. Both matter: a close match that two people share is not a
    # confident identification of either.
    span = max(MATCH_MAX - CONFIDENT_MAX, 1e-6)
    closeness = float(np.clip((MATCH_MAX - top['distance']) / span, 0.0, 1.0))
    separation = float(np.clip(margin / max(MARGIN_MIN * 2, 1e-6), 0.0, 1.0))
    conf = 0.0 if decision == 'unknown' else closeness * separation

    return {'decision': decision, 'reason': reason, 'pid': top['pid'],
            'name': top['name'], 'distance': top['distance'],
            'which': top['which'], 'runner_up': second, 'margin':
            None if margin == float('inf') else float(margin),
            'confidence': round(conf, 4),
            'all': [{k: b[k] for k in ('name', 'distance', 'which')}
                    for b in best[:4]]}


# --- learning ---------------------------------------------------------------
def learn(store, result, emb, source=None):
    """Add a confidently matched fingerprint to that person's set.

    Only confident matches. A profile that learns from uncertain ones slowly
    absorbs whoever it has been confusing itself with, and every wrong addition
    makes the next wrong match likelier, so the failure compounds rather than
    averaging out. A borderline match is counted and discarded.
    """
    p = store['people'].get(result.get('pid') or '')
    if p is None:
        return False, 'no such person'
    p['stats']['matched'] += 1
    p['stats']['last_seen'] = time.time()
    if result['decision'] != 'confident':
        p['stats']['borderline'] += 1
        return False, 'not confident, discarded'

    e = np.asarray(emb, dtype=np.float64)
    n = np.linalg.norm(e)
    e = (e / n if n else e)
    if result['which'] == 'learned':
        p['learned'][result['index']]['strength'] += MATCH_BONUS
        p['learned'][result['index']]['matches'] += 1
    p['learned'].append({'vec': e.tolist(), 'strength': 1.0, 'matches': 1,
                         'created': time.time(), 'source': source,
                         'distance_at_add': result['distance']})
    p['stats']['confident'] += 1
    p['stats']['since_consolidate'] += 1
    if (p['stats']['since_consolidate'] >= CONSOLIDATE_EVERY
            or len(p['learned']) > MAX_LEARNED):
        consolidate(p)
    return True, 'added'


def consolidate(person):
    """Fade, merge, bound. The anchor is untouched by all three.

    Fading multiplies every learned strength by DECAY, so a fingerprint that
    stops being matched loses ground to ones that keep being matched. Merging
    repeatedly folds the two closest learned fingerprints into their
    strength-weighted mean until the set is inside MAX_LEARNED, which turns a
    long tail of near-duplicates into a small number of modes rather than
    throwing recent material away. Dropping removes anything that has faded
    below MIN_STRENGTH.

    The anchor never fades, never merges and is never dropped, so the profile
    can always be reset to what was actually enrolled.
    """
    L = person['learned']
    for f in L:
        f['strength'] *= DECAY
    L = [f for f in L if f['strength'] >= MIN_STRENGTH]

    while len(L) > MAX_LEARNED:
        V = np.array([f['vec'] for f in L])
        d = 1.0 - V @ V.T
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a, b = L[int(i)], L[int(j)]
        wa, wb = a['strength'], b['strength']
        m = (np.array(a['vec']) * wa + np.array(b['vec']) * wb) / max(wa + wb, 1e-9)
        nn = np.linalg.norm(m)
        merged = {'vec': (m / nn if nn else m).tolist(),
                  'strength': wa + wb,
                  'matches': a.get('matches', 0) + b.get('matches', 0),
                  'created': min(a.get('created', 0), b.get('created', 0)),
                  'source': 'merged', 'distance_at_add': None}
        L = [f for k, f in enumerate(L) if k not in (int(i), int(j))] + [merged]

    person['learned'] = L
    person['stats']['since_consolidate'] = 0
    person['stats']['consolidations'] = person['stats'].get('consolidations', 0) + 1
    return person


# --- the pipeline entry point ----------------------------------------------
def identify_utterances(rows, window_spans, window_emb, store=None,
                        learning=True, source=None):
    """Match every utterance in a recording and label its words.

    The fingerprint for an utterance is the strength-free mean of the
    diarization window embeddings that overlap it, which costs nothing because
    those embeddings already exist. A live caller with no diarization pass
    would call fingerprint() on the utterance span instead; the result is the
    same shape and the cost is in question 6 of the report.
    """
    store = load() if store is None else store
    if not rows:
        return {}, store
    starts = np.array([s[0] for s in window_spans]) if len(window_spans) else np.zeros(0)
    ends = np.array([s[1] for s in window_spans]) if len(window_spans) else np.zeros(0)

    out = {}
    for sent in sorted({r['sentence'] for r in rows}):
        members = [r for r in rows if r['sentence'] == sent]
        a, b = members[0]['start'], members[-1]['end']
        m = (ends > a) & (starts < b) if len(starts) else np.zeros(0, dtype=bool)
        if not m.any():
            res = {'decision': 'unknown', 'reason': 'no usable audio window',
                   'pid': None, 'name': None, 'distance': None,
                   'confidence': 0.0}
            emb = None
        else:
            emb = window_emb[m].mean(axis=0)
            nn = np.linalg.norm(emb)
            emb = emb / nn if nn else emb
            res = match(store, emb)
            if learning and res['decision'] == 'confident':
                learn(store, res, emb, source=source)
        out[sent] = res
        for r in members:
            r['person'] = res.get('name')
            r['person_id'] = res.get('pid')
            r['person_decision'] = res['decision']
            r['person_distance'] = res.get('distance')
            r['person_confidence'] = res.get('confidence', 0.0)
    return out, store


def same_voice_report(paths):
    """Measure the same-voice half of the threshold, which is the only half
    the corpus can supply.

    For each single-speaker recording this takes every utterance-sized chunk,
    holds out the first 30 seconds as a stand-in enrolment anchor, and reports
    how far the remaining chunks land from it. That distribution is what a
    correct match has to accept. The other half, how far a different person
    lands, cannot be measured without a recording of a different person, and
    the threshold is not settled until it is.
    """
    rows = []
    for p in paths:
        emb, spans = enrolment_fingerprints(p)
        if len(emb) < 6:
            continue
        anchor_n = max(3, sum(1 for s in spans if s[1] <= 30.0))
        if len(emb) - anchor_n < 3:          # nothing left to test against
            continue
        anchor = emb[:anchor_n].mean(axis=0)
        anchor /= max(np.linalg.norm(anchor), 1e-9)
        rest = emb[anchor_n:]
        # utterance-sized fingerprints: mean of three consecutive windows
        chunks = np.array([rest[i:i + 3].mean(axis=0)
                           for i in range(0, len(rest) - 2, 3)])
        chunks = np.array([c / max(np.linalg.norm(c), 1e-9) for c in chunks])
        d = 1.0 - chunks @ anchor
        rows.append((p, len(chunks), d))
    return rows


# --- commands ---------------------------------------------------------------
def print_person(p):
    L = p['learned']
    print(f"{p['name']}  id {p['id']}")
    print(f"  anchor {len(p['anchor'])} fingerprint(s), learned {len(L)} "
          f"of at most {MAX_LEARNED}")
    print(f"  matched {p['stats']['matched']}, confident "
          f"{p['stats']['confident']}, borderline {p['stats']['borderline']}, "
          f"consolidations {p['stats'].get('consolidations', 0)}")
    if L:
        s = sorted(f['strength'] for f in L)
        print(f"  learned strengths {s[0]:.2f} to {s[-1]:.2f}, "
              f"total {sum(s):.1f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    cmd = sys.argv[1]
    store = load()

    if cmd == "enroll":
        name, paths = sys.argv[2], sys.argv[3:]
        if not paths:
            print("enroll needs at least one wav")
            return 1
        pid, p = find(store, name)
        if p is None:
            p = new_person(name)
            store['people'][p['id']] = p
        for path in paths:
            emb, spans = enrolment_fingerprints(path)
            for v, s in zip(emb, spans):
                p['anchor'].append({'vec': v.tolist(), 'source': path,
                                    'span': [float(s[0]), float(s[1])],
                                    'created': time.time()})
            print(f"{path}: {len(emb)} anchor fingerprint(s)")
        save(store)
        print_person(p)
        return 0

    if cmd == "identify":
        for path in sys.argv[2:]:
            emb, spans = enrolment_fingerprints(path)
            if not len(emb):
                print(f"{path}: no usable speech")
                continue
            chunks = [emb[i:i + 3].mean(axis=0)
                      for i in range(0, max(len(emb) - 2, 1), 3)]
            print(f"{path}: {len(chunks)} utterance-sized fingerprint(s)")
            for k, c in enumerate(chunks):
                r = match(store, c)
                print(f"  chunk {k}: {r['decision']}"
                      + (f", {r['name']} at {r['distance']:.3f}"
                         if r['name'] else "")
                      + f", confidence {r['confidence']:.2f}  ({r['reason']})")
        return 0

    if cmd == "list":
        if not store['people']:
            print(f"nobody enrolled in {PEOPLE_PATH.name}")
            return 0
        for p in store['people'].values():
            if len(sys.argv) > 2 and p['name'].lower() != sys.argv[2].lower():
                continue
            print_person(p)
        return 0

    if cmd == "consolidate":
        for p in store['people'].values():
            if len(sys.argv) > 2 and p['name'].lower() != sys.argv[2].lower():
                continue
            before = len(p['learned'])
            consolidate(p)
            print(f"{p['name']}: {before} -> {len(p['learned'])} learned")
        save(store)
        return 0

    if cmd == "reset":
        pid, p = find(store, sys.argv[2])
        if p is None:
            print("no such person")
            return 1
        p['learned'] = []
        p['stats'].update({'confident': 0, 'borderline': 0,
                           'since_consolidate': 0})
        save(store)
        print(f"{p['name']}: learning dropped, {len(p['anchor'])} anchor "
              f"fingerprint(s) kept")
        return 0

    if cmd == "forget":
        pid, p = find(store, sys.argv[2])
        if p is None:
            print("no such person")
            return 1
        del store['people'][pid]
        save(store)
        print(f"forgot {p['name']}")
        return 0

    print(__doc__.strip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
