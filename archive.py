"""The audio the live path used to throw away.

Until now a segment was measured and dropped. That made two things impossible.
Nothing could ever be re-transcribed by a better recogniser, so every
transcription error was permanent even though the audio that would settle it
had existed a second earlier. And the third rung of the memory ladder --
gist, then transcript, then the recording -- had nothing to stand on.

FORMAT. FLAC, mono, 16 kHz, 16-bit, which is the rate the gate and the
recogniser already work at, so nothing is resampled on the way in or out.
Lossless, because the point of keeping it is that a better model can read it
later, and a lossy archive would mean every future reading is of a
reconstruction. The project already says the raw archive is inviolable.
Measured over natural1, turns1, pod2 and drag1 it costs 61.6 MB per hour of
speech against 115.2 MB for the same audio as WAV, a ratio of 0.53. Only
speech is stored, because the gate has already dropped the rest: across those
four recordings speech was 70.9 percent of wall-clock, so an hour of being in
the room costs about 44 MB rather than 62.

ADDRESSING. Content-addressed, flat, per the storage design: the name of a
blob is the BLAKE2b-256 digest of its samples. The digest is taken over the
16-bit PCM rather than over the encoded file, so re-encoding the same audio
under a different FLAC version addresses the same blob and two identical
segments are stored once.

WHOSE. Everyone's. A run is stored whole, before anything has decided who was
speaking, and the record that points at it carries the person the recogniser
matched, or unknown. Storing only the owner would mean the other half of
every conversation could never be re-read, and it was already settled that
everyone in a room is recorded and tagged. Who may later retrieve a blob is a
question for the memory layer, which has the person on the record to answer
it with; it is not a question storage can answer by discarding.
"""

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "store" / "audio"
SR = 16000
SUFFIX = ".flac"
FANOUT = 2          # first two hex characters become a directory


def to_pcm16(samples):
    """The canonical byte form a digest is taken over."""
    a = np.asarray(samples, dtype='float32')
    return (np.clip(a, -1.0, 1.0) * 32767.0).astype('<i2')


def digest(samples):
    return hashlib.blake2b(to_pcm16(samples).tobytes(), digest_size=32).hexdigest()


def path_for(d):
    return ROOT / d[:FANOUT] / (d + SUFFIX)


def put(samples, sr=SR):
    """Store one run of speech and return the record that points at it.

    Writing is skipped when the blob is already there, which is what
    content-addressing buys: the same audio arriving twice costs one write.
    """
    import soundfile as sf
    if sr != SR:
        raise ValueError(f"the archive stores {SR} Hz only, got {sr}")
    pcm = to_pcm16(samples)
    d = hashlib.blake2b(pcm.tobytes(), digest_size=32).hexdigest()
    p = path_for(d)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        sf.write(tmp, pcm, SR, format='FLAC', subtype='PCM_16')
        tmp.replace(p)          # so a crash mid-write leaves no half blob
    return {'blob': d, 'format': 'flac', 'sample_rate': SR,
            'seconds': round(len(pcm) / SR, 3), 'bytes': p.stat().st_size}


def get(d):
    """The samples back, as float32 at SR."""
    import soundfile as sf
    p = path_for(d)
    if not p.exists():
        raise FileNotFoundError(f"no blob {d} at {p}")
    x, sr = sf.read(p, dtype='float32', always_2d=False)
    if sr != SR:
        raise ValueError(f"blob {d} is {sr} Hz, expected {SR}")
    return x


def locate(rec):
    """The audio for one stored utterance: (samples, start_s_within_blob).

    A record points at the whole run it came out of, not at a clip of itself,
    so several utterances in one run share one blob. The offset is where this
    utterance starts inside that run.
    """
    a = rec.get('audio')
    if not a:
        raise KeyError("record has no audio block")
    x = get(a['blob'])
    off = a.get('offset', 0.0)
    return x[int(off * SR):int((off + (rec['end'] - rec['start'])) * SR)], off


def usage():
    """What the archive currently holds."""
    n = total = 0
    for p in ROOT.rglob("*" + SUFFIX):
        n += 1
        total += p.stat().st_size
    return {'blobs': n, 'bytes': total, 'mb': round(total / 1e6, 2)}


# --- blobs nothing points at ------------------------------------------------
# A blob is written when a segment is measured; its observation is written a
# moment later. Anything that stops in between -- a crash, an interrupt, a
# session run before the store existed at all -- leaves audio on disk that no
# record refers to. That audio is unreachable by incognito.forget(), which
# works from observations outward, so it is audio the person has no control
# that can delete. That is the whole reason this exists.
#
# GRACE. A blob younger than GRACE_S is never swept, because "written but not
# yet recorded" and "written and never recorded" look identical from here and
# only time separates them. The live path writes the observation within
# milliseconds of the blob, so an hour is enormous; it is set that way because
# the cost of waiting is a little disk and the cost of being wrong is deleting
# audio that was about to be referenced.
GRACE_S = 3600.0


def referenced(db):
    """Every blob digest some observation still points at."""
    return {r[0] for r in db.execute(
        "SELECT DISTINCT audio_blob FROM observations "
        "WHERE audio_blob IS NOT NULL")}


def orphans(db, grace_s=GRACE_S, now=None):
    """Blobs on disk that no observation refers to and that are old enough to
    be sure about. Returns a list of (digest, bytes, age_seconds)."""
    import time as _t
    now = _t.time() if now is None else now
    keep = referenced(db)
    out = []
    for p in sorted(ROOT.rglob("*" + SUFFIX)):
        d = p.stem
        if d in keep:
            continue
        st = p.stat()
        age = now - st.st_mtime
        if age < grace_s:
            continue
        out.append((d, st.st_size, age))
    return out


def sweep(db, apply=False, grace_s=GRACE_S, now=None):
    """Report unreferenced blobs, and delete them when asked.

    Reporting is the default and deleting is the flag, the same way
    incognito.plan_forget precedes incognito.forget: this removes recordings,
    and a destructive default is not something a person can safely re-run.

    Each removal leaves a tombstone, so a device that had synced the blob
    learns it is gone rather than offering it back.
    """
    import time as _t
    found = orphans(db, grace_s=grace_s, now=now)
    freed = 0
    removed = []
    if apply:
        stamp = _t.time()
        for d, size, _age in found:
            p = path_for(d)
            if p.exists():
                p.unlink()
                freed += size
                removed.append(d)
            db.execute("INSERT OR REPLACE INTO tombstones(id,kind,reason,at,"
                       "device,hlc) VALUES(?,?,?,?,?,?)",
                       (d, 'audio', 'swept: no observation referred to it',
                        stamp, db.device, db.clock.tick()))
        db.commit()
        # empty fan-out directories left behind, so the archive does not
        # accumulate 256 empty folders forever
        for sub in sorted(ROOT.iterdir()):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
    return {'found': len(found), 'bytes': sum(f[1] for f in found),
            'removed': len(removed), 'freed': freed,
            'oldest_age_s': max((f[2] for f in found), default=0.0),
            'digests': [f[0] for f in found]}


def main():
    import sys
    import memory as mem
    args = sys.argv[1:]
    apply = '--apply' in args
    grace = float(args[args.index('--grace') + 1]) if '--grace' in args else GRACE_S
    db = mem.open()
    u = usage()
    r = sweep(db, apply=apply, grace_s=grace)
    print(f"archive: {u['blobs']} blob(s), {u['mb']} MB")
    print(f"referenced by an observation: {len(referenced(db))}")
    print(f"unreferenced and older than {grace/3600:.2f} h: {r['found']} "
          f"({r['bytes']/1e6:.2f} MB, oldest {r['oldest_age_s']/3600:.1f} h)")
    if apply:
        print(f"removed {r['removed']} blob(s), freed {r['freed']/1e6:.2f} MB, "
              f"tombstoned each")
        print(f"archive now: {usage()['blobs']} blob(s), {usage()['mb']} MB")
    elif r['found']:
        print("nothing removed. Re-run with --apply to delete them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
