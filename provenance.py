"""What a stored number was produced under, recorded next to the number.

The thresholds in markers.LEVEL_THRESHOLDS moved four times in two rounds --
0.962/1.408, 1.042/1.518, 0.973/1.491, 1.143/1.659 -- and every move was
correct. What was wrong is that each one was found by hand, afterwards, by
noticing that something underneath had changed. The baseline had the same
fault in a worse form: it was accumulated over the whole-file measurement
path, prosody_core.measure then began segmenting first, and nothing anywhere
noticed that the reference and the words being scored against it no longer
came from the same pipeline. dur_resid_sd was wrong by a factor of two for as
long as it took someone to ask.

The fix is not cleverness, it is bookkeeping. Anything derived from
measurement carries a fingerprint of the measurement configuration it was
derived under, and the configuration is recomputed and compared every time
that value is loaded.

WHAT COUNTS AS CONFIGURATION. Every module that owns constants which move the
numbers declares them in its own PROVENANCE tuple, so that adding a constant
and forgetting to watch it is a visible omission in the file being edited
rather than an invisible one in this file. To that this module adds the two
things that are not module constants: the recogniser, identified by the
digest of bridge/daemon.swift because a change to the bridge changes the word
spans, and whether prosody_core.measure segments before transcribing, read
off its signature default rather than restated here.

WHAT IS NOT WATCHED. Paths, filenames, and anything that only affects how a
number is displayed. Comments and docstrings are deliberately not watched: a
digest over whole source files would trip on an edited comment, and a check
that cries wolf is a check that gets switched off.
"""

import hashlib
import inspect
import json

CONFIG_VERSION = 1          # bump when the shape of the record itself changes


def _module_constants():
    """Every watched constant, module by module, from the modules themselves."""
    import baseline
    import prosody_core
    import score
    import segment
    out = {}
    for mod in (prosody_core, segment, score, baseline):
        name = mod.__name__
        for k in getattr(mod, 'PROVENANCE', ()):
            v = getattr(mod, k)
            if isinstance(v, (set, frozenset)):
                v = sorted(v)
            elif isinstance(v, tuple):
                v = list(v)
            out[f"{name}.{k}"] = v
    return out


def _transcriber():
    """Which recogniser, identified by the source of the bridge it runs."""
    import speech
    try:
        src = speech.SOURCE.read_bytes()
        d = hashlib.blake2b(src, digest_size=8).hexdigest()
    except OSError:
        d = 'missing'
    return {'transcriber.engine': 'SpeechAnalyzer',
            'transcriber.bridge_source': d}


def _segmented_default():
    """Whether measure() segments before transcribing, read off its signature
    so this cannot claim one thing while the code does another."""
    import prosody_core
    p = inspect.signature(prosody_core.measure).parameters.get('segmented')
    return {'measure.segmented': None if p is None else p.default}


def current():
    """The configuration this process would measure under, right now."""
    d = {'config_version': CONFIG_VERSION}
    d.update(_transcriber())
    d.update(_segmented_default())
    d.update(_module_constants())
    return d


def canonical(d):
    return json.dumps(d, sort_keys=True, separators=(',', ':'), default=str)


def digest(d):
    return hashlib.blake2b(canonical(d).encode(), digest_size=16).hexdigest()


def differences(stored, now=None):
    """Which keys differ, as (key, was, is) triples. Empty means a match."""
    now = current() if now is None else now
    out = []
    for k in sorted(set(stored) | set(now)):
        a, b = stored.get(k, '<absent>'), now.get(k, '<absent>')
        if canonical(a) != canonical(b):
            out.append((k, a, b))
    return out


def content_digest(stats):
    """A stored baseline's identity, over its numbers and not its provenance.

    This is what a derived value cites when it says which baseline it came
    from. It has to exclude the provenance block, or stamping the block would
    change the identity it records.
    """
    body = {k: v for k, v in stats.items() if k != 'provenance'}
    return hashlib.blake2b(canonical(body).encode(), digest_size=16).hexdigest()


def stamp(stats):
    """The block to store beside a freshly derived set of numbers."""
    c = current()
    return {'config': c, 'config_digest': digest(c)}


class Stale(Exception):
    """A stored value was derived under a configuration this run does not use.

    Raised rather than warned. The four threshold moves this exists to prevent
    all happened in the presence of printed warnings, in headers nobody was
    reading, because a warning still lets the numbers out and the numbers look
    fine. Refusing to produce them is the only signal that cannot be skimmed
    past.
    """


def _report(what, whence, diffs, remedy):
    lines = [f"{what} was derived under a different configuration.",
             f"  stored in: {whence}",
             f"  {len(diffs)} setting(s) differ, oldest value first:"]
    for k, a, b in diffs[:12]:
        lines.append(f"    {k}: was {a!r}, now {b!r}")
    if len(diffs) > 12:
        lines.append(f"    ... and {len(diffs) - 12} more")
    lines += ["", "Every number derived from it is measured against a reference",
              "from a different pipeline, so it is wrong in a way that looks",
              "plausible. Nothing will run until this is resolved.", "",
              "To resolve:", remedy]
    return "\n".join(lines)


def check_baseline(speaker, stats, path):
    """Refuse to hand back a baseline built under a different configuration."""
    p = stats.get('provenance')
    if p is None:
        raise Stale(_report(
            f"The baseline for {speaker!r}", path,
            [('provenance', '<not recorded>', digest(current()))],
            f"  python baseline.py reset {speaker}\n"
            f"  python baseline.py add {speaker} <the wav files it was built from>\n"
            f"  ...then rederive the marking thresholds:\n"
            f"  python markers.py levels <the corpus files>"))
    diffs = differences(p['config'])
    if diffs:
        raise Stale(_report(
            f"The baseline for {speaker!r}", path, diffs,
            f"  python baseline.py reset {speaker}\n"
            f"  python baseline.py add {speaker} "
            f"{' '.join(stats.get('files') or ['<the wav files>'])}\n"
            f"  ...then rederive the marking thresholds:\n"
            f"  python markers.py levels <the corpus files>"))


def check_levels(level_provenance, stats, speaker, path):
    """Refuse to mark words with thresholds cut against a different baseline."""
    diffs = differences(level_provenance.get('config', {}))
    if diffs:
        raise Stale(_report(
            "markers.LEVEL_THRESHOLDS", "markers.py, beside LEVEL_THRESHOLDS",
            diffs, "  python markers.py levels <the corpus files>\n"
                   "  ...and paste the block it prints into markers.py"))
    was = level_provenance.get('baseline_digest')
    now = content_digest(stats)
    if was != now:
        raise Stale(_report(
            "markers.LEVEL_THRESHOLDS",
            "markers.py, beside LEVEL_THRESHOLDS",
            [(f"baseline({speaker}) content", was, now)],
            "  The configuration matches but the baseline itself has been\n"
            "  rebuilt since the thresholds were cut against it. Rederive:\n"
            "  python markers.py levels <the corpus files>\n"
            "  ...and paste the block it prints into markers.py"))
