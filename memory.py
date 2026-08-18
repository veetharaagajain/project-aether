"""The store behind Aether: what happened, and what was worked out from it.

Two kinds of thing, and keeping them apart is the whole design.

An OBSERVATION is something that happened. Written once, never modified. Right
now every observation is speech, but it carries a type because sight and the
rest arrive later and must not need a second shape. This is the record, and it
is the only thing in here that cannot be reconstructed.

A BELIEF is something worked out from observations. "He decided against
Parakeet" is not a thing anyone said. Beliefs carry a weight that fades unless
returned to, a certainty that is not the same as that weight, and the time the
conclusion was drawn, which is not the time the speech happened. Every belief
points back at the observations that produced it, so the entire belief layer
can be dropped and rebuilt from the record; rebuild_check() demonstrates that
rather than claiming it.

Beliefs are revised by being replaced, never edited, so what the system used to
think stays recoverable and answerable. Only the bookkeeping fields -- weight,
when it was last returned, how often -- ever change on a stored belief. Its
statement, its certainty and its provenance are fixed at write.

Connections between beliefs are typed with free text and not with an
enumeration, because the useful set is not known yet. Every distinct type is
counted in edge_types as it is used, so in a month there is a measured list to
promote instead of a guessed one.

IDENTITY. Ids are ULIDs: a millisecond timestamp followed by randomness,
sortable by creation time and unique without anyone coordinating. Every row
also carries the device that wrote it and a hybrid logical clock, which is a
wall clock that can only move forward and a counter that breaks ties within a
millisecond. Two devices that were both offline can union their rows and order
them without a server. Deletion is the one thing a union cannot express, so
deletions leave tombstones. The syncing itself is not built.

EMBEDDINGS. Vector search lives in the same file, through sqlite-vec, so
retrieval by meaning works from the start. The model that produced the vectors
is recorded in meta, and open() refuses a store whose vectors came out of a
different one, for the reason provenance.py exists: distances between vectors
from two models are still numbers and still rank results, wrongly.
"""

import json
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "store" / "memory.db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS = 384
SCHEMA_VERSION = 1

# Decay is applied when a belief is read rather than on a timer, so nothing has
# to run in the background for the numbers to be right. HALF_LIFE_DAYS is how
# long an untouched belief takes to lose half its weight; RETURN_BONUS is what
# being returned to adds back. See decayed_weight() and touch_belief().
HALF_LIFE_DAYS = 30.0
RETURN_BONUS = 0.25
WEIGHT_CEILING = 4.0
FAINT = 0.05          # below this a belief is faint: still stored, ranked last

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


# --- identity and time ------------------------------------------------------
def ulid(now_ms=None):
    """Sortable, coordination-free id: 48 bits of milliseconds, 80 of chance."""
    t = int(time.time() * 1000) if now_ms is None else int(now_ms)
    n = (t << 80) | secrets.randbits(80)
    return "".join(_CROCKFORD[(n >> (5 * i)) & 31] for i in range(25, -1, -1))


def ulid_time_ms(s):
    """The millisecond a ULID was minted, back out of the id itself."""
    n = 0
    for ch in s[:10]:
        n = n * 32 + _CROCKFORD.index(ch)
    return n


class Clock:
    """A hybrid logical clock: wall time that cannot go backwards.

    Ordering rows by wall clock alone breaks when two devices disagree about
    the time, and ordering by a counter alone loses any relation to when things
    actually happened. This keeps both, and formats them so that plain string
    comparison is the ordering.
    """

    def __init__(self, device):
        self.device = device
        self.last_ms = 0
        self.counter = 0

    def tick(self, observed=None):
        now = int(time.time() * 1000)
        if observed:
            try:
                o_ms, o_ct, _ = observed.split("-")
                now = max(now, int(o_ms))
            except ValueError:
                pass
        if now > self.last_ms:
            self.last_ms, self.counter = now, 0
        else:
            self.counter += 1
        return f"{self.last_ms:013d}-{self.counter:05d}-{self.device}"


# --- schema -----------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Something that happened. Append-only: there is no UPDATE anywhere in this
-- module that touches this table, and forget() is the only DELETE.
CREATE TABLE IF NOT EXISTS observations (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,          -- 'speech' now; 'sight' later
  started_at    REAL NOT NULL,          -- epoch seconds, when it happened
  ended_at      REAL NOT NULL,
  session       TEXT,
  person_id     TEXT,
  person        TEXT,
  person_decision TEXT,
  speaker       INTEGER,
  text          TEXT NOT NULL,          -- plain words, for search and embedding
  body          TEXT NOT NULL,          -- the full record as JSON, weights and all
  audio_blob    TEXT,                   -- digest into archive.py, may be NULL
  audio_offset  REAL,
  audio_seconds REAL,
  config_digest TEXT,                   -- which pipeline measured it
  device        TEXT NOT NULL,
  hlc           TEXT NOT NULL,
  written_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS obs_time ON observations(started_at);
CREATE INDEX IF NOT EXISTS obs_person ON observations(person_id);
CREATE INDEX IF NOT EXISTS obs_session ON observations(session);

-- Something worked out. statement, certainty and provenance never change once
-- written; a revision is a new row plus a 'replaces' edge.
CREATE TABLE IF NOT EXISTS beliefs (
  id            TEXT PRIMARY KEY,
  statement     TEXT NOT NULL,
  certainty     REAL NOT NULL,          -- how firmly held, 0..1, fixed at write
  weight        REAL NOT NULL,          -- how live it is, decays, mutable
  formed_at     REAL NOT NULL,          -- when the conclusion was drawn
  about         TEXT,                   -- optional subject handle, free text
  author        TEXT NOT NULL,          -- which caller wrote it
  last_touched  REAL NOT NULL,          -- when weight was last recomputed
  times_returned INTEGER NOT NULL DEFAULT 0,
  device        TEXT NOT NULL,
  hlc           TEXT NOT NULL,
  written_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS bel_formed ON beliefs(formed_at);

-- Every connection, in one table, because a belief points at observations and
-- at other beliefs and both are edges. type is free text on purpose.
CREATE TABLE IF NOT EXISTS edges (
  id         TEXT PRIMARY KEY,
  src_kind   TEXT NOT NULL,             -- 'belief' | 'observation'
  src_id     TEXT NOT NULL,
  dst_kind   TEXT NOT NULL,
  dst_id     TEXT NOT NULL,
  type       TEXT NOT NULL,
  created_at REAL NOT NULL,
  device     TEXT NOT NULL,
  hlc        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edge_src ON edges(src_id, type);
CREATE INDEX IF NOT EXISTS edge_dst ON edges(dst_id, type);
CREATE INDEX IF NOT EXISTS edge_type ON edges(type);

-- The measured census of connection types. Not a constraint, a record.
CREATE TABLE IF NOT EXISTS edge_types (
  type       TEXT PRIMARY KEY,
  n          INTEGER NOT NULL,
  first_seen REAL NOT NULL,
  last_seen  REAL NOT NULL
);

-- What a union of two devices cannot express on its own.
CREATE TABLE IF NOT EXISTS tombstones (
  id        TEXT PRIMARY KEY,
  kind      TEXT NOT NULL,
  reason    TEXT NOT NULL,
  at        REAL NOT NULL,
  device    TEXT NOT NULL,
  hlc       TEXT NOT NULL
);

-- Who asked, what they asked, what left the store. Content is deliberately not
-- copied in here: see gate.py.
CREATE TABLE IF NOT EXISTS access_log (
  id         TEXT PRIMARY KEY,
  at         REAL NOT NULL,
  caller     TEXT NOT NULL,
  tool       TEXT NOT NULL,
  arguments  TEXT NOT NULL,
  decision   TEXT NOT NULL,             -- 'allowed' | 'denied'
  reason     TEXT,
  n_returned INTEGER NOT NULL DEFAULT 0,
  returned   TEXT NOT NULL DEFAULT '[]' -- ids only, never content
);
CREATE INDEX IF NOT EXISTS log_at ON access_log(at);
CREATE INDEX IF NOT EXISTS log_caller ON access_log(caller);

-- Manual approval. A request waits here until a person decides, or until it
-- times out. Content is deliberately NOT stored: the row keeps ids and counts,
-- the same rule the access log follows, and the preview of what would actually
-- be handed over travels over the transient channel to the viewer and is never
-- written down. An approval queue holding the text of everything anyone asked
-- for would be a second copy of the person's speech under none of the same
-- rules, and forget() would have to reach it.
CREATE TABLE IF NOT EXISTS approvals (
  id         TEXT PRIMARY KEY,
  at         REAL NOT NULL,
  caller     TEXT NOT NULL,
  tool       TEXT NOT NULL,
  arguments  TEXT NOT NULL,
  n_would_return INTEGER NOT NULL DEFAULT 0,
  ids        TEXT NOT NULL DEFAULT '[]',
  decision   TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|denied|timeout
  decided_at REAL,
  decided_by TEXT,
  standing_minutes REAL
);
CREATE INDEX IF NOT EXISTS appr_pending ON approvals(decision, at);

-- A kind of request approved in advance, so the person is not asked the same
-- question forty times. The grain is (caller, tool): finer than that and the
-- standing grant becomes a policy language, which is the judgement layer this
-- round is explicitly not building.
CREATE TABLE IF NOT EXISTS standing_approvals (
  caller     TEXT NOT NULL,
  tool       TEXT NOT NULL,
  granted_at REAL NOT NULL,
  expires_at REAL,
  note       TEXT,
  PRIMARY KEY (caller, tool)
);

-- Flat per-caller rules. The interesting version needs judgement and judgement
-- needs a model; this is the table that stands in for it.
-- A caller cannot exist without a secret: see gate.add_caller, which generates
-- one when none is given and has no path that leaves these NULL. They are
-- nullable in the schema only so that a store written before secrets existed
-- can be opened, and gate.authenticate refuses NULL, so such a caller cannot
-- connect rather than being trusted by default.
CREATE TABLE IF NOT EXISTS callers (
  caller      TEXT PRIMARY KEY,
  can_read    INTEGER NOT NULL DEFAULT 0,
  can_write   INTEGER NOT NULL DEFAULT 0,
  note        TEXT,
  added_at    REAL NOT NULL,
  secret_salt BLOB,
  secret_hash BLOB,
  secret_set_at REAL
);

-- Capture state and its history. Append-only; state() reads the latest row.
CREATE TABLE IF NOT EXISTS capture_state (
  id      TEXT PRIMARY KEY,
  at      REAL NOT NULL,
  mode    TEXT NOT NULL,                -- 'capturing' | 'paused'
  note    TEXT,
  device  TEXT NOT NULL
);
"""

# distance_metric=cosine, not the vec0 default of L2. The embeddings are
# length-normalised, so L2 and cosine rank identically, but L2 comes back on a
# 0..2 scale and every "similarity = 1 - distance" downstream would be silently
# negative. Saying which metric is meant is cheaper than remembering.
VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS obs_vec USING vec0(
  id TEXT PRIMARY KEY,
  embedding float[{EMBED_DIMS}] distance_metric=cosine
);
CREATE VIRTUAL TABLE IF NOT EXISTS bel_vec USING vec0(
  id TEXT PRIMARY KEY,
  embedding float[{EMBED_DIMS}] distance_metric=cosine
);
"""


class StaleVectors(Exception):
    """The stored vectors came out of a different embedding model."""


class Store(sqlite3.Connection):
    """A connection that also knows which device it is and what time it is.

    A subclass rather than a wrapper so that every sqlite3 method stays
    available unchanged; the store is a database and nothing is gained by
    hiding that behind a facade.
    """

    device = None
    clock = None


def _device_id(db):
    row = db.execute("SELECT value FROM meta WHERE key='device'").fetchone()
    if row:
        return row[0]
    d = uuid.uuid4().hex[:8]
    db.execute("INSERT INTO meta(key,value) VALUES('device',?)", (d,))
    return d


def open(path=None, check_vectors=True):
    """The store, created if absent, with sqlite-vec loaded."""
    import sqlite_vec
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(p), factory=Store)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(SCHEMA)
    db.executescript(VEC_SCHEMA)
    # stores written before callers had secrets. Adding the columns leaves
    # every existing caller with NULL, which authenticate() treats as
    # unusable -- the failure mode asked for.
    have = {r[1] for r in db.execute("PRAGMA table_info(callers)")}
    for col, decl in (('secret_salt', 'BLOB'), ('secret_hash', 'BLOB'),
                      ('secret_set_at', 'REAL')):
        if col not in have:
            db.execute(f"ALTER TABLE callers ADD COLUMN {col} {decl}")

    want = {"schema_version": str(SCHEMA_VERSION),
            "embed_model": EMBED_MODEL, "embed_dims": str(EMBED_DIMS)}
    for k, v in want.items():
        row = db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        if row is None:
            db.execute("INSERT INTO meta(key,value) VALUES(?,?)", (k, v))
        elif row[0] != v and check_vectors:
            n = db.execute("SELECT count(*) FROM obs_vec").fetchone()[0]
            if n:
                raise StaleVectors(
                    f"This store's {k} is {row[0]!r}; this build uses {v!r}, and "
                    f"there are {n} stored vectors.\nDistances between vectors "
                    "from two models are still numbers and still rank results,\n"
                    "wrongly. Re-embed everything or open a different store.")
            db.execute("UPDATE meta SET value=? WHERE key=?", (v, k))
    db.commit()
    # The gate protects the MCP surface; the filesystem protects the file. A
    # local process that can read this database does not need to authenticate
    # to anything, so it is made unreadable to other accounts here rather than
    # left to the umask.
    try:
        p.chmod(0o600)
        p.parent.chmod(0o700)
    except OSError:
        pass
    db.device = _device_id(db)
    db.clock = Clock(db.device)
    db.commit()
    return db


# --- embeddings -------------------------------------------------------------
_MODEL = None


def embedder():
    """Loaded once, lazily. Small, local, CPU: this ranks text by meaning, it
    does not reason about it, and nothing in this round does."""
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _MODEL


def embed(texts):
    import numpy as np
    v = embedder().encode(list(texts), normalize_embeddings=True,
                          show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def _vec_bytes(v):
    return v.astype("float32").tobytes()


# --- writing ----------------------------------------------------------------
def add_observation(db, kind, started_at, ended_at, text, body,
                    session=None, person_id=None, person=None,
                    person_decision=None, speaker=None, audio=None,
                    config_digest=None, embed_now=True):
    """Record something that happened. There is no update counterpart."""
    oid = ulid(started_at * 1000)
    a = audio or {}
    db.execute(
        "INSERT INTO observations(id,kind,started_at,ended_at,session,person_id,"
        "person,person_decision,speaker,text,body,audio_blob,audio_offset,"
        "audio_seconds,config_digest,device,hlc,written_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, kind, started_at, ended_at, session, person_id, person,
         person_decision, speaker, text, json.dumps(body, separators=(',', ':')),
         a.get('blob'), a.get('offset'), a.get('seconds'), config_digest,
         db.device, db.clock.tick(), time.time()))
    if embed_now and text.strip():
        db.execute("INSERT INTO obs_vec(id,embedding) VALUES(?,?)",
                   (oid, _vec_bytes(embed([text])[0])))
    db.commit()
    return oid


def add_answer(db, text, cites, author, model=None, prompt=None,
               started_at=None, ended_at=None, session=None, about=None):
    """What a model said, recorded as an observation like anything else.

    A model answering is a thing that happened and the being witnessed it, so
    it is an observation by the same definition speech is. kind is 'answer'.

    The reason is not auditing. A belief fades in prominence, and when an old
    one is revisited the reasoning that produced it should still be underneath
    it, immutable and undecayed -- so a belief cites both the speech that
    prompted it and the answer that produced it, and neither is a summary of
    the other. person carries the model, with person_decision 'model' so
    nothing can mistake it for a recognised human voice.

    cites is required and must name real observations. An answer with no
    citations is a claim from nowhere, and -- the sharper reason -- forget()
    reaches model output by following exactly these edges. An uncited answer
    would be the one thing in the store that repeats the person's words back
    and cannot be found by the control meant to delete them.
    """
    if not cites:
        raise ValueError(
            "a model answer must cite the observations it drew on: without "
            "them forget() cannot reach it, and it repeats the person's words")
    for oid in cites:
        if db.execute("SELECT 1 FROM observations WHERE id=?",
                      (oid,)).fetchone() is None:
            raise LookupError(f"cannot cite {oid}: no such observation")
    now = time.time()
    oid = add_observation(
        db, 'answer', started_at if started_at is not None else now,
        ended_at if ended_at is not None else now,
        text=text,
        body={'model': model, 'prompt': prompt, 'about': about,
              'cites': list(cites)},
        session=session, person=model or author, person_decision='model',
        speaker=None, audio=None, config_digest=None)
    for src in cites:
        link(db, 'observation', oid, 'observation', src, 'cites')
    return oid


def note_edge_type(db, t, when):
    db.execute(
        "INSERT INTO edge_types(type,n,first_seen,last_seen) VALUES(?,1,?,?) "
        "ON CONFLICT(type) DO UPDATE SET n=n+1, last_seen=excluded.last_seen",
        (t, when, when))


def link(db, src_kind, src_id, dst_kind, dst_id, type):
    """One connection. type is free text and is counted as it is used."""
    now = time.time()
    eid = ulid(now * 1000)
    db.execute("INSERT INTO edges(id,src_kind,src_id,dst_kind,dst_id,type,"
               "created_at,device,hlc) VALUES(?,?,?,?,?,?,?,?,?)",
               (eid, src_kind, src_id, dst_kind, dst_id, type, now,
                db.device, db.clock.tick()))
    note_edge_type(db, type, now)
    db.commit()
    return eid


def add_belief(db, statement, certainty, sources, author, about=None,
               weight=1.0, replaces=None, links=(), formed_at=None):
    """Write a conclusion, with the observations it was drawn from.

    sources is required and must be non-empty. A belief with no provenance
    cannot be rebuilt, cannot be checked, and cannot be reached by forget(); it
    would be the one thing in here that is neither record nor derivable, so it
    is refused rather than accepted and regretted.
    """
    if not sources:
        raise ValueError(
            "a belief needs the observations it came from: without them the "
            "belief layer stops being rebuildable and forget() cannot reach it")
    # about is required, for the same shape of reason sources is. A belief with
    # no subject is findable only by embedding similarity, which is the fuzzy
    # path and the one that fails on the questions that matter -- everything
    # this person has concluded about their health, all at once, exactly. The
    # cost of requiring it is one word from the writer; the cost of leaving it
    # optional is that it is set sometimes, which is worse than never, because
    # a filter that silently covers part of the store looks like it covers all
    # of it.
    #
    # No enumeration, deliberately. Free text and censused after the fact, the
    # same treatment edge types get in edge_types: after real use there is a
    # measured vocabulary to promote rather than a guessed one. See
    # about_census().
    about = (about or '').strip()
    if not about:
        raise ValueError(
            "a belief needs an 'about': one or two words for what it concerns, "
            "free text, so it can be found by subject and not only by "
            "resemblance")
    now = time.time()
    formed = now if formed_at is None else formed_at
    bid = ulid(formed * 1000)
    db.execute(
        "INSERT INTO beliefs(id,statement,certainty,weight,formed_at,about,"
        "author,last_touched,times_returned,device,hlc,written_at) "
        "VALUES(?,?,?,?,?,?,?,?,0,?,?,?)",
        (bid, statement, float(certainty), float(weight), formed, about, author,
         now, db.device, db.clock.tick(), now))
    db.execute("INSERT INTO bel_vec(id,embedding) VALUES(?,?)",
               (bid, _vec_bytes(embed([statement])[0])))
    for oid in sources:
        link(db, 'belief', bid, 'observation', oid, 'came-from')
    if replaces:
        link(db, 'belief', bid, 'belief', replaces, 'replaces')
    for t, kind, target in links:
        link(db, 'belief', bid, kind, target, t)
    db.commit()
    return bid


def about_census(db):
    """What the free-text subjects have actually turned into.

    The same reason edge_types exists: nothing constrains the value, so the
    only way to learn whether it is carrying meaning or filling up with
    'general' is to count it.
    """
    return [dict(r) for r in db.execute(
        "SELECT about, count(*) AS n, min(formed_at) AS first_seen, "
        "max(formed_at) AS last_seen FROM beliefs GROUP BY about ORDER BY n DESC")]


# --- decay ------------------------------------------------------------------
def decayed_weight(weight, last_touched, now=None):
    """Exponential decay by elapsed time, computed at read.

    Nothing runs on a timer. A belief's stored weight is only ever correct as
    of last_touched, and this is what turns it into the weight right now, so a
    store that sat closed for a year comes back with the right numbers rather
    than the numbers it was frozen at.
    """
    now = time.time() if now is None else now
    days = max(now - last_touched, 0.0) / 86400.0
    return float(weight) * (0.5 ** (days / HALF_LIFE_DAYS))


def touch_belief(db, bid, now=None):
    """Returning to a belief is what keeps it alive. Decay first, then add."""
    now = time.time() if now is None else now
    r = db.execute("SELECT weight,last_touched,times_returned FROM beliefs "
                   "WHERE id=?", (bid,)).fetchone()
    if r is None:
        return None
    w = min(decayed_weight(r['weight'], r['last_touched'], now) + RETURN_BONUS,
            WEIGHT_CEILING)
    db.execute("UPDATE beliefs SET weight=?,last_touched=?,times_returned=? "
               "WHERE id=?", (w, now, r['times_returned'] + 1, bid))
    db.commit()
    return w


# --- reading ----------------------------------------------------------------
def replaced_by(db, bid):
    r = db.execute("SELECT src_id FROM edges WHERE dst_id=? AND type='replaces'",
                   (bid,)).fetchone()
    return r['src_id'] if r else None


def current_version(db, bid, _seen=None):
    """Follow the replacement chain to whatever is current now."""
    seen = _seen or set()
    while bid not in seen:
        seen.add(bid)
        nxt = replaced_by(db, bid)
        if not nxt:
            return bid
        bid = nxt
    return bid


def get_observation(db, oid):
    r = db.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d['body'] = json.loads(d['body'])
    return d


def get_belief(db, bid, reader=None):
    """One belief, and a return counted whenever anything reads it through the
    gate.

    WHY NOT WHO. This briefly excluded reads by the belief's own author, to
    stop a caller keeping its conclusions alive by looping over them. That rule
    was wrong, and it was wrong for the same reason the approval exemption
    beside it was: a caller is a name, not a continuing mind. "claude" next
    month is a fresh process with no memory of what "claude" concluded before,
    so it reading its own old belief is not housekeeping -- it is a mind
    finding something it did not know, which is precisely the use that should
    count. The rule blocked the common case to prevent the rare one.

    What would actually separate them is WHEN, not WHO: a re-read minutes later
    inside one conversation is housekeeping, a read a month later is use. That
    needs a session concept for callers, which does not exist here yet.

    So: reader set means something returned to this belief, and it counts.
    reader=None means nobody is returning to anything -- the viewer, the
    rebuild check, and every internal lookup -- and none of those move a
    weight. A superseded belief never counts either, because reading what the
    system used to think, to see what changed, is not evidence the old version
    is useful.
    """
    r = db.execute("SELECT * FROM beliefs WHERE id=?", (bid,)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d['weight_now'] = decayed_weight(d['weight'], d['last_touched'])
    d['faint'] = d['weight_now'] < FAINT
    d['sources'] = [x['dst_id'] for x in db.execute(
        "SELECT dst_id FROM edges WHERE src_id=? AND type='came-from'", (bid,))]
    d['replaced_by'] = replaced_by(db, bid)
    d['superseded'] = d['replaced_by'] is not None
    d['current'] = current_version(db, bid)
    d['replaces'] = [x['dst_id'] for x in db.execute(
        "SELECT dst_id FROM edges WHERE src_id=? AND type='replaces'", (bid,))]
    d['links'] = [dict(x) for x in db.execute(
        "SELECT type,dst_kind,dst_id FROM edges WHERE src_id=? "
        "AND type NOT IN ('came-from','replaces')", (bid,))]
    d['return_counted'] = bool(reader and not d['superseded'])
    if d['return_counted']:
        d['weight_now'] = touch_belief(db, bid)
        d['times_returned'] += 1
    return d


def search(db, query, limit=10, kinds=('observation', 'belief'),
           include_superseded=False):
    """Retrieval by meaning, over both kinds, ranked into one list.

    Superseded beliefs are out by default: asking what the system thinks should
    not return four generations of what it used to think. They are reachable by
    asking for them, and by following any answer's chain.
    """
    q = _vec_bytes(embed([query])[0])
    out = []
    if 'observation' in kinds:
        for r in db.execute(
                "SELECT v.id, v.distance, o.text, o.started_at, o.person, o.kind "
                "FROM obs_vec v JOIN observations o ON o.id=v.id "
                "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance",
                (q, limit)):
            out.append({'kind': 'observation', 'id': r['id'],
                        'distance': r['distance'], 'text': r['text'],
                        'at': r['started_at'], 'person': r['person'],
                        'observation_kind': r['kind'], 'score': 1.0 - r['distance']})
    if 'belief' in kinds:
        for r in db.execute(
                "SELECT v.id, v.distance, b.statement, b.certainty, b.weight, "
                "b.last_touched, b.formed_at FROM bel_vec v "
                "JOIN beliefs b ON b.id=v.id "
                "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance",
                (q, limit)):
            sup = replaced_by(db, r['id'])
            if sup and not include_superseded:
                continue
            w = decayed_weight(r['weight'], r['last_touched'])
            out.append({'kind': 'belief', 'id': r['id'],
                        'distance': r['distance'], 'text': r['statement'],
                        'certainty': r['certainty'], 'weight': w,
                        'faint': w < FAINT, 'superseded': bool(sup),
                        'replaced_by': sup, 'at': r['formed_at'],
                        # a faint belief is not hidden, only ranked behind
                        'score': (1.0 - r['distance']) * (0.5 + min(w, 1.0) / 2)})
    out.sort(key=lambda x: -x['score'])
    return out[:limit]


# --- can the belief layer be thrown away and rebuilt ------------------------
def observation_fingerprint(db):
    """A digest over the record, to prove it is untouched by belief work."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for r in db.execute("SELECT id,kind,started_at,ended_at,text,body,"
                        "audio_blob,audio_offset,config_digest FROM observations "
                        "ORDER BY id"):
        h.update(repr(tuple(r)).encode())
    return h.hexdigest()


def derivations(db):
    """Every belief reduced to what a rebuild would need: the statement, the
    certainty, and the ids of the observations it was drawn from.

    This is not a backup of the belief layer. It is the shape of its input: a
    real deriver reads the observations and produces these, and this function
    exists so that the round trip can be run without one.
    """
    out = []
    for b in db.execute("SELECT id,statement,certainty,about,author,formed_at "
                        "FROM beliefs ORDER BY formed_at, id"):
        src = [r['dst_id'] for r in db.execute(
            "SELECT dst_id FROM edges WHERE src_id=? AND type='came-from' "
            "ORDER BY dst_id", (b['id'],))]
        rep = [r['dst_id'] for r in db.execute(
            "SELECT dst_id FROM edges WHERE src_id=? AND type='replaces'",
            (b['id'],))]
        out.append({'statement': b['statement'], 'certainty': b['certainty'],
                    'about': b['about'], 'author': b['author'],
                    'formed_at': b['formed_at'], 'sources': src,
                    'replaces': rep, 'old_id': b['id']})
    return out


def drop_belief_layer(db):
    """Delete every belief, every edge and every belief vector.

    Observations are not mentioned. That is the point being demonstrated.
    """
    n = db.execute("SELECT count(*) FROM beliefs").fetchone()[0]
    e = db.execute("SELECT count(*) FROM edges").fetchone()[0]
    db.execute("DELETE FROM bel_vec")
    db.execute("DELETE FROM beliefs")
    db.execute("DELETE FROM edges")
    db.commit()
    return {'beliefs': n, 'edges': e}


def rebuild_belief_layer(db, derivs):
    """Put beliefs back from derivations and the observations they name.

    Every source id is resolved against the observations table before anything
    is written, so a rebuild that has lost its record fails here rather than
    quietly producing beliefs that point at nothing.
    """
    remap = {}
    for d in sorted(derivs, key=lambda x: x['formed_at']):
        for oid in d['sources']:
            if db.execute("SELECT 1 FROM observations WHERE id=?",
                          (oid,)).fetchone() is None:
                raise LookupError(
                    f"cannot rebuild {d['statement']!r}: it was drawn from "
                    f"observation {oid}, which is not in the record")
        new = add_belief(db, d['statement'], d['certainty'], d['sources'],
                         author=d['author'], about=d['about'],
                         formed_at=d['formed_at'],
                         replaces=remap.get((d['replaces'] or [None])[0]))
        remap[d['old_id']] = new
    return remap


def rebuild_check(db, verbose=True):
    """Run the round trip and report, rather than asserting it is possible.

    Four things are checked. Every belief has provenance. Every provenance
    pointer resolves to an observation that is actually there. The record is
    byte-identical before and after the belief layer is destroyed and remade.
    And the remade layer says the same things, with the same certainties, drawn
    from the same observations, in the same replacement order.

    What this does NOT show is that a deriver could invent the same sentences
    from the observations unaided; that needs a model and there is none in this
    round. What it shows is the property that has to hold for such a deriver to
    be possible at all: the record is sufficient and the belief layer is
    free-standing.
    """
    report = {}
    orphans = [r['id'] for r in db.execute(
        "SELECT b.id FROM beliefs b WHERE NOT EXISTS (SELECT 1 FROM edges e "
        "WHERE e.src_id=b.id AND e.type='came-from')")]
    dangling = [dict(r) for r in db.execute(
        "SELECT e.src_id, e.dst_id FROM edges e WHERE e.type='came-from' "
        "AND NOT EXISTS (SELECT 1 FROM observations o WHERE o.id=e.dst_id)")]
    report['beliefs_without_provenance'] = orphans
    report['provenance_pointing_nowhere'] = dangling

    before = observation_fingerprint(db)
    n_obs = db.execute("SELECT count(*) FROM observations").fetchone()[0]
    derivs = derivations(db)
    was = [(d['statement'], round(d['certainty'], 6), tuple(d['sources']))
           for d in derivs]

    dropped = drop_belief_layer(db)
    report['dropped'] = dropped
    report['observations_after_drop'] = db.execute(
        "SELECT count(*) FROM observations").fetchone()[0]
    report['record_survived_drop'] = observation_fingerprint(db) == before

    rebuild_belief_layer(db, derivs)
    now = [(d['statement'], round(d['certainty'], 6), tuple(d['sources']))
           for d in derivations(db)]
    report['record_survived_rebuild'] = observation_fingerprint(db) == before
    report['rebuilt_beliefs'] = len(now)
    report['identical'] = sorted(was) == sorted(now)
    report['observations'] = n_obs
    report['record_fingerprint'] = before
    report['ok'] = (not orphans and not dangling and report['identical']
                    and report['record_survived_drop']
                    and report['record_survived_rebuild'])
    if verbose:
        print(f"record: {n_obs} observation(s), fingerprint {before}")
        print(f"beliefs without provenance: {len(orphans)}")
        print(f"provenance pointing nowhere: {len(dangling)}")
        print(f"dropped: {dropped['beliefs']} belief(s), {dropped['edges']} edge(s)")
        print(f"observations after the drop: {report['observations_after_drop']}")
        print(f"record fingerprint unchanged by the drop: "
              f"{report['record_survived_drop']}")
        print(f"rebuilt from the record: {report['rebuilt_beliefs']} belief(s)")
        print(f"record fingerprint unchanged by the rebuild: "
              f"{report['record_survived_rebuild']}")
        print(f"statements, certainties and sources identical: "
              f"{report['identical']}")
        print(f"OK: {report['ok']}")
    return report
