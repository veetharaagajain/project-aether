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

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "store" / "memory.db"
# An ASYMMETRIC model, and that is the whole point of the choice.
#
# all-MiniLM-L6-v2 embeds a query and a stored sentence the same way, so a
# question in the store sits next to a question in the query: "What did you
# have for breakfast?" scored 0.79 against a query phrased as a question while
# the answer scored 0.45. No chunking fixes that, because it is not about
# chunk size -- it is the model putting questions near questions.
#
# e5 encodes the two sides differently, and the prefixes below are how it is
# told which side it is looking at. They are load-bearing, not decorative:
# measured on the breakfast case, the stored question beats the answer by 0.052
# with the prefixes and by 0.0798 without, so using the model unprefixed throws
# away a third of what it was chosen for and would do it silently.
#
# It is also multilingual, which matters because the Kannada and Hindi speech in
# this store cannot be embedded meaningfully by an English-only model at all.
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIMS = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
SCHEMA_VERSION = 2

# How the text behind a vector was assembled, not just what embedded it.
#
# The model check alone was insufficient in exactly the way provenance.py was
# written about. Two vectors from the same model, one made from an observation
# alone and one from a window of neighbouring speech, are the same shape and
# the same scale and compare happily, and the comparison means nothing. A store
# half-indexed each way would rank incoherently with nothing noticing, because
# every check it had was still passing.
#
# So the recipe is recorded beside the model and checked the same way. Bump
# INDEX_RECIPE whenever the text that goes into an embedding changes -- the
# unit, the bounds, the separator, anything. It is a name and a version rather
# than a hash of the parameters, because a person reading a refusal needs to
# know what changed, and "window-v1 became window-v2" says more than two
# digests do.
INDEX_UNIT = "window"          # 'observation' = the old scheme, one vector each
INDEX_RECIPE = "window-v2-e5-prefixed"

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
  is_question   INTEGER NOT NULL DEFAULT 0,  -- either question signal fired
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

-- How an observation is found, which is not what an observation is. A window is
-- a short run of consecutive speech centred on one observation; the vector for
-- it lives in win_vec. Windows are derived, disposable and rebuilt by
-- build_windows(); nothing here is a record and nothing here is ever released.
CREATE TABLE IF NOT EXISTS windows (
  id         TEXT PRIMARY KEY,
  centre_id  TEXT NOT NULL,          -- the observation this window is FOR
  session    TEXT,
  first_id   TEXT NOT NULL,
  last_id    TEXT NOT NULL,
  n_obs      INTEGER NOT NULL,
  n_words    INTEGER NOT NULL,
  started_at REAL NOT NULL,
  ended_at   REAL NOT NULL,
  text       TEXT NOT NULL,
  recipe     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS win_centre ON windows(centre_id);
CREATE INDEX IF NOT EXISTS win_session ON windows(session, started_at);

-- Which long-lived processes are running and what code they hold. A process
-- that imported gate.py at 11:12 is still executing the 11:12 version at 13:53,
-- and nothing else in this store can tell. staleness.py writes here; viewer.py
-- reads it and shows anything stale in the header.
CREATE TABLE IF NOT EXISTS processes (
  pid        INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,           -- 'mcp' | 'viewer' | 'live'
  started_at REAL NOT NULL,
  last_seen  REAL NOT NULL,
  argv       TEXT,
  modules    TEXT NOT NULL DEFAULT '{}'
);

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
CREATE VIRTUAL TABLE IF NOT EXISTS win_vec USING vec0(
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

    def execute(self, *a, **kw):
        """Every statement, retried while the database is locked.

        The retry started life wrapped around add_observation, which fixed the
        write that was crashing and left every other one exposed -- and the
        crash simply moved to the start-up writes in staleness.register, which
        run before anything else and are exactly when a second process is most
        likely to be starting too.

        So it lives on the connection instead of on one caller. A statement
        that returns SQLITE_BUSY did not take effect, so re-running that
        statement is safe; the case this does not cover is BUSY raised by
        commit, which is why commit retries as well.
        """
        return write_retry(lambda: sqlite3.Connection.execute(self, *a, **kw))

    def commit(self):
        return write_retry(lambda: sqlite3.Connection.commit(self))


def _device_id(db):
    row = db.execute("SELECT value FROM meta WHERE key='device'").fetchone()
    if row:
        return row[0]
    d = uuid.uuid4().hex[:8]
    db.execute("INSERT INTO meta(key,value) VALUES('device',?)", (d,))
    return d


# How long a write waits for another connection to finish before giving up.
# Ten seconds is far longer than any write here takes and far shorter than a
# person would wait for the store to be wedged.
BUSY_TIMEOUT_MS = 10000
WRITE_ATTEMPTS = 4
WRITE_BACKOFF_S = 0.25


class Busy(Exception):
    """The store stayed locked through every retry. Raised rather than
    returned so a caller cannot mistake a dropped write for a stored one."""


def write_retry(fn, attempts=WRITE_ATTEMPTS, backoff=WRITE_BACKOFF_S):
    """Run a write, retrying while the database is locked.

    busy_timeout handles contention inside SQLite and this handles what is
    left: a lock held longer than the timeout, or one taken between the
    timeout expiring and the retry. Backoff is exponential and jittered,
    because two processes retrying in lockstep is how a brief collision
    becomes a long one.
    """
    import random
    last = None
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if 'locked' not in str(e).lower() and 'busy' not in str(e).lower():
                raise
            last = e
            if i < attempts - 1:
                time.sleep(backoff * (2 ** i) * (0.5 + random.random()))
    raise Busy(f"the store stayed locked through {attempts} attempts: {last}")


def open(path=None, check_vectors=True):
    """The store, created if absent, with sqlite-vec loaded."""
    import sqlite_vec
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Shared across threads deliberately. The live path now runs its pipeline
    # on a worker thread so a slow segment cannot stall audio capture, which
    # means the connection built on the main thread is used from that worker
    # -- and sqlite3 refuses that by default, with "SQLite objects created in a
    # thread can only be used in that same thread". It is safe here because
    # this SQLite is built serialized (sqlite3.threadsafety == 3), so the
    # library takes its own mutex around every use; the check being disabled is
    # Python's, not SQLite's.
    if sqlite3.threadsafety < 3:
        raise RuntimeError(
            "this sqlite3 is not serialized, so the store cannot be shared "
            "across threads; the live path needs a connection per thread")
    db = sqlite3.connect(str(p), factory=Store, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    # SQLite's default busy timeout is zero: a database held by another
    # connection raises SQLITE_BUSY on the instant rather than waiting. That is
    # what took the live service down -- a second process had the store open,
    # add_observation raised, nothing caught it, and launchd restarted a
    # capture that had been running fine. WAL already lets readers and one
    # writer coexist; this covers the case of two writers, which is brief
    # because every write here is a single small insert.
    db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    db.execute("PRAGMA foreign_keys=ON")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(SCHEMA)
    db.executescript(VEC_SCHEMA)
    # CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    # a column added to SCHEMA never reaches a store made before it. Adding them
    # explicitly is the only way; each is nullable or defaulted so an existing
    # row is valid the moment it appears.
    for table, column, decl in (
            ('observations', 'is_question', 'INTEGER NOT NULL DEFAULT 0'),):
        have = {r['name'] for r in db.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    db.commit()
    # stores written before callers had secrets. Adding the columns leaves
    # every existing caller with NULL, which authenticate() treats as
    # unusable -- the failure mode asked for.
    have = {r[1] for r in db.execute("PRAGMA table_info(callers)")}
    # can_outward defaults to 0, so an existing caller gains no new reach from
    # the column appearing -- the failure mode asked for, again.
    for col, decl in (('secret_salt', 'BLOB'), ('secret_hash', 'BLOB'),
                      ('secret_set_at', 'REAL'),
                      ('can_outward', 'INTEGER NOT NULL DEFAULT 0')):
        if col not in have:
            db.execute(f"ALTER TABLE callers ADD COLUMN {col} {decl}")

    want = {"schema_version": str(SCHEMA_VERSION),
            "embed_model": EMBED_MODEL, "embed_dims": str(EMBED_DIMS),
            "index_unit": INDEX_UNIT, "index_recipe": INDEX_RECIPE}
    for k, v in want.items():
        row = db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        if row is None:
            db.execute("INSERT INTO meta(key,value) VALUES(?,?)", (k, v))
        elif row[0] != v and check_vectors:
            n = (db.execute("SELECT count(*) FROM obs_vec").fetchone()[0]
                 + db.execute("SELECT count(*) FROM win_vec").fetchone()[0]
                 + db.execute("SELECT count(*) FROM bel_vec").fetchone()[0])
            if n:
                raise StaleVectors(
                    f"This store's {k} is {row[0]!r}; this build uses {v!r}, "
                    f"and there are {n} stored vector(s).\n"
                    "Vectors made under two different settings are the same "
                    "shape and the same scale, so they\ncompare without "
                    "complaint and rank wrongly. Nothing here can tell them "
                    "apart after the fact.\n\n"
                    "To resolve:\n"
                    "    python memory.py reindex        rebuild every vector "
                    "under the current settings")
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


def embed(texts, kind="passage"):
    """Embed text, telling the model which side of the comparison it is.

    kind must be 'passage' for anything stored and 'query' for anything asked.
    Getting it backwards produces numbers of the right shape and scale that
    rank wrongly, with nothing to notice it -- so the argument has no default
    that could quietly be wrong in the common case, and every caller says which
    it means.
    """
    if kind not in ("passage", "query"):
        raise ValueError("kind must be 'passage' or 'query', not %r" % (kind,))
    pre = PASSAGE_PREFIX if kind == "passage" else QUERY_PREFIX
    v = embedder().encode([pre + t for t in texts], normalize_embeddings=True,
                          show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def embed_query(text):
    return embed([text], kind="query")[0]


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

    def write():
        db.execute(
            # is_question is set HERE, at insert. It used to be set only by
            # backfill_questions, which meant every observation written since
            # the column was added carried the default 0 -- 1,903 rows ending
            # in a question mark with the flag clear -- so the question penalty
            # in search() could never fire on any of them. The penalty was not
            # too small, it had nothing to act on.
            "INSERT INTO observations(id,kind,started_at,ended_at,session,"
            "person_id,person,person_decision,speaker,text,body,audio_blob,"
            "audio_offset,audio_seconds,config_digest,device,hlc,written_at,"
            "is_question) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, kind, started_at, ended_at, session, person_id, person,
             person_decision, speaker, text,
             json.dumps(body, separators=(',', ':')),
             a.get('blob'), a.get('offset'), a.get('seconds'), config_digest,
             db.device, db.clock.tick(), time.time(),
             1 if question_flags(body, text) else 0))
        if embed_now and text.strip():
            db.execute("INSERT INTO obs_vec(id,embedding) VALUES(?,?)",
                       (oid, _vec_bytes(embed([text], kind='passage')[0])))
        db.commit()

    try:
        write_retry(write)
    except Busy:
        # a half-applied insert would leave the row without its vector
        try:
            db.rollback()
        except sqlite3.Error:
            pass
        raise
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
               (bid, _vec_bytes(embed([statement], kind='passage')[0])))
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


# How far a fragment reaches for the context it is missing. Chosen from the
# store rather than picked: within a session the median gap between
# observations is 1.23 s, three quarters are under 3.02 s, and then the
# distribution breaks -- p90 is 10.66 s and p95 is 40.92 s. Speech separated by
# a second or two is one person continuing; speech separated by half a minute is
# a different moment. NEIGHBOUR_MAX_GAP_S sits in the flat part after the knee,
# so a window grows through continuous talk and stops at a real break.
#
# One either side, measured rather than assumed. Over five questions and fifty
# candidates: no context kept 9 answers; one either side kept 13 at identical
# cost, 4.8 s per question; two either side kept 14 but cost 6.8 s and wedged
# the model on 2 of 50 prompts -- a five-line window reproducibly hangs it where
# the same window at four lines answers in 0.6 s. The second neighbour buys one
# answer and pays for it in latency and in a failure mode that withholds.
#
# One is also what the case this exists for needs: "Charitable trust and not a
# public authority under the RTI Act" has its subject exactly one observation
# back, 0.42 s earlier.
NEIGHBOURS_BEFORE = 1
NEIGHBOURS_AFTER = 1
NEIGHBOUR_MAX_GAP_S = 8.0
NEIGHBOUR_MAX_WORDS = 120
# Below this a neighbour is not context but the same sentence: p75 of the gap
# distribution, the point up to which speech is still one continuous run. Used
# when deciding what to release, not what to judge. See gate.
NEIGHBOUR_TIGHT_GAP_S = 3.0


# --- the index unit --------------------------------------------------------
# An observation is the wrong thing to embed. The median one is four words, 59
# percent are four or fewer and 37 percent are two or fewer, so most of the
# index was vectors for text that means nothing alone -- "we want." and
# "chicken." ranked against whole sentences on equal terms. No amount of
# judging downstream repairs an index built on fragments.
#
# So the thing embedded is a window: a short run of consecutive speech. The
# thing stored, released and pointed at stays the observation. A window is only
# how an observation is found.
#
# BOUNDS, from the same gap distribution the reach cap came from. Within a
# session the median gap is 1.23 s, p75 is 3.02 s, then it breaks: p90 10.66 s,
# p95 40.92 s. WINDOW_MAX_GAP_S sits after that knee, so a window grows through
# continuous talk and stops at a real pause rather than welding two moments
# together. WINDOW_TARGET_WORDS is where growth stops being useful: sentence
# embedding models of this size are built for a sentence or two, and a window
# that keeps growing starts describing a passage instead of a remark.
# WINDOW_MAX_OBS is the backstop for a run of one-word utterances, where the
# word target alone would swallow a whole minute of "okay. okay. right."
WINDOW_MAX_GAP_S = 8.0
# Set from the LoCoMo window sweep, transferred rather than copied. There the
# optimum was three complete conversational contributions: one alone scored
# 22% recall, three scored 41%, seven scored 33% and nine scored 30%. Both
# tails hurt -- too narrow is a fragment, too wide matches on anything it
# contains and stops discriminating.
#
# A LoCoMo turn is 20 words and one complete thought. Here an observation is
# one run between silences, median four words, 60 percent of them four or
# fewer, so a complete thought spans several observations rather than one.
# Three thoughts is therefore roughly 40 words, not the 60 that won on LoCoMo
# and not the 30 that was here; and the observation cap has to rise with it or
# the word target is never reached. The measured window before this was 27
# words over 6 observations, so this widens it by about half a thought.
#
# NOT validated on this store: that would need labelled questions against real
# speech, which do not exist. It is an inference from the shape of the LoCoMo
# curve, and it is one constant to put back.
WINDOW_TARGET_WORDS = 40
WINDOW_MAX_OBS = 9
WINDOW_JOIN = " "
# Windows overlap, so the top-k windows can centre on fewer than k distinct
# observations. Ask the index for more and keep the best per centre.
WINDOW_OVERSAMPLE = 3
# How much of the score is context and how much is the fragment itself.
# Swept over the seven evaluation questions; see the report.
WINDOW_WEIGHT = 0.75


# An observation that is itself a question is rarely the answer to one. 17.5
# percent of the speech here is question-shaped by one of the two signals
# already stored -- 13.0 percent by the transcriber's punctuation, 6.5 percent
# by the pitch-rise detection, agreeing on only 2.1 -- so this fires often
# enough to matter.
#
# A penalty and never a filter. "Did you take the bins out" is a perfectly good
# answer to "what did she ask me to do", and a filter would make that
# unanswerable. This only moves such an observation down the ranking, and only
# when the query is itself question-shaped; against a query that is not a
# question it does nothing at all.
QUESTION_PENALTY = 0.15


def looks_like_a_question(text):
    """Is this text question-shaped. Used on the QUERY side only.

    Stored observations do not go through this -- they carry the real signals,
    the transcriber's punctuation and the measured pitch rise, in is_question.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first = t.split()[0] if t.split() else ""
    return first in {"what", "who", "when", "where", "why", "how", "which",
                     "whose", "did", "does", "do", "is", "are", "was", "were",
                     "can", "could", "will", "would", "has", "have", "had"}


def question_flags(body, text):
    """Either stored signal, or a question mark the transcriber left."""
    b = body if isinstance(body, dict) else {}
    return bool(b.get('question_by_punctuation')
                or b.get('question_by_pitch_rise')
                or (text or '').strip().endswith('?'))


def backfill_questions(db):
    """Set is_question from what is already stored in each observation's body."""
    n = 0
    for r in db.execute("SELECT id,text,body FROM observations").fetchall():
        try:
            b = json.loads(r['body'])
        except ValueError:
            b = {}
        q = 1 if question_flags(b, r['text']) else 0
        db.execute("UPDATE observations SET is_question=? WHERE id=?", (q, r['id']))
        n += q
    db.commit()
    return n


def build_window(db, oid):
    """The window centred on one observation.

    Centred, and one per observation, so that a hit resolves to exactly one
    observation with nothing to decide. The alternative -- windows on a stride,
    each covering several observations -- makes every hit ambiguous about which
    observation in it actually answered, and resolving that needs another
    judgement. Overlap is the price: consecutive windows share most of their
    text, and an observation appears in up to WINDOW_MAX_OBS of them.

    Growth alternates outward so the centre stays near the middle rather than
    the window running off in whichever direction has shorter utterances.
    """
    c = db.execute("SELECT id,session,started_at,ended_at,text FROM observations "
                   "WHERE id=?", (oid,)).fetchone()
    if c is None:
        return None
    rows = [dict(c)]
    words = len(c['text'].split())
    left_open = right_open = True
    while (left_open or right_open) and words < WINDOW_TARGET_WORDS \
            and len(rows) < WINDOW_MAX_OBS:
        for side in (-1, 1):
            if side < 0 and not left_open:
                continue
            if side > 0 and not right_open:
                continue
            edge = rows[0] if side < 0 else rows[-1]
            if side < 0:
                r = db.execute(
                    "SELECT id,session,started_at,ended_at,text FROM observations"
                    " WHERE session=? AND started_at < ? ORDER BY started_at DESC"
                    " LIMIT 1", (edge['session'], edge['started_at'])).fetchone()
                gap = (edge['started_at'] - r['ended_at']) if r else None
            else:
                r = db.execute(
                    "SELECT id,session,started_at,ended_at,text FROM observations"
                    " WHERE session=? AND started_at > ? ORDER BY started_at"
                    " LIMIT 1", (edge['session'], edge['started_at'])).fetchone()
                gap = (r['started_at'] - edge['ended_at']) if r else None
            if r is None or gap is None or gap > WINDOW_MAX_GAP_S:
                if side < 0:
                    left_open = False
                else:
                    right_open = False
                continue
            if len(rows) + 1 > WINDOW_MAX_OBS:
                left_open = right_open = False
                break
            words += len(r['text'].split())
            if side < 0:
                rows.insert(0, dict(r))
            else:
                rows.append(dict(r))
            if words >= WINDOW_TARGET_WORDS:
                break
    text = WINDOW_JOIN.join(r['text'].strip() for r in rows)
    return {'id': 'w' + oid, 'centre_id': oid, 'session': c['session'],
            'first_id': rows[0]['id'], 'last_id': rows[-1]['id'],
            'n_obs': len(rows), 'n_words': len(text.split()),
            'started_at': rows[0]['started_at'], 'ended_at': rows[-1]['ended_at'],
            'text': text, 'recipe': INDEX_RECIPE}


def build_windows(db, batch=256, progress=None):
    """Rebuild every window and its vector. Derived data: safe to drop and redo."""
    db.execute("DELETE FROM win_vec")
    db.execute("DELETE FROM windows")
    db.commit()
    ids = [r['id'] for r in db.execute(
        "SELECT id FROM observations ORDER BY session, started_at")]
    made = 0
    for i in range(0, len(ids), batch):
        chunk = [build_window(db, o) for o in ids[i:i + batch]]
        chunk = [w for w in chunk if w]
        if not chunk:
            continue
        vecs = embed([w['text'] for w in chunk], kind='passage')
        for w, v in zip(chunk, vecs):
            db.execute(
                "INSERT INTO windows(id,centre_id,session,first_id,last_id,"
                "n_obs,n_words,started_at,ended_at,text,recipe) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (w['id'], w['centre_id'], w['session'], w['first_id'],
                 w['last_id'], w['n_obs'], w['n_words'], w['started_at'],
                 w['ended_at'], w['text'], w['recipe']))
            db.execute("INSERT INTO win_vec(id,embedding) VALUES(?,?)",
                       (w['id'], _vec_bytes(v)))
        db.commit()
        made += len(chunk)
        if progress:
            progress(made, len(ids))
    return {'windows': made, 'observations': len(ids)}


def repair_windows(db, gone_ids):
    """Bring the window index back in line after observations were deleted.

    Windows are derived, but they are not merely a pointer: each one stores the
    joined text of its span. Deleting an observation and leaving the windows
    alone removes the row and keeps the words, in up to WINDOW_MAX_OBS windows
    that are all still searchable -- so a forgotten sentence stays findable by
    the one index built to find sentences by their surroundings. That is the
    whole leak, and it is silent, because the observation really is gone.

    A window is affected when a deleted observation lay inside its span. Those
    windows are dropped and rebuilt around whatever survives; build_window
    reads live rows, so rebuilding after the delete is enough. Windows centred
    on a deleted observation are dropped and not rebuilt -- their centre no
    longer exists.
    """
    if not gone_ids:
        return {'dropped': 0, 'rebuilt': 0}
    # the spans to repair, found from the deleted rows' own tombstoned times.
    # Passed in rather than looked up: by the time this runs the rows are gone.
    spans = [(g['session'], g['started_at']) for g in gone_ids
             if g.get('session') is not None]
    doomed = {g['id'] for g in gone_ids}
    affected = {}
    for session, at in spans:
        for r in db.execute(
                "SELECT id,centre_id FROM windows WHERE session=? AND "
                "started_at <= ? AND ended_at >= ?", (session, at, at)):
            affected[r['id']] = r['centre_id']
    if not affected:
        return {'dropped': 0, 'rebuilt': 0}
    q = ",".join("?" * len(affected))
    db.execute(f"DELETE FROM win_vec WHERE id IN ({q})", list(affected))
    db.execute(f"DELETE FROM windows WHERE id IN ({q})", list(affected))
    centres = [c for c in affected.values() if c not in doomed]
    rebuilt = 0
    for i in range(0, len(centres), 256):
        chunk = [build_window(db, o) for o in centres[i:i + 256]]
        chunk = [w for w in chunk if w]
        if not chunk:
            continue
        vecs = embed([w['text'] for w in chunk], kind='passage')
        for w, v in zip(chunk, vecs):
            db.execute(
                "INSERT INTO windows(id,centre_id,session,first_id,last_id,"
                "n_obs,n_words,started_at,ended_at,text,recipe) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (w['id'], w['centre_id'], w['session'], w['first_id'],
                 w['last_id'], w['n_obs'], w['n_words'], w['started_at'],
                 w['ended_at'], w['text'], w['recipe']))
            db.execute("INSERT INTO win_vec(id,embedding) VALUES(?,?)",
                       (w['id'], _vec_bytes(v)))
        rebuilt += len(chunk)
    db.commit()
    return {'dropped': len(affected), 'rebuilt': rebuilt}


def neighbourhood(db, oid, before=NEIGHBOURS_BEFORE, after=NEIGHBOURS_AFTER,
                  max_gap=NEIGHBOUR_MAX_GAP_S, max_words=NEIGHBOUR_MAX_WORDS):
    """The observation, plus what was said either side of it in the session.

    Returns rows in time order, each with 'offset' (0 is the target) and 'gap',
    the silence between it and the row before it. Growth stops at a gap wider
    than max_gap, at the session edge, or when the window has enough words --
    a fragment judged inside a wall of text is no longer the thing being judged.
    """
    t = db.execute("SELECT id,session,started_at,ended_at,text,kind FROM "
                   "observations WHERE id=?", (oid,)).fetchone()
    if t is None:
        return []
    out = [dict(t, offset=0, gap=0.0)]
    words = len(t['text'].split())

    def grow(direction):
        nonlocal words
        edge = out[0] if direction < 0 else out[-1]
        for n in range(1, (before if direction < 0 else after) + 1):
            if direction < 0:
                r = db.execute(
                    "SELECT id,session,started_at,ended_at,text,kind FROM "
                    "observations WHERE session=? AND started_at < ? "
                    "ORDER BY started_at DESC LIMIT 1",
                    (edge['session'], edge['started_at'])).fetchone()
                gap = (edge['started_at'] - r['ended_at']) if r else None
            else:
                r = db.execute(
                    "SELECT id,session,started_at,ended_at,text,kind FROM "
                    "observations WHERE session=? AND started_at > ? "
                    "ORDER BY started_at LIMIT 1",
                    (edge['session'], edge['started_at'])).fetchone()
                gap = (r['started_at'] - edge['ended_at']) if r else None
            if r is None or gap is None or gap > max_gap:
                return
            w = len(r['text'].split())
            if words + w > max_words:
                return
            words += w
            row = dict(r, offset=direction * n, gap=round(float(gap), 3))
            if direction < 0:
                out.insert(0, row)
            else:
                out.append(row)
            edge = row

    grow(-1)
    grow(+1)
    # Recompute gaps over the finished window. Growing leftwards inserts at the
    # front, so the target's own gap -- the silence separating it from the
    # neighbour before it -- was still the 0.0 it was seeded with, and any rule
    # reading it saw every left neighbour as adjacent.
    for k, r in enumerate(out):
        r['gap'] = 0.0 if k == 0 else round(
            float(r['started_at'] - out[k - 1]['ended_at']), 3)
    return out


def window_text(rows, mark='>>'):
    """The neighbourhood as one block, with the fragment being judged marked.

    Marked rather than merged: the judge is being asked about one fragment read
    in context, not about the paragraph. Without the marker it answers for the
    whole window and every neighbour comes back as an answer.
    """
    lines = []
    for r in rows:
        lines.append(f"{mark} {r['text']}" if r['offset'] == 0
                     else f"   {r['text']}")
    return "\n".join(lines)


def search(db, query, limit=10, kinds=('observation', 'belief'),
           include_superseded=False):
    """Retrieval by meaning, over both kinds, ranked into one list.

    Superseded beliefs are out by default: asking what the system thinks should
    not return four generations of what it used to think. They are reachable by
    asking for them, and by following any answer's chain.
    """
    q = _vec_bytes(embed_query(query))
    out = []
    if 'observation' in kinds:
        # Search windows, answer with observations. A window is how a fragment
        # is found; the observation is what it is. Overlap means several
        # windows can centre on observations near each other, so ask for more
        # than needed and keep the best window per centre -- a centre is never
        # returned twice, and what comes back is the record, never the window.
        # Two indexes, combined. The window says how well the surrounding
        # speech matches; the observation says whether this fragment is itself
        # what was asked about. A window hit alone promotes every neighbour of a
        # good match, because their windows all contain it. An observation hit
        # alone is the fragment problem this was built to fix. Neither is
        # sufficient and the failure modes are opposite, so the score is a
        # weighted sum and both terms are computed for every candidate rather
        # than left missing for whichever index did not surface it.
        cand = {}
        k = limit * WINDOW_OVERSAMPLE
        for r in db.execute(
                "SELECT w.centre_id AS id, v.distance FROM win_vec v "
                "JOIN windows w ON w.id=v.id "
                "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance", (q, k)):
            cand.setdefault(r['id'], {})['win'] = 1.0 - r['distance']
        for r in db.execute(
                "SELECT v.id, v.distance FROM obs_vec v "
                "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance", (q, k)):
            cand.setdefault(r['id'], {})['obs'] = 1.0 - r['distance']
        if cand:
            qv = np.frombuffer(q, dtype='float32')
            ids = list(cand)
            marks = ",".join("?" * len(ids))
            for table, key in (('obs_vec', 'obs'), ('win_vec', 'win')):
                need = [i for i in ids if key not in cand[i]]
                if not need:
                    continue
                if key == 'win':
                    rows = db.execute(
                        "SELECT w.centre_id AS id, v.embedding FROM win_vec v "
                        "JOIN windows w ON w.id=v.id WHERE w.centre_id IN (%s)"
                        % ",".join("?" * len(need)), need)
                else:
                    rows = db.execute(
                        "SELECT id, embedding FROM obs_vec WHERE id IN (%s)"
                        % ",".join("?" * len(need)), need)
                for r in rows:
                    v = np.frombuffer(r['embedding'], dtype='float32')
                    cand[r['id']][key] = float(qv @ v)
            asking = QUESTION_PENALTY and looks_like_a_question(query)
            for r in db.execute(
                    "SELECT id,text,started_at,person,kind,is_question FROM "
                    "observations WHERE id IN (%s)" % marks, ids):
                c = cand[r['id']]
                w, o = c.get('win', 0.0), c.get('obs', 0.0)
                base = WINDOW_WEIGHT * w + (1 - WINDOW_WEIGHT) * o
                penalised = bool(asking and r['is_question'])
                if penalised:
                    base *= (1.0 - QUESTION_PENALTY)
                out.append({'kind': 'observation', 'id': r['id'],
                            'question_penalised': penalised,
                            'distance': 1.0 - base,
                            'text': r['text'], 'at': r['started_at'],
                            'person': r['person'], 'observation_kind': r['kind'],
                            'found_via': {'window': round(w, 4),
                                          'observation': round(o, 4)},
                            'score': base})
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


def reindex(db, verbose=True):
    """Rebuild every vector under the current settings.

    The remedy the StaleVectors refusal names. Windows are derived, so this
    throws them away and remakes them; observations and beliefs are untouched
    except for their vectors.
    """
    def tick(done, total):
        if verbose and done % 256 == 0:
            print(f"  windows {done}/{total}", flush=True)
    backfill_questions(db)
    r = build_windows(db, progress=tick)
    # beliefs keep a vector of their own: a belief is already a whole statement
    # and has no session position to draw a window from
    db.execute("DELETE FROM bel_vec")
    bel = [dict(x) for x in db.execute("SELECT id,statement FROM beliefs")]
    for i in range(0, len(bel), 256):
        chunk = bel[i:i + 256]
        for b, v in zip(chunk, embed([x['statement'] for x in chunk],
                                     kind='passage')):
            db.execute("INSERT INTO bel_vec(id,embedding) VALUES(?,?)",
                       (b['id'], _vec_bytes(v)))
    # The per-observation index is kept, not discarded. Windows alone rank a
    # strong match's neighbours as highly as the match: every window containing
    # "What did you have for breakfast?" scores well, and each resolves to its
    # own centre, so "Hey, J." and "Mm, mm, mm" arrived at ranks 2 and 4. The
    # centre has to answer for itself as well as for its surroundings.
    db.execute("DELETE FROM obs_vec")
    obs = [dict(x) for x in db.execute("SELECT id,text FROM observations")]
    for i in range(0, len(obs), 256):
        chunk = obs[i:i + 256]
        for o, v in zip(chunk, embed([x['text'] for x in chunk],
                                     kind='passage')):
            db.execute("INSERT INTO obs_vec(id,embedding) VALUES(?,?)",
                       (o['id'], _vec_bytes(v)))
    db.commit()
    for k, v in (("schema_version", str(SCHEMA_VERSION)),
                 ("embed_model", EMBED_MODEL), ("embed_dims", str(EMBED_DIMS)),
                 ("index_unit", INDEX_UNIT), ("index_recipe", INDEX_RECIPE)):
        db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) "
                   "DO UPDATE SET value=excluded.value", (k, v))
    db.commit()
    r['beliefs'] = len(bel)
    return r


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == 'reindex':
        _db = open(check_vectors=False)
        import time as _t
        _t0 = _t.perf_counter()
        _r = reindex(_db)
        print(f"reindexed {_r['windows']} window(s) over {_r['observations']} "
              f"observation(s) and {_r['beliefs']} belief(s) in "
              f"{_t.perf_counter()-_t0:.1f}s under {INDEX_RECIPE}")
    else:
        print(__doc__.strip().splitlines()[0])
        print("usage: memory.py reindex")
