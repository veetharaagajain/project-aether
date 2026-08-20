"""A window onto the store. Read-only, except pause and resume.

Everything this project does has been judged from terminal output for weeks,
which is why one decision got made properly and the rest were guesses. This is
a page that shows what is happening, what is in memory, and what has left it.

No framework: http.server from the standard library, threaded so a long-lived
event stream does not block the other requests. The page is one HTML file with
inline CSS and JS, served as-is. There is no build step and nothing to bundle.

LOCALHOST, for the reason the MCP server binds the same way, plus one more. A
Host header check rejects requests that arrive addressed to anything but
localhost, because binding 127.0.0.1 alone does not stop a page on another site
from pointing a name that resolves to 127.0.0.1 at this port and reading the
replies. Binding is about who can connect; the header check is about who can be
tricked into connecting for them.

READ-ONLY, except /api/pause and /api/resume. Those are here because stopping
capture is the one control a person wants in a hurry, and making them walk to a
terminal for it defeats the purpose. Nothing else mutates: there is no endpoint
that writes a belief, forgets a window, or registers a caller. Forget in
particular is deliberately absent -- it destroys audio, and a destructive
button on an auto-refreshing debug page is an accident waiting to happen.

  python viewer.py [--port 8800]

Then open http://127.0.0.1:8800/. It works with or without live.py running; see
channel.py for why neither side depends on the other.
"""

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import channel as ch
import singleton

HOST = "127.0.0.1"
PORT = 8800
PAGE = Path(__file__).resolve().parent / "viewer.html"
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
KEEPALIVE_S = 15.0
MAX_LIVE = 200          # utterances kept for a page that opens mid-session

# --- the fan-out ------------------------------------------------------------
# One thread reads the socket; every open event stream has a queue. A slow or
# dead browser fills its queue and is dropped rather than backing anything up.
_subs = []
_subs_lock = threading.Lock()
_recent = []
_recent_lock = threading.Lock()


def _fanout(event):
    with _recent_lock:
        _recent.append(event)
        del _recent[:-MAX_LIVE]
    with _subs_lock:
        dead = []
        for q in _subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subs.remove(q)


def _reader():
    rx = ch.listener()
    while True:
        try:
            data, _ = rx.recvfrom(65536)
            _fanout(json.loads(data))
        except (OSError, ValueError):
            continue


def subscribe():
    q = queue.Queue(maxsize=64)
    with _subs_lock:
        _subs.append(q)
    return q


def unsubscribe(q):
    with _subs_lock:
        if q in _subs:
            _subs.remove(q)


# --- what the page is allowed to read ---------------------------------------
def _db():
    """A connection per thread: ThreadingHTTPServer serves each request on its
    own thread and sqlite3 connections are thread-bound."""
    import memory as mem
    local = threading.current_thread()
    d = getattr(local, '_aether_db', None)
    if d is None:
        d = mem.open()
        local._aether_db = d
    return d


def api_state():
    import incognito as inc
    db = _db()
    s = inc.state(db)
    n_live = 0
    with _recent_lock:
        n_live = len(_recent)
    # The thresholds travel with the state rather than being written into the
    # page. A constant copied into viewer.html would be a fifth place for the
    # marking levels to live, and the whole provenance argument of the last two
    # rounds was about exactly that: a number duplicated is a number that will
    # silently disagree with itself.
    import markers as mk
    import staleness as st
    stale = [p for p in st.living(db) if p['stale']]
    return {'stale_processes': stale,
            'capturing': s['recording'], 'since': s['since'],
            'note': s['note'], 'banner': inc.banner(db),
            'levels': mk.LEVEL_THRESHOLDS,
            'mark_min_words': mk.MARK_MIN_WORDS,
            'observations': db.execute(
                "SELECT count(*) FROM observations").fetchone()[0],
            'beliefs': db.execute("SELECT count(*) FROM beliefs").fetchone()[0],
            'calls': db.execute("SELECT count(*) FROM access_log").fetchone()[0],
            'live_buffered': n_live,
            'history': [dict(r) for r in db.execute(
                "SELECT at, mode, note FROM capture_state "
                "ORDER BY at DESC LIMIT 8")]}


def api_memory(limit=40, q=None):
    """Observations and beliefs. Text search is a LIKE, not the embedder: this
    is a filter box on a list, and loading a 90 MB model to narrow a table
    would make the page slow for no gain."""
    import memory as mem
    db = _db()
    like = f"%{q}%" if q else None
    obs = [dict(r) for r in db.execute(
        "SELECT id,kind,started_at,ended_at,person,person_decision,text,"
        "audio_blob IS NOT NULL AS has_audio,speaker FROM observations "
        + ("WHERE text LIKE ? " if like else "")
        + "ORDER BY started_at DESC LIMIT ?",
        ((like, limit) if like else (limit,)))]
    beliefs = []
    for r in db.execute(
            "SELECT id FROM beliefs "
            + ("WHERE statement LIKE ? " if like else "")
            + "ORDER BY formed_at DESC LIMIT ?",
            ((like, limit) if like else (limit,))):
        b = mem.get_belief(db, r['id'])   # reader=None: a view never returns to anything
        if b:
            beliefs.append({
                'id': b['id'], 'statement': b['statement'],
                'certainty': b['certainty'], 'weight': round(b['weight_now'], 4),
                'faint': b['faint'], 'formed_at': b['formed_at'],
                'about': b['about'], 'author': b['author'],
                'times_returned': b['times_returned'],
                'n_sources': len(b['sources']),
                'superseded': b['superseded'], 'replaced_by': b['replaced_by'],
                'replaces': b['replaces'], 'current': b['current']})
    return {'observations': obs, 'beliefs': beliefs}


def api_belief(bid):
    """One belief with its record underneath it, resolved.

    reader=None throughout. Being looked at in a debugger is not the same as
    being returned to in answer to a question, and letting a page that
    auto-refreshes strengthen beliefs would corrupt the decay it is meant to
    display.
    """
    import memory as mem
    db = _db()
    b = mem.get_belief(db, bid)
    if b is None:
        t = db.execute("SELECT * FROM tombstones WHERE id=?", (bid,)).fetchone()
        return {'gone': True, 'id': bid,
                'reason': t['reason'] if t else 'no such belief'}
    src = []
    for oid in b['sources']:
        o = db.execute(
            "SELECT id,started_at,person,text,audio_blob IS NOT NULL AS "
            "has_audio FROM observations WHERE id=?", (oid,)).fetchone()
        if o:
            src.append(dict(o))
        else:
            t = db.execute("SELECT reason FROM tombstones WHERE id=?",
                           (oid,)).fetchone()
            src.append({'id': oid, 'gone': True,
                        'reason': t['reason'] if t else 'missing'})
    chain = []
    for rid in b['replaces']:
        r = db.execute("SELECT id,statement,certainty,formed_at FROM beliefs "
                       "WHERE id=?", (rid,)).fetchone()
        if r:
            chain.append(dict(r))
    return {'id': b['id'], 'statement': b['statement'],
            'certainty': b['certainty'], 'weight': round(b['weight_now'], 4),
            'faint': b['faint'], 'formed_at': b['formed_at'],
            'about': b['about'], 'author': b['author'],
            'times_returned': b['times_returned'],
            'superseded': b['superseded'], 'replaced_by': b['replaced_by'],
            'current': b['current'], 'sources': src, 'replaces': chain,
            'links': [{'type': l['type'], 'id': l['dst_id'],
                       'kind': l['dst_kind']} for l in b['links']]}


def api_observation(oid):
    import memory as mem
    db = _db()
    o = mem.get_observation(db, oid)
    if o is None:
        t = db.execute("SELECT * FROM tombstones WHERE id=?", (oid,)).fetchone()
        return {'gone': True, 'id': oid,
                'reason': t['reason'] if t else 'no such observation'}
    return {'id': o['id'], 'kind': o['kind'], 'text': o['text'],
            'started_at': o['started_at'], 'ended_at': o['ended_at'],
            'person': o['person'], 'person_decision': o['person_decision'],
            'has_audio': bool(o['audio_blob']),
            'config_digest': o['config_digest'],
            'record': o['body']}


def api_log(limit=80):
    """The access log. Deliberately does not join anything back to content.

    The stored row holds ids because content in a log would be a second copy of
    the person's speech under none of the same rules. Resolving those ids to
    text here would rebuild exactly what the log avoids, so the page shows the
    count and the ids and stops.
    """
    db = _db()
    out = []
    for r in db.execute("SELECT id,at,caller,tool,arguments,decision,reason,"
                        "n_returned,returned FROM access_log "
                        "ORDER BY at DESC LIMIT ?", (limit,)):
        try:
            args = json.loads(r['arguments'])
        except ValueError:
            args = {}
        args.pop('secret', None)        # belt and braces; log() already strips
        out.append({'at': r['at'], 'caller': r['caller'], 'tool': r['tool'],
                    'decision': r['decision'], 'reason': r['reason'],
                    'n_returned': r['n_returned'],
                    'arguments': args,
                    'returned': json.loads(r['returned'] or '[]')})
    return out


def api_recent():
    with _recent_lock:
        return list(_recent)


def api_processes():
    """Every long-lived process, and whether it is running code that no longer
    exists on disk.

    This is the answer to the failure that produced it: the MCP server ran two
    hours of stale code and the only record was a log line nobody read. The
    viewer is where a person looks, so this is what the header reads.
    """
    import staleness as st
    db = _db()
    st.reap(db)
    ps = st.living(db)
    return {'processes': ps,
            'stale': [p for p in ps if p['stale']],
            'now': time.time()}


def api_approvals():
    """Pending requests, plus whether approval is on and what stands.

    The pending rows carry ids and counts only -- the preview of what would be
    handed over arrives over the live channel and is held in the page, never
    written down. A queue holding the text of everything anyone asked for would
    be a second copy of the person's speech under none of the same rules.
    """
    import gate
    db = _db()
    return {'mode': gate.approval_mode(db),
            'timeout_s': gate.APPROVAL_TIMEOUT_S,
            'default_standing_minutes': gate.STANDING_DEFAULT_MINUTES,
            'pending': gate.pending(db),
            'standing': [dict(r) for r in db.execute(
                "SELECT * FROM standing_approvals ORDER BY caller, tool")],
            'recent': [dict(r) for r in db.execute(
                "SELECT * FROM approvals WHERE decision!='pending' "
                "ORDER BY at DESC LIMIT 30")]}


def decide_approval(aid, decision, standing_minutes=None):
    import gate
    r = gate.decide_approval(_db(), aid, decision, by='viewer',
                             standing_minutes=standing_minutes)
    return r or {'error': 'no such pending request'}


def set_approval_mode(on):
    import gate
    return {'mode': gate.set_approval_mode(_db(), on)}


def revoke_standing(caller=None, tool=None):
    import gate
    gate.revoke_standing(_db(), caller, tool)
    return api_approvals()


def set_capture(on, note):
    import incognito as inc
    db = _db()
    s = inc.resume(db, note) if on else inc.pause(db, note)
    ev = {'kind': 'capture', 'at': time.time(), 'capturing': s['recording'],
          'banner': inc.banner(db), 'source': 'viewer'}
    _fanout(ev)     # so the page updates even with no live.py running
    return api_state()


# --- the server -------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "aether-viewer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass        # the access log this project cares about is the other one

    def _local(self):
        host = (self.headers.get('Host') or '').rsplit(':', 1)[0]
        return host in ALLOWED_HOSTS or host == ''

    def _send(self, code, body, ctype="application/json", extra=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _query(self):
        from urllib.parse import parse_qs, urlparse
        return {k: v[0] for k, v in
                parse_qs(urlparse(self.path).query).items()}

    def do_GET(self):
        if not self._local():
            return self._send(403, {'error': 'localhost only'})
        path = self.path.split('?')[0]
        try:
            if path == '/':
                return self._send(200, PAGE.read_text(), "text/html; charset=utf-8")
            if path == '/events':
                return self._events()
            if path == '/api/state':
                return self._send(200, api_state())
            if path == '/api/recent':
                return self._send(200, api_recent())
            if path == '/api/memory':
                q = self._query()
                return self._send(200, api_memory(
                    limit=min(int(q.get('limit', 40)), 500), q=q.get('q')))
            if path == '/api/processes':
                return self._send(200, api_processes())
            if path == '/api/approvals':
                return self._send(200, api_approvals())
            if path == '/api/log':
                q = self._query()
                return self._send(200, api_log(
                    limit=min(int(q.get('limit', 80)), 500)))
            if path.startswith('/api/belief/'):
                return self._send(200, api_belief(path.rsplit('/', 1)[1]))
            if path.startswith('/api/observation/'):
                return self._send(200, api_observation(path.rsplit('/', 1)[1]))
            return self._send(404, {'error': 'no such path'})
        except Exception as e:                          # noqa: BLE001
            return self._send(500, {'error': f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._local():
            return self._send(403, {'error': 'localhost only'})
        path = self.path.split('?')[0]
        n = int(self.headers.get('Content-Length') or 0)
        if n:
            self.rfile.read(n)
        try:
            if path == '/api/pause':
                return self._send(200, set_capture(False, 'paused from the viewer'))
            if path == '/api/resume':
                return self._send(200, set_capture(True, 'resumed from the viewer'))
            q = self._query()
            if path == '/api/approve':
                mins = q.get('standing_minutes')
                return self._send(200, decide_approval(
                    q.get('id'), 'approved',
                    float(mins) if mins else None))
            if path == '/api/deny':
                return self._send(200, decide_approval(q.get('id'), 'denied'))
            if path == '/api/approval-mode':
                return self._send(200, set_approval_mode(q.get('on') == '1'))
            if path == '/api/revoke-standing':
                return self._send(200, revoke_standing(q.get('caller'),
                                                       q.get('tool')))
            return self._send(404, {'error': 'no such path'})
        except Exception as e:                          # noqa: BLE001
            return self._send(500, {'error': f"{type(e).__name__}: {e}"})

    def _events(self):
        """Server-sent events: one long response, flushed per event.

        SSE rather than a websocket because this only ever pushes one way and
        the browser reconnects on its own. A keepalive comment goes out every
        KEEPALIVE_S so a proxy or a sleeping laptop does not silently hold a
        dead stream open.
        """
        q = subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b": open\n\n")
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=KEEPALIVE_S)
                    payload = json.dumps(ev, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            unsubscribe(q)


def main():
    a = sys.argv[1:]
    port = int(a[a.index('--port') + 1]) if '--port' in a else PORT
    host = a[a.index('--host') + 1] if '--host' in a else HOST
    if host not in ALLOWED_HOSTS:
        raise SystemExit(f"refusing to bind {host}: this page shows one "
                         f"person's speech, their memory and every access to "
                         f"it. Bind {HOST}.")
    # Before the reader thread touches the socket. channel.listener() unlinks a
    # stale socket file before binding, which is correct after a crash and
    # wrong while someone is still using it: a second viewer took the live
    # stream and the first went deaf, with no error at either end. The port
    # would have caught a same-port collision a moment later, but only after
    # the theft, and not at all on a different port.
    try:
        singleton.take(
            'viewer', 'viewer',
            "Two at once means the second takes the live stream from the\n"
            "first, which then shows nothing and does not say why.",
            process='viewer.py')
    except singleton.AlreadyRunning as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 1
    import staleness as st
    d = _db()
    st.reap(d)
    st.register(d, 'viewer')
    st.heartbeat(lambda: __import__('memory').open())
    threading.Thread(target=_reader, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"aether viewer on http://{host}:{port}/")
    print(f"listening for the live path on {ch.SOCKET_PATH}")
    print("read-only except pause and resume. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    sys.exit(main() or 0)
