"""The transient channel between the running live path and anything watching.

The viewer needs to show words as they are measured. There were three ways to
connect them and the choice matters, so here is the reasoning rather than the
result.

Polling the store does not work. Storage is what pause switches off, so a
paused session would show a blank page -- and watching is exactly what a person
wants while capture is off, to confirm that it is. Watching and recording have
to be independent or the pause indicator cannot be trusted.

Tailing live_records.jsonl does not work either, for the same reason plus a
worse one: it is a file, so anything written to it survives, and using it to
feed a live view would mean pausing still leaves a transcript on disk. (It did,
until this was built. live.py now gates that file on capture along with
everything else.)

So: a unix datagram socket, and nothing persisted. Datagram rather than stream
because publishing must never block the measurement loop and must never fail
when nothing is listening -- sendto on a socket with no reader is a no-op here,
and every error is swallowed. Unix rather than UDP on loopback because a unix
socket has no port and cannot be reached from the network at all, which is a
stronger version of the same constraint the MCP server binds for. It lives
inside store/, which memory.open() has already chmodded to 0700.

The consequence, stated plainly: live.py runs perfectly well with nothing
listening, and the viewer runs perfectly well with no live.py -- it just has an
empty live pane and says so. Neither depends on the other.
"""

import json
import socket
from pathlib import Path

SOCKET_PATH = Path(__file__).resolve().parent / "store" / "live.sock"

# macOS gives a unix datagram socket a 2048-byte send buffer by default, and a
# datagram larger than the buffer is refused outright with EMSGSIZE rather than
# fragmented. Since publish() swallows errors so it can never disturb the
# measurement loop, that combination lost every utterance silently -- the first
# version of this shipped and only the small state events arrived. Both ends now
# ask for a larger buffer and the size actually granted is what the limit is
# read from, rather than a number written here and hoped for.
WANT_BUF = 1 << 18
_TX = None
_LIMIT = 2048


def _sized(sock, opt):
    """Ask for WANT_BUF, then report what the kernel actually gave."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, opt, WANT_BUF)
    except OSError:
        pass
    return sock.getsockopt(socket.SOL_SOCKET, opt)


def publish(event):
    """Fire and forget. Never raises, never blocks, never retries.

    A dropped frame costs one line of a live view. Anything that could make the
    measurement loop wait on a viewer would be a worse trade than that. The one
    thing worth spending a little care on is not dropping frames for a reason
    that is fixable, which is what the buffer sizing above is about.
    """
    global _TX, _LIMIT
    try:
        if _TX is None:
            _TX = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            _TX.setblocking(False)
            _LIMIT = _sized(_TX, socket.SO_SNDBUF)
        data = json.dumps(event, separators=(',', ':'), default=str).encode()
        if len(data) > _LIMIT:
            # a frame too big to send is reported rather than vanishing, so the
            # page shows that something was measured and not what
            data = json.dumps({'kind': event.get('kind', 'unknown'),
                               'at': event.get('at'), 'oversize': len(data),
                               'limit': _LIMIT}).encode()
        _TX.sendto(data, str(SOCKET_PATH))
    except (OSError, TypeError, ValueError):
        pass


def listener():
    """The receiving end, for the viewer. Replaces any stale socket file."""
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        SOCKET_PATH.unlink()
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    _sized(s, socket.SO_RCVBUF)
    s.bind(str(SOCKET_PATH))
    SOCKET_PATH.chmod(0o600)
    return s
