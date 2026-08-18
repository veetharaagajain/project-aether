"""Aether as an MCP server. The only way in.

Four tools, bound to localhost. Every call names its caller, because the rules
are per caller and an unnamed one is refused rather than defaulted.

The caller name is an argument rather than transport-derived authentication,
which is not the same thing and is not pretended to be: on a loopback-only
socket with no auth, a caller could name itself anything. That is honest for
this round -- the rules are flat and the interesting version needs a model --
but it means this must never be exposed beyond localhost, which serve()
enforces rather than documents.

  python aether_mcp.py                       stdio, for a local MCP client
  python aether_mcp.py --http [--port N]     streamable http on 127.0.0.1
  python aether_mcp.py --add-caller NAME [--read] [--write] [--note T]
                                             register one, print its secret
  python aether_mcp.py --callers             show who may do what
  python aether_mcp.py --log [N]             what has left the store
  python aether_mcp.py --state               capturing or paused
"""

import os
import sys
import threading

from mcp.server.mcpserver import MCPServer

import gate
import incognito as inc
import memory as mem

# One connection per thread, not one shared. The MCP runtime dispatches tool
# calls on worker threads, and a sqlite3 connection may only be used from the
# thread that made it -- the alternative, check_same_thread=False plus a lock
# around every call, serialises the server to hide a problem WAL already
# solves. Each connection carries its own hybrid clock; ids stay unique because
# a ULID's 80 random bits do not depend on the clock.
_LOCAL = threading.local()

# WHERE THE CALLER IDENTITY COMES FROM, AND WHY IT DEPENDS ON THE TRANSPORT.
#
# Over stdio there is exactly one client: the process that spawned this one, on
# the far end of a pipe nothing else can open. The transport is already the
# boundary, so the identity may come from the environment the parent set --
# AETHER_CALLER and AETHER_SECRET -- and the model never has to be told a
# secret, never puts one in a tool call, and never has one sitting in a
# conversation it might repeat.
#
# Over HTTP the opposite holds. One server faces many local clients, which is
# the entire reason the secret exists, and a server-side default identity would
# hand its own credentials to whoever connected first. So the fallback is
# refused there and every call must carry its own.
#
# _TRANSPORT is set by serve() before anything is served.
_TRANSPORT = None


def identity(caller, secret):
    """Resolve who is calling, from the arguments or from the environment."""
    if caller and secret:
        return caller, secret
    if _TRANSPORT == 'stdio':
        return (caller or os.environ.get('AETHER_CALLER', ''),
                secret or os.environ.get('AETHER_SECRET', ''))
    # http, or an unknown transport: no inherited identity
    return caller or '', secret or ''


def db():
    d = getattr(_LOCAL, 'db', None)
    if d is None:
        d = _LOCAL.db = mem.open()
    return d


server = MCPServer(
    name="aether",
    instructions=(
        "Aether is the single point of contact for one person's memory. It "
        "holds observations, which are things that happened and are never "
        "modified, and beliefs, which are conclusions drawn from them. Search "
        "first; fetch by id second. Every belief you write must name the "
        "observations it came from."))


@server.tool()
def search_memory(query: str, limit: int = 10,
                  kinds: list[str] | None = None,
                  include_superseded: bool = False,
                  caller: str = "", secret: str = "") -> list[dict]:
    """Search memory by meaning. Returns observations and beliefs, ranked.

    caller and secret are normally supplied by the environment and should
    be left out. Pass them only when connecting over HTTP, where each
    caller must identify itself.

    kinds may be ["observation"], ["belief"], or both. Superseded beliefs are
    excluded unless asked for.
    """
    caller, secret = identity(caller, secret)
    return gate.search_memory(db(), caller, secret, query, limit=limit,
                              kinds=kinds,
                              include_superseded=include_superseded)


@server.tool()
def fetch_observation(id: str, with_body: bool = True,
                      caller: str = "", secret: str = "") -> dict:
    """Fetch one observation by id: what was said, when, by whom, and with
    with_body the per-word emphasis weights."""
    caller, secret = identity(caller, secret)
    return gate.fetch_observation(db(), caller, secret, id,
                                  with_body=with_body)


@server.tool()
def fetch_belief(id: str, follow: bool = True,
                 caller: str = "", secret: str = "") -> dict:
    """Fetch one belief by id, with the observations it came from and whether
    it has since been replaced."""
    caller, secret = identity(caller, secret)
    return gate.fetch_belief(db(), caller, secret, id, follow=follow)


@server.tool()
def write_belief(statement: str, certainty: float, sources: list[str],
                 about: str, replaces: str | None = None,
                 links: list[dict] | None = None,
                 caller: str = "", secret: str = "") -> dict:
    """Record a conclusion drawn from observations.

    sources must name observations that exist, and should include both the
    speech that prompted the conclusion and any model answer that produced it.
    about is required: one or two words for what this concerns, free text, so
    the belief is findable by subject and not only by resemblance. certainty is
    0 to 1 and is how firmly the conclusion is held, which is not the same as
    how often it is used. To revise an earlier belief, write a new one with replaces set to
    its id; the old one is kept and stays fetchable.
    """
    caller, secret = identity(caller, secret)
    return gate.write_belief(db(), caller, secret, statement, certainty,
                             sources, about=about, replaces=replaces,
                             links=links)


@server.tool()
def record_answer(text: str, cites: list[str], model: str | None = None,
                  about: str | None = None,
                  caller: str = "", secret: str = "") -> dict:
    """Record what you just said as an observation, so the reasoning survives.

    A model answering is a thing that happened, and it belongs in the record by
    the same definition speech does. Record it and then cite it from any belief
    you write, alongside the speech that prompted it: a belief fades, and when
    an old one is revisited the reasoning underneath it should still be there,
    immutable and undecayed.

    cites must name the observations you drew on and is required. It is what
    lets the person's forget control reach your answer -- your sentences repeat
    their words back, so an uncited answer would be the one thing in the store
    that cannot be deleted by the control meant to delete them.
    """
    caller, secret = identity(caller, secret)
    return gate.record_answer(db(), caller, secret, text, cites, model=model,
                              about=about)


def serve(http=False, host=gate.DEFAULT_HOST, port=gate.DEFAULT_PORT):
    global _TRANSPORT
    _TRANSPORT = 'http' if http else 'stdio'
    if http:
        if host not in gate.LOCAL_ONLY:
            raise SystemExit(
                f"refusing to bind {host}. Anything that can reach this can "
                f"read a person's life; bind {gate.DEFAULT_HOST} and tunnel if "
                f"you need it elsewhere.")
        print(f"aether on http://{host}:{port}/mcp  "
              f"[{inc.banner(db())}]", file=sys.stderr)
        import functools

        import anyio
        anyio.run(functools.partial(server.run_streamable_http_async,
                                    host=host, port=port))
    else:
        who = os.environ.get('AETHER_CALLER')
        print(f"aether on stdio as {who or '(no AETHER_CALLER set)'}  "
              f"[{inc.banner(db())}]", file=sys.stderr)
        import anyio
        anyio.run(server.run_stdio_async)


def main():
    a = sys.argv[1:]
    if '--add-caller' in a:
        name = a[a.index('--add-caller') + 1]
        note = a[a.index('--note') + 1] if '--note' in a else None
        secret = gate.add_caller(db(), name, can_read='--read' in a,
                                 can_write='--write' in a, note=note)
        print(f"caller {name!r} registered: read={'--read' in a} "
              f"write={'--write' in a}")
        print(f"secret: {secret}")
        print("Shown once and not recoverable -- only its scrypt hash is "
              "stored. Re-run this to issue a new one.")
        return
    if '--callers' in a:
        for c in gate.callers(db()):
            flag = '' if c['has_secret'] else '   NO SECRET: cannot connect'
            print(f"  {c['caller']:<20} read {c['can_read']!s:<5} "
                  f"write {c['can_write']!s:<5} {c['note'] or ''}{flag}")
        return
    if '--state' in a:
        print(inc.banner(db()))
        return
    if '--log' in a:
        i = a.index('--log')
        n = int(a[i + 1]) if len(a) > i + 1 and a[i + 1].isdigit() else 30
        for r in gate.recent_access(db(), limit=n):
            import time as t
            print(f"  {t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(r['at']))}  "
                  f"{r['caller']:<17}{r['tool']:<19}{r['decision']:<8}"
                  f"{r['n_returned']:>3} returned  {r['reason'] or ''}")
        return
    port = gate.DEFAULT_PORT
    if '--port' in a:
        port = int(a[a.index('--port') + 1])
    host = a[a.index('--host') + 1] if '--host' in a else gate.DEFAULT_HOST
    serve(http='--http' in a, host=host, port=port)


if __name__ == "__main__":
    main()
