"""Run the live path and the viewer as launchd services.

Both of these died whenever a terminal closed or the laptop slept, and
restarting them by hand was the main friction in using any of this. launchd
starts them at login, restarts them when they exit, and keeps their output in
files rather than in a terminal that is gone.

Two things make this work beyond writing a plist.

The first is that a process must actually exit when it breaks. macOS tears the
input device down across sleep and PortAudio does not raise, so the old capture
loop sat awake and deaf forever, holding the lock. live.Capture now raises
AudioStalled after STALL_S of nothing and main returns EXIT_STALLED, which is
the entire recovery mechanism: launchd reopens the process against whatever
device exists on the other side of the wake.

The second is that a supervised process and a hand-started one must not both
run. They cannot, because singleton.take already refuses the second -- but a
refusal is an exit, and a supervisor that restarts on every exit would spin.
So the exit code says which happened: EXIT_LOCKED means back off, anything
else means restart. launchd has no "unless it exited with 4" switch, so
ThrottleInterval carries it: a refused start costs one log line every
THROTTLE seconds instead of a hot loop.

usage:
  service.py install [live|viewer|both]
  service.py uninstall [live|viewer|both]
  service.py start|stop|restart [live|viewer|both]
  service.py status
  service.py logs [live|viewer] [-n LINES]
  service.py tail [live|viewer]
"""

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
AGENTS = Path.home() / "Library" / "LaunchAgents"
LOGS = Path.home() / "Library" / "Logs" / "aether"
UV = Path.home() / ".local" / "bin" / "uv"

# Long enough that a refused start (someone is running it by hand) costs one
# line a minute rather than a hot loop, short enough that a real crash comes
# back while you are still looking at it.
THROTTLE = 30

SERVICES = {
    'live': {
        'label': 'com.aether.live',
        # Pinned to the built-in microphone by name, not left to the system
        # default. The first install took the default, which was a connected
        # pair of AirPods, and Bluetooth inputs in the wrong profile deliver
        # digital silence: 15,000 blocks of exactly zero, on time, with no
        # error anywhere. By name rather than by index because indices move
        # when devices come and go. Override at install time with:
        #     service.py install live -- --device "Some Other Microphone"
        'args': ['live.py', '--device', 'MacBook Air Microphone'],
        'what': 'the live path: capture, measure, store',
    },
    'viewer': {
        'label': 'com.aether.viewer',
        'args': ['viewer.py'],
        'what': 'the viewer: a local web page onto the store',
    },
    'consolidate': {
        'label': 'com.aether.consolidate',
        'args': ['consolidate.py', 'nightly', '--prefer', 'claude-api'],
        'what': 'the nightly pass: yesterday\'s episodes into beliefs',
        # a job, not a daemon. It runs, finishes, and exits; KeepAlive would
        # restart it in a loop and spend money doing it.
        'schedule': {'Hour': 4, 'Minute': 15},
    },
}


def plist_path(name):
    return AGENTS / f"{SERVICES[name]['label']}.plist"


def build_plist(name, extra_args=()):
    s = SERVICES[name]
    args = list(extra_args) if extra_args else list(s['args'][1:])
    sched = s.get('schedule')
    out = {
        'Label': s['label'],
        'ProgramArguments': [str(UV), 'run', 'python', s['args'][0], *args],
        'WorkingDirectory': str(PROJECT),
        # A scheduled job must not RunAtLoad -- that would fire it on every
        # login and on every reinstall -- and must not KeepAlive, which would
        # restart it the moment it finished and spend a day's budget in a
        # minute. StartCalendarInterval alone: run at the hour, exit, wait.
        'RunAtLoad': not sched,
        'KeepAlive': not sched,
        'ThrottleInterval': THROTTLE,
        # Interactive keeps the audio thread out of the low-priority band that
        # App Nap and the task policy would otherwise put a background agent
        # in. A throttled capture loop drops frames.
        'ProcessType': 'Interactive',
        'StandardOutPath': str(LOGS / f"{name}.log"),
        'StandardErrorPath': str(LOGS / f"{name}.err"),
        'EnvironmentVariables': {
            'PATH': f"{UV.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            # No network at login. Every model is in the cache; without this a
            # slow or captive network turns start-up into a hang rather than a
            # failure, and there is nothing to fetch anyway.
            'HF_HUB_OFFLINE': '1',
            'PYTHONUNBUFFERED': '1',
        },
    }
    if sched:
        out['StartCalendarInterval'] = sched
        # HF_HUB_OFFLINE stays set: the models are cached and a nightly job
        # should not be the thing that discovers the network is captive.
    return out


def domain():
    return f"gui/{os.getuid()}"


def run(cmd, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def install(name, extra_args=()):
    AGENTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    p = plist_path(name)
    p.write_bytes(plistlib.dumps(build_plist(name, extra_args)))
    # bootout returns before the job is actually gone, and bootstrapping into
    # a domain that still holds the label fails with a bare "5: Input/output
    # error". Wait for it to disappear rather than racing it.
    label = SERVICES[name]['label']
    run(['launchctl', 'bootout', f"{domain()}/{label}"])
    for _ in range(50):
        if run(['launchctl', 'print', f"{domain()}/{label}"]).returncode:
            break
        time.sleep(0.2)
    r = run(['launchctl', 'bootstrap', domain(), str(p)])
    if r.returncode:
        print(f"  bootstrap failed: {r.stderr.strip() or r.stdout.strip()}")
        return False
    print(f"installed {SERVICES[name]['label']} -> {p}")
    print(f"  {SERVICES[name]['what']}")
    print(f"  logs: {LOGS / (name + '.log')} and .err")
    return True


def uninstall(name):
    run(['launchctl', 'bootout', f"{domain()}/{SERVICES[name]['label']}"])
    p = plist_path(name)
    if p.exists():
        p.unlink()
    print(f"removed {SERVICES[name]['label']}")


def ctl(name, what):
    label = SERVICES[name]['label']
    if what == 'start':
        r = run(['launchctl', 'kickstart', f"{domain()}/{label}"])
    elif what == 'stop':
        r = run(['launchctl', 'kill', 'TERM', f"{domain()}/{label}"])
    else:
        r = run(['launchctl', 'kickstart', '-k', f"{domain()}/{label}"])
    msg = (r.stderr or r.stdout).strip()
    print(f"{what} {label}: {'ok' if not r.returncode else msg}")


def status():
    print(f"launchd domain {domain()}")
    for name, s in SERVICES.items():
        label = s['label']
        installed = plist_path(name).exists()
        r = run(['launchctl', 'print', f"{domain()}/{label}"])
        state = pid = last = '-'
        for line in r.stdout.splitlines():
            t = line.strip()
            if t.startswith('state = '):
                state = t.split('=', 1)[1].strip()
            elif t.startswith('pid = '):
                pid = t.split('=', 1)[1].strip()
            elif t.startswith('last exit code = '):
                last = t.split('=', 1)[1].strip()
        print(f"  {name:<11} plist {'yes' if installed else 'no ':<3}  "
              f"state {state:<10} pid {pid:<8} last exit {last}")
        lock = PROJECT / 'store' / f"{name}.lock"
        if lock.exists():
            held = lock.read_text().strip()
            print(f"          lock: {held or '(empty)'}")
    print("  a held lock with no pid running is stale text, not a held lock: "
          "the flock is released by the kernel whatever happened to the holder")


def logs(name, n=40, follow=False):
    for suffix in ('log', 'err'):
        f = LOGS / f"{name}.{suffix}"
        if not f.exists():
            continue
        print(f"===== {f} =====")
        if follow:
            os.execvp('tail', ['tail', '-f', str(f)])
        print(run(['tail', '-n', str(n), str(f)]).stdout, end='')


SCHEDULED = {n for n, s in SERVICES.items() if s.get('schedule')}


def which(args):
    picked = [a for a in args if a in SERVICES]
    return picked or list(SERVICES)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'install':
        extra = rest[rest.index('--'):][1:] if '--' in rest else ()
        for n in which(rest):
            install(n, extra)
    elif cmd == 'uninstall':
        for n in which(rest):
            uninstall(n)
    elif cmd in ('start', 'stop', 'restart'):
        for n in which(rest):
            ctl(n, cmd)
    elif cmd == 'status':
        status()
    elif cmd in ('logs', 'tail'):
        n = int(rest[rest.index('-n') + 1]) if '-n' in rest else 40
        logs(which(rest)[0], n, follow=(cmd == 'tail'))
    else:
        print(__doc__.strip())
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
