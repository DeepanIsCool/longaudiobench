#!/usr/bin/env python3
"""Cost guard keyed on cell growth, not on completions.

Five generations of this, each a real failure:

1. Deadline only. A pass that failed all 13 models in 11 seconds, slept,
   exited 0 and was restarted by Runpod is a cycle, not a hang - it burned
   $3.04 over six hours while an 11 h deadline waited.
2. Stall on completions. Correct for a loop, wrong for a slow model: Omni-7B
   takes ~35 min for 840 cells and looks identical to a stall by that
   measure, so a healthy run was about to be killed at two-thirds of its
   ladder.
3. Stall on cell growth. A model whose setup runs long - MOSS builds a
   package from git then pulls 10 GB - writes no cells for 20+ minutes and
   looked exactly like a dead loop. Killed a healthy MOSS at minute 25.
4. Cells OR log growth. Right idea, but counted only results.jsonl and
   sweep.jsonl, so the question-only phase (which writes neither) read as
   zero progress for its whole duration.
5. This: any *.jsonl under any model directory, plus log bytes. A loop
   produces neither; a slow model produces one or the other.

    python scripts/runpod/watchdog.py <pod_id> <expected_ok> [deadline_min]
"""
import json
import os
import re
import subprocess
import sys
import time

KEY = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY is not set")
POD = sys.argv[1]
EXPECTED = int(sys.argv[2])
DEADLINE_MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 300
STALL_MIN = 40
U = f"https://{POD}-8000.proxy.runpod.net"


def curl(args, t=40):
    try:
        return subprocess.run(["curl", "-s", "--max-time", str(t)] + args,
                              capture_output=True, text=True,
                              timeout=t + 20).stdout
    except Exception:
        return ""


def alive():
    """Unreadable replies count as alive. The first watchdog grepped the first
    40 bytes for '"id"', the reply began '{"consumerUserId"', and it declared
    the pod gone at minute 1 - leaving it running unprotected."""
    raw = curl(["-H", f"Authorization: Bearer {KEY}", f"https://rest.runpod.io/v1/pods/{POD}"])
    if not raw:
        return True
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return True
    return d.get("id") == POD and d.get("desiredStatus") != "TERMINATED"


def kill(why):
    print(f"TERMINATING ({why})", flush=True)
    curl(["-X", "DELETE", "-H", f"Authorization: Bearer {KEY}",
          f"https://rest.runpod.io/v1/pods/{POD}"], 60)


def state():
    """Total cells written across every model directory, plus log size."""
    models = sorted(set(re.findall(r'href="([a-z0-9_]+)/"', curl([f"{U}/"]))))
    cells = 0
    for m in models:
        for f in re.findall(r'href="([^"]+\.jsonl)"', curl([f"{U}/{m}/"])):
            body = curl([f"{U}/{m}/{f}"], 30)
            if body and "<!DOCTYPE" not in body[:20]:
                cells += body.count('{"item_id"')
    log = curl([f"{U}/run.log"])
    return (cells, len(log), len(set(re.findall(r"(\w+)_OK$", log, re.M))),
            "RUN_COMPLETE" in log)


last_sig, last_change = None, time.time()
for minute in range(1, DEADLINE_MIN + 1):
    time.sleep(60)
    if not alive():
        print(f"pod gone at minute {minute}", flush=True)
        sys.exit(0)
    cells, logsize, ok, complete = state()
    sig = (cells, logsize)
    if sig != last_sig:
        last_sig, last_change = sig, time.time()
    idle = (time.time() - last_change) / 60
    if minute % 5 == 0:
        print(f"minute {minute}: {ok}/{EXPECTED} OK, {cells} cells, "
              f"log {logsize}B, {idle:.0f} min idle", flush=True)
    if complete or ok >= EXPECTED:
        kill(f"all {ok}/{EXPECTED} done")
        sys.exit(0)
    if idle >= STALL_MIN:
        kill(f"no cells and no log growth for {idle:.0f} min at {cells} cells")
        sys.exit(1)
kill(f"deadline {DEADLINE_MIN} min")
