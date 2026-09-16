#!/usr/bin/env python3
"""Mirror the pod's whole output tree into a local directory, repeatedly.

An earlier monitor banked a model only once a DONE marker appeared, so a phase
that wrote a different marker banked nothing. This copies anything served,
whatever it is called and whether or not it finished; partial files are
refetched next pass. A pod's volume dies with the pod, so "copy it now" beats
"copy it when it is finished".

    python scripts/runpod/pull.py <pod_id> --dest results/exp2 [--minutes 180]
"""
import argparse
import os
import re
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("pod")
ap.add_argument("--dest", required=True)
ap.add_argument("--minutes", type=int, default=180)
ap.add_argument("--every", type=int, default=90, help="seconds between passes")
a = ap.parse_args()
U = f"https://{a.pod}-8000.proxy.runpod.net"


def get(path, out=None):
    args = ["curl", "-s", "--max-time", "60", f"{U}/{path}"]
    if out:
        args += ["-o", out, "-w", "%{http_code}"]
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return ""


def listing(path=""):
    return re.findall(r'href="([^"?/][^"]*)"', get(path))


def mirror(remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for f in listing(remote_dir):
        if f.endswith("/"):
            mirror(f"{remote_dir}{f}", os.path.join(local_dir, f.rstrip("/")))
            continue
        out = os.path.join(local_dir, f)
        prev = os.path.getsize(out) if os.path.exists(out) else -1
        code = get(f"{remote_dir}{f}", out).strip()
        now = os.path.getsize(out) if os.path.exists(out) else 0
        if code != "200" or now == 0:
            if os.path.exists(out):
                os.remove(out)
        elif now != prev:
            print(f"pulled {remote_dir}{f} {now}B", flush=True)


deadline = time.time() + a.minutes * 60
while time.time() < deadline:
    mirror("", a.dest)
    time.sleep(a.every)
print("pull done", flush=True)
