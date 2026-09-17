#!/usr/bin/env python3
"""Runpod control with a hard spend ceiling.

The danger is not a failed run, it is a pod nobody stopped. Every launch here
carries a wall-clock deadline and the watchdog terminates on it whether the
work finished or not. Balance is checked before each launch and never assumed.

Credentials come from the environment, never from this file:

    RUNPOD_API_KEY   required
    HF_TOKEN         forwarded into the pod's environment at launch
    KAGGLE_JSON      contents of kaggle.json; split into KAGGLE_USERNAME and
                     KAGGLE_KEY before forwarding, because a JSON blob did not
                     survive Runpod env -> shell intact and the pod's kaggle CLI
                     went anonymous (403 on a private dataset, $0.03 of pod)

README.md in this directory has the three export lines that load them from
the gitignored files at the repo root. Nothing here is ever written with a
literal token in it - the repo is public and a pushed key is scraped within
minutes.

    python scripts/runpod/rp.py status
    python scripts/runpod/rp.py guard
    python scripts/runpod/rp.py launch --name undertone --script experiments/02_prominence_2x2.sh
    python scripts/runpod/rp.py stop                  # keep the volume; resume later
    python scripts/runpod/rp.py kill                  # delete pods and volumes

Use ONE --name for every step. A stopped pod of that name is resumed with
the new step's script, so the 10-20 GB of weights and the packs are pulled
once, and a stock wait happens at most once.
"""
import argparse
import json
import os
import subprocess
import sys

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"

# A reserve, not a budget: refuse to launch unless this much is still on the
# account after the planned spend. It was set to 6.00 when the balance was
# 10.00 and then blocked a $1 job at a 5.64 balance; the ceiling has to track
# what is left, not what there was. Override per launch with --reserve.
# The whole balance is spendable; this is the floor a launch may not
# breach, not money set aside. It is small because the error budget is
# enforced elsewhere - a stall bills ~$0.25 before the watchdog kills it, a
# restart loop ~$0.02 - and the pre-launch guard only has to stop a launch
# that could not complete.
DEFAULT_RESERVE_USD = 1.00
# 48 GB, sm86, $0.33-0.35/h on community cloud. Either is fine: same
# architecture generation, same dtype path, same headroom. The API takes a
# list and gives whichever has stock.
DEFAULT_GPU = "NVIDIA A40,NVIDIA RTX A6000"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REPO = "https://github.com/DeepanIsCool/longaudiobench.git"


def key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY is not set. See scripts/runpod/README.md.")
    return k


def _curl(args, timeout=60):
    out = subprocess.run(["curl", "-s", "--max-time", str(timeout)] + args,
                         capture_output=True, text=True, timeout=timeout + 15)
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return {"_raw": out.stdout, "_err": out.stderr}


def gql(query):
    return _curl(["-X", "POST", GQL, "-H", f"Authorization: Bearer {key()}",
                  "-H", "Content-Type: application/json",
                  "-d", json.dumps({"query": query})])


def balance():
    d = gql("query { myself { clientBalance currentSpendPerHr } }")
    me = (d.get("data") or {}).get("myself") or {}
    return me.get("clientBalance"), me.get("currentSpendPerHr")


def pods():
    """Running pods only. Stopped ones (desiredStatus EXITED) are listed by
    all_pods() and are not a reason to refuse a launch."""
    return [p for p in all_pods() if p.get("desiredStatus") == "RUNNING"]


def terminate(pod_id):
    """Delete the pod and its volume. Only for a pod that is never coming
    back; everything else should `stop`."""
    subprocess.run(["curl", "-s", "-X", "DELETE", "--max-time", "60",
                    "-H", f"Authorization: Bearer {key()}", f"{REST}/pods/{pod_id}"],
                   capture_output=True, text=True, timeout=75)
    return pod_id


def stop(pod_id):
    """Stop the pod, keeping its volume - the weights, the pack, the ASR cache
    - and its place on the host. Billing drops to volume storage (cents a
    day). A later `resume` gets the GPU back without a stock wait when the
    host still has one, which is most of the time on secure cloud. Four
    twins launches were terminated when they should have been stopped, and
    each cost an hour's wait for a new card."""
    return _curl(["-X", "POST", "-H", f"Authorization: Bearer {key()}",
                  f"{REST}/pods/{pod_id}/stop"], 60)


def resume(pod_id):
    """Start a stopped pod. Its dockerStartCmd runs again from the top: the
    repo is re-cloned at the same tag and the step script re-runs, skipping
    every model with a DONE marker."""
    return _curl(["-X", "POST", "-H", f"Authorization: Bearer {key()}",
                  f"{REST}/pods/{pod_id}/start"], 120)


def all_pods():
    d = _curl(["-H", f"Authorization: Bearer {key()}", f"{REST}/pods"])
    return d if isinstance(d, list) else d.get("pods", [])


def stop_all():
    stopped = [stop(p["id"]) and p["id"] for p in pods()]
    print(f"stopped {len(stopped)} pod(s): {stopped or '(none were running)'}")
    return stopped


def kill_all():
    """Terminate everything, running or stopped. Safe to run twice."""
    killed = [terminate(p["id"]) for p in all_pods()]
    print(f"terminated {len(killed)} pod(s): {killed or '(none existed)'}")
    return killed


def guard(reserve=DEFAULT_RESERVE_USD, planned=0.0):
    """Refuse to launch when balance - planned spend would breach the reserve."""
    bal, rate = balance()
    print(f"balance ${bal}   current spend ${rate}/hr")
    if bal is None:
        sys.exit("could not read balance - not launching")
    if bal - planned < reserve:
        sys.exit(f"balance ${bal} minus planned ${planned:.2f} is under the "
                 f"${reserve:.2f} reserve - not launching")
    live = pods()
    if live:
        sys.exit(f"{len(live)} pod already running: {[p['id'] for p in live]}. "
                 "Stop it before launching another.")
    print("guard passed: balance sufficient, no pods running")


def launch(name, script, gpu=DEFAULT_GPU, image=DEFAULT_IMAGE, volume_gb=80,
           disk_gb=40, reserve=DEFAULT_RESERVE_USD, planned=0.0, ref="main",
           cloud="COMMUNITY"):
    """Create one pod that clones the repo at `ref` and runs `script`.

    Secrets reach the pod as environment variables set on the pod itself,
    which Runpod stores encrypted. The start command below never contains
    them; it reads them from the environment the same way the local scripts
    do.

    NOTE: the create-pod body follows the REST v1 schema as of the last run.
    If Runpod rejects a field name, check https://rest.runpod.io/v1 docs -
    the API has renamed fields before and this file cannot be tested
    without a live key.
    """
    guard(reserve, planned)
    hf = os.environ.get("HF_TOKEN", "")
    kg = os.environ.get("KAGGLE_JSON", "")
    if not hf:
        print("warning: HF_TOKEN unset; gated models (Gemma, Llama) will fail")
    try:
        kj = json.loads(kg) if kg else {}
        kaggle_user, kaggle_key = kj["username"], kj["key"]
    except (json.JSONDecodeError, KeyError):
        sys.exit("KAGGLE_JSON must be the contents of kaggle.json (username + key)")
    start = (
        "bash -lc '"
        f"rm -rf /workspace/repo && git clone --depth 1 --branch {ref} {REPO} /workspace/repo && "
        "cd /workspace/repo && export UNDERTONE_CODE_SHA=$(git rev-parse HEAD) && "
        f"bash {script}'"
    )
    body = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [g.strip() for g in gpu.split(",") if g.strip()],
        "gpuCount": 1,
        "cloudType": cloud,
        # Never a spot pod: an interruption mid-ladder throws away the cells
        # since the last pull and the model load before them.
        "interruptible": False,
        # Every step starts by pulling 10-20 GB of weights. A slow node bills
        # for the wait.
        "minDownloadMbps": 500,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "containerDiskInGb": disk_gb,
        "ports": ["8000/http"],
        "env": {
            "HF_TOKEN": hf, "HUGGING_FACE_HUB_TOKEN": hf,
            "KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key,
            # Which item pack the step scores. bootstrap.sh downloads it into
            # its own directory, so packs never unzip over each other.
            "PACK_DATASET": os.environ.get("PACK_DATASET", "deepansadhukhanjeet/undertone-item-pack"),
            # Optional model-list override for a step script (see 03/04).
            "MODELS": os.environ.get("MODELS", ""),
            # Optional: a Kaggle dataset of a previous pod's output tree,
            # restored into $OUT before the step runs. See bootstrap.sh.
            "RESTORE_DATASET": os.environ.get("RESTORE_DATASET", ""),
            "HF_HOME": "/workspace/hf",
            "UNDERTONE_ASR_CACHE": "/workspace/asr_cache",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "dockerStartCmd": ["bash", "-lc", start],
    }
    # A stopped pod of this name is resumed with the new step's start command
    # and fresh env, keeping its volume - weights, pack, ASR cache - and its
    # host. One pod carries every step; a stock wait happens at most once.
    for p in all_pods():
        if p.get("name") == name and p.get("desiredStatus") == "EXITED":
            patch = _curl(["-X", "PATCH", f"{REST}/pods/{p['id']}",
                           "-H", f"Authorization: Bearer {key()}",
                           "-H", "Content-Type: application/json",
                           "-d", json.dumps({"dockerStartCmd": body["dockerStartCmd"],
                                             "env": body["env"]})], 120)
            started = resume(p["id"])
            if started.get("id") == p["id"] or started.get("desiredStatus") == "RUNNING":
                print(f"launched {p['id']}  {name}  (resumed; volume kept; step {script})")
                print(f"  proxy: https://{p['id']}-8000.proxy.runpod.net/")
                return p["id"]
            print(f"resume of {p['id']} failed: {json.dumps(started)[:300]}\n"
                  f"  (patch said: {json.dumps(patch)[:120]}) - falling through to create")
    d = _curl(["-X", "POST", f"{REST}/pods", "-H", f"Authorization: Bearer {key()}",
               "-H", "Content-Type: application/json", "-d", json.dumps(body)], 120)
    pod_id = d.get("id")
    if not pod_id:
        sys.exit(f"launch failed: {json.dumps(d)[:600]}")
    print(f"launched {pod_id}  {name}  {d.get('machine', {}).get('gpuDisplayName') or gpu}  {cloud}")
    print(f"  proxy: https://{pod_id}-8000.proxy.runpod.net/")
    print(f"  now run:  python scripts/runpod/watchdog.py {pod_id} <expected_ok> <deadline_min>")
    print(f"       and:  python scripts/runpod/pull.py {pod_id} --dest results/{name}")
    return pod_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("stop", help="stop every running pod, keeping volumes")
    sub.add_parser("kill", help="terminate every pod, running or stopped, deleting volumes")
    r = sub.add_parser("resume"); r.add_argument("pod_id")
    g = sub.add_parser("guard")
    g.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_USD)
    g.add_argument("--planned", type=float, default=0.0)
    l = sub.add_parser("launch")
    l.add_argument("--name", required=True)
    l.add_argument("--script", required=True, help="repo-relative bash script the pod runs")
    l.add_argument("--gpu", default=DEFAULT_GPU, help="comma-separated; any with stock is taken")
    l.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    l.add_argument("--image", default=DEFAULT_IMAGE)
    l.add_argument("--ref", default="main", help="git ref to clone; use a tag for paper runs")
    l.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_USD)
    l.add_argument("--planned", type=float, default=0.0,
                   help="expected spend of this launch; guard refuses if it breaches the reserve")
    a = ap.parse_args()
    if a.cmd == "status":
        bal, rate = balance()
        ps = all_pods()
        print(f"balance ${bal}   spend ${rate}/hr   pods: "
              f"{sum(p.get('desiredStatus') == 'RUNNING' for p in ps)} running, "
              f"{sum(p.get('desiredStatus') == 'EXITED' for p in ps)} stopped")
        for p in ps:
            print(f"  {p['id']}  {p.get('name')}  {p.get('desiredStatus')}  "
                  f"{(p.get('machine') or {}).get('gpuDisplayName', '?')}")
    elif a.cmd == "stop":
        stop_all()
    elif a.cmd == "resume":
        print(json.dumps(resume(a.pod_id))[:300])
    elif a.cmd == "kill":
        kill_all()
    elif a.cmd == "guard":
        guard(a.reserve, a.planned)
    elif a.cmd == "launch":
        launch(a.name, a.script, a.gpu, a.image, reserve=a.reserve,
               planned=a.planned, ref=a.ref, cloud=a.cloud)


if __name__ == "__main__":
    main()
