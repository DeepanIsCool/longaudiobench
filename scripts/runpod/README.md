# Runpod orchestration

Everything needed to run a scored experiment on a rented GPU and get the rows
back, with the spend guarded. Five files:

| file | runs where | does |
| --- | --- | --- |
| `rp.py` | laptop | balance, guard, launch, kill. The only thing that talks to the Runpod API. |
| `watchdog.py` | laptop | terminates the pod on completion, stall, or deadline. **Always running while a pod is.** |
| `pull.py` | laptop | mirrors the pod's output tree locally every 90 s. Nothing waits for a DONE marker. |
| `bootstrap.sh` | pod | sourced by every `experiments/*.sh`; venv-per-model, pins, http server, secrets from env. |
| `run_model.py` | pod | one model, one process: `--ladder`, `--sweep arm:levels`, `--qo`. Resumes. |
| `pins.py` | pod | prints a model's exact pip set from `make_notebooks.META`. Never retype pins. |

## Credentials

Three environment variables. Load them from the gitignored files at the repo
root; never paste a token into a script or a shell history you might commit.

```bash
export RUNPOD_API_KEY=$(cat .rp_key)
export HF_TOKEN=$(cat .hf_token)
export KAGGLE_JSON=$(cat kaggle.json)
```

`rp.py launch` forwards `HF_TOKEN` and `KAGGLE_JSON` into the pod's
environment, where Runpod stores them encrypted. The pod's start command
never contains them.

## The loop

```bash
python scripts/runpod/rp.py status                       # balance, running pods
python scripts/runpod/rp.py launch --name 2x2 \
    --script experiments/02_prominence_2x2.sh --planned 4.00
#   -> prints <pod_id>. In two more terminals, immediately:
python scripts/runpod/watchdog.py <pod_id> 11 360        # expected OKs, deadline min
python scripts/runpod/pull.py <pod_id> --dest results/exp02_2x2 --minutes 360
```

`--planned` is the expected spend; `guard` refuses if `balance - planned`
would breach the floor (`$1.00`). Pass `--ref paper-run-N` to clone a tag
rather than `main` for anything that goes in the paper.

**Use one `--name` for every step.** The watchdog *stops* the pod on
completion (or stall, or restart loop) rather than terminating it. A stopped
pod keeps its volume — weights, packs, ASR cache — and its host, and bills
cents a day for storage. The next `launch` with the same name patches in the
new step's script and tag and resumes it: no re-download, no stock wait.
`rp.py kill` is the only thing that deletes a pod, and it is never automatic.

When the watchdog prints `STOPPING (all N/N done)` the rows are in the
snapshot and pull directories. `rp.py status` shows the pod as stopped.

## Why it looks like this

Each rule in `bootstrap.sh` and `watchdog.py` carries the failure that
produced it, in comments. The short version: a pod that exits non-zero is
restarted by Runpod and re-runs from the top; pip accumulates into one
site-packages so one model's floors break every model after it; a watchdog
that counts the wrong thing kills healthy runs or lets dead ones bill. The
first paper run lost $3.04 of $10 to the first of these before any of this
existed.

## Adding an experiment

Copy the shortest `experiments/*.sh`, change the model list and the
`run_model` flags, put the launch/watch/pull lines in its header comment, and
add it to `experiments/README.md` with a cost. Nothing else changes.
