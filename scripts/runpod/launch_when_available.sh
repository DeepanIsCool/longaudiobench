#!/usr/bin/env bash
# Retry a launch until a GPU has stock, then arm the watchdog and puller in
# the same breath. Waiting for stock is free; a pod without a watchdog is not.
#
#   scripts/runpod/launch_when_available.sh <name> <script> <planned_usd> <expected_ok> <deadline_min> <dest>
#
# Alternates community and secure cloud each attempt. Reads credentials
# from .rp_key / .hf_token / kaggle.json at the repo root.
set -u
NAME=$1; SCRIPT=$2; PLANNED=$3; EXPECTED=$4; DEADLINE=$5; DEST=$6
MAX_TRIES=${MAX_TRIES:-72}     # 72 x 5 min = 6 h
cd "$(dirname "$0")/../.."
export RUNPOD_API_KEY=$(cat .rp_key) HF_TOKEN=$(cat .hf_token) KAGGLE_JSON=$(cat kaggle.json)
PY=.venv/bin/python
for i in $(seq 1 "$MAX_TRIES"); do
  CLOUD=$([ $((i % 2)) -eq 1 ] && echo COMMUNITY || echo SECURE)
  OUT=$($PY scripts/runpod/rp.py launch --name "$NAME" --script "$SCRIPT" \
        --planned "$PLANNED" --ref "${REF:-paper-run-7}" --cloud "$CLOUD" 2>&1)
  if POD=$(echo "$OUT" | grep -oE '^launched [a-z0-9]+' | awk '{print $2}') && [ -n "$POD" ]; then
    echo "$OUT"
    echo "$POD" > .pod_id
    nohup $PY scripts/runpod/watchdog.py "$POD" "$EXPECTED" "$DEADLINE" > "results/watchdog_${NAME}.log" 2>&1 &
    nohup $PY scripts/runpod/pull.py "$POD" --dest "$DEST" --minutes "$DEADLINE" > "results/pull_${NAME}.log" 2>&1 &
    echo "watchdog and pull armed for $POD (logs: results/watchdog_${NAME}.log, results/pull_${NAME}.log)"
    exit 0
  fi
  if echo "$OUT" | grep -q "no instances currently available"; then
    [ $((i % 6)) -eq 0 ] && echo "try $i/$MAX_TRIES: still no A40/A6000 stock on either cloud"; sleep 300; continue
  fi
  echo "$OUT"; echo "launch refused for a reason other than stock - stopping"; exit 1
done
echo "no stock after $MAX_TRIES tries"; exit 1
