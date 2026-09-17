#!/usr/bin/env bash
# The second pod: Aero + Audio-Flamingo on all three packs. Runs alongside
# chain.sh; same wait/bank/verify/retry loop, different pod name and image.
set -u
cd "$(dirname "$0")/.."
export IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
RESTORE=deepansadhukhanjeet/undertone-restore-t28
V1=deepansadhukhanjeet/undertone-item-pack
V2=deepansadhukhanjeet/undertone-item-pack-v2
B6=deepansadhukhanjeet/undertone-item-pack-600
STEPS=(
  "06_t28_v1|$V1|56bd324cf6a3|v1|2|300|1.20"
  "06_t28_v2|$V2|0b14538c9854|v2|2|600|3.00"
  "06_t28_600|$B6|7e4fd5c5c30c|600|2|300|1.20"
)
wait_for_stop() {
  local POD; POD=$(cat .pod_id_torch28 2>/dev/null || true); [ -z "$POD" ] && return
  while pgrep -f "watchdog.py $POD" >/dev/null; do sleep 60; done
  pkill -f "pull.py $POD" 2>/dev/null || true
  echo "$(date -u +%FT%TZ) pod $POD stopped: $(tail -1 results/watchdog_torch28.log)"
}
done_count() { find "$1" -maxdepth 2 -name DONE 2>/dev/null | wc -l | tr -d ' '; }
launch() {  # name pack fp expected deadline planned restore
  OUT=/workspace/out_$1 PACK_DATASET=$2 FP=$3 RESTORE_DATASET=$7 POD_ID_FILE=.pod_id_torch28 \
    scripts/runpod/launch_when_available.sh torch28 experiments/06_newer_torch.sh "$6" "$4" "$5" "results/exp_$1"
}
for spec in "${STEPS[@]}"; do
  IFS='|' read -r NAME PACK FP TAG EXPECTED DEADLINE PLANNED <<<"$spec"
  DEST=results/exp_$NAME
  if [ "$(done_count "$DEST")" -ge "$EXPECTED" ]; then echo "$NAME: already done, skipping"; continue; fi
  echo "$(date -u +%FT%TZ) === $NAME: launch ==="
  launch "$NAME" "$PACK" "$FP" "$EXPECTED" "$DEADLINE" "$PLANNED" "" || { echo "$NAME: launch failed"; exit 1; }
  wait_for_stop
  scripts/runpod/bank.sh "$DEST" "$TAG" | tail -2
  N=$(done_count "$DEST"); echo "$(date -u +%FT%TZ) $NAME: $N/$EXPECTED DONE"
  if [ "$N" -lt "$EXPECTED" ]; then
    echo "$NAME: short; retrying once"
    launch "$NAME" "$PACK" "$FP" "$EXPECTED" "$DEADLINE" "$PLANNED" "$RESTORE" || exit 1
    wait_for_stop; scripts/runpod/bank.sh "$DEST" "$TAG" | tail -2
    echo "$(date -u +%FT%TZ) $NAME after retry: $(done_count "$DEST")/$EXPECTED DONE"
  fi
done
echo "$(date -u +%FT%TZ) TORCH28 CHAIN COMPLETE"
