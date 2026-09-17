#!/usr/bin/env bash
# Run the remaining steps back to back on one pod, unattended.
#
#   experiments/chain.sh            # from step 2's completion onward
#
# For each step: wait for the running watchdog to exit (the pod stopped),
# bank the rows, count DONE markers against the models expected; if short,
# relaunch the same step once (DONE markers skip finished models, so a retry
# only redoes what failed); then launch the next step. Every launch goes
# through launch_when_available.sh, which resumes the stopped pod when its
# host has a GPU and otherwise creates a fresh one and restores.
set -u
cd "$(dirname "$0")/.."
RESTORE=deepansadhukhanjeet/undertone-restore   # set only on retries, see run_step
V1=deepansadhukhanjeet/undertone-item-pack
V2=deepansadhukhanjeet/undertone-item-pack-v2
B6=deepansadhukhanjeet/undertone-item-pack-600

# step | script | pack dataset | FP | bank tag | expected DONE | deadline min | planned $
STEPS=(
  "03_all_items|experiments/03_all_items.sh|$V2|0b14538c9854|v2|12|1500|6.00"
  "03b_twins|experiments/03b_twins_all_items.sh|$V2|0b14538c9854|v2|4|300|1.20"
  "04_band_600|experiments/04_band_600.sh|$B6|7e4fd5c5c30c|600|13|900|5.40"
  "05_2x2_v1|experiments/02_prominence_2x2.sh|$V1|56bd324cf6a3|v1|11|400|1.10"
  "05_2x2_v2|experiments/02_prominence_2x2.sh|$V2|0b14538c9854|v2|11|1000|5.30"
  "05_2x2_600|experiments/02_prominence_2x2.sh|$B6|7e4fd5c5c30c|600|11|400|1.30"
)

wait_for_stop() {  # until no watchdog is running for the current pod
  local POD; POD=$(cat .pod_id 2>/dev/null || true)
  [ -z "$POD" ] && return
  while pgrep -f "watchdog.py $POD" >/dev/null; do sleep 60; done
  pkill -f "pull.py $POD" 2>/dev/null || true
  echo "$(date -u +%FT%TZ) pod $POD stopped: $(tail -1 results/watchdog_undertone.log)"
}

done_count() { find "$1" -maxdepth 2 -name DONE 2>/dev/null | wc -l | tr -d ' '; }

run_step() {  # name script pack fp tag expected deadline planned
  local NAME=$1 SCRIPT=$2 PACK=$3 FP=$4 TAG=$5 EXPECTED=$6 DEADLINE=$7 PLANNED=$8
  local DEST=results/exp_$NAME
  echo "$(date -u +%FT%TZ) === $NAME: launch ($PLANNED planned, $EXPECTED models, $DEADLINE min) ==="
  # First launch: a fresh output directory on the volume and no restore. A
  # restore here would bring the previous step's DONE markers along.
  OUT=/workspace/out_$NAME PACK_DATASET=$PACK FP=$FP RESTORE_DATASET= \
    scripts/runpod/launch_when_available.sh undertone "$SCRIPT" "$PLANNED" "$EXPECTED" "$DEADLINE" "$DEST" || { echo "$NAME: launch failed"; return 1; }
  wait_for_stop
  scripts/runpod/bank.sh "$DEST" "$TAG" | tail -2
  local N; N=$(done_count "$DEST")
  echo "$(date -u +%FT%TZ) $NAME: $N/$EXPECTED DONE"
  if [ "$N" -lt "$EXPECTED" ]; then
    echo "$NAME: short by $((EXPECTED-N)); retrying once"
    # Retry: bank.sh just pushed this step's tree as the restore dataset, so a
    # fresh pod continues from it; a resumed pod has it on the volume anyway.
    OUT=/workspace/out_$NAME PACK_DATASET=$PACK FP=$FP RESTORE_DATASET=$RESTORE \
      scripts/runpod/launch_when_available.sh undertone "$SCRIPT" "$PLANNED" "$EXPECTED" "$DEADLINE" "$DEST" || return 1
    wait_for_stop
    scripts/runpod/bank.sh "$DEST" "$TAG" | tail -2
    N=$(done_count "$DEST"); echo "$(date -u +%FT%TZ) $NAME after retry: $N/$EXPECTED DONE"
  fi
}

# Step 2 is already running: wait for it, bank it, and verify before moving on.
echo "$(date -u +%FT%TZ) waiting for the running step 2 to stop"
wait_for_stop
scripts/runpod/bank.sh results/exp03_all v2 | tail -2
N=$(done_count results/exp03_all); echo "$(date -u +%FT%TZ) 03_all_items: $N/12 DONE"
if [ "$N" -lt 12 ]; then
  IFS='|' read -r NAME SCRIPT PACK FP TAG EXPECTED DEADLINE PLANNED <<<"${STEPS[0]}"
  echo "03_all_items short; retrying once"
  # Step 2 started under the default OUT=/workspace/out; keep it so DONE
  # markers on a resumed volume still apply.
  OUT=/workspace/out PACK_DATASET=$PACK FP=$FP RESTORE_DATASET=$RESTORE \
    scripts/runpod/launch_when_available.sh undertone "$SCRIPT" "$PLANNED" "$EXPECTED" "$DEADLINE" results/exp03_all || exit 1
  wait_for_stop; scripts/runpod/bank.sh results/exp03_all v2 | tail -2
  echo "$(date -u +%FT%TZ) 03_all_items after retry: $(done_count results/exp03_all)/12 DONE"
fi
for spec in "${STEPS[@]:1}"; do
  IFS='|' read -r NAME SCRIPT PACK FP TAG EXPECTED DEADLINE PLANNED <<<"$spec"
  run_step "$NAME" "$SCRIPT" "$PACK" "$FP" "$TAG" "$EXPECTED" "$DEADLINE" "$PLANNED" || { echo "CHAIN STOPPED at $NAME"; exit 1; }
done
echo "$(date -u +%FT%TZ) CHAIN COMPLETE"
export RUNPOD_API_KEY=$(cat .rp_key); .venv/bin/python scripts/runpod/rp.py status | head -1
