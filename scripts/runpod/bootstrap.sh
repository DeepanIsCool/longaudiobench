#!/usr/bin/env bash
# Pod-side driver. Sourced by every experiments/*.sh; never run directly.
#
# Provides:
#   setup            once per pod: http server for pull/watchdog, HF + Kaggle
#                    auth from env, torch pin so pip cannot replace it, pack
#   run_model KEY    fresh venv with KEY's exact pins, then run_model.py "$@"
#   finish           summary + RUN_COMPLETE marker + exit 0
#
# Rules this file exists to enforce, each learned the expensive way:
#
#   No `set -e`.   A non-zero exit makes Runpod restart the container and
#                  re-run everything from the top. Six hours and $3.04 went
#                  that way once. Each model reports its own status; the
#                  script always ends exit 0 and the watchdog owns the deadline.
#   One venv per model.  pip accumulates into one site-packages. A torchao
#                  floor on one model replaced torch for every model after it;
#                  11 of 13 failed with `module 'torch' has no attribute 'int1'`.
#   Pins from pins.py.  Never typed by hand. See pins.py.
#   Constraint torch.  A pip constraints file holding the image's torch so no
#                  model's requirements can swap it.
#   Secrets from env.  Set on the pod by rp.py launch. Nothing here has a
#                  literal token in it.
#   No `set -u` either.  An unbound variable would exit non-zero, and a
#                  non-zero exit is a restart, and a restart is a loop. A
#                  typo should fail one model's run_model call and move on.
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export UNDERTONE_ASR_CACHE=${UNDERTONE_ASR_CACHE:-/workspace/asr_cache}
REPO=${REPO:-/workspace/repo}
OUT=${OUT:-/workspace/out}
PACK=${PACK:-/workspace/pack}
PACK_DATASET=${PACK_DATASET:-deepansadhukhanjeet/undertone-item-pack}

setup() {
  mkdir -p "$OUT" "$HF_HOME" "$UNDERTONE_ASR_CACHE"
  # Kaggle auth is KAGGLE_USERNAME + KAGGLE_KEY in the environment, set by
  # rp.py launch. No kaggle.json is written: a JSON blob passed through the
  # pod env came out unusable and the CLI silently went anonymous.
  ( cd "$OUT" && python -m http.server 8000 >/dev/null 2>&1 & )
  exec > >(tee -a "$OUT/run.log") 2>&1
  # The watchdog counts this exact banner. Two of them means Runpod restarted
  # the container, and the watchdog terminates the pod on sight.
  echo "=== RUN START $(basename "$0") on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date -u +%FT%TZ) ==="
  echo "code sha: ${UNDERTONE_CODE_SHA:-unset}"
  TORCH_V=$(python -c "import torch;print(torch.__version__.split('+')[0])")
  printf 'torch==%s\n' "$TORCH_V" > /workspace/constraints.txt
  pip install -q kaggle 2>&1 | tail -1
  if [ ! -f "$PACK/item_pack/item_pack.jsonl" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    mkdir -p "$PACK"
    kaggle datasets download "$PACK_DATASET" -p "$PACK" --unzip 2>&1 | tail -1
  fi
  # No pack means every model fails after a venv build each. Say so once
  # and finish; the watchdog terminates on RUN_COMPLETE within a minute.
  # find, not ls: `ls a/*/x b/x` exits non-zero when any argument is
  # missing, so a pack unzipped flat read as missing and the third twins
  # launch aborted a successful download. $0.02.
  if [ -z "$(find "$PACK" -maxdepth 3 -name item_pack.jsonl 2>/dev/null)" ]; then
    echo "PACK MISSING under $PACK after download of $PACK_DATASET - aborting before any model"
    finish
  fi
}

run_model() {
  local KEY=$1; shift
  if [ -f "$OUT/$KEY/DONE" ]; then echo "skip $KEY (DONE)"; return; fi
  echo; echo "######## $KEY ########"; date -u +"start %H:%M:%S UTC"
  local PINS; PINS=$(cd "$REPO" && python scripts/runpod/pins.py "$KEY")
  local V=/workspace/venv
  rm -rf "$V" && python -m venv --system-site-packages "$V"
  # shellcheck disable=SC2086
  "$V/bin/pip" install -q --constraint /workspace/constraints.txt $PINS 2>&1 | tail -1
  ( cd "$REPO" && "$V/bin/python" scripts/runpod/run_model.py "$KEY" \
      --pack "$PACK" --out "$OUT" --repo "$REPO" "$@" )
  date -u +"end   %H:%M:%S UTC"
  rm -rf "$HF_HOME/hub" "$V" 2>/dev/null   # weights and venv; the ASR cache stays
}

finish() {
  echo; echo "############ SUMMARY ############"
  grep -aE "_OK$|_FAILED$" "$OUT/run.log" | sort | uniq -c
  echo "RUN_COMPLETE"
  # The watchdog terminates on RUN_COMPLETE within a minute. This sleep is
  # only so pull.py gets a last pass first; if the watchdog is somehow not
  # running, the pod exits on its own and Runpod's restart hits the
  # DONE-marker skips and the banner count, and is terminated on sight.
  sleep 900
  exit 0
}
