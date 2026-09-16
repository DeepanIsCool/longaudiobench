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
set -u
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export UNDERTONE_ASR_CACHE=${UNDERTONE_ASR_CACHE:-/workspace/asr_cache}
REPO=${REPO:-/workspace/repo}
OUT=${OUT:-/workspace/out}
PACK=${PACK:-/workspace/pack}
PACK_DATASET=${PACK_DATASET:-sadhukhandeepan/undertone-item-pack}

setup() {
  mkdir -p "$OUT" "$HF_HOME" "$UNDERTONE_ASR_CACHE" ~/.kaggle
  if [ -n "${KAGGLE_JSON:-}" ]; then
    printf '%s' "$KAGGLE_JSON" > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  fi
  ( cd "$OUT" && python -m http.server 8000 >/dev/null 2>&1 & )
  exec > >(tee -a "$OUT/run.log") 2>&1
  echo "=== $(basename "$0") on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
  echo "code sha: ${UNDERTONE_CODE_SHA:-unset}"
  TORCH_V=$(python -c "import torch;print(torch.__version__.split('+')[0])")
  printf 'torch==%s\n' "$TORCH_V" > /workspace/constraints.txt
  pip install -q kaggle 2>&1 | tail -1
  if [ ! -f "$PACK/item_pack/item_pack.jsonl" ] && [ -n "${KAGGLE_JSON:-}" ]; then
    mkdir -p "$PACK"
    kaggle datasets download "$PACK_DATASET" -p "$PACK" --unzip 2>&1 | tail -1
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
  # Give pull.py time for a last pass before the watchdog terminates the pod.
  sleep 600
  exit 0
}
