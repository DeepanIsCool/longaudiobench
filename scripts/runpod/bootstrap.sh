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
#   Pins from pins.txt.  Generated on the laptop by pins.py --write, never
#                  typed by hand, never computed on the pod.
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
PACK_DATASET=${PACK_DATASET:-deepansadhukhanjeet/undertone-item-pack}
# One directory per dataset: v1, v2 and the 600 s band all unzip flat with
# an item_pack.jsonl at the root, and a second pack into one directory
# would overwrite the first's.
PACK=${PACK:-/workspace/pack_$(basename "$PACK_DATASET")}

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
  TORCH_V=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null)
  case "$TORCH_V" in
    [0-9]*.[0-9]*) printf 'torch==%s\n' "$TORCH_V" > /workspace/constraints.txt ;;
    *) echo "torch version unreadable from the image python ('$TORCH_V') - aborting before any model"; finish ;;
  esac
  echo "constraints: $(cat /workspace/constraints.txt)"
  pip install -q kaggle 2>&1 | tail -1
  echo "pack: $PACK_DATASET -> $PACK"
  if [ -z "$(find "$PACK" -maxdepth 3 -name item_pack.jsonl 2>/dev/null)" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    mkdir -p "$PACK"
    kaggle datasets download "$PACK_DATASET" -p "$PACK" --unzip 2>&1 | tail -1
  else
    echo "pack already on the volume"
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
  # From the committed file, with grep. No Python runs on the pod before the
  # venv exists - that is what took the fourth twins launch down.
  local PINS; PINS=$(grep "^${KEY}|" "$REPO/scripts/runpod/pins.txt" | cut -d'|' -f2-)
  if [ -z "$PINS" ]; then
    echo "$KEY: pins.py printed nothing - not building a venv for it"; echo "${KEY}_FAILED"; return
  fi
  echo "pins: $PINS"
  local V=/workspace/venv
  rm -rf "$V" && python -m venv --system-site-packages "$V"
  # Full pip output to a file the puller mirrors. The first twins run showed
  # only pip's upgrade notice via tail -1 while installing nothing in six
  # seconds; the reason was in the lines tail threw away.
  mkdir -p "$OUT/$KEY"
  # shellcheck disable=SC2086
  "$V/bin/pip" install --constraint /workspace/constraints.txt $PINS > "$OUT/$KEY/pip.log" 2>&1
  local PIPRC=$?
  grep -vE "^\s*$|Requirement already satisfied|^\[notice\]" "$OUT/$KEY/pip.log" | tail -4
  if [ $PIPRC -ne 0 ]; then
    echo "$KEY: pip exited $PIPRC - see $KEY/pip.log"; echo "${KEY}_FAILED"; return
  fi
  # The venv must be able to import the first pin before a model load is
  # attempted. transformers is in every pin set.
  if ! "$V/bin/python" -c "import transformers" 2>/dev/null; then
    echo "$KEY: venv cannot import transformers after pip succeeded - see $KEY/pip.log"
    "$V/bin/pip" freeze | head -20; echo "${KEY}_FAILED"; return
  fi
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
