#!/usr/bin/env bash
# Every existing task on the 338 new items, one model load each. ~$14.50
# at $0.49/h. This is what makes every table in the paper the same 408
# items for every model.
#
#   ladder L1-L4       12 models
#   sweep needle       11 models (Aero has no recorded competitor spans)
#   question-only      the same 4 models as the original control
#
# Per model, run_model.py loads weights once and runs all three; the ladder
# is ~4.5 s/cell, the sweep ~1 s/cell. Ordered slowest first so a deadline
# cut costs the cheapest models.
#
# Launch:  PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-v2 \
#          python scripts/runpod/rp.py launch --name undertone \
#              --script experiments/03_all_items.sh --planned 14.50
# Watch:   python scripts/runpod/watchdog.py <pod> 12 2400
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp03_all --minutes 2400
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=0b14538c9854
QO="gemma3n_e2b moss_audio_4b_instruct phi4_multimodal qwen2_5_omni_7b"
for KEY in moss_audio_8b_thinking qwen2_5_omni_7b gemma3n_e4b gemma3n_e2b \
           qwen2_audio_7b moss_audio_4b_thinking qwen2_5_omni_3b \
           phi4_multimodal moss_audio_8b_instruct moss_audio_4b_instruct \
           voxtral_mini_3b aero_1_audio; do
  EXTRA=""
  [ "$KEY" != "aero_1_audio" ] && EXTRA="--sweep needle:default"
  case " $QO " in *" $KEY "*) EXTRA="$EXTRA --qo";; esac
  # shellcheck disable=SC2086
  run_model "$KEY" --fingerprint $FP --ladder $EXTRA
done
finish
