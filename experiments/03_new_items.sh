#!/usr/bin/env bash
# Step 3 of the $20 plan. ~$7 for the 279 new items (12 models, ladder only).
#
# The ladder on the *expansion pack only*. This is what resolves the two
# claims that failed the cluster-robust test (abstention rises with context,
# 9/12 p=.15; quiet items abstain more, 8/11 p=.23) - they are underpowered
# at 70 items, and no model count fixes that.
#
# The expansion pack exists: data/item_pack_v2/ (279 items, fingerprint
# d2815553b0ff), merged from Kaggle runs 03 and 04 at paper-run-3. C1 86,
# P2 88, P4 75, P3 21, P1 9, plus 30 nulls. Disjoint from v1 by window.
#
# Upload it once (private, ~450 MB), then launch:
#
#   kaggle datasets create -p data/item_pack_v2 --dir-mode zip
#
# Merge at analysis time by concatenating rows; the fingerprint on every row
# says which pack it came from.
#
# Launch:  PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-v2 \
#          python scripts/runpod/rp.py launch --name items \
#              --script experiments/03_new_items.sh --planned 7.00
# Watch:   python scripts/runpod/watchdog.py <pod> 12 1200
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp03_items --minutes 1200
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
for KEY in moss_audio_8b_thinking qwen2_5_omni_7b gemma3n_e4b gemma3n_e2b \
           qwen2_audio_7b moss_audio_4b_thinking qwen2_5_omni_3b \
           phi4_multimodal moss_audio_8b_instruct moss_audio_4b_instruct \
           voxtral_mini_3b aero_1_audio; do
  run_model "$KEY" --ladder
done
finish
