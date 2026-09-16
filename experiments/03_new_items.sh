#!/usr/bin/env bash
# Step 3 of the $20 plan. ~$4 for ~130 new items; scales with the pack.
#
# The ladder on the *expansion pack only*. This is what resolves the two
# claims that failed the cluster-robust test (abstention rises with context,
# 9/12 p=.15; quiet items abstain more, 8/11 p=.23) - they are underpowered
# at 70 items, and no model count fixes that.
#
# The expansion pack is built locally first (CPU, free) and pushed to Kaggle
# as its own dataset, so nothing is re-scored:
#
#   python scripts/build_item_pack.py --out data/item_pack_v2 --meetings 120 \
#       --exclude-pack data/item_pack/item_pack.jsonl \
#       --share C1=0.30,P1=0.25,P2=0.25,P3=0.10,P4=0.10 --target 200
#   kaggle datasets create -p data/item_pack_v2   # then set PACK_DATASET below
#
# Merge at analysis time by concatenating rows; the fingerprint on every row
# says which pack it came from.
#
# Launch:  PACK_DATASET=<user>/undertone-item-pack-v2 \
#          python scripts/runpod/rp.py launch --name items \
#              --script experiments/03_new_items.sh --planned 4.00
# Watch:   python scripts/runpod/watchdog.py <pod> 12 600
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp03_items --minutes 600
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
for KEY in moss_audio_8b_thinking qwen2_5_omni_7b gemma3n_e4b gemma3n_e2b \
           qwen2_audio_7b moss_audio_4b_thinking qwen2_5_omni_3b \
           phi4_multimodal moss_audio_8b_instruct moss_audio_4b_instruct \
           voxtral_mini_3b aero_1_audio; do
  run_model "$KEY" --ladder
done
finish
