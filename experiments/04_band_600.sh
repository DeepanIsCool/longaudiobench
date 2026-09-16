#!/usr/bin/env bash
# Step 4 of the $20 plan. ~$6 for 12 models; drop to 6 if step 3 ran long.
#
# The ladder at a 600 s duration band. Every item so far is a 5-minute
# haystack, so "needle type explains more variance than recording length"
# is untestable. A second band makes it a two-way model, positioned against
# every benchmark that varies only duration.
#
# Memory: Omni-7B peaks at 19.4 GiB for 300 s; audio tokens scale linearly
# with memory-efficient attention, so ~35 GiB at 600 s. An A40 fits. Do not
# attempt 1200 s here - that is H100 territory and out of budget.
#
# The 600 s pack is a separate harvest with --band-cap 600. Same meetings
# are fine - the windows are different lengths, so no cell is repeated.
#
#   python scripts/build_item_pack.py --out data/item_pack_600 --band-cap 600 \
#       --meetings 120 --target 70
#
# Launch:  PACK_DATASET=<user>/undertone-item-pack-600 \
#          python scripts/runpod/rp.py launch --name band600 \
#              --script experiments/04_band_600.sh --planned 6.00
# Watch:   python scripts/runpod/watchdog.py <pod> 12 720
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp04_band600 --minutes 720
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
# If the budget is short, keep the first six: they span the RetrievalCost
# range (+.357 Gemma-E4B down to +.043 Voxtral) and both matched pairs.
MODELS=${MODELS:-"gemma3n_e4b voxtral_mini_3b moss_audio_8b_thinking moss_audio_8b_instruct \
  qwen2_5_omni_7b qwen2_5_omni_3b \
  moss_audio_4b_thinking moss_audio_4b_instruct gemma3n_e2b qwen2_audio_7b \
  phi4_multimodal aero_1_audio"}
for KEY in $MODELS; do
  run_model "$KEY" --ladder
done
finish
