#!/usr/bin/env bash
# The 600 s band: ladder for the 9 models that can hear it, then the four
# text twins on the same 85 items. ~$3.60 at $0.49/h.
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
# The 600 s pack exists: data/item_pack_600/, 85 items (P1 21, P2 19, P3 14,
# P4 25, C1 6, incl. 10 nulls), from Kaggle notebook 06 at paper-run-5 -
# all scenario meetings at --band-cap 600, target 180 pre-filter. Window ids
# carry the band (ES2006d_b600_w1), so nothing collides with the 300 s packs.
# Uploaded as deepansadhukhanjeet/undertone-item-pack-600.
#
# P1 note: 21 quiet items here against 20 in the whole 300 s corpus. Longer
# windows have more dynamic range for the bottom-decile energy test to work
# with. That is the first real P1 sample size the benchmark has had.
#
# Launch:  PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-600 \
#          python scripts/runpod/rp.py launch --name undertone \
#              --script experiments/04_band_600.sh --planned 3.60
# Watch:   python scripts/runpod/watchdog.py <pod> 13 720
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp04_band600 --minutes 720
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
# Nine models, not twelve: both Gemma-3n and Qwen2-Audio have 30 s encoder
# caps, so their 600 s rows would be the same truncation as their 300 s rows
# and say nothing about duration. Ordered so the first six span the
# RetrievalCost range and both matched pairs, if the budget forces a cut.
MODELS=${MODELS:-"moss_audio_8b_thinking moss_audio_8b_instruct qwen2_5_omni_7b voxtral_mini_3b \
  moss_audio_4b_thinking moss_audio_4b_instruct \
  qwen2_5_omni_3b phi4_multimodal aero_1_audio"}
for KEY in $MODELS; do
  run_model "$KEY" --ladder
done
for KEY in cascaded_whisper_llm cascaded_whisper_mistral_7b \
           cascaded_whisper_llama31_8b cascaded_whisper_gemma2_9b; do
  run_model "$KEY" --ladder --skip-preflight
done
finish
