#!/usr/bin/env bash
# Step 2 of the $20 plan. ~$4, ~5 h on an A40. The novelty upgrade.
#
# Three sweeps per model on the current 70 items, sharing one sweep file per
# arm so the needle arm's resume never marks the competitor arm done:
#
#   needle:boost        +3/+6/+9 dB on the answer. If a louder answer repairs
#                       the failure, prominence was what was missing - the
#                       fix, not just the break.
#   competitor:default  the same 0..-60 ladder applied to the loud competitor
#                       with the answer untouched. If accuracy rises as the
#                       loud thing gets quieter, the prior is *relative*
#                       prominence; if not, it is the answer's own audibility.
#   needle:calibrated   0..-12 in 2 dB steps, the region where Table 3b
#                       shows the dose is delivered to within 0.5 dB. A
#                       psychometric curve fitted here gives a threshold in
#                       dB per model.
#
# Together with the existing needle:default sweep that is the 2x2. Sweep
# cells use short contrast windows, so they are cheap: ~1 s each.
#
# Launch:  python scripts/runpod/rp.py launch --name 2x2 \
#              --script experiments/02_prominence_2x2.sh --planned 4.00
# Watch:   python scripts/runpod/watchdog.py <pod> 11 360
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp02_2x2 --minutes 360
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=56bd324cf6a3
# Slowest first, so a deadline kill costs the cheapest models, not the dearest.
for KEY in moss_audio_8b_thinking qwen2_5_omni_7b gemma3n_e4b gemma3n_e2b \
           qwen2_audio_7b moss_audio_4b_thinking qwen2_5_omni_3b \
           phi4_multimodal moss_audio_8b_instruct moss_audio_4b_instruct \
           voxtral_mini_3b; do
  run_model "$KEY" --fingerprint $FP \
    --sweep needle:boost --sweep competitor:default --sweep needle:calibrated
done
finish
