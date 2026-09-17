#!/usr/bin/env bash
# The prominence 2x2, on every item: run once per pack.
#
#   needle:boost       +3/+6/+9 dB on the answer. If a louder answer repairs
#                      the failure, prominence was what was missing.
#   competitor:coarse  0/-6/-12/-24/removed on the loud competitor, answer
#                      untouched. If accuracy rises as the loud thing gets
#                      quieter, the prior is *relative* prominence.
#
#   needle:calibrated  0..-12 in 2 dB steps, the region where the stimulus
#                      audit shows the dose is delivered to within 0.5 dB.
#                      A psychometric threshold per model comes from here.
#
# Level 0 for every item already exists in the main sweep (needle:default),
# as do -6 and -12; the three arms add 11 new levels. ~$7.70 across 493
# items and 11 models.
#
# Launch, one pack at a time, same pod:
#   PACK_DATASET=deepansadhukhanjeet/undertone-item-pack     FP=56bd324cf6a3 \
#   python scripts/runpod/rp.py launch --name undertone --script experiments/02_prominence_2x2.sh --planned 1.10
#   PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-v2  FP=0b14538c9854 ... --planned 5.30
#   PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-600 FP=7e4fd5c5c30c ... --planned 1.30
# Watch:   python scripts/runpod/watchdog.py <pod> 11 600
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=${FP:-56bd324cf6a3}
MODELS=${MODELS:-"moss_audio_8b_thinking qwen2_5_omni_7b gemma3n_e4b gemma3n_e2b \
  qwen2_audio_7b moss_audio_4b_thinking qwen2_5_omni_3b \
  phi4_multimodal moss_audio_8b_instruct moss_audio_4b_instruct \
  voxtral_mini_3b"}
for KEY in $MODELS; do
  run_model "$KEY" --fingerprint "$FP" --sweep needle:boost --sweep competitor:coarse --sweep needle:calibrated
done
finish
