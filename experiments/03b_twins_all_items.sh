#!/usr/bin/env bash
# The four text twins on the 338 new items. ~$1.10. Text-only; the first
# twin pays for Whisper on 338 x 4 windows, the rest read the cache.
#
# Launch:  PACK_DATASET=deepansadhukhanjeet/undertone-item-pack-v2 \
#          python scripts/runpod/rp.py launch --name undertone \
#              --script experiments/03b_twins_all_items.sh --planned 1.10
# Watch:   python scripts/runpod/watchdog.py <pod> 4 240
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=0b14538c9854
for KEY in cascaded_whisper_llm cascaded_whisper_mistral_7b \
           cascaded_whisper_llama31_8b cascaded_whisper_gemma2_9b; do
  run_model "$KEY" --fingerprint $FP --ladder --skip-preflight
done
finish
