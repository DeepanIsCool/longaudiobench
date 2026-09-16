#!/usr/bin/env bash
# Step 1 of the $20 plan. ~$1, ~40 min on an A40. Cannot fail expensively.
#
# Three more text models behind the identical Whisper front end, on the same
# 70 items with the same letter randomisation as the audio models. The first
# cascade (Qwen2.5-7B-Instruct) is the text twin of one audio model; these
# make the salience-prior result a claim about text LLMs, not about one lab.
#
# The first twin pays for ASR on all 280 windows; the transcript cache
# (UNDERTONE_ASR_CACHE) hands the same transcripts to the other two, so they
# cost only the text model. The original cascade is re-run first so all four
# share one commit - it ran on 0948b549, the audio models on 68c75bc.
#
# Launch:  python scripts/runpod/rp.py launch --name twins \
#              --script experiments/01_text_twins.sh --planned 1.00
# Watch:   python scripts/runpod/watchdog.py <pod> 4 120
# Pull:    python scripts/runpod/pull.py <pod> --dest results/exp01_twins
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=56bd324cf6a3
for KEY in cascaded_whisper_llm cascaded_whisper_mistral_7b \
           cascaded_whisper_llama31_8b cascaded_whisper_gemma2_9b; do
  run_model "$KEY" --fingerprint $FP --ladder --skip-preflight
done
finish
