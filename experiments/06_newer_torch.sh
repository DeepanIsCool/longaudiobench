#!/usr/bin/env bash
# Aero-1-Audio and Audio-Flamingo-Next need torch >= 2.5: transformers 5.16
# refuses anything older (is_torch_available() returns False, and the
# processor load dies with an empty ImportError), and Aero's only successful
# run was on torch 2.10. The main pod's image is torch 2.4.1, held there by
# the constraints file so no other model's floor can replace it. These two
# run on a second pod with a torch 2.8 image, full protocol, one pack per
# launch. Aero's rows from the main pod, if any, are superseded so all of
# its rows share one torch.
#
#   IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#   PACK_DATASET=... FP=... python scripts/runpod/rp.py launch --name torch28 \
#       --image $IMAGE --script experiments/06_newer_torch.sh --planned 1.50
source "$(dirname "$0")/../scripts/runpod/bootstrap.sh"
setup
FP=${FP:-56bd324cf6a3}
MODELS=${MODELS:-"audio_flamingo_next aero_1_audio"}
for KEY in $MODELS; do
  run_model "$KEY" --fingerprint "$FP" --ladder --sweep needle:default \
    --sweep needle:boost --sweep competitor:coarse --sweep needle:calibrated
done
finish
