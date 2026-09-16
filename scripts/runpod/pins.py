#!/usr/bin/env python3
"""Print the exact pip set the notebook generator holds for one model.

Phase 0 failed because a pip line was hand-typed: transformers==4.57.1 for
Phi-4, whose notebook pins 4.48.2. Its remote code then hit
'Phi4MMModel' object has no attribute 'prepare_inputs_for_generation'. The
generator is the single source of truth for pins; nothing retypes them.

    python scripts/runpod/pins.py qwen2_5_omni_7b
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import make_notebooks as mk  # noqa: E402

key = sys.argv[1]
meta = mk.META.get(key, {})
pins = list(meta.get("pip", mk.BASE_PIP))
if key.startswith("cascaded_"):
    # Mirrors build_cascaded_notebook: the control adds the ASR engine to
    # the base set. The text twins share this exactly.
    pins.append("faster-whisper>=1.0.0")
print(" ".join(f'"{p}"' for p in pins))
