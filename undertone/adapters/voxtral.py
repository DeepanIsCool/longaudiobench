"""Voxtral-Mini-3B-2507 -- 4.68B, ~9.5GB fp16, fits one T4.

Longest documented audio window in the roster: 32k context gives 30 min for
transcription and **40 min for understanding**, which covers every band we run.

Mistral ship this for vLLM first, but the vLLM server path cannot expose a
next-token logit row cleanly, and uniform letter-logit scoring across all
thirteen models is the whole point of the protocol.  So this uses the
Transformers ``VoxtralForConditionalGeneration`` path instead.  ``mistral-common``
is still needed for the tokenizer.

The chat template takes audio by path rather than by array, so each window is
written to a temp wav -- one disk write against a multi-minute prefill.
"""

from __future__ import annotations

import os

import numpy as np

from .base import (
    SAMPLE_RATE,
    ModelAdapter,
    as_temp_wav,
    move_to_device,
    primary_device,
    register,
)


@register
class VoxtralMini3B(ModelAdapter):
    key = "voxtral_mini_3b"
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    # Mistral document 40 min, which assumes vLLM paged attention. Under HF
    # sdpa on a 15 GB T4 the 30-minute cells OOM trying to allocate 60.78 GiB:
    # attention over ~45k audio tokens is quadratic. 600 s is what actually runs.
    max_audio_s = 600.0
    documented_max_audio_s = 2400.0
    primary = "logits"
    notes = (
        "Transformers path (not vLLM) so letter logits are readable. "
        "Documented 40 min; achievable ~10 min on a T4 under sdpa - the 30 min "
        "cells OOM at 60.78 GiB. Needs `pip install mistral-common[audio]`."
    )

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model = self.place(VoxtralForConditionalGeneration.from_pretrained(
            self.model_id, **self.load_kwargs()))

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        path = as_temp_wav(audio, sr)
        try:
            conversation = [{
                "role": "user",
                "content": [{"type": "audio", "path": path},
                            {"type": "text", "text": prompt}],
            }]
            inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )
