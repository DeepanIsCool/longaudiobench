"""Gemma-3n E2B / E4B.

**Gated repos.**  Accept the licence at huggingface.co/google/gemma-3n-E2B-it
and -E4B-it, then add the HF token as a Kaggle secret named ``HF_TOKEN``.
Without that these two notebooks cannot run at all.

Audio ceiling is **30 s**: the USM-based encoder ships configured for 30 s
clips at ~6.25 tokens/s (one token per 160 ms).  Google note the encoder is
streaming and not fundamentally capped, but the released implementation is, so
30 s is what we report.  Like Qwen2-Audio, these two are L1 instruments -- they
anchor the perception ceiling that makes a low L3 score interpretable.

Params are 5.44B (E2B) and 7.85B (E4B) raw, despite the "effective 2B/4B"
naming; VRAM follows the raw count, so E4B needs both T4s.
"""

from __future__ import annotations

import numpy as np

from .base import SAMPLE_RATE, ModelAdapter, move_to_device, primary_device, register


class _Gemma3n(ModelAdapter):
    max_audio_s = 30.0
    primary = "logits"
    audio_float_keys = ("input_features", "input_features_mask")

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        from ..env import hf_token

        token = hf_token()
        if not token:
            raise RuntimeError(
                f"{self.model_id} is gated. Accept the licence on the Hub, then "
                "provide a token: HF_TOKEN in the environment, a Kaggle secret "
                "named HF_TOKEN, or .hf_token at the repo root (gitignored).")
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self.tokenizer = self.processor.tokenizer
        self.model = self.place(Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id, token=token, **self.load_kwargs()))

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        messages = [{
            "role": "user",
            "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )


@register
class Gemma3nE2B(_Gemma3n):
    key = "gemma3n_e2b"
    model_id = "google/gemma-3n-E2B-it"
    notes = "Gated. 30 s encoder cap (~6.25 audio tokens/s). 5.44B raw params."


@register
class Gemma3nE4B(_Gemma3n):
    key = "gemma3n_e4b"
    model_id = "google/gemma-3n-E4B-it"
    notes = "Gated. 30 s encoder cap. 7.85B raw params -> sharded over 2xT4."
