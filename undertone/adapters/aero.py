"""Aero-1-Audio -- 1.5B LLM (Qwen2.5-1.5B) + Whisper-style encoder, ~4GB fp16.

The only model in the roster that fits comfortably on one T4 with headroom, and
the cheapest way to exercise the full 15-minute ladder.  Card claims continuous
ASR and understanding up to 15 min without splitting; AudioMarathon Table 8
agrees (900 s, 22.5k tokens at 25 tok/s).

Two card details that bite:
  * the card recommends ``flash_attention_2``.  T4 is sm75 -- flash-attn-2 needs
    sm80+, so ``sdpa`` here, not the documented default.
  * ``eos_token_id=151645`` must be passed explicitly to ``generate``.
  * the message content type is ``audio_url`` with the literal string
    ``"placeholder"``; the real array goes through the processor separately.
  * the card's snippet says ``lmms-lab/Aero-1-Audio-1.5B`` while the repo is
    ``lmms-lab/Aero-1-Audio``; both are tried at load.
"""

from __future__ import annotations

import numpy as np

from .base import (
    SAMPLE_RATE,
    ModelAdapter,
    call_processor,
    move_to_device,
    primary_device,
    register,
)


@register
class Aero1Audio(ModelAdapter):
    key = "aero_1_audio"
    model_id = "lmms-lab/Aero-1-Audio"
    fallback_ids = ("lmms-lab/Aero-1-Audio-1.5B",)
    max_audio_s = 900.0
    primary = "logits"
    eos_token_id = 151645
    notes = "sdpa not flash-attn-2 on sm75. 15 min ceiling, 25 audio tokens/s."

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        last: Exception | None = None
        for repo in (self.model_id, *self.fallback_ids):
            try:
                self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
                self.model = self.place(AutoModelForCausalLM.from_pretrained(
                    repo, attn_implementation="sdpa", trust_remote_code=True,
                    **self.load_kwargs()))
                self.model_id = repo
                self.tokenizer = self.processor.tokenizer
                return
            except Exception as exc:  # noqa: BLE001 - try the documented alias
                last = exc
        raise RuntimeError(f"could not load Aero-1-Audio from any known id: {last}")

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        messages = [{
            "role": "user",
            "content": [{"type": "audio_url", "audio": "placeholder"},
                        {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Named explicitly: this processor takes **kwargs, so `audio=` is
        # accepted and silently dropped, which is how it returned identical
        # logits for different clips.
        inputs = call_processor(self.processor, text, audio, sr, audio_kwarg="audios")
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )

    def generate_kwargs(self) -> dict:
        return {"eos_token_id": self.eos_token_id}
