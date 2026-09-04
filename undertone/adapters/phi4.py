"""Phi-4-multimodal-instruct -- 5.57B, ~11GB fp16, fits one T4.

Card facts that drive this adapter:
  * ``transformers==4.48.2``.  Its remote code is version-sensitive, so this
    model gets its own notebook pin rather than sharing the roster's.
  * ``_attn_implementation="eager"`` for pre-Ampere GPUs; T4 is sm75.
  * Prompt format is literal: ``<|user|><|audio_1|>{q}<|end|><|assistant|>``.
  * Audio is passed as ``audios=[(array, sample_rate)]`` -- a tuple, not a bare
    array, unlike every other model in the roster.
  * "maximum audio length is suggested to be 40s [for QA] ... for summarization
    tasks, 30 mins".  We use the 30 min ceiling and record the caveat: the QA
    guidance is 40 s, so long-band results for this model carry an asterisk.
"""

from __future__ import annotations

import numpy as np

from .base import SAMPLE_RATE, ModelAdapter, move_to_device, primary_device, register


@register
class Phi4Multimodal(ModelAdapter):
    key = "phi4_multimodal"
    model_id = "microsoft/Phi-4-multimodal-instruct"
    max_audio_s = 1800.0
    primary = "logits"
    audio_float_keys = ("input_audio_embeds", "audio_attention_mask")
    notes = (
        "transformers==4.48.2, eager attention (sm75). Card suggests 40 s for QA "
        "and 30 min only for summarization: long-band cells are marked as "
        "exceeding the documented QA guidance."
    )
    qa_guidance_s = 40.0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = self.place(AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, _attn_implementation="eager",
            **self.load_kwargs()))
        self.generation_config = GenerationConfig.from_pretrained(self.model_id)

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        text = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
        inputs = self.processor(text=text, audios=[(audio, sr)], return_tensors="pt")
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )

    def generate_kwargs(self) -> dict:
        return {"generation_config": self.generation_config}
