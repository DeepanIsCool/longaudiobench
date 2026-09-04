"""Audio-Flamingo Next -- 8.27B, ~16.5GB fp16, sharded over 2xT4.

Native long-audio support to **30 minutes**: the released processor is
configured for 1800 s, processed internally in 30 s windows, with Rotary Time
Embeddings for timestamp awareness.  That makes it the only model in the roster
whose documented ceiling reaches our longest band exactly.

Card caveats handled here:
  * the example loads in bfloat16; sm75 has no bf16 compute, so fp16.
  * ``batch["input_features"]`` must be cast to the model dtype explicitly --
    the processor does not do it.
  * the conversation is a list **of lists** (batched), unlike every other chat
    template in the roster.
  * audio goes in by path, so each window is written to a temp wav.
  * architecture string is ``musicflamingo``; ``AutoModel`` is the documented
    entry point.
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
class AudioFlamingoNext(ModelAdapter):
    key = "audio_flamingo_next"
    model_id = "nvidia/audio-flamingo-next-hf"
    max_audio_s = 1800.0
    primary = "logits"
    notes = (
        "Processor configured for 1800 s, 30 s internal windows. Card uses bf16; "
        "fp16 here for sm75, with input_features cast to model dtype."
    )

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model = self.place(AutoModel.from_pretrained(
            self.model_id, **self.load_kwargs()))

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        path = as_temp_wav(audio, sr)
        try:
            conversation = [[{
                "role": "user",
                "content": [{"type": "text", "text": prompt},
                            {"type": "audio", "path": path}],
            }]]
            batch = self.processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=True
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        batch = dict(batch)
        return move_to_device(
            batch, primary_device(self.model), self.model.dtype, self.audio_float_keys
        )
