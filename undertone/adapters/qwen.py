"""Qwen adapters.

Qwen2-Audio-7B-Instruct -- 8.40B, ~16.8GB fp16, **hard 30 s cap**.
  Its Whisper-large-v3 encoder is fixed at 3000 mel frames (10 ms/frame), so
  everything past 30 s is dropped by the feature extractor with no warning.
  This is the model that made the retired ANiH sweep measure a truncation
  artifact: latency stayed flat at ~3.1 s whether the nominal clip was 60 s or
  900 s.  Here the cap is applied up-front and recorded.

Qwen2.5-Omni-3B / 7B -- 32k context at 25 audio tokens/s.  AudioMarathon
  Table 8 reports ~21 min of audio.  The Talker (speech synthesis) head is dead
  weight for a text MCQ and costs ~2GB, so it is disabled at load.
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

QWEN_OMNI_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


@register
class Qwen2Audio7B(ModelAdapter):
    key = "qwen2_audio_7b"
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    max_audio_s = 30.0
    primary = "logits"
    notes = (
        "Hard 30 s encoder cap (3000 mel frames). L1 only; L2/L3/L4 run "
        "truncated and are reported in the truncation table, never as zeros."
    )

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        # fp16 on CUDA and MPS (sm75 has no bf16 compute), fp32 on CPU;
        # device_map="auto" only where accelerate can shard -- 16.8 GB does not
        # fit one T4. flash-attn-2 needs sm80+, so sdpa.
        self.model = self.place(Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, attn_implementation="sdpa", **self.load_kwargs()))

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "audio_url": "audio"},
                        {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = call_processor(self.processor, text, audio, sr, padding=True)
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )


class _Qwen25Omni(ModelAdapter):
    primary = "logits"
    audio_float_keys = ("input_features", "feature_attention_mask")

    def load(self) -> None:
        import transformers
        from transformers import Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

        # Prefer the Thinker on its own. We want text, never speech, so loading
        # the full Thinker+Talker only to disable the Talker wastes ~2 GB and
        # walks into a config incompatibility ("Qwen2_5OmniTalkerConfig has no
        # attribute pad_token_id") that killed both Omni adapters on the first
        # smoke run.
        thinker_cls = getattr(transformers, "Qwen2_5OmniThinkerForConditionalGeneration", None)
        if thinker_cls is not None:
            try:
                self.model = self.place(thinker_cls.from_pretrained(
                    self.model_id, attn_implementation="sdpa", **self.load_kwargs()))
                self.loaded_via = "Thinker"
                return
            except Exception:  # noqa: BLE001 - fall back to the full model
                pass

        from transformers import Qwen2_5OmniForConditionalGeneration

        self.model = self.place(Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_id, attn_implementation="sdpa", **self.load_kwargs()))
        self.loaded_via = "full"
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": QWEN_OMNI_SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio},
                                         {"type": "text", "text": prompt}]},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = call_processor(self.processor, text, audio, sr,
                               padding=True, use_audio_in_video=False)
        return move_to_device(
            dict(inputs), primary_device(self.model), self.cast_dtype(), self.audio_float_keys
        )

    def forward_logits(self, inputs: dict):
        """The top-level Omni model wraps Thinker+Talker; logits live on Thinker."""
        import torch

        from .base import assert_finite

        thinker = getattr(self.model, "thinker", self.model)
        with torch.no_grad():
            out = thinker(**inputs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits[0, -1, :].float()
        assert_finite(row, f"{self.key} next-token logits")
        return row

    def generate_kwargs(self) -> dict:
        return {"return_audio": False, "use_audio_in_video": False}


@register
class Qwen25Omni3B(_Qwen25Omni):
    key = "qwen2_5_omni_3b"
    model_id = "Qwen/Qwen2.5-Omni-3B"
    max_audio_s = 1260.0    # 32k ctx at 25 tok/s, minus prompt headroom
    notes = "Talker disabled. ~21 min audio ceiling (AudioMarathon Table 8)."


@register
class Qwen25Omni7B(_Qwen25Omni):
    key = "qwen2_5_omni_7b"
    model_id = "Qwen/Qwen2.5-Omni-7B"
    max_audio_s = 1260.0
    notes = "Talker disabled; ~14GB fp16 after that, still sharded over 2xT4."
