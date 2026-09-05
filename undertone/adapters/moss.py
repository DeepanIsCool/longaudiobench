"""MOSS-Audio 4B / 8B, Instruct and Thinking (four checkpoints).

Two concrete defects to work around on Kaggle, both confirmed from the repo
files rather than the card:

1.  ``processor_config.json`` sets ``"mel_dtype": "bfloat16"``.  T4 is sm75 and
    has no bf16 compute, so the mel batch is overridden to fp16 after load and
    before the first ``__call__``.  Without this the processor allocates a bf16
    tensor that the encoder then has to touch.

2.  ``MossAudioProcessor.__call__`` neither chunks nor truncates -- it runs the
    whole mel through in one go (see ``processing_moss_audio.py``: no length
    guard anywhere).  The ceiling is therefore the LLM context,
    ``max_position_embeddings = 40960``, consumed at 12.5 audio tokens/s plus
    time markers every 2 s.  ``estimate_tokens`` reproduces that arithmetic so
    an over-long window fails loudly instead of OOM-ing halfway through a sweep.

The Thinking variants are trained to reason before answering, so their first
generated token is a thought, not an answer.  For those two the primary scorer
is free generation with the reasoning block stripped; letter logits are still
recorded, as a diagnostic rather than the headline number.
"""

from __future__ import annotations

import numpy as np

from .base import (
    SAMPLE_RATE,
    ModelAdapter,
    move_to_device,
    primary_device,
    register,
)

AUDIO_TOKENS_PER_SECOND = 12.5      # processing_moss_audio.py
TIME_MARKER_EVERY_SECONDS = 2
CONTEXT_LIMIT = 40960               # config.json language_config
PROMPT_HEADROOM = 1024              # question + four options + answer


def estimate_tokens(seconds: float) -> int:
    """Audio tokens plus time-marker digit tokens for a window of ``seconds``."""
    audio_tokens = int(seconds * AUDIO_TOKENS_PER_SECOND)
    markers = range(TIME_MARKER_EVERY_SECONDS, int(seconds) + 1, TIME_MARKER_EVERY_SECONDS)
    marker_tokens = sum(len(str(second)) for second in markers)
    return audio_tokens + marker_tokens


def max_seconds() -> float:
    """Largest window that fits the context, by bisection on estimate_tokens."""
    budget = CONTEXT_LIMIT - PROMPT_HEADROOM
    lo, hi = 0.0, 10_000.0
    while hi - lo > 1.0:
        mid = (lo + hi) / 2.0
        if estimate_tokens(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


class _MossAudio(ModelAdapter):
    # Config-derived, not documented by the authors; the smoke test measures the
    # real ceiling empirically and the results table reports the measured value.
    max_audio_s = 1800.0
    audio_float_keys = ("audio_data",)

    def load(self) -> None:
        """MOSS is not a plain transformers load.

        The Hub repo ships config, processor and weights but **no modeling
        file**, and `moss_audio` is registered in no transformers release --
        4.57.1 and 5.0.0 both reject MossAudioConfig. The model class lives in
        OpenMOSS's own package:

            pip install git+https://github.com/OpenMOSS/MOSS-Audio.git

        installed *without* its ``torch-runtime`` extra, which would pin
        torch==2.9.1+cu128 over Kaggle's existing build. Their pyproject already
        pins transformers==4.57.1, which is the roster's shared pin.
        """
        from ..env import torch_dtype

        MossAudioModel, MossAudioProcessor = self._import_moss()

        self.processor = MossAudioProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, enable_time_marker=True)
        # processor_config.json declares bfloat16; sm75 has no bf16 compute.
        if hasattr(self.processor, "config"):
            self.processor.config.mel_dtype = torch_dtype(self.hardware)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or \
            getattr(self.processor, "_base_tokenizer", None)

        self.model = self.place(MossAudioModel.from_pretrained(
            self.model_id, trust_remote_code=True, **self.load_kwargs()))
        self.model.eval()
        self.loaded_via = "OpenMOSS/MOSS-Audio"

    @staticmethod
    def _import_moss():
        """Import from the installed package, else from a shallow clone."""
        try:
            from src.modeling_moss_audio import MossAudioModel
            from src.processing_moss_audio import MossAudioProcessor

            return MossAudioModel, MossAudioProcessor
        except ImportError:
            pass

        import subprocess
        import sys
        from pathlib import Path

        # The package installs its modules under the top-level name `src`, so a
        # clone on sys.path is an equivalent fallback when pip is unavailable.
        target = Path("/kaggle/temp/MOSS-Audio")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/OpenMOSS/MOSS-Audio.git", str(target)],
                           check=True)
        sys.path.insert(0, str(target))
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor

        return MossAudioModel, MossAudioProcessor

    def build_inputs(self, audio, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        import torch

        seconds = len(audio) / sr
        needed = estimate_tokens(seconds)
        if needed > CONTEXT_LIMIT - PROMPT_HEADROOM:
            raise ValueError(
                f"{self.key}: {seconds:.0f}s of audio needs ~{needed} tokens, over the "
                f"{CONTEXT_LIMIT} context. Cap is ~{max_seconds():.0f}s.")

        inputs = self.processor(text=prompt, audios=[audio], return_tensors="pt")
        inputs = dict(inputs)
        inputs = move_to_device(inputs, primary_device(self.model), None, ())
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        # Required by MossAudioModel.forward and not produced by the processor.
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id
        return inputs

    def estimated_audio_tokens(self, seconds: float) -> int:
        return estimate_tokens(seconds)


@register
class MossAudio4BInstruct(_MossAudio):
    key = "moss_audio_4b_instruct"
    model_id = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    primary = "logits"
    prefers_single_device = True     # 10.4 GB fits one T4; sharding breaks it
    notes = ("Needs OpenMOSS's own package - the Hub repo has no modeling file "
             "and no transformers release registers moss_audio. "
             "mel_dtype forced to fp16 (sm75). ~5.2B, 10.4GB, one T4.")


@register
class MossAudio4BThinking(_MossAudio):
    key = "moss_audio_4b_thinking"
    model_id = "OpenMOSS-Team/MOSS-Audio-4B-Thinking"
    primary = "freegen"
    prefers_single_device = True
    strip_reasoning = True
    # Room to finish reasoning and still emit the letter. At 8 tokens the model
    # was cut off mid-<think> and every cell came back unparseable.
    generation_budget = 1024
    notes = "Thinking variant: first token is a thought, so free-gen is primary."


@register
class MossAudio8BInstruct(_MossAudio):
    key = "moss_audio_8b_instruct"
    model_id = "OpenMOSS-Team/MOSS-Audio-8B-Instruct"
    primary = "logits"
    notes = "9.05B, ~18GB fp16 -> sharded over 2xT4. mel_dtype forced to fp16."


@register
class MossAudio8BThinking(_MossAudio):
    key = "moss_audio_8b_thinking"
    model_id = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
    primary = "freegen"
    strip_reasoning = True
    generation_budget = 1024
    notes = "Thinking variant on 2xT4; free-gen primary, logits diagnostic."
