"""The one per-model surface.

Everything above this file -- prompts, option shuffling, ladder windows,
scoring, checkpointing, output schema -- is shared, which is what lets thirteen
notebooks each be written against its own model's documentation and still
produce numbers that compare.

Three constraints every adapter inherits from Kaggle's 2xT4 (sm75):

1.  **No bf16 compute.**  Weights trained in bf16 (MOSS-Audio, Audio-Flamingo
    Next, Qwen2.5-Omni) load in fp16 here.  ``assert_finite`` exists because
    that conversion can overflow, and a NaN logit row silently argmaxes to "A".
2.  **No flash-attention-2.**  ``sdpa`` or ``eager`` only, whatever the card
    recommends.
3.  **Two devices.**  ``device_map="auto"`` shards the 7-9B models, so *every*
    input tensor has to be moved, not just ``input_ids``.  That bug is already
    documented in the retired ``QwenAudioBaseline``; it is fixed once, here.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_audio(path: str, start: float = 0.0, end: float | None = None,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at ``sr``, optionally a ``[start, end)`` slice in seconds."""
    import librosa

    duration = None if end is None else max(0.0, end - start)
    audio, _ = librosa.load(path, sr=sr, mono=True, offset=start, duration=duration)
    return np.asarray(audio, dtype=np.float32)


@dataclass(frozen=True)
class Truncation:
    audio: np.ndarray
    truncated: bool
    seconds_seen: float
    seconds_offered: float


def apply_cap(audio: np.ndarray, max_audio_s: float, sr: int = SAMPLE_RATE) -> Truncation:
    """Cut audio to the model's documented cap, and say so.

    Deliberately explicit rather than letting a feature extractor crop in
    silence: Qwen2-Audio's extractor drops everything past 30 s with no signal,
    which is how the retired ANiH runs spent GPU hours measuring a truncation
    artifact and reporting it as retrieval accuracy.
    """
    offered = len(audio) / sr
    cap = int(round(max_audio_s * sr))
    if len(audio) <= cap:
        return Truncation(audio, False, offered, offered)
    return Truncation(audio[:cap], True, cap / sr, offered)


def primary_device(model) -> Any:
    """Where inputs must live for an accelerate-sharded model."""
    import torch

    try:
        embed = model.get_input_embeddings()
        if embed is not None and hasattr(embed, "weight"):
            return embed.weight.device
    except (AttributeError, NotImplementedError):
        pass
    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(inputs: dict, device, dtype=None, float_keys: tuple[str, ...] = ()) -> dict:
    """Move every tensor; cast only the named float tensors.

    Casting indiscriminately would turn ``input_ids`` into floats.  ``float_keys``
    names the audio-feature tensors that must match the model dtype -- fp16 on
    a T4, whatever the model card's example says.
    """
    import torch

    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if dtype is not None and (key in float_keys) and value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out


def assert_finite(tensor, what: str = "logits") -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"non-finite values in {what}: bf16-trained weights overflowing in fp16 "
            f"is the usual cause on sm75. Try float32 for the audio tower, or the "
            f"smaller variant."
        )


def free_memory(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# adapter protocol
# --------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Subclasses supply ``load`` and ``build_inputs``; the rest is shared."""

    key: str = ""                    # short name, used in filenames
    model_id: str = ""               # HF repo id
    max_audio_s: float = 30.0        # documented cap; see the roster table
    primary: str = "logits"          # "logits" or "freegen"
    strip_reasoning: bool = False    # True for the Thinking variants
    audio_float_keys: tuple[str, ...] = ("input_features", "audio_data", "input_audio_embeds")
    notes: str = ""                  # documented caveat, carried into results
    # What the card claims vs what this hardware can actually run. Voxtral
    # documents 40 minutes; on a T4 the L3 cells OOM trying to allocate 60 GiB,
    # because naive O(n^2) attention over ~45k audio tokens is not what the
    # figure assumes. Table 4 reports both, or it reports a ceiling nobody can
    # reach.
    documented_max_audio_s: float | None = None

    # Set when a model cannot run on this hardware at all. A model that is
    # absent because the GPU is too small is a different fact from one that ran
    # and scored badly, and Table 4 has to say which.
    hardware_blocked: str | None = None
    # Some models must not be sharded. MOSS's masked_scatter_ needs source and
    # target co-resident, and accelerate splitting a 10.4 GB model across two
    # 15.6 GB T4s produced "source is on cuda:1, different from other tensors on
    # cuda:0". If it fits on one device, put it on one device.
    prefers_single_device: bool = False

    # For a model that does NOT fit one GPU but still breaks when split across
    # two: keep every GPU tensor on cuda:0 and let the overflow go to CPU.
    # Slower than a two-GPU split, but it runs, which a device mismatch does not.
    single_gpu_with_cpu_overflow: bool = False

    # How many tokens generation gets. Eight is plenty for a model that answers
    # with a letter, and nowhere near enough for one that reasons first: the
    # Thinking variants emitted "<think>\nFor this question, I need" and were cut
    # off before reaching an answer, so strip_cot removed the whole fragment and
    # every cell came back unparseable.
    generation_budget: int = 8
    is_control: bool = False         # a baseline, not one of the thirteen models

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._letter_ids: dict[str, list[int]] | None = None
        self._hardware = None

    # -- hardware, resolved once and recorded ------------------------------

    @property
    def hardware(self):
        from ..env import resolve_hardware

        if self._hardware is None:
            self._hardware = resolve_hardware()
        return self._hardware

    def load_kwargs(self, **extra) -> dict:
        """``from_pretrained`` kwargs for whatever this machine is.

        fp16 on CUDA and MPS, fp32 on CPU (torch leaves fp16 CPU kernels
        unimplemented and they fail mid-forward rather than at load).
        ``device_map="auto"`` only under accelerate on CUDA; elsewhere the model
        is moved by ``place()`` after loading.
        """
        from ..env import torch_dtype

        kwargs = {"torch_dtype": torch_dtype(self.hardware), **extra}
        if self.hardware.device_map:
            if self.prefers_single_device and self.hardware.backend == "cuda":
                kwargs["device_map"] = "cuda:0"
            elif self.single_gpu_with_cpu_overflow and self.hardware.backend == "cuda":
                kwargs["device_map"] = "auto"
                # 11 GiB, not 14. At 14 the weights fill a 14.56 GiB card and
                # MOSS-8B OOM'd mid-sweep with 88 MiB free: it passed smoke on a
                # 20 s clip and died once the audio got longer. Activations for
                # a 5-minute window need room the weights were holding.
                kwargs["max_memory"] = {0: "11GiB", "cpu": "32GiB"}
            else:
                kwargs["device_map"] = self.hardware.device_map
        return kwargs

    def place(self, model):
        """Move a model that accelerate did not shard. No-op on CUDA."""
        import torch

        if self.hardware.device_map is None:
            model = model.to(torch.device(self.hardware.backend))
        return model.eval()

    def cast_dtype(self):
        from ..env import torch_dtype

        return torch_dtype(self.hardware)

    # -- required per model -------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Populate ``self.model`` / ``self.processor`` / ``self.tokenizer``."""

    @abstractmethod
    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        """Model-ready inputs, already on the right device and dtype."""

    # -- shared -------------------------------------------------------------

    @property
    def letter_ids(self) -> dict[str, list[int]]:
        from ..scoring import letter_token_ids

        if self._letter_ids is None:
            tok = self.tokenizer or getattr(self.processor, "tokenizer", None)
            if tok is None:
                raise RuntimeError(f"{self.key}: no tokenizer available for letter ids")
            self._letter_ids = letter_token_ids(tok)
        return self._letter_ids

    def forward_logits(self, inputs: dict):
        """Next-token logit row.  Override if the model's forward is unusual."""
        import torch

        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits[0, -1, :].float()
        assert_finite(row, f"{self.key} next-token logits")
        return row

    def score_letters(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict[str, float]:
        from ..scoring import letter_logits

        inputs = self.build_inputs(audio, prompt, sr)
        return letter_logits(self.forward_logits(inputs), self.letter_ids)

    def generate(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE,
                 max_new_tokens: int | None = None) -> str:
        import torch

        inputs = self.build_inputs(audio, prompt, sr)
        n_prompt = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.generation_budget,
                do_sample=False,          # deterministic: reruns must match exactly
                temperature=None,
                top_p=None,
                **self.generate_kwargs(),
            )
        tok = self.tokenizer or self.processor
        return tok.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0].strip()

    def generate_kwargs(self) -> dict:
        return {}

    def unload(self) -> None:
        model, processor, tokenizer = self.model, self.processor, self.tokenizer
        self.model = self.processor = self.tokenizer = None
        self._letter_ids = None
        free_memory(model, processor, tokenizer)

    # -- introspection used by the smoke test -------------------------------

    def describe(self) -> dict[str, Any]:
        from ..env import versions

        return {
            "key": self.key,
            "model_id": self.model_id,
            "max_audio_s": self.max_audio_s,
            "documented_max_audio_s": self.documented_max_audio_s or self.max_audio_s,
            "primary": self.primary,
            "strip_reasoning": self.strip_reasoning,
            "notes": self.notes,
            # Recorded so a results table can prove it is homogeneous. Two
            # backends in one table compares machines, not models.
            "backend": self.hardware.backend,
            "dtype": self.hardware.dtype,
            "hardware": self.hardware.detail,
            "signature": self.hardware.signature,
            "hardware_blocked": self.hardware_blocked,
            "versions": versions(),
        }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    if not cls.key:
        raise ValueError(f"{cls.__name__} must set a non-empty `key`")
    if cls.key in _REGISTRY and _REGISTRY[cls.key] is not cls:
        raise ValueError(f"duplicate adapter key {cls.key!r}")
    _REGISTRY[cls.key] = cls
    return cls


def get_adapter(key: str) -> ModelAdapter:
    if key not in _REGISTRY:
        raise KeyError(f"unknown adapter {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def list_adapters(include_controls: bool = False) -> list[str]:
    """Registered model keys.

    Controls (the cascaded ASR+LLM baseline) are excluded by default: they are
    a validity check, not a model under test, and folding them into "the roster"
    would misreport how many audio models were evaluated.
    """
    return sorted(k for k, cls in _REGISTRY.items()
                  if include_controls or not cls.is_control)


def list_controls() -> list[str]:
    return sorted(k for k, cls in _REGISTRY.items() if cls.is_control)


# --------------------------------------------------------------------------
# processor call shims
# --------------------------------------------------------------------------

def call_processor(processor, text: str, audio: np.ndarray, sr: int = SAMPLE_RATE,
                   audio_kwarg: str | None = None, **kw):
    """Call a processor whose audio kwarg name varies, and verify it landed.

    ``audios=`` was renamed to ``audio=`` mid-4.5x, so both names are in play
    across the roster. Trying one then the other is not enough: a processor with
    ``**kwargs`` *accepts* the wrong name and silently drops the audio. Aero did
    exactly that -- it loaded, produced a valid prompt, and returned identical
    logits for two different clips, which would have yielded a full table of
    numbers measuring nothing but a text prior.

    So the result is checked for an audio feature tensor, and a call that
    produced none is treated as a failure rather than a success.
    """
    names = [audio_kwarg] if audio_kwarg else ["audio", "audios"]
    last: Exception | None = None
    for key in names:
        try:
            out = processor(text=text, **{key: [audio]}, sampling_rate=sr,
                            return_tensors="pt", **kw)
        except TypeError as exc:      # wrong kwarg name for this version
            last = exc
            continue
        if _has_audio_features(out):
            return out
        last = ValueError(
            f"processor accepted {key}= but produced no audio features")
    raise TypeError(
        f"processor never ingested the audio (tried {names}): {last}")


AUDIO_FEATURE_KEYS = (
    "input_features", "audio_data", "input_audio_embeds", "audio_values",
    "input_audio_features", "audio_input_features", "audio_embeds",
)


def _has_audio_features(batch) -> bool:
    """Did the processor actually encode audio, or just the text?"""
    try:
        keys = set(batch.keys())
    except AttributeError:
        return False
    return any(k in keys for k in AUDIO_FEATURE_KEYS)


def as_temp_wav(audio: np.ndarray, sr: int = SAMPLE_RATE, tmpdir: str | None = None) -> str:
    """Write audio to a temp wav and return the path.

    Some processors only take a path or URL for audio (Voxtral's chat template,
    Audio-Flamingo Next).  Rather than reverse-engineer their array plumbing,
    hand them a file -- the cost is one disk write per window, which is noise
    next to a multi-minute prefill.
    """
    import tempfile

    import soundfile as sf

    fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmpdir)
    fd.close()
    sf.write(fd.name, audio, sr, subtype="PCM_16")
    return fd.name
