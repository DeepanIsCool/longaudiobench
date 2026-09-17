"""Gemini through the API: the frontier point the open roster cannot supply.

Every finding in the paper is "for open models of 3-11 B". One closed model
either extends the claim or bounds it, and either is the stronger paper.
This adapter runs the identical protocol - same items, same rendered prompt,
same letter-to-role randomisation, same ladder windows - through the Gemini
API, and differs in exactly one documented way: there are no letter logits
over the wire, so the cell is scored by **generation with strict parsing**.
The row carries ``scorer: "freegen"`` and a one-hot ``logit_scores`` so the
tables can tell.

Cost control, in order of what matters:

*   **Uploads are cached per audio.** L3 and L4 hand every item of a
    recording the same 300 s window, so 338 items are ~110 uploads, not
    1352. Cache is keyed on the audio bytes and persisted to disk so a
    resumed run reuses it (Gemini keeps files 48 h).
*   **One call per cell.** ``score_letters`` makes the call and remembers
    the text; ``generate`` returns it without a second call.
*   **Backoff on 429/503**, a floor on the interval between calls
    (``GEMINI_RPM``), and every failure is a per-cell error row, never a
    dead run.

No GPU. This runs on a Kaggle CPU kernel or a laptop.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np

from ..env import Hardware
from .base import SAMPLE_RATE, ModelAdapter, as_temp_wav, register

LETTERS = "ABCD"


class GeminiAudio(ModelAdapter):
    """Base for the Gemini variants; subclasses set ``key`` and ``model_id``."""

    max_audio_s = 34200.0            # ~9.5 h per the API docs; nothing here approaches it
    documented_max_audio_s = 34200.0
    primary = "freegen"
    generation_budget = 8
    is_api = True
    notes = ("Closed model via the Gemini API. Scored by generation with strict "
             "single-letter parsing; the API exposes no letter logits.")

    def __init__(self) -> None:
        super().__init__()
        self.client = None
        self._files: dict[str, str] = {}     # sha1(audio) -> uploaded file name
        self._last: tuple[str, str] | None = None   # (cell key, text)
        self._t_last = 0.0
        self.calls = 0

    # -- an API has no hardware; say so in the signature, do not fake one ------

    @property
    def hardware(self):
        if self._hardware is None:
            self._hardware = Hardware(backend="api", dtype="none", device_map=None,
                                      detail=self.model_id, total_memory_gb=0.0,
                                      supports_bf16=False)
        return self._hardware

    def load(self) -> None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = self.model_id          # runner reads adapter.model for truthiness
        cache = os.environ.get("GEMINI_UPLOAD_CACHE")
        if cache and os.path.exists(cache):
            try:
                self._files = json.load(open(cache))
            except (OSError, ValueError):
                self._files = {}

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        return {"audio": audio, "prompt": prompt, "sr": sr}

    # -- uploads ----------------------------------------------------------------

    def _upload(self, audio: np.ndarray, sr: int) -> Any:
        digest = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
        name = self._files.get(digest)
        if name:
            try:
                f = self.client.files.get(name=name)
                if getattr(f, "state", None) in (None, "ACTIVE") or str(getattr(f, "state", "")).endswith("ACTIVE"):
                    return f
            except Exception:  # noqa: BLE001 - expired or gone; re-upload
                pass
        path = as_temp_wav(audio, sr)
        try:
            f = self._retry(lambda: self.client.files.upload(file=path))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        # Larger uploads are processed asynchronously; wait until usable.
        for _ in range(60):
            state = str(getattr(f, "state", "ACTIVE"))
            if state.endswith("ACTIVE"):
                break
            if state.endswith("FAILED"):
                raise RuntimeError(f"upload failed: {name}")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        self._files[digest] = f.name
        cache = os.environ.get("GEMINI_UPLOAD_CACHE")
        if cache:
            try:
                json.dump(self._files, open(cache, "w"))
            except OSError:
                pass
        return f

    # -- calls ------------------------------------------------------------------

    def _retry(self, fn, tries: int = 6):
        delay = 15.0
        for attempt in range(tries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classify by message, the SDK's types vary
                msg = str(exc)
                retryable = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503",
                                                   "UNAVAILABLE", "500", "DEADLINE"))
                if not retryable or attempt == tries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 240.0)

    def _call(self, audio: np.ndarray, prompt: str, sr: int) -> str:
        from google.genai import types

        rpm = float(os.environ.get("GEMINI_RPM", "10"))
        gap = 60.0 / rpm
        wait = self._t_last + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        f = self._upload(audio, sr)
        # Thinking off. The 3.x Flash models think by default and spent the
        # whole budget on thoughts - finish_reason MAX_TOKENS, text None, 5
        # thought tokens, 0 output. With thinking_budget=0 the same cell is
        # 'A' in one token. This also makes the rows comparable to the
        # instruct audio models, not the thinking variants. A model that
        # refuses to disable thinking gets a budget large enough to finish.
        cfgs = [
            types.GenerateContentConfig(temperature=0.0, max_output_tokens=16,
                                        thinking_config=types.ThinkingConfig(thinking_budget=0)),
            types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
        ]
        last_exc = None
        for cfg in cfgs:
            try:
                resp = self._retry(lambda: self.client.models.generate_content(
                    model=self.model_id, contents=[f, prompt], config=cfg))
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if "thinking" not in str(exc).lower():
                    raise
        else:
            raise last_exc
        self._t_last = time.time()
        self.calls += 1
        return (getattr(resp, "text", None) or "").strip()

    def _cell_key(self, audio: np.ndarray, prompt: str) -> str:
        h = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        h.update(prompt.encode())
        return h.hexdigest()

    def score_letters(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict[str, float]:
        """One API call; a one-hot over the parsed letter so the shared tables
        have a ``logit_scores`` shape. ``is_degenerate`` sees a flat 0/1 and the
        row's ``scorer`` field says freegen; nothing downstream mistakes this
        for a logit."""
        from ..scoring import parse_free_letter

        key = self._cell_key(audio, prompt)
        text = self._call(audio, prompt, sr)
        self._last = (key, text)
        letter = parse_free_letter(text)
        return {L: (1.0 if L == letter else 0.0) for L in LETTERS}

    def generate(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE,
                 max_new_tokens: int | None = None) -> str:
        key = self._cell_key(audio, prompt)
        if self._last and self._last[0] == key:
            return self._last[1]
        text = self._call(audio, prompt, sr)
        self._last = (key, text)
        return text


@register
class GeminiFlashLite(GeminiAudio):
    key = "gemini_3_1_flash_lite"
    model_id = "gemini-3.1-flash-lite"
    notes = GeminiAudio.notes + " Flash-Lite tier: the cheap bulk run."


@register
class GeminiFlash(GeminiAudio):
    key = "gemini_3_5_flash"
    model_id = "gemini-3.5-flash"
    notes = GeminiAudio.notes + " Flash tier."


@register
class GeminiPro(GeminiAudio):
    key = "gemini_3_1_pro"
    model_id = "gemini-3.1-pro-preview"
    notes = GeminiAudio.notes + " Pro tier: the frontier point."
