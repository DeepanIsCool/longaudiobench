"""OpenAI audio models through the API: the second closed family.

Same protocol as the Gemini adapter - identical items, prompt, letter
randomisation, windows - scored by generation with strict parsing because
there are no letter logits over the wire (``scorer: "freegen"``).

What is different from Gemini, and why:

*   **No file upload.** Chat Completions takes the audio inline as base64
    WAV in the message, so there is nothing to cache; a 300 s window is a
    9.6 MB WAV, well inside the request cap.
*   **A hard spend cap.** ``OPENAI_BUDGET_USD`` (default 4.80). Every reply
    carries ``usage.prompt_tokens_details.audio_tokens``; the adapter keeps
    a running dollar total from the published per-token prices and refuses
    to make a call once the total would pass the cap. The refusal is a
    normal per-cell error, so the runner stops after five and the rows
    already written are kept. The total is also persisted to
    ``OPENAI_SPEND_FILE`` so a resumed run continues the same tally.
*   **Backoff on 429/5xx**, one call per cell, ``generate`` reuses the text.

No GPU; runs on a laptop.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from typing import Any

import numpy as np

from ..env import Hardware
from .base import SAMPLE_RATE, ModelAdapter, register

LETTERS = "ABCD"

# USD per 1M tokens (developers.openai.com/api/docs/pricing, 2026-09).
PRICES = {
    "gpt-audio-mini": {"audio_in": 10.0, "text_in": 0.60, "text_out": 2.40},
    "gpt-audio": {"audio_in": 32.0, "text_in": 2.50, "text_out": 10.0},
}


class BudgetExceeded(RuntimeError):
    pass


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OpenAIAudio(ModelAdapter):
    """Base for the OpenAI audio variants; subclasses set ``key``/``model_id``."""

    max_audio_s = 1800.0            # a 600 s window is the longest we send
    documented_max_audio_s = 1800.0
    primary = "freegen"
    generation_budget = 8
    is_api = True
    notes = ("Closed model via the OpenAI Chat Completions API, audio inline as WAV. "
             "Scored by generation with strict single-letter parsing; no letter logits.")

    def __init__(self) -> None:
        super().__init__()
        self.client = None
        self._last: tuple[str, str] | None = None
        self._t_last = 0.0
        self.calls = 0
        self.spend_usd = 0.0
        self.audio_tokens = 0

    @property
    def hardware(self):
        if self._hardware is None:
            self._hardware = Hardware(backend="api", dtype="none", device_map=None,
                                      detail=self.model_id, total_memory_gb=0.0,
                                      supports_bf16=False)
        return self._hardware

    # -- lifecycle ----------------------------------------------------------------

    def load(self) -> None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key, timeout=120.0, max_retries=0)
        self.model = self.model_id
        self.budget = float(os.environ.get("OPENAI_BUDGET_USD", "4.80"))
        self.spend_file = os.environ.get("OPENAI_SPEND_FILE")
        if self.spend_file and os.path.exists(self.spend_file):
            try:
                d = json.load(open(self.spend_file))
                self.spend_usd = float(d.get("spend_usd", 0.0))
                self.audio_tokens = int(d.get("audio_tokens", 0))
            except (OSError, ValueError):
                pass

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        return {"audio": audio, "prompt": prompt, "sr": sr}

    # -- money ----------------------------------------------------------------------

    def _cost(self, usage: Any) -> float:
        p = PRICES[self.model_id]
        details = getattr(usage, "prompt_tokens_details", None)
        a = int(getattr(details, "audio_tokens", 0) or 0)
        t_in = int(getattr(usage, "prompt_tokens", 0) or 0) - a
        t_out = int(getattr(usage, "completion_tokens", 0) or 0)
        self.audio_tokens += a
        return (a * p["audio_in"] + t_in * p["text_in"] + t_out * p["text_out"]) / 1e6

    def _persist(self) -> None:
        if self.spend_file:
            try:
                json.dump({"spend_usd": self.spend_usd, "audio_tokens": self.audio_tokens,
                           "calls": self.calls, "model": self.model_id},
                          open(self.spend_file, "w"))
            except OSError:
                pass

    _sess_tokens = 0
    _sess_seconds = 0.0

    def _estimate(self, seconds: float) -> float:
        # ~10 audio tokens per second until this process has measured its own
        # rate. Only this process's tokens and seconds: the persisted total
        # spans earlier runs whose call count is not persisted, and dividing
        # that by a fresh call count once priced a 20 s clip at $13 and
        # refused every cell.
        rate = self._sess_tokens / self._sess_seconds if self._sess_seconds else 10.0
        return seconds * rate * PRICES[self.model_id]["audio_in"] / 1e6

    # -- calls ------------------------------------------------------------------------

    def _retry(self, fn, tries: int = 6):
        delay = 10.0
        for attempt in range(tries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "insufficient_quota" in msg or "billing" in msg.lower():
                    raise BudgetExceeded(f"account quota: {msg[:200]}") from exc
                retryable = any(s in msg for s in ("429", "500", "502", "503", "504",
                                                   "rate_limit", "overloaded", "timed out",
                                                   "Connection"))
                if not retryable or attempt == tries - 1:
                    raise
                print(f"[{self.key}] {msg[:90]!r} - retry in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

    def _call(self, audio: np.ndarray, prompt: str, sr: int) -> str:
        seconds = len(audio) / sr
        projected = self.spend_usd + self._estimate(seconds)
        if projected > self.budget:
            raise BudgetExceeded(f"spend ${self.spend_usd:.3f} + est ${self._estimate(seconds):.3f} "
                                 f"would pass the cap ${self.budget:.2f}")
        rpm = float(os.environ.get("OPENAI_RPM", "30"))
        wait = self._t_last + 60.0 / rpm - time.time()
        if wait > 0:
            time.sleep(wait)
        b64 = _wav_b64(audio, sr)
        resp = self._retry(lambda: self.client.chat.completions.create(
            model=self.model_id,
            modalities=["text"],
            temperature=0.0,
            max_completion_tokens=16,
            messages=[{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                {"type": "text", "text": prompt},
            ]}],
        ))
        self._t_last = time.time()
        self.calls += 1
        before = self.audio_tokens
        self.spend_usd += self._cost(resp.usage)
        self._sess_tokens += self.audio_tokens - before
        self._sess_seconds += seconds
        self._persist()
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return text

    def _cell_key(self, audio: np.ndarray, prompt: str) -> str:
        h = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        h.update(prompt.encode())
        return h.hexdigest()

    def score_letters(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict[str, float]:
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
class GPTAudioMini(OpenAIAudio):
    key = "gpt_audio_mini"
    model_id = "gpt-audio-mini"
    notes = OpenAIAudio.notes + " Mini tier."


@register
class GPTAudio(OpenAIAudio):
    key = "gpt_audio"
    model_id = "gpt-audio"
    notes = OpenAIAudio.notes + " Flagship tier."
