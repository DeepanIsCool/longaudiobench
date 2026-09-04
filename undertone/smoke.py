"""Pre-flight checks for every adapter.

The point of this module is to spend ~15 minutes of quota discovering that an
adapter is broken, instead of six hours discovering it from a table of zeros.
Every check here corresponds to a failure mode that has actually happened in
this project or is documented on the model card:

  loads              -- gated repo, wrong auto-class, OOM on 2xT4
  finite             -- bf16-trained weights overflowing in fp16 on sm75
  discriminates      -- all four letter logits equal => wrong token ids or a
                        prompt the model never sees as a question
  parses             -- free generation that never emits a bare letter
  deterministic      -- sampling left on; reruns of a "fixed" config diverged
                        by 2x in the retired suite (v21 0.14 vs v23 0.28)
  truncates_loudly   -- over-long audio must set the flag, not vanish silently
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import numpy as np

from .adapters.base import ModelAdapter, apply_cap
from .items import MCQItem
from .ladder import window_for
from .protocol import render
from .scoring import argmax_letter, is_degenerate, parse_free_letter

SAMPLE_RATE = 16000


def demo_item(lang: str = "en") -> MCQItem:
    """A throwaway item whose only job is to exercise the prompt path."""
    questions = {
        "en": "Which dose did the speaker mention?",
        "hi": "वक्ता ने कौन सी खुराक बताई?",
        "bn": "বক্তা কোন ডোজটির কথা বলেছেন?",
    }
    options = {
        "en": ("five milligrams", "fifty milligrams", "fifteen milligrams"),
        "hi": ("पाँच मिलीग्राम", "पचास मिलीग्राम", "पंद्रह मिलीग्राम"),
        "bn": ("পাঁচ মিলিগ্রাম", "পঞ্চাশ মিলিগ্রাম", "পনেরো মিলিগ্রাম"),
    }
    correct, salience, recency = options[lang]
    return MCQItem(
        item_id=f"smoke_{lang}",
        recording_id="smoke",
        lang=lang,
        category="P3",
        sector="smoke",
        audio_path="<in-memory>",
        duration_band=300,
        needle_start=5.0,
        needle_end=9.0,
        question=questions[lang],
        options={"correct": correct, "salience": salience,
                 "recency": recency, "absent": "placeholder"},
    )


def demo_audio(seconds: float = 20.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Real speech if librosa's example cache is reachable, else band-limited noise.

    Speech is preferable -- some audio towers behave differently on pure noise --
    but the checks below are about plumbing, so a fallback keeps the smoke test
    runnable offline.
    """
    want = int(seconds * sr)
    try:
        import librosa

        audio, _ = librosa.load(librosa.ex("libri1"), sr=sr, mono=True)
    except Exception:  # noqa: BLE001 - no network, no pooch cache
        rng = np.random.default_rng(0)
        audio = rng.normal(0, 0.05, size=sr * 5).astype(np.float32)
    if len(audio) < want:
        audio = np.tile(audio, int(np.ceil(want / len(audio))))
    return np.asarray(audio[:want], dtype=np.float32)


def purge_cache(model_id: str, cache_root: str | None = None) -> float:
    """Delete one model's weights from the HF cache. Returns GB freed.

    Kaggle gives ~60 GB of disk and the roster is ~130 GB of checkpoints, so a
    sequential smoke test over all thirteen fills the disk partway through and
    fails on a download rather than on anything interesting. Purging after each
    model is what makes the full sweep runnable at all.
    """
    import shutil
    from pathlib import Path

    root = Path(cache_root or os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = root / "hub" if (root / "hub").is_dir() else root
    target = hub / ("models--" + model_id.replace("/", "--"))
    if not target.is_dir():
        return 0.0
    freed = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(target, ignore_errors=True)
    return round(freed, 2)


def disk_free_gb(path: str = "/kaggle/temp") -> float:
    import shutil
    from pathlib import Path

    target = path if Path(path).is_dir() else "/"
    return round(shutil.disk_usage(target).free / 1e9, 1)


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(
                sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def smoke_adapter(
    adapter: ModelAdapter,
    langs: tuple[str, ...] = ("en", "hi", "bn"),
    clip_seconds: float = 20.0,
    check_truncation: bool = True,
    keep_loaded: bool = False,
) -> dict[str, Any]:
    """Run every check against one adapter.  Never raises; returns a report.

    ``keep_loaded`` leaves the model in place for a sweep to reuse. Unloading and
    reloading a 17 GB checkpoint costs ~140 s and, worse, the first copy is not
    reliably released: a model notebook OOM'd on its sweep immediately after
    passing every check, because the check's model was still resident.
    """
    report: dict[str, Any] = {
        **adapter.describe(),
        "checks": {},
        "failures": [],
        "loaded_via": None,
    }

    def record(name: str, ok: bool, detail: Any = None) -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass

    started = time.time()
    try:
        adapter.load()
        report["loaded_via"] = getattr(adapter, "loaded_via", "declared class")
        record("loads", True, f"{time.time() - started:.0f}s")
    except Exception as exc:  # noqa: BLE001
        record("loads", False, f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc(limit=6)
        return report

    try:
        audio = demo_audio(clip_seconds)
        capped = apply_cap(audio, adapter.max_audio_s)
        first_scores: dict[str, float] | None = None

        for lang in langs:
            item = demo_item(lang)
            prompt = render(item, window_for(item, "L1"), seed=0).prompt
            scores = adapter.score_letters(capped.audio, prompt)

            finite = all(np.isfinite(v) for v in scores.values())
            record(f"finite[{lang}]", finite, scores)
            record(f"discriminates[{lang}]", not is_degenerate(scores),
                   f"spread={max(scores.values()) - min(scores.values()):.3f}")

            text = adapter.generate(capped.audio, prompt, max_new_tokens=8)
            letter = parse_free_letter(text, strip_reasoning=adapter.strip_reasoning)
            record(f"parses[{lang}]", letter is not None, repr(text[:120]))

            if lang == langs[0]:
                first_scores = scores

        # Does the model actually hear the audio? The single most dangerous
        # silent failure in this project: an adapter that builds a valid prompt
        # but never attaches the audio still loads, still discriminates between
        # letters, still parses -- and produces a whole benchmark measuring a
        # text prior. Two different clips through the same prompt must not give
        # identical logits.
        item = demo_item(langs[0])
        prompt = render(item, window_for(item, "L1"), seed=0).prompt
        rng = np.random.default_rng(1)
        other = rng.normal(0, 0.1, size=int(clip_seconds * SAMPLE_RATE)).astype(np.float32)
        scores_a = adapter.score_letters(capped.audio, prompt)
        scores_b = adapter.score_letters(apply_cap(other, adapter.max_audio_s).audio, prompt)
        delta = max(abs(scores_a[k] - scores_b[k]) for k in scores_a)
        record("hears_audio", delta > 1e-3,
               {"max_logit_delta": round(delta, 5),
                "note": "identical logits on different audio means the audio is "
                        "not reaching the model"})

        # Same input twice must give the same answer, or nothing downstream is
        # reproducible and every cross-model comparison is noise.
        item = demo_item(langs[0])
        prompt = render(item, window_for(item, "L1"), seed=0).prompt
        again = adapter.score_letters(capped.audio, prompt)
        record("deterministic",
               first_scores is not None and argmax_letter(first_scores) == argmax_letter(again),
               {"first": argmax_letter(first_scores) if first_scores else None,
                "again": argmax_letter(again)})

        if check_truncation:
            over = demo_audio(adapter.max_audio_s + 60.0)
            cut = apply_cap(over, adapter.max_audio_s)
            record("truncates_loudly",
                   cut.truncated and cut.seconds_seen <= adapter.max_audio_s + 1e-6,
                   {"offered": cut.seconds_offered, "seen": cut.seconds_seen})

        report["peak_vram_gb"] = _peak_vram_gb()
    except Exception as exc:  # noqa: BLE001
        record("inference", False, f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc(limit=6)
    finally:
        if not keep_loaded:
            adapter.unload()

    report["ok"] = not report["failures"]
    report["seconds"] = round(time.time() - started, 1)
    return report


def measure_ceiling(
    adapter: ModelAdapter,
    candidates: tuple[float, ...] = (30, 120, 300, 600, 900, 1200, 1800),
) -> dict[str, Any]:
    """Longest window that actually runs, as opposed to the documented figure.

    MOSS-Audio's ceiling is inferred from config.json (12.5 tokens/s against a
    40 960 context) and is not documented by its authors; this is how the
    truncation table gets a measured number instead of an inferred one.
    Expensive -- opt in, and only where the documented figure is a guess.
    """
    if adapter.model is None:
        adapter.load()
    item = demo_item("en")
    prompt = render(item, window_for(item, "L1"), seed=0).prompt

    results: list[dict[str, Any]] = []
    largest_ok = 0.0
    for seconds in candidates:
        if seconds > adapter.max_audio_s:
            results.append({"seconds": seconds, "ok": False, "why": "over declared cap"})
            continue
        started = time.time()
        try:
            adapter.score_letters(demo_audio(seconds), prompt)
            elapsed = round(time.time() - started, 1)
            results.append({"seconds": seconds, "ok": True, "elapsed_s": elapsed})
            largest_ok = seconds
        except Exception as exc:  # noqa: BLE001
            results.append({"seconds": seconds, "ok": False,
                            "why": f"{type(exc).__name__}: {exc}"})
            break
    return {"key": adapter.key, "declared_max_audio_s": adapter.max_audio_s,
            "measured_max_audio_s": largest_ok, "trials": results}
