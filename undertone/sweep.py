"""The prominence gain sweep.

Everything else in this project is observational: find items that happen to be
quiet or masked, report what models do. Any duration benchmark could add a
prominence column tomorrow and match it.

This is the one experiment that manipulates prominence directly. Same speaker,
same words, same room, same recording -- only the loudness ratio between the
answer and its loud competitor changes, in steps. Measuring where a model flips
from the correct answer to the loud one turns "models prefer loud things" from
a rate into **a threshold in dB**, per model.

Two design points that matter:

*   **The window must contain both mentions.** An L1 window is centred on the
    needle and usually excludes the competitor entirely - and a salience trap
    cannot operate if the salient thing was never heard. The sweep uses a
    contrast window spanning the target and its nearest competing mention,
    which is the minimal context in which the effect can exist at all.

*   **Level 0 is the control.** The unedited window is swept alongside the
    edited ones, so the curve starts from what the recording actually did rather
    than from an assumption about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .harvest.construct import SAMPLE_RATE, apply_gain_edits, GainEdit, measure_contrast
from .items import MCQItem
from .protocol import render
from .ladder import Window

# Attenuation applied to the answer, in dB. 0 is the untouched control; the
# steps are coarse because a psychometric curve needs range more than
# resolution at this sample size.
DEFAULT_LEVELS = (0.0, -3.0, -6.0, -9.0, -12.0)

CONTRAST_PAD = 3.0      # seconds of context either side of the two mentions
MAX_CONTRAST_WINDOW = 90.0


@dataclass(frozen=True)
class SweepCell:
    item_id: str
    level_db: float
    achieved_contrast_db: float
    window_start: float
    window_end: float


def contrast_window(item: MCQItem, competitor_start: float,
                    competitor_end: float) -> Window:
    """The smallest window holding both the answer and its loud competitor.

    Without the competitor in earshot there is nothing for the model to be
    lured by, and the sweep would measure plain audibility instead.
    """
    lo = max(0.0, min(item.needle_start, competitor_start) - CONTRAST_PAD)
    hi = min(float(item.duration_band),
             max(item.needle_end, competitor_end) + CONTRAST_PAD)
    if hi - lo > MAX_CONTRAST_WINDOW:
        # Too far apart to hold in one window: centre on the needle and accept
        # that this item cannot carry the sweep.
        mid = item.needle_mid
        lo = max(0.0, mid - MAX_CONTRAST_WINDOW / 2)
        hi = min(float(item.duration_band), lo + MAX_CONTRAST_WINDOW)
    return Window(lo, hi, oracle=False)


def sweep_item(adapter, item: MCQItem, audio: np.ndarray, window: Window,
               competitor_spans: list[tuple[float, float]],
               levels: tuple[float, ...] = DEFAULT_LEVELS,
               seed: int = 0, sr: int = SAMPLE_RATE) -> list[dict[str, Any]]:
    """Score one item at each attenuation level. Returns one row per level."""
    rendered = render(item, window, seed)
    target = (item.needle_start - window.start, item.needle_end - window.start)
    local_competitors = [(s - window.start, e - window.start)
                         for s, e in competitor_spans
                         if window.start <= s and e <= window.end]

    rows: list[dict[str, Any]] = []
    for level in levels:
        edited = audio if level == 0.0 else apply_gain_edits(
            audio, [GainEdit(target[0], target[1], level)], sr)
        contrast = (measure_contrast(edited, target, local_competitors[0], sr)
                    if local_competitors else float("nan"))

        row: dict[str, Any] = {
            "item_id": item.item_id,
            "recording_id": item.recording_id,
            "category": item.category,
            "lang": item.lang,
            "model_key": adapter.key,
            "signature": adapter.hardware.signature,
            "condition": "SWEEP",
            "level_db": level,
            "achieved_contrast_db": contrast,
            "competitors_in_window": len(local_competitors),
            "is_null": item.is_null,
            "correct_role": item.correct_role,
            "letter_to_role": rendered.letter_to_role,
            "error": None,
        }
        try:
            from .scoring import argmax_letter, is_degenerate, letter_logits

            scores = adapter.score_letters(edited, rendered.prompt, sr)
            letter = argmax_letter(scores)
            row.update(letter_chosen=letter,
                       role_chosen=rendered.letter_to_role[letter],
                       logit_degenerate=is_degenerate(scores))
            row["correct"] = row["role_chosen"] == item.correct_role
        except Exception as exc:  # noqa: BLE001 - one level must not kill a curve
            row.update(error=f"{type(exc).__name__}: {exc}",
                       role_chosen=None, letter_chosen=None)
        rows.append(row)
    return rows


def flip_threshold(rows: list[dict[str, Any]]) -> float:
    """Attenuation at which this item's answer is lost to the competitor.

    The headline number: how much quieter the right answer has to be before a
    model takes the loud wrong one. NaN when the model never flips within the
    swept range, which is itself informative and must not read as zero.
    """
    ordered = sorted((r for r in rows if r.get("role_chosen")),
                     key=lambda r: -r["level_db"])
    for row in ordered:
        if row["role_chosen"] == "salience":
            return float(row["level_db"])
    return float("nan")


def curve(rows: list[dict[str, Any]], levels: tuple[float, ...] = DEFAULT_LEVELS
          ) -> list[dict[str, Any]]:
    """Trap rate and accuracy at each level, pooled over items."""
    out = []
    for level in levels:
        at = [r for r in rows if r["level_db"] == level and r.get("role_chosen")]
        non_null = [r for r in at if not r["is_null"]]
        out.append({
            "level_db": level,
            "n": len(at),
            "accuracy": (sum(r["role_chosen"] == r["correct_role"] for r in at) / len(at)
                         if at else float("nan")),
            "salience_rate": (sum(r["role_chosen"] == "salience" for r in non_null)
                              / len(non_null) if non_null else float("nan")),
            "mean_contrast_db": (float(np.nanmean([r["achieved_contrast_db"] for r in at]))
                                 if at else float("nan")),
        })
    return out
