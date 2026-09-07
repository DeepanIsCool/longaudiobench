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
# 0 is the untouched control. The first run stopped at -12 dB and moved only 3
# of 70 answers, with the gain edit verifiably landing (achieved contrast rose
# 5.2 -> 16.7 dB across the steps), so the range was too narrow to reach a
# threshold rather than the effect being absent. -18 and -24 extend it.
#
# -60 dB is not a level on the curve: it is the needle-removed control, ~0.001
# amplitude and inaudible. If accuracy there matches accuracy at 0 dB, the model
# was never reading the needle and every other number for that item measures a
# prior rather than a retrieval. Nothing else in the benchmark tests that
# per item.
NEEDLE_REMOVED_DB = -60.0
DEFAULT_LEVELS = (0.0, -3.0, -6.0, -9.0, -12.0, -18.0, -24.0, NEEDLE_REMOVED_DB)

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
    # The removal control is excluded: at -60 dB the needle is gone, so a
    # salience answer there says nothing about a threshold. The first run
    # reported "flip threshold: 0.0 dB" for a model that never flipped, because
    # an item already answering salience at the untouched control counted as a
    # flip at 0.
    ordered = sorted((r for r in rows if r.get("role_chosen")
                      and r["level_db"] != NEEDLE_REMOVED_DB),
                     key=lambda r: -r["level_db"])
    if ordered and ordered[0]["role_chosen"] == "salience":
        return float("nan")   # already lost before any attenuation
    for row in ordered:
        if row["role_chosen"] == "salience":
            return float(row["level_db"])
    return float("nan")


def needle_necessity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Does removing the answer from the audio change what the model says?

    Accuracy at -60 dB is the per-item question-only floor. If it matches
    accuracy at 0 dB, the model answered without reading the needle and that
    item measures a prior, not a retrieval.
    """
    intact = [r for r in rows if r["level_db"] == 0.0 and r.get("role_chosen")]
    removed = [r for r in rows if r["level_db"] == NEEDLE_REMOVED_DB
               and r.get("role_chosen")]
    if not intact or not removed:
        return {"n_intact": len(intact), "n_removed": len(removed),
                "acc_intact": float("nan"), "acc_removed": float("nan"),
                "audio_dependence": float("nan")}
    a = sum(r["role_chosen"] == r["correct_role"] for r in intact) / len(intact)
    b = sum(r["role_chosen"] == r["correct_role"] for r in removed) / len(removed)
    return {"n_intact": len(intact), "n_removed": len(removed),
            "acc_intact": a, "acc_removed": b, "audio_dependence": a - b}


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


# The competitor's location is recorded as a single timestamp, so the span is
# reconstructed around it. Mentions in these recordings run about a second;
# the pad only has to be wide enough that the gain edit and the contrast
# measurement land on the mention rather than beside it.
COMPETITOR_SPAN_S = 1.2


def competitor_spans(item: MCQItem) -> list[tuple[float, float]]:
    """Where the loud competitor is, from the item's own provenance.

    Returns an empty list when the item never recorded one - those items are
    skipped rather than swept, because a salience trap cannot operate when the
    salient thing is not in the window.
    """
    at = (item.provenance or {}).get("salience_at")
    if at is None:
        return []
    half = COMPETITOR_SPAN_S / 2
    return [(max(0.0, float(at) - half), float(at) + half)]


def run_sweep(adapter, pack, out_path, levels: tuple[float, ...] = DEFAULT_LEVELS,
              seed: int = 0, audio_root: str = ".", progress: bool = True):
    """Score every sweepable item at every attenuation level.

    Mirrors runner.run_model: appends one row per (item, level), resumes from
    what is already on disk, and stamps the pack fingerprint and code sha so a
    curve cannot silently mix item packs or code versions.
    """
    import json
    import os
    from pathlib import Path

    from .adapters.base import load_audio

    items = [it for it in pack if competitor_spans(it)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = pack.fingerprint

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pack_fingerprint") == fingerprint and not row.get("error"):
                done.add((row["item_id"], row.get("level_db")))

    skipped = len(list(pack)) - len(items)
    if progress:
        print(f"[{adapter.key}] sweep: {len(items)} items x {len(levels)} levels, "
              f"{skipped} items have no recorded competitor and are skipped")

    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(items, 1):
            todo = [lv for lv in levels if (item.item_id, lv) not in done]
            if not todo:
                continue
            audio = load_audio(os.path.join(audio_root, item.audio_path))
            window = contrast_window(item, *competitor_spans(item)[0])
            clip = audio[int(window.start * SAMPLE_RATE):int(window.end * SAMPLE_RATE)]
            for row in sweep_item(adapter, item, clip, window,
                                  competitor_spans(item), tuple(todo), seed):
                row["pack_fingerprint"] = fingerprint
                row["code_sha"] = os.environ.get("UNDERTONE_CODE_SHA")
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            if progress and n % 10 == 0:
                print(f"  {n}/{len(items)} items, {written} rows")
    if progress:
        print(f"[{adapter.key}] sweep wrote {written} rows to {out_path}")
    return out_path
