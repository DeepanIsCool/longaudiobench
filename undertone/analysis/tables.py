"""Paper tables, built from the runner's JSONL rows.

Three rules that the retired analysis broke and this one does not:

*  **Truncated cells never enter an accuracy table.** They get their own table
   (per-model input limits and truncation coverage) because "this model cannot
   hear the audio" and "this model heard it and got it wrong" are different
   findings.
*  **Unverified items are excluded and counted.** An item nobody has listened to
   is a proposal, not evidence.
*  **No composite score.** The paper plan says so explicitly, and the retired
   suite's composites hid which term was zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..items import CATEGORIES, CATEGORY_LABEL, LANGS
from ..ladder import CONDITIONS
from ..scoring import accuracy, cluster_bootstrap_ci, ladder_costs, salience_trap, summarize


class MixedHardware(RuntimeError):
    """Two backends in one results table."""


def signatures(rows: Sequence[dict]) -> set[str]:
    return {r["signature"] for r in rows if r.get("signature")}


def usable(rows: Sequence[dict], require_verified: bool = True,
           allow_mixed_hardware: bool = False) -> list[dict]:
    """Rows that may appear in a results table.

    Raises on mixed backends unless explicitly allowed. Running one model on a
    Mac and twelve on a T4 would make every cross-model comparison partly a
    comparison of kernels -- and the whole point of one shared protocol is that
    the only thing differing between rows is the model.
    """
    keep = [r for r in rows if not r.get("error") and not r.get("truncated")
            and r.get("role_chosen") is not None]
    if require_verified:
        keep = [r for r in keep if r.get("verified", True)]

    found = signatures(keep)
    if len(found) > 1 and not allow_mixed_hardware:
        by_signature = {
            sig: sorted({r["model_key"] for r in keep if r.get("signature") == sig})
            for sig in sorted(found)}
        raise MixedHardware(
            "results came from more than one backend/dtype and cannot share a "
            f"table: {by_signature}. Re-run the odd ones out on the same hardware, "
            "or pass allow_mixed_hardware=True and report the split in Limitations.")
    return keep


def _cell(rows: Sequence[dict], ci: bool) -> dict[str, Any]:
    out = summarize(rows)
    if ci and rows:
        lo, hi = cluster_bootstrap_ci(rows, accuracy)
        out["acc_ci"] = (round(lo, 3), round(hi, 3))
        lo, hi = cluster_bootstrap_ci(rows, salience_trap)
        out["salience_ci"] = (round(lo, 3), round(hi, 3))
    return out


def table1_main(rows: Sequence[dict], condition: str = "L3",
                ci: bool = True) -> list[dict]:
    """Model x category at one ladder condition, with the trap rates alongside.

    The salience-trap column is the headline: accuracy says a model failed,
    the trap rate says it failed *by selecting the loud competing mention*,
    which is the mechanism claim.
    """
    scoped = [r for r in usable(rows) if r["condition"] == condition]
    out: list[dict] = []
    for model in sorted({r["model_key"] for r in scoped}):
        for category in CATEGORIES:
            cell = [r for r in scoped if r["model_key"] == model and r["category"] == category]
            if not cell:
                continue
            stats = _cell(cell, ci)
            out.append({
                "model": model, "category": category,
                "label": CATEGORY_LABEL[category], "n": stats["n"],
                "accuracy": round(stats["accuracy"], 3),
                "acc_ci": stats.get("acc_ci"),
                "salience_trap": round(stats["salience_trap_rate"], 3),
                "salience_ci": stats.get("salience_ci"),
                "recency_trap": round(stats["recency_trap_rate"], 3),
                "fabrication": round(stats["fabrication_rate"], 3),
            })
    return out


def table1_nulls(rows: Sequence[dict], condition: str = "L3") -> list[dict]:
    """Null-item accuracy: can the model say "not mentioned" when it is true?"""
    scoped = [r for r in usable(rows) if r["condition"] == condition and r["is_null"]]
    return [{"model": m,
             "n": len([r for r in scoped if r["model_key"] == m]),
             "null_accuracy": round(
                 accuracy([r for r in scoped if r["model_key"] == m]), 3)}
            for m in sorted({r["model_key"] for r in scoped})]


def table2_ladder(rows: Sequence[dict]) -> list[dict]:
    """RetrievalCost and LongContextCost per model per category.

    Both are measured against L1, so a model that never perceived the needle
    cannot be credited with "losing it in context" -- which is the whole reason
    L1 is not optional.
    """
    rows = usable(rows)
    out: list[dict] = []
    for model in sorted({r["model_key"] for r in rows}):
        for category in CATEGORIES:
            scoped = [r for r in rows if r["model_key"] == model and r["category"] == category]
            if not scoped:
                continue
            costs = ladder_costs({c: [r for r in scoped if r["condition"] == c]
                                  for c in CONDITIONS})
            out.append({"model": model, "category": category,
                        **{k: (round(v, 3) if v == v else None) for k, v in costs.items()}})
    return out


def table4_truncation(rows: Sequence[dict], adapters: dict[str, float] | None = None) -> list[dict]:
    """Per-model input limits and how much of the grid they could actually reach.

    Preempts "you scored truncation as failure". Nothing here is scored at all:
    the column is coverage, and a model that reaches 25% of the grid is reported
    as reaching 25% of the grid.
    """
    out: list[dict] = []
    for model in sorted({r["model_key"] for r in rows}):
        scoped = [r for r in rows if r["model_key"] == model]
        truncated = [r for r in scoped if r.get("truncated")]
        by_condition = {
            c: round(1 - sum(1 for r in scoped
                             if r["condition"] == c and r.get("truncated"))
                     / max(1, sum(1 for r in scoped if r["condition"] == c)), 3)
            for c in CONDITIONS}
        limit = (adapters or {}).get(model)
        if limit is None and scoped:
            limit = max((r.get("seconds_seen", 0) for r in scoped), default=0)
        out.append({
            "model": model,
            "max_audio_s": limit,
            "cells": len(scoped),
            "truncated": len(truncated),
            "coverage": round(1 - len(truncated) / max(1, len(scoped)), 3),
            **{f"coverage_{c}": v for c, v in by_condition.items()},
        })
    return out


def table_language(rows: Sequence[dict], condition: str = "L3") -> list[dict]:
    """Category x language, which is where F3' lives.

    F3' predicts a higher salience-trap rate on P3 in Hindi and Bengali than in
    English: prominence there is carried by focus particles and word order that
    an acoustic salience prior cannot see. Note this is a weaker claim than the
    paper plan's original F3, which needed a tone language.
    """
    scoped = [r for r in usable(rows) if r["condition"] == condition]
    out: list[dict] = []
    for model in sorted({r["model_key"] for r in scoped}):
        for lang in LANGS:
            for category in CATEGORIES:
                cell = [r for r in scoped if r["model_key"] == model
                        and r["lang"] == lang and r["category"] == category]
                if not cell:
                    continue
                stats = summarize(cell)
                out.append({"model": model, "lang": lang, "category": category,
                            "n": stats["n"],
                            "accuracy": round(stats["accuracy"], 3),
                            "salience_trap": round(stats["salience_trap_rate"], 3)})
    return out


def scorer_gap(rows: Sequence[dict]) -> list[dict]:
    """Letter-logit answer vs free-generation answer.

    The gap separates "the model does not know" from "the model knows but will
    not emit a bare letter" -- an instruction-following artifact that the
    retired regex parsers silently scored as a wrong answer.
    """
    out: list[dict] = []
    for model in sorted({r["model_key"] for r in rows}):
        scoped = [r for r in usable(rows) if r["model_key"] == model
                  and r.get("logit_role") and r.get("gen_letter") is not None]
        if not scoped:
            continue
        agree = sum(1 for r in scoped if r["logit_letter"] == r["gen_letter"])
        all_rows = [r for r in rows if r["model_key"] == model and not r.get("error")]
        out.append({
            "model": model, "n": len(scoped),
            "agreement": round(agree / len(scoped), 3),
            "logit_accuracy": round(accuracy(
                [{**r, "role_chosen": r["logit_role"]} for r in scoped]), 3),
            "gen_accuracy": round(accuracy(
                [{**r, "role_chosen": r.get("gen_role")} for r in scoped]), 3),
            "unparseable_rate": round(
                sum(1 for r in all_rows if r.get("gen_letter") is None)
                / max(1, len(all_rows)), 3),
        })
    return out


def sanity_checks(rows: Sequence[dict]) -> list[str]:
    """Things that invalidate the tables above. Empty list means clean."""
    problems: list[str] = []
    scoped = usable(rows, allow_mixed_hardware=True)
    if not scoped:
        return ["no usable rows at all"]

    degenerate = [r for r in scoped if r.get("logit_degenerate")]
    if degenerate:
        problems.append(
            f"{len(degenerate)} cells had all four letter logits equal -- that is a "
            "broken measurement, not a 25% baseline")

    for model in sorted({r["model_key"] for r in scoped}):
        cells = [r for r in scoped if r["model_key"] == model]
        for letter in "ABCD":
            rate = sum(1 for r in cells if r.get("letter_chosen") == letter) / len(cells)
            if rate > 0.60:
                problems.append(
                    f"{model} picked {letter} on {rate:.0%} of cells -- letter-position "
                    "bias at that level makes its role rates uninterpretable")

    found = signatures(rows)
    if len(found) > 1:
        problems.append(
            f"rows come from {len(found)} different backends ({sorted(found)}) -- "
            "they cannot share a results table")

    unverified = [r for r in rows if r.get("verified") is False]
    if unverified:
        problems.append(
            f"{len(unverified)} rows come from unverified items and were excluded; "
            "verify them or the affected cells cannot be reported")
    return problems
