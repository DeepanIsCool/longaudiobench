"""Scoring.

The retired pipeline read free text with regexes and folded every parse failure
into a zero (``metrics/__init__.py`` dropped ``inf`` and then defaulted to
``0.0``, which is why every cascaded ``timestamp_error`` in ``results_v25`` reads
exactly ``0.000`` -- that number meant "nothing parsed", not "no error").

Two scorers replace it:

*  ``letter_logits``  -- one forward pass, compare the next-token distribution
   over A/B/C/D.  Deterministic, generation-free, no regex, and it makes the
   salience-trap rate exactly P(salience role).  Primary for every model whose
   answer is meant to be the next token.
*  ``parse_free_letter`` -- strict single-letter parse of generated text, with
   ``None`` reported as its own ``unparseable`` bucket and never as a wrong
   answer.  Primary only for the chain-of-thought variants, where the first
   token is a thought rather than an answer.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from .items import LETTERS, ROLES

# Prefixes a tokenizer might produce for an option letter.  Whichever variant
# scores highest wins; models differ on whether the answer token carries a
# leading space.
_LETTER_VARIANTS = ("{L}", " {L}", "{L}.", "{L})", " {L}.", "\n{L}")


def letter_token_ids(tokenizer, letters: Sequence[str] = LETTERS) -> dict[str, list[int]]:
    """First-token ids for each option letter, across plausible surface forms."""
    out: dict[str, list[int]] = {}
    for letter in letters:
        ids: list[int] = []
        for pattern in _LETTER_VARIANTS:
            text = pattern.format(L=letter)
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:  # some fast tokenizers reject the kwarg
                encoded = tokenizer.encode(text)
            if encoded and encoded[0] not in ids:
                ids.append(int(encoded[0]))
        if not ids:
            raise ValueError(f"tokenizer produced no ids for option letter {letter!r}")
        out[letter] = ids
    return out


def letter_logits(
    next_token_logits,
    token_ids: dict[str, list[int]],
) -> dict[str, float]:
    """Best logit per option letter, from a 1-D next-token logit vector.

    ``next_token_logits`` is anything indexable by token id returning a float
    (a torch tensor row or a numpy array).
    """
    scores: dict[str, float] = {}
    for letter, ids in token_ids.items():
        best = -math.inf
        for tid in ids:
            value = float(next_token_logits[tid])
            if value > best:
                best = value
        scores[letter] = best
    return scores


def argmax_letter(scores: dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def is_degenerate(scores: dict[str, float], tol: float = 1e-6) -> bool:
    """All four letters equally likely => broken prompt or wrong token ids.

    The smoke test asserts against this: a model that cannot tell A from D is
    not producing a 25% baseline, it is producing no measurement at all.
    """
    values = list(scores.values())
    return (max(values) - min(values)) < tol


_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def parse_free_letter(text: str, strip_reasoning: bool = False) -> str | None:
    """Strict single-letter parse.  ``None`` means unparseable, not wrong."""
    if not text:
        return None
    if strip_reasoning:
        text = strip_cot(text)
    match = _LETTER_RE.search(text)
    return match.group(1) if match else None


_COT_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_COT_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_cot(text: str) -> str:
    """Drop ``<think>...</think>`` spans emitted by the Thinking variants.

    Also drops an unterminated trailing block, which is what a token budget cut
    short looks like -- leaving it in would let a letter mentioned mid-reasoning
    be read as the final answer.
    """
    text = _COT_BLOCK.sub(" ", text)
    text = _COT_OPEN.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def role_of(letter: str | None, letter_to_role: dict[str, str]) -> str | None:
    return letter_to_role.get(letter) if letter else None


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Rates over scored rows.

    Each row needs ``role_chosen`` (may be ``None``), ``correct_role`` and
    ``is_null``.  Truncated rows must be filtered out before calling this --
    they belong in the truncation table, not the accuracy table.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}

    scored = [r for r in rows if r.get("role_chosen") is not None]
    non_null = [r for r in scored if not r["is_null"]]
    null = [r for r in scored if r["is_null"]]

    def rate(subset: Sequence[dict], pred: Callable[[dict], bool]) -> float:
        return sum(1 for r in subset if pred(r)) / len(subset) if subset else float("nan")

    out: dict[str, float | int] = {
        "n": n,
        "n_scored": len(scored),
        "unparseable_rate": (n - len(scored)) / n,
        "accuracy": rate(scored, lambda r: r["role_chosen"] == r["correct_role"]),
        # The headline diagnostic: picking the loud competing mention.
        "salience_trap_rate": rate(non_null, lambda r: r["role_chosen"] == "salience"),
        "recency_trap_rate": rate(non_null, lambda r: r["role_chosen"] == "recency"),
        "fabrication_rate": rate(non_null, lambda r: r["role_chosen"] == "absent"),
        "null_accuracy": rate(null, lambda r: r["role_chosen"] == "absent"),
        "n_null": len(null),
    }
    # Letter-position bias, free at this point and worth reporting: a model that
    # prefers "A" regardless of content invalidates everything above it.
    for letter in LETTERS:
        out[f"letter_rate_{letter}"] = rate(scored, lambda r, L=letter: r.get("letter_chosen") == L)
    return out


def cluster_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    cluster_key: str = "recording_id",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling **clusters**, not rows.

    Items from one recording share a speaker, a room and a topic, so resampling
    items independently understates the interval.  ~7 items per recording here,
    so the difference is not cosmetic.
    """
    if not rows:
        return (float("nan"), float("nan"))

    buckets: dict[Any, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row[cluster_key], []).append(row)
    keys = list(buckets)

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        drawn: list[dict] = []
        for _ in range(len(keys)):
            drawn.extend(buckets[keys[rng.randrange(len(keys))]])
        value = statistic(drawn)
        if not math.isnan(value):
            samples.append(value)

    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[min(len(samples) - 1, int(alpha * len(samples)))]
    hi = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return (lo, hi)


def accuracy(rows: Sequence[dict[str, Any]]) -> float:
    scored = [r for r in rows if r.get("role_chosen") is not None]
    if not scored:
        return float("nan")
    return sum(1 for r in scored if r["role_chosen"] == r["correct_role"]) / len(scored)


def salience_trap(rows: Sequence[dict[str, Any]]) -> float:
    subset = [r for r in rows if r.get("role_chosen") is not None and not r["is_null"]]
    if not subset:
        return float("nan")
    return sum(1 for r in subset if r["role_chosen"] == "salience") / len(subset)


MIN_CELLS_FOR_A_COST = 10


def ladder_costs(by_condition: dict[str, Sequence[dict[str, Any]]],
                 min_cells: int = MIN_CELLS_FOR_A_COST) -> dict[str, float]:
    """RetrievalCost and LongContextCost, both relative to the L1 ceiling.

    A cost is withheld when either side rests on too few cells. Phi-4 reported
    RetrievalCost 0.422 off an acc_L3 of 0.0 that survived on a handful of rows
    after 162 errored -- arithmetically fine, and indistinguishable in a table
    from Omni-3B's 0.396, which came from 188 scorable cells and no errors. The
    per-condition n is returned so a reader can see which is which.
    """
    out: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cond in ("L1", "L2", "L3", "L4"):
        rows = [r for r in by_condition.get(cond, []) if r.get("role_chosen") is not None]
        counts[cond] = len(rows)
        out[f"acc_{cond}"] = accuracy(rows)
        out[f"n_{cond}"] = len(rows)

    def cost(other: str) -> float:
        if counts["L1"] < min_cells or counts[other] < min_cells:
            return float("nan")
        return out["acc_L1"] - out[f"acc_{other}"]

    out["RetrievalCost"] = cost("L3")
    out["LongContextCost"] = cost("L4")
    return out


__all__ = [
    "ROLES",
    "LETTERS",
    "letter_token_ids",
    "letter_logits",
    "argmax_letter",
    "is_degenerate",
    "parse_free_letter",
    "strip_cot",
    "role_of",
    "summarize",
    "cluster_bootstrap_ci",
    "accuracy",
    "salience_trap",
    "ladder_costs",
]
