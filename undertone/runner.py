"""Sweep runner: items x ladder conditions, resumable, one JSONL row per cell.

Kaggle sessions die at 12 h and the weekly quota is 30 GPU-h, so a sweep that
cannot resume is a sweep that has to be redone.  Every row is appended as soon
as it is scored and ``(item_id, condition)`` keys already present are skipped on
restart.

Truncation is recorded, never scored as zero.  A model whose documented ceiling
is below the window still runs -- what it does with a fragment (especially how
often it fabricates rather than picking "not mentioned") is a result -- but its
rows carry ``truncated: true`` and are excluded from the accuracy table.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .adapters.base import ModelAdapter, Truncation, apply_cap, load_audio
from .items import ItemPack, MCQItem
from .ladder import CONDITIONS, window_for
from .protocol import render
from .scoring import argmax_letter, is_degenerate, parse_free_letter

DEFAULT_SEED = 20260904


def _key(item_id: str, condition: str) -> str:
    return f"{item_id}|{condition}"


def completed_keys(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written final line from a killed session
            if row.get("error") is None and "item_id" in row:
                done.add(_key(row["item_id"], row["condition"]))
    return done


class _AudioCache:
    """One decode per distinct window.

    L3 and L4 share a window and differ only in the prompt, so this halves
    decode work outright.

    ponytail: decode-level caching only. Reusing the audio *prefix KV cache*
    across the ~7 items that share a recording would cut prefill by roughly the
    same factor again, but needs per-model cache surgery. Add it if quota binds.
    """

    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self._store: dict[tuple, Any] = {}

    def get(self, path: str, start: float, end: float):
        key = (path, round(start, 3), round(end, 3))
        if key not in self._store:
            if len(self._store) >= self.max_entries:
                self._store.pop(next(iter(self._store)))
            self._store[key] = load_audio(path, start, end)
        return self._store[key]


def run_model(
    adapter: ModelAdapter,
    pack: ItemPack | Sequence[MCQItem],
    out_path: str | Path,
    conditions: Iterable[str] = CONDITIONS,
    seed: int = DEFAULT_SEED,
    run_id: str = "pilot",
    also_generate: bool = True,
    max_new_tokens: int | None = None,
    audio_root: str | Path = ".",
    max_consecutive_errors: int = 5,
    progress: bool = True,
) -> Path:
    """Score every (item, condition) cell and append rows to ``out_path``."""
    items = list(pack)
    conditions = list(conditions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root = Path(audio_root)

    done = completed_keys(out_path)
    todo = [(it, c) for it in items for c in conditions if _key(it.item_id, c) not in done]
    if progress:
        print(f"[{adapter.key}] {len(todo)} cells to run, {len(done)} already done")
    if not todo:
        return out_path

    if adapter.model is None:
        adapter.load()

    cache = _AudioCache()
    consecutive_errors = 0

    with out_path.open("a", encoding="utf-8") as fh:
        for index, (item, condition) in enumerate(todo, 1):
            row = _run_cell(
                adapter, item, condition, seed, run_id, cache, audio_root,
                also_generate, max_new_tokens,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            if row.get("error"):
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[{adapter.key}] {consecutive_errors} consecutive failures; "
                        f"last: {row['error']}"
                    )
            else:
                consecutive_errors = 0

            if progress and (index % 20 == 0 or index == len(todo)):
                print(f"[{adapter.key}] {index}/{len(todo)}")

    return out_path


def _run_cell(
    adapter: ModelAdapter,
    item: MCQItem,
    condition: str,
    seed: int,
    run_id: str,
    cache: _AudioCache,
    audio_root: Path,
    also_generate: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    window = window_for(item, condition)
    rendered = render(item, window, seed)

    row: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "model_key": adapter.key,
        "model_id": adapter.model_id,
        # Stamped on every row so a results table can prove it is homogeneous.
        # CUDA and MPS do not produce identical logits; a table mixing them
        # compares machines, not models.
        "signature": adapter.hardware.signature,
        "backend": adapter.hardware.backend,
        "dtype": adapter.hardware.dtype,
        "item_id": item.item_id,
        "recording_id": item.recording_id,
        "lang": item.lang,
        "category": item.category,
        "sector": item.sector,
        "duration_band": item.duration_band,
        "condition": condition,
        "window_start": window.start,
        "window_end": window.end,
        "is_null": item.is_null,
        "correct_role": item.correct_role,
        "letter_to_role": rendered.letter_to_role,
        "error": None,
    }

    started = time.time()
    try:
        path = str(audio_root / item.audio_path) if not Path(item.audio_path).is_absolute() \
            else item.audio_path
        # Decode only what the model can actually ingest. A 30 s-capped model
        # was decoding the full 1800 s window and discarding 99% of it, which is
        # ~115 MB of float32 per cell for nothing.
        decode_end = min(window.end, window.start + adapter.max_audio_s)
        audio = cache.get(path, window.start, decode_end)
        capped = apply_cap(audio, adapter.max_audio_s)
        capped = Truncation(capped.audio, window.seconds > adapter.max_audio_s,
                            capped.seconds_seen, window.seconds)
        row.update(
            seconds_offered=round(capped.seconds_offered, 2),
            seconds_seen=round(capped.seconds_seen, 2),
            truncated=capped.truncated,
        )

        scores = adapter.score_letters(capped.audio, rendered.prompt)
        logit_letter = argmax_letter(scores)
        row.update(
            logit_scores={k: round(v, 4) for k, v in scores.items()},
            logit_letter=logit_letter,
            logit_role=rendered.letter_to_role[logit_letter],
            logit_degenerate=is_degenerate(scores),
        )

        if also_generate:
            text = adapter.generate(capped.audio, rendered.prompt,
                                    max_new_tokens=max_new_tokens)  # None -> budget
            gen_letter = parse_free_letter(text, strip_reasoning=adapter.strip_reasoning)
            row.update(
                gen_text=text,
                gen_letter=gen_letter,
                gen_role=rendered.letter_to_role.get(gen_letter) if gen_letter else None,
            )

        # The primary scorer decides role_chosen; the other is kept alongside so
        # the logits-vs-generation gap is measurable rather than assumed away.
        if adapter.primary == "freegen":
            row["scorer"] = "freegen"
            row["letter_chosen"] = row.get("gen_letter")
            row["role_chosen"] = row.get("gen_role")
        else:
            row["scorer"] = "logits"
            row["letter_chosen"] = logit_letter
            row["role_chosen"] = row["logit_role"]

        row["correct"] = row["role_chosen"] == item.correct_role
    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill a sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=4)
        row["role_chosen"] = None
        row["letter_chosen"] = None

    row["latency_ms"] = round((time.time() - started) * 1000, 1)
    return row


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def scorable(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that belong in the accuracy table: no error, not truncated."""
    return [r for r in rows if not r.get("error") and not r.get("truncated")]
