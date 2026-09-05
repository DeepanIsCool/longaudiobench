"""Item schema for UNDERTONE.

An item is one four-option multiple-choice question about one span ("the
needle") of one real recording.  The four options carry *roles*, not fixed
letters: the letter each role receives is drawn per run, so the diagnostic
signal is "which role did the model pick", never "which letter".

Roles (paper plan section 8.1):
    correct   the right answer
    salience  a louder / stressed / repeated competing mention  -> salience prior
    recency   the most recent mention of the topic              -> recency bias
    absent    "not mentioned in the recording"                  -> fabrication,
              and the correct role on null items
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

ROLES: tuple[str, ...] = ("correct", "salience", "recency", "absent")
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

# Five prominence categories.  P1-P4 share one mechanism; C1 is the contrast
# that has to fail *differently*, or the mechanism claim collapses into
# "hard things are hard".
CATEGORIES: tuple[str, ...] = ("P1", "P2", "P3", "P4", "C1")
CATEGORY_LABEL = {
    "P1": "quiet",
    "P2": "masked",
    "P3": "backgrounded",
    "P4": "corrected",
    "C1": "delivered",
}

LANGS: tuple[str, ...] = ("en", "hi", "bn")

# Duration bands in seconds (5 / 10 / 20 / 30 min).
BANDS: tuple[int, ...] = (300, 600, 1200, 1800)


@dataclass
class MCQItem:
    item_id: str
    recording_id: str            # clustering unit for bootstrap CIs
    lang: str
    category: str
    sector: str
    audio_path: str              # the full recording, band-length
    duration_band: int           # one of BANDS; audio_path is this long
    needle_start: float          # seconds into audio_path
    needle_end: float
    question: str
    options: dict[str, str]      # role -> option text; keys are exactly ROLES
    is_null: bool = False        # True => "absent" is the correct role
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lang not in LANGS:
            raise ValueError(f"{self.item_id}: lang {self.lang!r} not in {LANGS}")
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.item_id}: category {self.category!r} not in {CATEGORIES}")
        if self.duration_band not in BANDS:
            raise ValueError(f"{self.item_id}: duration_band {self.duration_band} not in {BANDS}")
        missing = set(ROLES) - set(self.options)
        if missing:
            raise ValueError(f"{self.item_id}: options missing roles {sorted(missing)}")
        extra = set(self.options) - set(ROLES)
        if extra:
            raise ValueError(f"{self.item_id}: options have unknown roles {sorted(extra)}")
        if not 0.0 <= self.needle_start < self.needle_end:
            raise ValueError(
                f"{self.item_id}: bad needle span {self.needle_start}-{self.needle_end}"
            )
        if self.needle_end > self.duration_band:
            raise ValueError(
                f"{self.item_id}: needle ends at {self.needle_end}s, past the "
                f"{self.duration_band}s band"
            )

    @property
    def correct_role(self) -> str:
        return "absent" if self.is_null else "correct"

    @property
    def needle_mid(self) -> float:
        return 0.5 * (self.needle_start + self.needle_end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MCQItem":
        return cls(**d)


@dataclass
class ItemPack:
    """A set of items plus the build metadata a reviewer will ask for."""

    items: list[MCQItem]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MCQItem]:
        return iter(self.items)

    @property
    def fingerprint(self) -> str:
        """Content hash of the pack, stamped onto every result row.

        The pack builder writes to one place and the model notebooks read from
        another, joined only by a manual dataset upload. Skip that upload and
        every model runs happily against stale audio and reports plausible
        numbers -- nothing errors, and it is only visible if you notice the
        results are byte-identical to the previous round. This makes that
        loud instead of silent.
        """
        import hashlib

        h = hashlib.sha256()
        for item in sorted(self.items, key=lambda i: i.item_id):
            h.update(item.item_id.encode())
            h.update(str(item.duration_band).encode())
            h.update(item.options["correct"].encode())
        return h.hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            meta = {**self.meta, "fingerprint": self.fingerprint,
                    "n_items": len(self.items)}
            fh.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
            for item in self.items:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ItemPack":
        items: list[MCQItem] = []
        meta: dict[str, Any] = {}
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "__meta__" in obj:
                    meta = obj["__meta__"]
                else:
                    items.append(MCQItem.from_dict(obj))
        return cls(items=items, meta=meta)

    def filter(self, **kw: Any) -> "ItemPack":
        """Subset by any scalar attribute, e.g. ``pack.filter(lang="en")``."""
        out = [it for it in self.items if all(getattr(it, k) == v for k, v in kw.items())]
        return ItemPack(items=out, meta={**self.meta, "filtered_by": kw})

    def counts(self, *keys: str) -> dict[tuple, int]:
        tally: dict[tuple, int] = {}
        for it in self.items:
            k = tuple(getattr(it, key) for key in keys)
            tally[k] = tally.get(k, 0) + 1
        return tally
