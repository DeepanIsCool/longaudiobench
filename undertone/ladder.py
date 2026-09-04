"""The four-condition ladder (paper plan section 8.2).

    L1  isolated      ~20 s window containing the needle   -> perception ceiling
    L2  local         ~2 min window centred on the needle  -> perception + light search
    L3  full          the whole band-length recording      -> perception + retrieval
    L4  oracle        the whole recording, told where      -> perception under context load

    RetrievalCost    = Acc(L1) - Acc(L3)
    LongContextCost  = Acc(L1) - Acc(L4)

Without L1 a 20% L3 score is uninterpretable, which is what makes every finding
falsifiable.  L1 is therefore never optional, and it is the one condition every
model in the roster can run -- including the 30 s-capped ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from .items import MCQItem

CONDITIONS: tuple[str, ...] = ("L1", "L2", "L3", "L4")

L1_SECONDS = 20.0
L2_SECONDS = 120.0


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    oracle: bool          # True => the prompt names the needle's location

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def contains(self, start: float, end: float) -> bool:
        return self.start <= start and end <= self.end


def _centred(mid: float, span: float, total: float, needle: tuple[float, float]) -> Window:
    """A ``span``-second window centred on ``mid``, clamped into ``[0, total]``.

    Widened if necessary so the needle is always inside: a 25 s aside cannot be
    asked about through a 20 s window, and silently cropping it would turn a
    perception ceiling into a perception floor.
    """
    n_start, n_end = needle
    span = max(span, n_end - n_start)
    if span >= total:
        return Window(0.0, total, oracle=False)

    start = mid - span / 2.0
    start = max(0.0, min(start, total - span))
    end = start + span

    # Clamping to the recording edges can still push the needle out when the
    # needle sits very near a boundary; slide the window onto it.
    if n_start < start:
        start, end = n_start, n_start + span
    elif n_end > end:
        end = min(total, n_end)
        start = end - span
    return Window(max(0.0, start), min(total, end), oracle=False)


def window_for(item: MCQItem, condition: str) -> Window:
    """Audio window and oracle flag for one item under one ladder condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")

    total = float(item.duration_band)
    needle = (item.needle_start, item.needle_end)

    if condition == "L1":
        return _centred(item.needle_mid, L1_SECONDS, total, needle)
    if condition == "L2":
        return _centred(item.needle_mid, L2_SECONDS, total, needle)
    if condition == "L3":
        return Window(0.0, total, oracle=False)
    return Window(0.0, total, oracle=True)  # L4
