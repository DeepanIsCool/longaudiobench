"""The P3-constructed arm.

The paper plan (section 12) names this as the backstop for its own top risk:

    "Natural yield for the flat-aside-plus-loud-repeated-distractor conjunction
     is the primary risk. Split P3 into P3-natural (harvested) and
     P3-constructed ... State in Limitations that the F2 mechanism claim rests
     partly on the constructed arm."

That risk is now measured rather than anticipated. Across 60 AMI meetings and
47 windows, P3 yields ~10 items after the leak filter -- against 22 for P2 and
21 for P4 -- and on the first full ladder its salience-trap rate came out
*lowest* of the five categories, where F2 predicts highest. Ten items cannot
settle that either way.

The plan's own construction is purpose-recorded sessions with confederates. With
no studio, this does the nearest defensible thing: it takes a real aside and a
real competing mention **from the same window of the same recording** and
applies a gain envelope so the prominence contrast P3 is defined by actually
exists. No splicing, no inserted content, no TTS -- every word was spoken by
that speaker in that room, and only the relative loudness changes.

What this is not: it is not evidence that the contrast occurs naturally at this
strength. Items carry ``constructed: True`` and their gains, the analysis
reports the two arms separately, and the paper says the mechanism claim leans on
the constructed one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

# How far apart the two mentions are pushed. 12 dB is a large but ordinary
# difference between an aside and a stressed repetition; beyond that the
# attenuated span stops being speech a listener could recover, which would make
# the item a perception test rather than a retrieval one.
TARGET_ATTENUATION_DB = -9.0
COMPETITOR_BOOST_DB = 3.0
# Short enough not to eat the contrast it is protecting: at 0.25 s a one-second
# span is half ramp, which cost ~5 dB of the 12 dB being applied.
RAMP_SECONDS = 0.04          # avoid a click at the envelope edges


@dataclass(frozen=True)
class GainEdit:
    start: float
    end: float
    gain_db: float

    def as_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "gain_db": self.gain_db}


def _ramped_gain(n: int, gain_db: float, sr: int) -> np.ndarray:
    """A gain curve that eases in and out rather than stepping."""
    gain = 10.0 ** (gain_db / 20.0)
    curve = np.full(n, gain, dtype=np.float32)
    ramp = min(int(RAMP_SECONDS * sr), n // 2)
    if ramp > 0:
        up = np.linspace(1.0, gain, ramp, dtype=np.float32)
        curve[:ramp] = up
        curve[-ramp:] = up[::-1]
    return curve


def apply_gain_edits(audio: np.ndarray, edits: list[GainEdit],
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply gain envelopes in place on a copy. Nothing is cut or inserted."""
    out = np.array(audio, dtype=np.float32, copy=True)
    for edit in edits:
        lo = max(0, int(edit.start * sr))
        hi = min(len(out), int(edit.end * sr))
        if hi <= lo:
            continue
        span = out[lo:hi] * _ramped_gain(hi - lo, edit.gain_db, sr)
        # Clip protection scoped to the edited span. Normalising the whole
        # signal would quietly rescale audio nobody edited - the item would no
        # longer be the recording it claims to be, everywhere except where the
        # manipulation was declared.
        peak = float(np.max(np.abs(span))) if span.size else 0.0
        if peak > 1.0:
            span *= 0.99 / peak
        out[lo:hi] = span
    return out


def construct_p3(audio: np.ndarray, target_start: float, target_end: float,
                 competitor_spans: list[tuple[float, float]],
                 sr: int = SAMPLE_RATE,
                 attenuation_db: float = TARGET_ATTENUATION_DB,
                 boost_db: float = COMPETITOR_BOOST_DB
                 ) -> tuple[np.ndarray, list[dict]]:
    """Make the P3 prominence contrast explicit in real audio.

    The target (the aside carrying the correct answer) is attenuated; every
    mention of the competing value is boosted. Returns the edited audio and the
    edit list, which goes into provenance so the manipulation is auditable and
    exactly reversible in description.
    """
    edits = [GainEdit(target_start, target_end, attenuation_db)]
    edits += [GainEdit(s, e, boost_db) for s, e in competitor_spans]
    return apply_gain_edits(audio, edits, sr), [e.as_dict() for e in edits]


def measure_contrast(audio: np.ndarray, target: tuple[float, float],
                     competitor: tuple[float, float],
                     sr: int = SAMPLE_RATE) -> float:
    """Competitor minus target level, in dB. The audit number for the arm.

    Reported per constructed item so a reviewer can see the contrast actually
    achieved rather than the gain requested -- they differ whenever a span was
    already loud or already quiet.
    """
    def level(span: tuple[float, float]) -> float:
        lo, hi = int(span[0] * sr), int(span[1] * sr)
        chunk = audio[max(0, lo):max(0, hi)]
        if chunk.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        return 20.0 * np.log10(max(rms, 1e-9))

    return round(level(competitor) - level(target), 2)
