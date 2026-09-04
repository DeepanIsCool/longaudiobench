"""Acoustic features that decide which category a candidate belongs to.

Every feature is computed from the real recording.  Nothing is inserted and
nothing is mixed: the prominence differences these measure are differences the
speakers themselves produced.

Deliberately cheap -- RMS energy, gold overlap from segment timings, and an
optional F0 pass.  A heavier stack (pyannote OSD, torchcrepe) would give tidier
numbers, but candidate proposal only has to be good enough to put clips in front
of a human, who then confirms or rejects.  The precision that matters is the
verification step's, not this one's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources import Recording, Segment

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SegmentFeatures:
    index: int
    rms_db: float
    energy_percentile: float    # 0-1 within this recording
    overlap_ratio: float        # fraction of the segment covered by another speaker
    f0_range_semitones: float   # 0.0 when F0 was not computed
    f0_percentile: float        # rank within this recording's measured ranges
    is_quiet: bool
    is_masked: bool
    is_flat: bool


def _rms_db(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def overlap_ratio(segment: Segment, others: list[Segment]) -> float:
    """Fraction of a segment during which a *different* speaker is also talking.

    Gold, from the transcript's own timings -- no detector, no threshold to
    tune, and no way for a detection error to become a mislabelled item.
    """
    if segment.duration <= 0:
        return 0.0
    covered = 0.0
    for other in others:
        if other is segment or other.speaker == segment.speaker:
            continue
        lo, hi = max(segment.start, other.start), min(segment.end, other.end)
        if hi > lo:
            covered += hi - lo
    return min(1.0, covered / segment.duration)


def f0_range_semitones(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Pitch span of a segment; a flat span is the P3 signature.

    Returns 0.0 when librosa is unavailable or the segment is unvoiced, and
    callers treat 0.0 as "unknown" rather than "flat" -- guessing flat would
    manufacture P3 candidates out of silence.
    """
    try:
        import librosa

        f0 = librosa.yin(chunk, fmin=60, fmax=350, sr=sr)
        voiced = f0[np.isfinite(f0) & (f0 > 0)]
        if voiced.size < 10:
            return 0.0
        lo, hi = np.percentile(voiced, [10, 90])
        return float(12.0 * np.log2(max(hi, 1e-6) / max(lo, 1e-6)))
    except Exception:  # noqa: BLE001 - feature is optional
        return 0.0


def segment_features(
    recording: Recording,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    quiet_percentile: float = 0.25,
    quiet_db_drop: float = 6.0,
    masked_ratio: float = 0.30,
    flat_percentile: float = 0.33,
    with_f0: bool = True,
) -> list[SegmentFeatures]:
    """Per-segment prominence features for one recording.

    Thresholds are relative to *this* recording, not absolute: a quiet aside in
    a loud meeting and a quiet aside in a quiet one are both quiet asides, and
    an absolute dB cut would only find the quiet meetings.

    "Quiet" needs a low rank **and** a real drop below the recording's median.
    Rank alone always fires -- in a uniformly loud recording the bottom quartile
    is still a quartile -- which would fill P1 with segments that are not quiet
    at all and turn the category into noise.
    """
    levels: list[float] = []
    chunks: list[np.ndarray] = []
    for segment in recording.segments:
        lo = int(max(0.0, segment.start) * sr)
        hi = int(min(recording.duration, segment.end) * sr)
        chunk = audio[lo:hi] if hi > lo else np.zeros(0, dtype=np.float32)
        chunks.append(chunk)
        levels.append(_rms_db(chunk))

    order = np.argsort(np.argsort(np.asarray(levels)))
    n = max(1, len(levels))
    median_db = float(np.median(levels)) if levels else -120.0

    # F0 range is ranked within this recording, like energy. An absolute
    # semitone cut is the wrong instrument: measured ranges on AMI run from 0.3
    # to 28 semitones with a median near 14, so a fixed 2.5 threshold selects a
    # handful of segments and P3 came out empty on the first real harvest.
    f0_values = [f0_range_semitones(chunks[i], sr) if with_f0 else 0.0
                 for i in range(len(recording.segments))]
    measured = [v for v in f0_values if v > 0]
    measured_sorted = sorted(measured)

    out: list[SegmentFeatures] = []
    for i, segment in enumerate(recording.segments):
        percentile = float(order[i]) / n
        overlap = overlap_ratio(segment, recording.segments)
        f0_range = f0_values[i]
        if f0_range > 0 and measured_sorted:
            rank = sum(1 for v in measured_sorted if v < f0_range)
            f0_pct = rank / len(measured_sorted)
        else:
            f0_pct = float("nan")     # unmeasured, never "flat"
        out.append(SegmentFeatures(
            index=i,
            rms_db=levels[i],
            energy_percentile=percentile,
            overlap_ratio=overlap,
            f0_range_semitones=f0_range,
            f0_percentile=f0_pct,
            is_quiet=(percentile <= quiet_percentile
                      and levels[i] <= median_db - quiet_db_drop
                      and overlap < masked_ratio),
            is_masked=overlap >= masked_ratio,
            # Bottom tercile of this recording's own measured pitch ranges.
            # NaN (unmeasured) is never flat -- guessing would manufacture P3.
            is_flat=f0_pct == f0_pct and f0_pct <= flat_percentile,
        ))
    return out


# Lexical markers of self-repair.  Their *absence* is what makes a P4 item
# audio-necessary: "twenty -- twelve milligrams" is ambiguous in text and
# unambiguous in audio, whereas "twenty, sorry, twelve" is solved by a text
# model and gets rejected by the leak filter.
REPAIR_MARKERS = {
    # "<disf>" is AMI's own disfluency marker, carried through from the
    # annotations: it marks the seam of a repair without saying it in words.
    "en": ("sorry", "i mean", "i meant", "no wait", "rather", "correction",
           "make that", "scratch that", "actually no", "<disf>"),
    "hi": ("माफ़ कीजिए", "मेरा मतलब", "नहीं नहीं", "बल्कि", "सुधार"),
    "bn": ("দুঃখিত", "আমি বলতে চেয়েছি", "না না", "বরং", "সংশোধন"),
}

HESITATION_MARKERS = {
    "en": ("um", "uh", "erm", "hmm", "i think", "i guess", "maybe", "not sure",
           "i'm not certain", "sort of", "kind of"),
    "hi": ("शायद", "पता नहीं", "मुझे लगता है", "हो सकता है"),
    "bn": ("হয়তো", "জানি না", "আমার মনে হয়", "হতে পারে"),
}


def has_marker(text: str, lang: str, markers: dict[str, tuple[str, ...]]) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers.get(lang, ()))
