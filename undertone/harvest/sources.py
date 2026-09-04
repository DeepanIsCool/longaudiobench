"""Recordings and their transcripts, before any item exists.

A ``Recording`` is a real continuous audio file plus its gold segmentation.
Nothing here inserts, splices or synthesises: every needle an item asks about is
a span that a real person really produced in that file.  That is the whole
difference from the retired suite, whose labels were drawn with
``random.choice`` before the audio existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Six sectors, partially crossed with language (paper plan section 7).  Stated
# up front rather than discovered: some language x sector cells will be thin.
SECTORS = ("meetings", "earnings", "news", "tutorials", "interviews", "service")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "Segment") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Recording:
    recording_id: str
    audio_path: str
    lang: str
    sector: str
    duration: float
    segments: list[Segment]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.segments = sorted(self.segments, key=lambda s: (s.start, s.end))

    @property
    def speakers(self) -> set[str]:
        return {s.speaker for s in self.segments}

    def within(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments if s.start < end and start < s.end]

    def band(self, band_seconds: int) -> "Recording":
        """The first ``band_seconds`` of this recording, segments clipped to fit.

        Bands are 5/10/20/30 min.  Taking a prefix rather than a random excerpt
        keeps the recording's own opening context intact, which matters for the
        recency distractor -- "most recent mention" has to mean something.
        """
        kept = [s for s in self.segments if s.end <= band_seconds]
        return Recording(
            recording_id=f"{self.recording_id}_b{band_seconds}",
            audio_path=self.audio_path,
            lang=self.lang,
            sector=self.sector,
            duration=float(band_seconds),
            segments=kept,
            meta={**self.meta, "band": band_seconds, "source_recording": self.recording_id},
        )


# --------------------------------------------------------------------------
# AMI
# --------------------------------------------------------------------------

AMI_AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "{meeting}/audio/{meeting}.Mix-Headset.wav"
)

# Measured yield on AMI: about 104 mentions per meeting collapse to ~51 attribute
# groups, of which roughly **one** has the three distinct competing values an
# item needs -- around 3.5 usable proposals per meeting. Reaching 180 items
# therefore takes roughly 50 meetings (~25 h of audio, ~7 GB of Mix-Headset wav).
#
# That is the paper plan's own "P3 natural yield below usable rate" risk, now
# with a number attached. The scenario meetings (ES/IS/TS) are remote-control
# design sessions: four speakers, heavy overlap, and constant argument about
# colours, materials and button counts, which is what produces competing values
# at all. EN* are naturally-occurring meetings and yield less.
def _ami_series(prefix: str, first: int, last: int, parts: str = "abcd") -> tuple[str, ...]:
    return tuple(f"{prefix}{n}{p}" for n in range(first, last + 1) for p in parts)


AMI_SCENARIO_MEETINGS = (
    _ami_series("ES", 2002, 2016)      # Edinburgh
    + _ami_series("IS", 1000, 1009)    # Idiap
    + _ami_series("TS", 3003, 3012)    # TNO
)

# A small default so an exploratory run is minutes, not hours. Pass
# --meetings "$(python -c 'from undertone.harvest.sources import
# AMI_SCENARIO_MEETINGS as m; print(" ".join(m))')" for the full harvest.
AMI_DEFAULT_MEETINGS = AMI_SCENARIO_MEETINGS[:15]


AMI_ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)

# Pause long enough to count as a turn boundary when grouping words. AMI's own
# segment layer exists, but it points at words through NITE stand-off links that
# need a second parse; grouping on silence gets the same spans for a fraction of
# the code.
TURN_GAP_SECONDS = 0.5


def download_ami_annotations(dest_dir: str) -> str:
    """Fetch and unpack the AMI manual annotations (~30 MB, CC-BY-4.0).

    This replaces streaming ``edinburghcstr/ami``: that parquet carries audio
    bytes on every row, so finding two meetings meant downloading a large
    fraction of a 100-hour corpus to throw the audio away. The annotation zip is
    30 MB, has exact word-level timings, and works offline afterwards.
    """
    import os
    import urllib.request
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    marker = os.path.join(dest_dir, "words")
    if os.path.isdir(marker):
        return dest_dir

    archive = os.path.join(dest_dir, "ami_public_manual.zip")
    if not os.path.exists(archive):
        urllib.request.urlretrieve(AMI_ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    return dest_dir


def _ami_words(path: str) -> list[tuple[float, float, str, bool]]:
    """(start, end, token, is_disfluency_marker) from one AMI words XML file."""
    import xml.etree.ElementTree as ET

    out: list[tuple[float, float, str, bool]] = []
    root = ET.parse(path).getroot()
    for node in root:
        tag = node.tag.split("}")[-1]
        start, end = node.get("starttime"), node.get("endtime")
        if start is None or end is None:
            continue
        try:
            start_f, end_f = float(start), float(end)
        except ValueError:
            continue
        if tag == "w":
            text = (node.text or "").strip()
            if text:
                out.append((start_f, end_f, text, False))
        elif tag == "disfmarker":
            # AMI marks interruption points explicitly. These are the seam of a
            # self-repair, and P4's audio-necessity depends on whether the repair
            # is *lexically* marked as well -- so the marker is carried through
            # rather than dropped.
            out.append((start_f, end_f, "<disf>", True))
    return sorted(out, key=lambda w: w[0])


def ami_segments_from_annotations(meeting_ids: Iterable[str],
                                  annotations_dir: str) -> dict[str, list[Segment]]:
    """Segment tables for the named meetings, from the manual annotations."""
    import glob
    import os

    out: dict[str, list[Segment]] = {}
    for meeting in meeting_ids:
        segments: list[Segment] = []
        pattern = os.path.join(annotations_dir, "words", f"{meeting}.*.words.xml")
        for path in sorted(glob.glob(pattern)):
            speaker = os.path.basename(path).split(".")[1]
            words = _ami_words(path)
            if not words:
                continue

            current: list[tuple[float, float, str, bool]] = []
            for word in words:
                if current and word[0] - current[-1][1] > TURN_GAP_SECONDS:
                    segments.append(_as_segment(current, speaker))
                    current = []
                current.append(word)
            if current:
                segments.append(_as_segment(current, speaker))

        if segments:
            out[meeting] = sorted(segments, key=lambda s: s.start)
    return out


def _as_segment(words: list[tuple[float, float, str, bool]], speaker: str) -> Segment:
    text = " ".join(w[2] for w in words if not w[3])
    if any(w[3] for w in words):
        # Kept in the text so the repair detector can see it and the leak filter
        # can tell a marked repair from an unmarked one.
        text += " <disf>"
    return Segment(words[0][0], words[-1][1], speaker, text.strip())


def ami_segments_from_hf(meeting_ids: Iterable[str], split: str = "train") -> dict[str, list[Segment]]:
    """Deprecated fallback: stream ``edinburghcstr/ami`` for the same tables.

    Kept only because it needs no separate download. It streams audio bytes it
    immediately discards, so finding a handful of meetings costs a large
    fraction of the corpus -- prefer ``ami_segments_from_annotations``.
    """
    from datasets import load_dataset

    wanted = set(meeting_ids)
    out: dict[str, list[Segment]] = {mid: [] for mid in wanted}

    stream = load_dataset("edinburghcstr/ami", "ihm", split=split, streaming=True)
    for row in stream:
        meeting = str(row.get("meeting_id") or row.get("audio_id", "")).split(".")[0][:7]
        if meeting not in wanted:
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        out[meeting].append(Segment(
            start=float(row["begin_time"]),
            end=float(row["end_time"]),
            speaker=str(row["speaker_id"]),
            text=text,
        ))
    return {mid: sorted(segs, key=lambda s: s.start) for mid, segs in out.items() if segs}


def download_ami_audio(meeting: str, dest_dir: str) -> str:
    """Fetch one meeting's Mix-Headset wav (CC-BY-4.0). ~150 MB for an hour."""
    import os
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{meeting}.Mix-Headset.wav")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(AMI_AUDIO_URL.format(meeting=meeting), dest)
    return dest


# --------------------------------------------------------------------------
# YODAS2 -- the multilingual long-form backbone
# --------------------------------------------------------------------------

# YODAS2 (CC-BY-3.0) is the long-form cut of YODAS: one row per whole video,
# unsegmented, at 24 kHz, with caption timings. That "unsegmented" is the reason
# it is here -- almost every other multilingual corpus ships pre-cut utterances,
# and an item has to point at a span of one continuous recording.
YODAS2_CONFIGS = {
    "hi": ("hi000", "hi001"),
    "bn": ("bn000", "bn001"),
    "en": ("en000",),
}

# Captions are YouTube's, so they are noisy. Two consequences, both stated
# rather than worked around:
#   * candidate proposal gets noisier -- which is what the verification pass is
#     for, and why nothing is an item until a person has listened to it;
#   * the "gold" transcript tier is not truly gold for hi/bn. The leak-filter
#     table reports this per language, because a weaker gold tier makes the
#     gold-leak rate an underestimate rather than a free pass.
YODAS_TRANSCRIPT_QUALITY = "youtube-captions"


def yodas_recordings(
    lang: str,
    sector: str = "tutorials",
    limit: int = 10,
    min_seconds: float = 320.0,
    max_seconds: float = 2400.0,
    audio_dir: str = "data/yodas_audio",
    configs: tuple[str, ...] | None = None,
) -> list[Recording]:
    """Long-form YODAS2 videos as ``Recording`` objects, audio written to disk.

    Streams rather than downloading a shard: a single YODAS2 config is far
    larger than Kaggle's disk, and we want a handful of long videos, not a
    corpus.
    """
    import os

    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    os.makedirs(audio_dir, exist_ok=True)
    out: list[Recording] = []

    for config in (configs or YODAS2_CONFIGS.get(lang, ())):
        if len(out) >= limit:
            break
        stream = load_dataset("espnet/yodas2", config, split="train", streaming=True)
        for row in stream:
            if len(out) >= limit:
                break
            duration = float(row.get("duration") or 0.0)
            if not (min_seconds <= duration <= max_seconds):
                continue

            segments = _yodas_segments(row)
            if len(segments) < 20:          # too sparse to hold competing mentions
                continue

            video_id = str(row["video_id"])
            path = os.path.join(audio_dir, f"{video_id}.flac")
            if not os.path.exists(path):
                audio = np.asarray(row["audio"]["array"], dtype=np.float32)
                sr = int(row["audio"]["sampling_rate"])
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sf.write(path, audio, 16000, format="FLAC")

            out.append(Recording(
                recording_id=f"yodas_{lang}_{video_id}",
                audio_path=path,
                lang=lang,
                sector=sector,
                duration=duration,
                segments=segments,
                meta={"config": config, "video_id": video_id,
                      "transcript_quality": YODAS_TRANSCRIPT_QUALITY},
            ))
    return out


def _yodas_segments(row: dict) -> list[Segment]:
    """Caption spans from a YODAS2 row.

    The ``utterances`` field has appeared both as a dict of parallel lists and
    as a list of dicts across YODAS2 revisions, so both are handled -- guessing
    one would break silently on the other and yield an empty segment table,
    which reads exactly like "this video had no speech".
    """
    utterances = row.get("utterances")
    if not utterances:
        return []

    if isinstance(utterances, dict):
        texts = utterances.get("text") or []
        starts = utterances.get("start") or utterances.get("begin_time") or []
        ends = utterances.get("end") or utterances.get("end_time") or []
        records = zip(starts, ends, texts)
    else:
        records = ((u.get("start", u.get("begin_time")), u.get("end", u.get("end_time")),
                    u.get("text", "")) for u in utterances)

    segments: list[Segment] = []
    for start, end, text in records:
        text = (text or "").strip()
        if not text or start is None or end is None or float(end) <= float(start):
            continue
        # YODAS2 carries no diarisation. A single speaker label means the
        # overlap feature is always zero, so P2 is unavailable for these
        # recordings -- stated here rather than discovered from an empty cell.
        segments.append(Segment(float(start), float(end), "spk", text))
    return sorted(segments, key=lambda s: s.start)


def available_categories(recording: Recording) -> set[str]:
    """Which prominence categories this source can express at all.

    A source without speaker labels cannot produce P2, and reporting a P2 cell
    of n=0 as "no items found" would hide the fact that the source could never
    have produced one.
    """
    if len(recording.speakers) <= 1:
        return {"P1", "P3", "P4", "C1"}
    return {"P1", "P2", "P3", "P4", "C1"}
