"""Harvesting items from real recordings.

Nothing in this package inserts, splices or synthesises audio.  Every item points
at a span a real person really produced, which is the one thing the retired
LongAudioBench tasks could not say about their labels.

    sources     recordings and their gold segmentations (AMI first)
    mentions    the competing figures that become the four options
    features    the acoustics that decide which prominence category applies
    build       candidate proposal and item assembly
    leakfilter  the two-tier audio-necessity gate
"""

from .asr import (
    Transcript,
    as_text,
    needle_recovery,
    recovered,
    transcribe,
    transcripts_for,
)
from .build import Candidate, propose, to_item
from .features import SegmentFeatures, segment_features
from .leakfilter import GATE, LeakReport, apply_filter, run_filter
from .mentions import Mention, find_mentions, group_by_kind
from .sources import (
    SECTORS,
    ami_segments_from_annotations,
    download_ami_annotations,
    YODAS2_CONFIGS,
    Recording,
    Segment,
    available_categories,
    yodas_recordings,
)

__all__ = [
    "Transcript", "as_text", "needle_recovery", "recovered",
    "transcribe", "transcripts_for",
    "Candidate", "propose", "to_item",
    "SegmentFeatures", "segment_features",
    "GATE", "LeakReport", "apply_filter", "run_filter",
    "Mention", "find_mentions", "group_by_kind",
    "SECTORS", "YODAS2_CONFIGS", "Recording", "Segment",
    "ami_segments_from_annotations", "download_ami_annotations",
    "available_categories", "yodas_recordings",
]
