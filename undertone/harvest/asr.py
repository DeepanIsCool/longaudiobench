"""ASR transcripts: the cascaded control and the audio-necessity evidence.

Two jobs, and the second is the interesting one.

**As the leak filter's second tier.** P1/P2/P3 are claims about cascaded
ASR+LLM systems, so they are gated on what an ASR transcript actually contains,
not on a reference transcript that has already solved the acoustic problem by
construction. See ``leakfilter.GATE``.

**As direct evidence.** ``needle_recovery`` asks a sharper question than "did a
text model get it right": *did the ASR transcribe the answer at all?* If Whisper
never wrote "five milligrams" anywhere near the needle, no downstream text model
could have found it, and the item is audio-necessary as a matter of fact rather
than as an inference from one model's score. That number goes in the paper.

faster-whisper rather than openai-whisper: the retired pipeline spent most of a
22-minute-per-task run inside large-v3 on a single T4, and CTranslate2 is several
times quicker for identical output. Decoding is greedy with a fixed beam so two
runs of the same audio give the same transcript -- the retired suite's unseeded
runs diverged by 2x on the same nominal config and nothing could be compared.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "large-v3-turbo"

# Whisper's language codes for the roster. Bengali is "bn", Hindi "hi".
WHISPER_LANG = {"en": "en", "hi": "hi", "bn": "bn"}


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    recording_id: str
    model: str
    lang: str
    text: str
    segments: list[ASRSegment] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def between(self, start: float, end: float, pad: float = 2.0) -> str:
        """Transcript text overlapping a time window, with a little slack.

        The pad exists because ASR segment boundaries drift by a second or so;
        without it a needle sitting on a boundary would look un-transcribed when
        it was merely split.
        """
        lo, hi = start - pad, end + pad
        return " ".join(s.text for s in self.segments
                        if s.start < hi and lo < s.end).strip()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return cls(
            recording_id=d["recording_id"], model=d["model"], lang=d["lang"],
            text=d["text"],
            segments=[ASRSegment(**s) for s in d.get("segments", [])],
            meta=d.get("meta", {}),
        )


def transcribe(
    audio_path: str | Path,
    recording_id: str,
    lang: str,
    cache_dir: str | Path = "data/asr_cache",
    model: str = DEFAULT_MODEL,
    duration: float | None = None,
    beam_size: int = 1,
) -> Transcript:
    """Transcribe one recording, caching the result.

    Transcribing a 30-minute recording is minutes of GPU; doing it twice because
    a notebook restarted is quota that a sweep needs. The cache key includes the
    model, so switching models does not silently reuse an old transcript.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{recording_id}.{model}.json"
    if cached.exists():
        return Transcript.from_dict(json.loads(cached.read_text(encoding="utf-8")))

    from faster_whisper import WhisperModel

    engine = _engine(model)
    segments, info = engine.transcribe(
        str(audio_path),
        language=WHISPER_LANG.get(lang, lang),
        beam_size=beam_size,
        # Deterministic: no temperature fallback, no sampling. Two runs of the
        # same audio must produce the same transcript or nothing downstream is
        # reproducible.
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,      # VAD would drop exactly the quiet needles we study
    )
    collected = [ASRSegment(float(s.start), float(s.end), s.text.strip())
                 for s in segments if s.text and s.text.strip()]

    transcript = Transcript(
        recording_id=recording_id, model=model, lang=lang,
        text=" ".join(s.text for s in collected).strip(),
        segments=collected,
        meta={"detected_language": getattr(info, "language", None),
              "duration": getattr(info, "duration", duration),
              "beam_size": beam_size},
    )
    cached.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False),
                      encoding="utf-8")
    return transcript


_ENGINES: dict[str, Any] = {}


def _engine(model: str):
    """One engine per model, reused. Loading large-v3-turbo is ~30 s."""
    if model not in _ENGINES:
        from faster_whisper import WhisperModel

        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        _ENGINES[model] = WhisperModel(
            model,
            device="cuda" if cuda else "cpu",
            # float16 on GPU; int8 on CPU, where float16 is unimplemented.
            compute_type="float16" if cuda else "int8",
        )
    return _ENGINES[model]


def unload() -> None:
    _ENGINES.clear()


# --------------------------------------------------------------------------
# audio-necessity evidence
# --------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.casefold()).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def recovered(transcript: Transcript, item, pad: float = 2.0) -> bool:
    """Did the ASR transcribe the correct answer anywhere near the needle?

    Token-level containment rather than exact string match: "five milligrams"
    and "5 mg" are the same answer, and holding ASR to a surface form would
    overstate audio-necessity.
    """
    window = _tokens(transcript.between(item.needle_start, item.needle_end, pad))
    if not window:
        return False
    wanted = _tokens(item.options["correct"])
    if not wanted:
        return False
    # A digit form counts too: whisper writes "5 mg" where the reference says
    # "five milligrams".
    joined = " ".join(window)
    return all(token in window for token in wanted) or " ".join(wanted) in joined


def needle_recovery(transcripts: dict[str, Transcript], items: Sequence) -> list[dict]:
    """Per-category recovery rate -- the audio-necessity table.

    A low rate for P1/P2 is the finding: the cascaded system provably could not
    have answered, because the words were never written down. That is stronger
    evidence than any single text model's score.
    """
    from ..items import CATEGORIES

    rows = []
    for category in CATEGORIES:
        subset = [i for i in items
                  if i.category == category and i.recording_id in transcripts]
        if not subset:
            continue
        hits = sum(1 for i in subset if recovered(transcripts[i.recording_id], i))
        rows.append({
            "category": category, "n": len(subset),
            "recovered": hits,
            "recovery_rate": round(hits / len(subset), 3),
            # The complement is the share of items no cascaded system could
            # answer regardless of how good its language model is.
            "unrecoverable_rate": round(1 - hits / len(subset), 3),
        })
    return rows


def transcripts_for(pack, audio_root: str | Path = ".", **kw) -> dict[str, Transcript]:
    """One transcript per distinct recording in a pack."""
    audio_root = Path(audio_root)
    out: dict[str, Transcript] = {}
    for item in pack:
        if item.recording_id in out:
            continue
        path = Path(item.audio_path)
        if not path.is_absolute():
            path = audio_root / path
        out[item.recording_id] = transcribe(
            path, item.recording_id, item.lang,
            duration=float(item.duration_band), **kw)
    return out


def as_text(transcripts: dict[str, Transcript]) -> dict[str, str]:
    """The shape ``leakfilter.run_filter`` wants for its ASR tier."""
    return {rid: t.text for rid, t in transcripts.items()}
