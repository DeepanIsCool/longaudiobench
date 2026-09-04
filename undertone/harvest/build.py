"""Candidate proposal and item assembly.

The pipeline, following the paper plan section 12 but sized for one person
rather than five annotation teams:

    1. propose      automatic, from acoustics + gold segment timings
    2. distractors  automatic, from competing mentions in the same recording
    3. question     templated, hand-edited later for situational plausibility
    4. leak filter  see leakfilter.py -- the audio-necessity gate
    5. verify       you listen to the clip and confirm the key

Only steps 1-3 live here.  Everything this module emits is a *proposal*: an
unverified item, marked as such in its provenance, which is why
``build_item_pack`` writes ``verified: false`` and the analysis refuses to
report unverified cells.

The distractors are the point.  They are not fabrications -- each is a real,
competing mention of the same quantity kind from the same recording, so a wrong
answer says which heuristic the model used rather than merely that it was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..items import MCQItem
from .features import (
    HESITATION_MARKERS,
    REPAIR_MARKERS,
    SegmentFeatures,
    has_marker,
)
from .mentions import Mention, find_mentions, group_by_kind
from .sources import Recording

# How the question names a quantity kind, per language.
KIND_PHRASE = {
    "en": {"currency": "amount of money", "percent": "percentage", "mass": "weight",
           "time_of_day": "time", "duration": "length of time", "count": "number",
           "colour": "colour", "material": "material", "shape": "shape"},
    "hi": {"currency": "रकम", "percent": "प्रतिशत", "mass": "वज़न",
           "time_of_day": "समय", "duration": "अवधि", "count": "संख्या",
           "colour": "रंग", "material": "सामग्री", "shape": "आकार"},
    "bn": {"currency": "পরিমাণ", "percent": "শতাংশ", "mass": "ওজন",
           "time_of_day": "সময়", "duration": "সময়কাল", "count": "সংখ্যা",
           "colour": "রং", "material": "উপাদান", "shape": "আকার"},
}

# "n:<noun>" kinds are open-ended, so the phrase is built from the noun itself.
COUNT_OF = {"en": "number of {noun}", "hi": "{noun} की संख्या",
            "bn": "{noun}-এর সংখ্যা"}


def kind_phrase(lang: str, kind: str) -> str:
    if kind.startswith("n:"):
        return COUNT_OF[lang].format(noun=kind[2:])
    return KIND_PHRASE[lang].get(kind, KIND_PHRASE[lang]["count"])

# Neutral about prominence on purpose: a question that said "the one mentioned
# quietly" would hand the model the manipulation being measured.
QUESTION = {
    "en": 'In the part of the recording about "{context}", what {kind} does the speaker give?',
    "hi": '"{context}" वाले हिस्से में वक्ता कौन सा {kind} बताता है?',
    "bn": '"{context}" অংশে বক্তা কোন {kind} উল্লেখ করেন?',
}

REPAIR_QUESTION = {
    "en": 'The speaker gives a {kind} about "{context}" and then changes it. '
          'Which one do they end up with?',
    "hi": 'वक्ता "{context}" के बारे में एक {kind} बताता है और फिर उसे बदल देता है। '
          'अंत में कौन सा रहता है?',
    "bn": 'বক্তা "{context}" সম্পর্কে একটি {kind} বলেন এবং তারপর সেটি বদলান। '
          'শেষ পর্যন্ত কোনটি থাকে?',
}

REPAIR_WINDOW_SECONDS = 6.0     # a self-repair follows its error closely
MIN_DISTINCT_VALUES = 3         # correct + salience + recency, all different
MARKER_WINDOW_CHARS = 45        # how close a hesitation marker must be to count


def _marker_near(text: str, span: tuple[int, int], lang: str,
                 markers: dict[str, tuple[str, ...]]) -> bool:
    """Is a marker adjacent to the mention, rather than anywhere in the segment?

    Segment-wide matching made C1 useless: 42% of AMI segments contain "um",
    "uh" or "I think" somewhere, so C1 claimed almost everything and P1/P2/P3
    came out empty. A marker three clauses away says nothing about how *this*
    value was delivered.
    """
    lo = max(0, span[0] - MARKER_WINDOW_CHARS)
    hi = min(len(text), span[1] + MARKER_WINDOW_CHARS)
    window = text[lo:hi].lower()
    return any(m in window for m in markers.get(lang, ()))


@dataclass
class Candidate:
    category: str
    target: Mention
    salience: Mention
    recency: Mention
    why: dict


MIN_CONTEXT_TOKENS = 3


def context_is_clean(context: str, *mentions: Mention) -> bool:
    """Reject a context that names any option.

    The first real harvest produced questions anchored on
    "have a bunch of options from , titanium , rubber , plastic" whose answer was
    "wood" -- a model could pick the one not listed without hearing anything, and
    the loud competitor was handed over in the prompt. The context has to
    identify *where* in the recording to listen, never *what* the answer is.
    """
    lowered = context.casefold()
    if len(lowered.split()) < MIN_CONTEXT_TOKENS:
        return False
    return not any(str(m.surface).casefold() in lowered for m in mentions)


def _repetition_count(mentions: Iterable[Mention], value: float) -> int:
    return sum(1 for m in mentions if m.value == value)


def _pick_salience(group: list[Mention], features: list[SegmentFeatures],
                   exclude: set[float]) -> Mention | None:
    """The loudest and/or most repeated competing mention.

    Loudness and repetition are combined rather than ranked, because the paper
    plan's salience prior is both ("loud repeated competing mention") and an item
    where they disagree is a weaker probe than one where they agree.
    """
    best, best_score = None, -1.0
    for mention in group:
        if mention.value in exclude:
            continue
        loudness = features[mention.segment_index].energy_percentile
        repeats = _repetition_count(group, mention.value)
        score = loudness + 0.5 * (repeats - 1)
        if score > best_score:
            best, best_score = mention, score
    return best


def _pick_recency(group: list[Mention], exclude: set[float]) -> Mention | None:
    """The latest competing mention -- the one a recency heuristic lands on."""
    for mention in reversed(group):
        if mention.value not in exclude:
            return mention
    return None


def _categorize(target: Mention, group: list[Mention], recording: Recording,
                features: list[SegmentFeatures]) -> tuple[str, dict] | None:
    segment = recording.segments[target.segment_index]
    feature = features[target.segment_index]
    text = segment.text
    lang = recording.lang

    # P4 first: an unmarked repair is the strongest evidence of intent, and it
    # would otherwise be miscategorised as whatever its acoustics look like.
    for other in group:
        if other is target or other.value == target.value:
            continue
        same_speaker = recording.segments[other.segment_index].speaker == segment.speaker
        gap = target.start - other.start
        if same_speaker and 0 < gap <= REPAIR_WINDOW_SECONDS:
            marked = has_marker(text, lang, REPAIR_MARKERS)
            return "P4", {
                "repaired_from": other.surface,
                "gap_seconds": round(gap, 2),
                # Lexically marked repairs are solved by a text model and will be
                # dropped by the gold-transcript filter; kept here so the leak
                # rate can be reported per subtype rather than silently avoided.
                "lexically_marked": marked,
            }

    # P3 before the other acoustic causes. Its signature is not flatness alone --
    # it is a flat aside *against a loud, repeated competing mention*, which is
    # the conjunction the paper plan names. Checking flatness last produced zero
    # P3 items on the first real harvest: AMI is ~50% overlapped, so every flat
    # aside was claimed as P2 or P1 before P3 was ever reached.
    loudest_repeat = max(
        (_repetition_count(group, m.value) for m in group if m.value != target.value),
        default=0)
    if (feature.is_flat and loudest_repeat >= 2
            and _repetition_count(group, target.value) == 1):
        return "P3", {"f0_range_semitones": round(feature.f0_range_semitones, 2),
                      "competitor_repeats": loudest_repeat}

    # C1 is lexical and is the residual. Ordering it last is what stops a
    # ubiquitous "um" from relabelling a masked or muttered mention -- and C1 has
    # to fail *differently* from P1-P4 or the mechanism claim collapses into
    # "hard things are hard".
    if feature.is_masked:
        return "P2", {"overlap_ratio": round(feature.overlap_ratio, 3)}
    if feature.is_quiet:
        return "P1", {"energy_percentile": round(feature.energy_percentile, 3),
                      "rms_db": round(feature.rms_db, 1)}

    span = (text.find(target.surface), text.find(target.surface) + len(target.surface))
    if span[0] >= 0 and _marker_near(text, span, lang, HESITATION_MARKERS):
        # NOTE: C1 in the paper plan asks about the *delivery* ("was there a
        # point where he sounded unsure?"), not about a value. This asks the
        # same value question with a hesitancy-marked target, which is a weaker
        # instrument. Flagged rather than papered over.
        return "C1", {"marker": "hesitation", "adjacent": True,
                      "question_type": "value_not_delivery"}
    return None


def propose(recording: Recording, features: list[SegmentFeatures],
            band: int) -> list[Candidate]:
    """Candidate items for one banded recording.  Proposals only."""
    mentions: list[Mention] = []
    for i, segment in enumerate(recording.segments):
        mentions.extend(find_mentions(segment.text, recording.lang, i,
                                      segment.start, segment.end))

    out: list[Candidate] = []
    for group in group_by_kind(mentions).values():
        if len({m.value for m in group}) < MIN_DISTINCT_VALUES:
            continue
        for target in group:
            if target.end > band or not target.context:
                continue
            categorized = _categorize(target, group, recording, features)
            if categorized is None:
                continue
            category, why = categorized

            salience = _pick_salience(group, features, exclude={target.value})
            if salience is None:
                continue
            recency = _pick_recency(group, exclude={target.value, salience.value})
            if recency is None:
                continue
            if not context_is_clean(target.context, target, salience, recency):
                continue
            out.append(Candidate(category, target, salience, recency, why))
    return out


def to_item(recording: Recording, candidate: Candidate, band: int,
            index: int) -> MCQItem:
    lang = recording.lang
    kind = kind_phrase(lang, candidate.target.kind)
    template = REPAIR_QUESTION if candidate.category == "P4" else QUESTION
    question = template[lang].format(context=candidate.target.context, kind=kind)

    # A needle needs enough span to be listenable in the L1 window; the mention
    # itself can be under a second.
    start = max(0.0, candidate.target.start - 1.0)
    end = min(float(band), max(candidate.target.end + 1.0, start + 2.0))

    return MCQItem(
        item_id=f"{recording.recording_id}_{candidate.category}_{index:03d}",
        recording_id=recording.recording_id,
        lang=lang,
        category=candidate.category,
        sector=recording.sector,
        audio_path=recording.audio_path,
        duration_band=band,
        needle_start=start,
        needle_end=end,
        question=question,
        options={
            "correct": candidate.target.surface,
            "salience": candidate.salience.surface,
            "recency": candidate.recency.surface,
            "absent": "placeholder",     # protocol.render substitutes the real text
        },
        is_null=False,
        provenance={
            "verified": False,           # nothing is a real item until you listen
            "leak_checked": False,
            "why": candidate.why,
            "quantity_kind": candidate.target.kind,
            "salience_at": round(candidate.salience.start, 2),
            "recency_at": round(candidate.recency.start, 2),
            "source_recording": recording.meta.get("source_recording",
                                                   recording.recording_id),
        },
    )


def absent_kinds(recording_kinds: set[str]) -> list[str]:
    """Quantity kinds this recording never mentions -- the basis for null items."""
    return [k for k in KIND_PHRASE["en"] if k not in recording_kinds]


def make_null(item: MCQItem, context: str, kind: str) -> MCQItem:
    """A null twin: same recording and context, a quantity kind never mentioned.

    Null items are what stop "not mentioned in the recording" from being a free
    wrong answer, and they are the only place a model can be scored for
    abstaining correctly. The distractors stay -- they are still real mentions,
    just of the wrong kind, which is exactly the trap.
    """
    data = item.to_dict()
    data.update(
        item_id=item.item_id + "_null",
        question=QUESTION[item.lang].format(context=context,
                                            kind=kind_phrase(item.lang, kind)),
        is_null=True,
        provenance={**item.provenance, "null_of": item.item_id,
                    "absent_kind": kind},
    )
    return MCQItem.from_dict(data)
