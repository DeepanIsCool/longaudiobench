"""Finding the values an item can be about.

An item asks which figure was given.  For that to have a wrong-but-plausible
distractor set, the recording has to contain *several* figures of the same kind:
one stated in a low-prominence way (the answer), one stated loudly or repeated
(the salience distractor), and whichever was said last (the recency distractor).

So mentions are grouped by **quantity kind** -- milligrams with milligrams,
percentages with percentages -- and an item is only possible where a group holds
at least three distinct values.  That constraint is doing real work: it is why
the distractors are competing mentions from the same recording rather than
fabrications, which is what makes a wrong answer diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DIGITS = {
    "en": "0123456789",
    "hi": "0123456789०१२३४५६७८९",
    "bn": "0123456789০১২৩৪৫৬৭৮৯",
}

_DIGIT_VALUE = {}
for _base, _chars in (("0", "0123456789"), ("०", "०१२३४५६७८९"), ("০", "০১২৩৪৫৬৭৮৯")):
    for _i, _ch in enumerate(_chars):
        _DIGIT_VALUE[_ch] = str(_i)

NUMBER_WORDS = {
    "en": {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    },
    "hi": {
        "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6,
        "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "बारह": 12, "पंद्रह": 15,
        "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "सौ": 100, "हज़ार": 1000,
    },
    "bn": {
        "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7,
        "আট": 8, "নয়": 9, "দশ": 10, "বারো": 12, "পনেরো": 15, "বিশ": 20,
        "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50, "একশ": 100, "হাজার": 1000,
    },
}

# An item needs a set of competing, mutually exclusive alternatives said about
# the same attribute.  Numbers with units are the cleanest instance, but they are
# rare in natural speech: a 21-minute AMI meeting yields about eleven, of which
# one kind has three distinct values.  So three mention types are recognised, in
# descending precision:
#
#   1. number + unit          "twelve milligrams"   -> kind = the unit class
#   2. number + head noun     "two buttons"         -> kind = "n:<noun>"
#   3. closed-class term      "red", "plastic"      -> kind = the class
#
# (2) and (3) are what make the pipeline work on real meeting speech at all.
UNITS = {
    "currency": r"(?:dollars?|euros?|pounds?|usd|eur|gbp|रुपये|रुपए|টাকা|\$|€|£|₹)",
    "percent": r"(?:percent|per cent|%|प्रतिशत|শতাংশ)",
    "mass": r"(?:milligrams?|mg|grams?|g|kilograms?|kg|मिलीग्राम|ग्राम|মিলিগ্রাম|গ্রাম)",
    "time_of_day": r"(?:o'clock|am|pm|बजे|টায়|টার)",
    "duration": r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
                r"मिनट|घंटे|दिन|साल|মিনিট|ঘণ্টা|দিন|বছর)",
    "count": r"(?:units?|items?|pieces?|people|copies|टुकड़े|लोग|জন|টি)",
}


# Head nouns that quantify nothing useful: grouping on these would put
# "two minutes" and "two people" in the same competing set.
_STOP_HEADS = {
    "en": {"of", "the", "a", "an", "and", "or", "is", "was", "to", "for", "in",
           "on", "at", "it", "that", "this", "there", "here", "so", "but", "we",
           "you", "i", "he", "she", "they", "thing", "things", "one", "ones"},
    "hi": {"का", "की", "के", "है", "हैं", "को", "में", "से", "और", "यह", "वह"},
    "bn": {"এর", "এই", "সেই", "এবং", "আছে", "থেকে", "করে", "হয়", "টা", "টি"},
}

# Closed-class alternatives that behave exactly like competing values: a design
# meeting picks one colour and rejects others, which is the same structure as
# picking one dose. High yield in AMI, and trivially checkable by a listener.
CATEGORICAL = {
    "colour": {
        "en": {"red", "blue", "green", "yellow", "black", "white", "orange",
               "purple", "grey", "gray", "brown", "pink", "silver", "gold"},
        "hi": {"लाल", "नीला", "हरा", "पीला", "काला", "सफ़ेद", "नारंगी", "गुलाबी"},
        "bn": {"লাল", "নীল", "সবুজ", "হলুদ", "কালো", "সাদা", "কমলা", "গোলাপি"},
    },
    "material": {
        "en": {"plastic", "rubber", "metal", "wood", "wooden", "titanium",
               "steel", "aluminium", "aluminum", "glass", "leather", "fabric"},
        "hi": {"प्लास्टिक", "रबर", "धातु", "लकड़ी", "स्टील", "काँच", "चमड़ा"},
        "bn": {"প্লাস্টিক", "রাবার", "ধাতু", "কাঠ", "স্টিল", "কাচ", "চামড়া"},
    },
    "shape": {
        "en": {"square", "round", "curved", "flat", "rectangular", "oval",
               "triangular", "circular", "straight"},
        "hi": {"चौकोर", "गोल", "घुमावदार", "सपाट", "आयताकार", "अंडाकार"},
        "bn": {"চৌকো", "গোল", "বাঁকা", "সমতল", "আয়তাকার", "ডিম্বাকার"},
    },
}


@dataclass(frozen=True)
class Mention:
    value: float | str
    surface: str        # exactly as spoken, for the option text
    kind: str           # one of UNITS
    unit: str           # the unit token as it appeared
    start: float        # seconds into the recording
    end: float
    segment_index: int
    context: str        # nearby content words, used to anchor the question


def _to_float(token: str) -> float | None:
    normalized = "".join(_DIGIT_VALUE.get(ch, ch) for ch in token)
    normalized = normalized.replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _word_value(word: str, lang: str) -> float | None:
    table = NUMBER_WORDS.get(lang, {})
    value = table.get(word.lower() if lang == "en" else word)
    return float(value) if value is not None else None


def _context_words(text: str, span: tuple[int, int], width: int = 6) -> str:
    """Content words either side of the mention, minus the number itself.

    This anchors the question ("...in connection with X, what figure?") so the
    answer is unique, without naming the prominence -- a question that said
    "the one said quietly" would hand the model the manipulation.
    """
    before = text[: span[0]].split()[-width:]
    after = text[span[1]:].split()[:width]
    return " ".join(before + after).strip()


def find_mentions(text: str, lang: str, segment_index: int,
                  start: float, end: float) -> list[Mention]:
    """Numeric mentions with a unit, located within one segment.

    Timing is linear within the segment: the exact word offset would need forced
    alignment, and for a 20 s L1 window a proportional estimate is inside the
    tolerance.  ponytail: swap in word-level alignment if L1 windows start
    missing needles.
    """
    digits = DIGITS.get(lang, DIGITS["en"])
    words = "|".join(re.escape(w) for w in NUMBER_WORDS.get(lang, {}))
    number = rf"(?:[{digits}][{digits},.]*" + (f"|{words}" if words else "") + ")"

    out: list[Mention] = []
    seen: set[tuple[int, int]] = set()
    covered: list[tuple[int, int]] = []
    for kind, unit_pattern in UNITS.items():
        pattern = re.compile(rf"({number})\s*({unit_pattern})", re.IGNORECASE)
        for match in pattern.finditer(text):
            if match.span() in seen:
                continue
            token = match.group(1)
            value = _to_float(token)
            if value is None:
                value = _word_value(token, lang)
            if value is None:
                continue
            seen.add(match.span())
            covered.append(match.span())
            fraction = match.start() / max(1, len(text))
            at = start + fraction * (end - start)
            out.append(Mention(
                value=value,
                surface=match.group(0).strip(),
                kind=kind,
                unit=match.group(2),
                start=at,
                end=min(end, at + 1.5),
                segment_index=segment_index,
                context=_context_words(text, match.span()),
            ))

    def uncovered(mention: Mention) -> bool:
        """Drop a "two buttons" hit that a "two milligrams" hit already covers."""
        return not any(mention.surface in text[a:b] for a, b in covered)

    out += [m for m in find_quantified(text, lang, segment_index, start, end)
            if uncovered(m)]
    out += find_categorical(text, lang, segment_index, start, end)
    return out


_HEAD_RE_CACHE: dict[str, "re.Pattern"] = {}


def _number_pattern(lang: str) -> str:
    digits = DIGITS.get(lang, DIGITS["en"])
    words = "|".join(re.escape(w) for w in NUMBER_WORDS.get(lang, {}))
    return rf"(?:[{digits}][{digits},.]*" + (f"|{words}" if words else "") + ")"


def find_quantified(text: str, lang: str, segment_index: int,
                    start: float, end: float) -> list[Mention]:
    """number + head noun, e.g. "two buttons" -> kind ``n:buttons``.

    Grouping by head noun is what gives competing alternatives in ordinary
    speech: "two buttons" and "six buttons" compete; "two buttons" and "two
    minutes" do not.
    """
    pattern = re.compile(rf"({_number_pattern(lang)})\s+([^\s,.;:!?]+)", re.IGNORECASE)
    stops = _STOP_HEADS.get(lang, set())
    out: list[Mention] = []
    for match in pattern.finditer(text):
        head = match.group(2).strip(".,;:!?").lower()
        if not head or head in stops or len(head) < 3:
            continue
        value = _to_float(match.group(1))
        if value is None:
            value = _word_value(match.group(1), lang)
        if value is None:
            continue
        fraction = match.start() / max(1, len(text))
        at = start + fraction * (end - start)
        out.append(Mention(value=value, surface=match.group(0).strip(),
                           kind=f"n:{head}", unit=head, start=at,
                           end=min(end, at + 1.5), segment_index=segment_index,
                           context=_context_words(text, match.span())))
    return out


def find_categorical(text: str, lang: str, segment_index: int,
                     start: float, end: float) -> list[Mention]:
    """Closed-class alternatives (colour, material, shape)."""
    out: list[Mention] = []
    for kind, tables in CATEGORICAL.items():
        vocabulary = tables.get(lang, set())
        if not vocabulary:
            continue
        pattern = re.compile(r"(?<!\w)(" + "|".join(
            re.escape(w) for w in sorted(vocabulary, key=len, reverse=True)) + r")(?!\w)",
            re.IGNORECASE)
        for match in pattern.finditer(text):
            term = match.group(1).lower()
            fraction = match.start() / max(1, len(text))
            at = start + fraction * (end - start)
            out.append(Mention(value=term, surface=match.group(1), kind=kind,
                               unit=kind, start=at, end=min(end, at + 1.5),
                               segment_index=segment_index,
                               context=_context_words(text, match.span())))
    return out


def group_by_kind(mentions: list[Mention]) -> dict[str, list[Mention]]:
    groups: dict[str, list[Mention]] = {}
    for mention in mentions:
        groups.setdefault(mention.kind, []).append(mention)
    return {k: sorted(v, key=lambda m: m.start) for k, v in groups.items()}


def distinct_values(mentions: list[Mention]) -> int:
    return len({m.value for m in mentions})
