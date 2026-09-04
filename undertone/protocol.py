"""Prompt construction and per-run option shuffling.

Two invariants the rest of the codebase leans on:

1.  Option *letters* are drawn per run from ``(item_id, seed)``, so a model
    cannot be right by preferring a letter.  What is scored is the **role** the
    model picked, which is what makes the salience-trap rate measurable at all.
2.  Every prompt ends on the answer cue, so the next-token distribution is the
    answer distribution.  That is what ``scoring.letter_logits`` reads, and it
    removes the format-following confound that made the retired regex parsers
    score parse failures as zeros.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .items import LETTERS, ROLES, MCQItem
from .ladder import Window

# Kept deliberately plain: three phrasings are swept later to kill the
# "prompting artifact" objection, and a baroque default would confound that.
INSTRUCTION = {
    "en": (
        "Listen to the recording, then answer the question about it. "
        "Reply with the letter of the correct option and nothing else."
    ),
    "hi": (
        "रिकॉर्डिंग सुनें और उसके बारे में पूछे गए प्रश्न का उत्तर दें। "
        "केवल सही विकल्प का अक्षर लिखें, और कुछ नहीं।"
    ),
    "bn": (
        "রেকর্ডিংটি শুনুন এবং সেটি সম্পর্কে জিজ্ঞাসিত প্রশ্নের উত্তর দিন। "
        "শুধুমাত্র সঠিক বিকল্পের অক্ষরটি লিখুন, অন্য কিছু নয়।"
    ),
}

ABSENT_OPTION = {
    "en": "Not mentioned in the recording",
    "hi": "रिकॉर्डिंग में इसका उल्लेख नहीं है",
    "bn": "রেকর্ডিংয়ে এর উল্লেখ নেই",
}

ORACLE_HINT = {
    "en": "The answer is somewhere between {start} and {end} in the recording.",
    "hi": "उत्तर रिकॉर्डिंग में {start} और {end} के बीच कहीं है।",
    "bn": "উত্তরটি রেকর্ডিংয়ের {start} থেকে {end} এর মধ্যে কোথাও আছে।",
}

ANSWER_CUE = {"en": "Answer:", "hi": "उत्तर:", "bn": "উত্তর:"}


def mmss(seconds: float) -> str:
    """``MM:SS`` for under an hour, ``H:MM:SS`` beyond it.

    The retired code emitted ``60:00`` for one hour and then parsed it back with
    ``\\d{1,2}:\\d{2}``, so an hour-long timestamp silently became 60 minutes of
    something else.  Bands top out at 30 min today; the hour branch exists so
    that stops being a latent trap if bands grow.
    """
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass(frozen=True)
class Rendered:
    prompt: str
    letter_to_role: dict[str, str]
    role_to_letter: dict[str, str]

    @property
    def letters(self) -> list[str]:
        return list(LETTERS)


def assign_letters(item: MCQItem, seed: int) -> dict[str, str]:
    """Deterministic role -> letter map for one item under one run seed."""
    roles = list(ROLES)
    random.Random(f"{seed}:{item.item_id}").shuffle(roles)
    return {role: letter for role, letter in zip(roles, LETTERS)}


def render(item: MCQItem, window: Window, seed: int) -> Rendered:
    lang = item.lang
    role_to_letter = assign_letters(item, seed)
    letter_to_role = {v: k for k, v in role_to_letter.items()}

    option_text = dict(item.options)
    # The absent option is protocol, not item content: force the canonical
    # phrasing so "D" never becomes identifiable by wording drift.
    option_text["absent"] = ABSENT_OPTION[lang]

    lines = [INSTRUCTION[lang], ""]
    if window.oracle:
        lines.append(
            ORACLE_HINT[lang].format(
                start=mmss(item.needle_start), end=mmss(item.needle_end)
            )
        )
        lines.append("")
    lines.append(item.question)
    lines.append("")
    for letter in LETTERS:
        lines.append(f"{letter}. {option_text[letter_to_role[letter]]}")
    lines.append("")
    lines.append(ANSWER_CUE[lang])

    return Rendered(
        prompt="\n".join(lines),
        letter_to_role=letter_to_role,
        role_to_letter=role_to_letter,
    )


def question_only(item: MCQItem, seed: int) -> Rendered:
    """The no-audio control.  Must land at ~25% or the distractors leak."""
    return render(item, Window(0.0, 0.0, oracle=False), seed)
