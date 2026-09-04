"""The audio-necessity gate.

The paper plan rejects any item a text model solves from the transcript.  Taken
literally against *gold* transcripts that would reject nearly every P1/P2/P3
item, because a perfect transcript of a muttered or overlapped utterance
contains the answer by construction -- the whole difficulty was acoustic, and a
gold transcript has already solved it.

So the filter runs in two tiers and each category is gated on the tier that
matches what it actually claims:

    tier      transcript                 gates          claim it defends
    ------    -----------------------    -----------    -------------------------
    gold      the reference transcript   P4, C1         nothing textual can solve
                                                        this, at all
    asr       Whisper on the same audio  P1, P2, P3     a cascaded ASR+LLM system
                                                        cannot solve this

P4 (unmarked self-repair) and C1 (delivery) are gated on gold because their
answer genuinely is not in the words: ASR normalises a repair away and never
records hesitancy.  P1/P2/P3 are gated on ASR because that is the system they
are a claim about -- "the needle was muttered, so the transcript is wrong" is
the finding, not a loophole.

Both rates are reported for every category regardless of which one gates, so a
reviewer can see the gold-leak rate we are not rejecting on.  Hiding it would be
the actual problem; the paper plan asks for exactly this table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..items import LETTERS, MCQItem
from ..protocol import ABSENT_OPTION, assign_letters

# Category -> which transcript tier decides rejection.
GATE = {"P1": "asr", "P2": "asr", "P3": "asr", "P4": "gold", "C1": "gold"}

# A text model deciding by majority vote over an ensemble; one model agreeing
# with the key by chance is not leakage.
DEFAULT_MAJORITY = 0.5

TextSolver = Callable[[str], str | None]
"""Takes a fully rendered text-only prompt, returns a letter or None.

Injected rather than imported so the notebook can point this at whatever is
available -- an 8B model on the same T4, or an API tier if one is configured.
The filter's strength is what earns the audio-necessity claim, so the stronger
the solver, the better the result.
"""


@dataclass
class LeakReport:
    per_item: dict[str, dict[str, bool]] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)

    def rate(self, items: Sequence[MCQItem], tier: str, category: str | None = None) -> float:
        subset = [i for i in items if category is None or i.category == category]
        seen = [self.per_item.get(i.item_id, {}).get(tier) for i in subset]
        seen = [s for s in seen if s is not None]
        return sum(seen) / len(seen) if seen else float("nan")

    def table(self, items: Sequence[MCQItem]) -> list[dict]:
        rows = []
        for category in ("P1", "P2", "P3", "P4", "C1"):
            subset = [i for i in items if i.category == category]
            if not subset:
                continue
            rows.append({
                "category": category,
                "n": len(subset),
                "gate": GATE[category],
                "gold_leak": round(self.rate(items, "gold", category), 3),
                "asr_leak": round(self.rate(items, "asr", category), 3),
                "rejected": sum(1 for i in subset if i.item_id in self.rejected),
            })
        return rows


def render_text_prompt(item: MCQItem, transcript: str, seed: int) -> tuple[str, dict[str, str]]:
    """The same question a model would hear, but with the words handed over.

    Uses the same option shuffling as the audio protocol, so a text model cannot
    be advantaged or disadvantaged by option order relative to the real run.
    """
    role_to_letter = assign_letters(item, seed)
    letter_to_role = {v: k for k, v in role_to_letter.items()}
    options = dict(item.options)
    options["absent"] = ABSENT_OPTION[item.lang]

    lines = [
        "Read the transcript and answer the question about it.",
        "Reply with the letter of the correct option and nothing else.",
        "",
        "TRANSCRIPT:",
        transcript.strip(),
        "",
        item.question,
        "",
    ]
    lines += [f"{L}. {options[letter_to_role[L]]}" for L in LETTERS]
    lines += ["", "Answer:"]
    return "\n".join(lines), letter_to_role


def run_filter(
    items: Sequence[MCQItem],
    gold_transcripts: dict[str, str],
    asr_transcripts: dict[str, str] | None,
    solvers: Sequence[TextSolver],
    seed: int = 0,
    majority: float = DEFAULT_MAJORITY,
) -> LeakReport:
    """Score every item on both tiers and mark the ones its gate rejects."""
    if not solvers:
        raise ValueError("the leak filter needs at least one text solver")

    report = LeakReport()
    tiers = {"gold": gold_transcripts}
    if asr_transcripts is not None:
        tiers["asr"] = asr_transcripts

    for item in items:
        verdicts: dict[str, bool] = {}
        for tier, transcripts in tiers.items():
            transcript = transcripts.get(item.recording_id)
            if transcript is None:
                continue
            prompt, letter_to_role = render_text_prompt(item, transcript, seed)
            hits = 0
            for solve in solvers:
                letter = solve(prompt)
                if letter and letter_to_role.get(letter) == item.correct_role:
                    hits += 1
            verdicts[tier] = (hits / len(solvers)) > majority
        report.per_item[item.item_id] = verdicts

        gate = GATE.get(item.category, "gold")
        if verdicts.get(gate):
            report.rejected.add(item.item_id)
    return report


def apply_filter(items: Sequence[MCQItem], report: LeakReport) -> list[MCQItem]:
    """Survivors, with the leak verdicts recorded in provenance."""
    kept: list[MCQItem] = []
    for item in items:
        if item.item_id in report.rejected:
            continue
        data = item.to_dict()
        data["provenance"] = {
            **item.provenance,
            "leak_checked": True,
            "leak": report.per_item.get(item.item_id, {}),
        }
        kept.append(MCQItem.from_dict(data))
    return kept
