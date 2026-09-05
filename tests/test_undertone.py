"""Unit tests for the UNDERTONE protocol.

Deliberately no model loading and no audio files: everything here is the logic
that decides what a number *means*, which is exactly where the retired suite
went wrong (constant labels, random verdicts, parse failures scored as zeros).
"""

from __future__ import annotations

import itertools
import math

import pytest

from undertone.items import BANDS, LETTERS, ROLES, ItemPack, MCQItem
from undertone.ladder import CONDITIONS, L1_SECONDS, L2_SECONDS, window_for
from undertone.protocol import assign_letters, mmss, render
from undertone import scoring


def make_item(**kw) -> MCQItem:
    base = dict(
        item_id="it_0001",
        recording_id="rec_a",
        lang="en",
        category="P3",
        sector="meetings",
        audio_path="audio/rec_a_300.flac",
        duration_band=300,
        needle_start=100.0,
        needle_end=104.0,
        question="What dose did she mention in passing?",
        options={
            "correct": "five milligrams",
            "salience": "fifty milligrams",
            "recency": "fifteen milligrams",
            "absent": "Not mentioned in the recording",
        },
    )
    base.update(kw)
    return MCQItem(**base)


# ---------------------------------------------------------------- items

class TestItem:
    def test_rejects_needle_past_the_band(self):
        with pytest.raises(ValueError, match="past the"):
            make_item(needle_start=299.0, needle_end=301.0)

    def test_rejects_missing_role(self):
        with pytest.raises(ValueError, match="missing roles"):
            make_item(options={"correct": "a", "salience": "b", "recency": "c"})

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="unknown roles"):
            opts = dict(make_item().options, distractor="x")
            make_item(options=opts)

    def test_null_item_flips_the_correct_role(self):
        assert make_item().correct_role == "correct"
        assert make_item(is_null=True).correct_role == "absent"

    def test_roundtrip_through_jsonl(self, tmp_path):
        pack = ItemPack([make_item(), make_item(item_id="it_0002", lang="bn")],
                        meta={"build": "test"})
        path = tmp_path / "pack.jsonl"
        pack.save(path)
        back = ItemPack.load(path)
        assert len(back) == 2
        # save() adds the fingerprint and item count to whatever meta it was given.
        assert back.meta["build"] == "test"
        assert back.meta["fingerprint"] == pack.fingerprint
        assert back.items[1].lang == "bn"
        assert back.items[0].options == pack.items[0].options


# ---------------------------------------------------------------- protocol

class TestSlotShuffling:
    def test_all_24_permutations_are_reachable(self):
        """A model that always answers one letter must not beat chance."""
        seen = set()
        for seed in range(2000):
            mapping = assign_letters(make_item(), seed)
            seen.add(tuple(mapping[r] for r in ROLES))
        assert len(seen) == math.factorial(4) == 24

    def test_assignment_is_a_bijection(self):
        mapping = assign_letters(make_item(), 7)
        assert sorted(mapping) == sorted(ROLES)
        assert sorted(mapping.values()) == sorted(LETTERS)

    def test_assignment_is_deterministic_per_item_and_seed(self):
        item = make_item()
        assert assign_letters(item, 11) == assign_letters(item, 11)

    def test_different_items_get_different_orders(self):
        a = assign_letters(make_item(item_id="x"), 3)
        b = assign_letters(make_item(item_id="y"), 3)
        assert any(a[r] != b[r] for r in ROLES)

    def test_key_survives_into_the_rendered_prompt(self):
        item = make_item()
        r = render(item, window_for(item, "L1"), seed=5)
        correct_letter = r.role_to_letter["correct"]
        assert f"{correct_letter}. five milligrams" in r.prompt
        assert r.letter_to_role[correct_letter] == "correct"

    def test_absent_option_uses_the_canonical_phrasing(self):
        """Wording drift would make "not mentioned" identifiable by style."""
        item = make_item(lang="hi", options=dict(make_item().options, absent="whatever"))
        r = render(item, window_for(item, "L1"), seed=1)
        assert "रिकॉर्डिंग में इसका उल्लेख नहीं है" in r.prompt
        assert "whatever" not in r.prompt

    def test_prompt_ends_on_the_answer_cue(self):
        """Letter-logit scoring reads the next token, so the cue must be last."""
        for lang, cue in (("en", "Answer:"), ("hi", "उत्तर:"), ("bn", "উত্তর:")):
            item = make_item(lang=lang)
            r = render(item, window_for(item, "L1"), seed=2)
            assert r.prompt.rstrip().endswith(cue)

    def test_oracle_hint_only_in_l4(self):
        item = make_item()
        for cond in ("L1", "L2", "L3"):
            assert "01:40" not in render(item, window_for(item, cond), 0).prompt
        assert "01:40" in render(item, window_for(item, "L4"), 0).prompt


class TestTimestamps:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "00:00"), (5, "00:05"), (65, "01:05"), (1800, "30:00"),
        (3600, "1:00:00"), (3725, "1:02:05"),
    ])
    def test_mmss(self, seconds, expected):
        """The retired code emitted 60:00 for an hour and parsed it back wrong."""
        assert mmss(seconds) == expected


# ---------------------------------------------------------------- ladder

class TestLadder:
    @pytest.mark.parametrize("cond", CONDITIONS)
    @pytest.mark.parametrize("start", [0.0, 1.0, 9.0, 150.0, 280.0, 295.0])
    def test_window_always_contains_the_needle(self, cond, start):
        item = make_item(needle_start=start, needle_end=start + 4.0)
        w = window_for(item, cond)
        assert w.contains(item.needle_start, item.needle_end), (cond, start, w)
        assert 0.0 <= w.start < w.end <= item.duration_band

    def test_l1_is_about_twenty_seconds(self):
        w = window_for(make_item(), "L1")
        assert w.seconds == pytest.approx(L1_SECONDS)

    def test_l1_is_centred_when_there_is_room(self):
        item = make_item()
        w = window_for(item, "L1")
        assert (w.start + w.end) / 2 == pytest.approx(item.needle_mid)

    def test_l2_is_about_two_minutes(self):
        assert window_for(make_item(), "L2").seconds == pytest.approx(L2_SECONDS)

    def test_window_widens_for_a_needle_longer_than_the_window(self):
        """A 25 s aside cannot be asked about through a 20 s hole."""
        item = make_item(needle_start=100.0, needle_end=125.0)
        w = window_for(item, "L1")
        assert w.seconds >= 25.0
        assert w.contains(100.0, 125.0)

    def test_l3_and_l4_are_the_full_band(self):
        item = make_item()
        for cond in ("L3", "L4"):
            w = window_for(item, cond)
            assert (w.start, w.end) == (0.0, float(item.duration_band))
        assert window_for(item, "L4").oracle
        assert not window_for(item, "L3").oracle

    def test_short_band_collapses_to_the_whole_recording(self):
        item = make_item(duration_band=300, needle_start=10.0, needle_end=12.0)
        assert window_for(item, "L2").seconds == pytest.approx(L2_SECONDS)

    def test_unknown_condition_raises(self):
        with pytest.raises(ValueError, match="unknown condition"):
            window_for(make_item(), "L5")


# ---------------------------------------------------------------- scoring

class TestLetterScoring:
    def test_letter_logits_picks_the_max_variant(self):
        # letter -> ids; " B" scores highest, so B wins even though A's bare id
        # is above A's spaced id.
        ids = {"A": [0, 1], "B": [2, 3], "C": [4], "D": [5]}
        logits = [0.1, 0.2, 0.0, 9.0, -1.0, -1.0]
        scores = scoring.letter_logits(logits, ids)
        assert scores == {"A": 0.2, "B": 9.0, "C": -1.0, "D": -1.0}
        assert scoring.argmax_letter(scores) == "B"

    def test_degenerate_detection(self):
        assert scoring.is_degenerate({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
        assert not scoring.is_degenerate({"A": 1.0, "B": 1.1, "C": 1.0, "D": 1.0})

    @pytest.mark.parametrize("text,expected", [
        ("B", "B"), ("B.", "B"), (" C ", "C"), ("Answer: D", "D"),
        ("(A)", "A"), ("A. five milligrams", "A"),
        ("", None), ("None of these", None), ("BAD", None), ("cab", None),
    ])
    def test_free_letter_parsing_is_strict(self, text, expected):
        assert scoring.parse_free_letter(text) == expected

    def test_unparseable_is_none_not_a_wrong_answer(self):
        """The retired metrics folded parse failures into zeros."""
        assert scoring.parse_free_letter("I cannot determine this") is None

    def test_strip_cot_removes_closed_and_dangling_blocks(self):
        assert scoring.strip_cot("<think>maybe A or B</think> C") == "C"
        assert scoring.parse_free_letter(
            "<think>could be A</think>\nD", strip_reasoning=True) == "D"
        # A budget-truncated block must not leak its letters.
        assert scoring.parse_free_letter(
            "<think>probably A because", strip_reasoning=True) is None


def rows_for(roles, correct="correct", is_null=False, recording="r0"):
    return [
        {"role_chosen": role, "correct_role": correct, "is_null": is_null,
         "letter_chosen": "A", "recording_id": recording}
        for role in roles
    ]


class TestSummarize:
    def test_rates_split_by_role(self):
        out = scoring.summarize(rows_for(
            ["correct", "correct", "salience", "salience", "salience",
             "recency", "absent", None]))
        assert out["n"] == 8
        assert out["n_scored"] == 7
        assert out["unparseable_rate"] == pytest.approx(1 / 8)
        assert out["accuracy"] == pytest.approx(2 / 7)
        assert out["salience_trap_rate"] == pytest.approx(3 / 7)
        assert out["recency_trap_rate"] == pytest.approx(1 / 7)
        assert out["fabrication_rate"] == pytest.approx(1 / 7)

    def test_null_items_are_correct_when_absent_is_chosen(self):
        out = scoring.summarize(
            rows_for(["absent", "absent", "correct"], correct="absent", is_null=True))
        assert out["null_accuracy"] == pytest.approx(2 / 3)
        # Null items must not inflate the fabrication rate.
        assert math.isnan(out["fabrication_rate"])

    def test_empty_input(self):
        assert scoring.summarize([]) == {"n": 0}


class TestBootstrap:
    def test_resamples_recordings_not_items(self):
        """Items in one recording share speaker, room and topic."""
        rows = (rows_for(["correct"] * 10, recording="r1")
                + rows_for(["salience"] * 10, recording="r2"))
        lo, hi = scoring.cluster_bootstrap_ci(rows, scoring.accuracy, n_bootstrap=500, seed=1)
        # Two clusters, all-or-nothing within each => the interval must span
        # essentially 0 to 1. Item-level resampling would give ~0.5 +/- 0.2.
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

    def test_deterministic_for_a_fixed_seed(self):
        rows = rows_for(["correct", "salience", "recency", "correct"])
        a = scoring.cluster_bootstrap_ci(rows, scoring.accuracy, n_bootstrap=200, seed=3)
        b = scoring.cluster_bootstrap_ci(rows, scoring.accuracy, n_bootstrap=200, seed=3)
        assert a == b

    def test_empty_input_is_nan(self):
        lo, hi = scoring.cluster_bootstrap_ci([], scoring.accuracy)
        assert math.isnan(lo) and math.isnan(hi)


class TestLadderCosts:
    def test_costs_are_relative_to_the_l1_ceiling(self):
        out = scoring.ladder_costs({
            "L1": rows_for(["correct"] * 8 + ["salience"] * 2),   # 0.8
            "L3": rows_for(["correct"] * 3 + ["salience"] * 7),   # 0.3
            "L4": rows_for(["correct"] * 6 + ["salience"] * 4),   # 0.6
        })
        assert out["RetrievalCost"] == pytest.approx(0.5)
        assert out["LongContextCost"] == pytest.approx(0.2)


# ---------------------------------------------------------------- adapters

class TestAdapterRoster:
    def test_thirteen_adapters_registered(self):
        from undertone import adapters

        keys = adapters.list_adapters()
        assert len(keys) == 13, keys

    def test_every_adapter_declares_a_real_ceiling(self):
        from undertone import adapters
        from undertone.adapters.base import ModelAdapter

        for key in adapters.list_adapters():
            cls = type(adapters.get_adapter(key))
            assert issubclass(cls, ModelAdapter)
            assert cls.model_id, key
            assert cls.max_audio_s > 0, key
            assert cls.primary in {"logits", "freegen"}, key
            assert cls.notes, f"{key} has no documented caveat"

    def test_thinking_variants_score_by_generation(self):
        """Their first token is a thought, so letter logits are not the answer."""
        from undertone import adapters

        for key in adapters.list_adapters():
            adapter = adapters.get_adapter(key)
            if "thinking" in key:
                assert adapter.primary == "freegen", key
                assert adapter.strip_reasoning, key
            else:
                assert adapter.primary == "logits", key

    def test_thirty_second_models_are_flagged_as_such(self):
        from undertone import adapters

        capped = {k for k in adapters.list_adapters()
                  if adapters.get_adapter(k).max_audio_s <= 30.0}
        assert capped == {"qwen2_audio_7b", "gemma3n_e2b", "gemma3n_e4b"}


class TestMossTokenBudget:
    def test_rate_matches_the_shipped_processor(self):
        from undertone.adapters.moss import AUDIO_TOKENS_PER_SECOND, estimate_tokens

        assert AUDIO_TOKENS_PER_SECOND == 12.5
        # 12.5 audio tokens/s plus one time marker every 2 s.
        assert estimate_tokens(100) == 1250 + sum(len(str(s)) for s in range(2, 101, 2))

    def test_longest_band_fits_the_context(self):
        from undertone.adapters.moss import CONTEXT_LIMIT, estimate_tokens, max_seconds

        assert estimate_tokens(1800) < CONTEXT_LIMIT
        assert max_seconds() > 1800

    def test_over_long_window_would_be_rejected(self):
        from undertone.adapters.moss import CONTEXT_LIMIT, PROMPT_HEADROOM, estimate_tokens

        assert estimate_tokens(4000) > CONTEXT_LIMIT - PROMPT_HEADROOM


# ---------------------------------------------------------------- truncation

class TestTruncation:
    def test_cap_reports_what_was_dropped(self):
        import numpy as np

        from undertone.adapters.base import apply_cap

        audio = np.zeros(16000 * 900, dtype="float32")   # 15 min
        out = apply_cap(audio, max_audio_s=30.0)
        assert out.truncated
        assert out.seconds_seen == pytest.approx(30.0)
        assert out.seconds_offered == pytest.approx(900.0)
        assert len(out.audio) == 16000 * 30

    def test_no_cap_no_flag(self):
        import numpy as np

        from undertone.adapters.base import apply_cap

        out = apply_cap(np.zeros(16000 * 10, dtype="float32"), max_audio_s=30.0)
        assert not out.truncated
        assert out.seconds_seen == out.seconds_offered == pytest.approx(10.0)


class TestGenerationBudget:
    def test_thinking_variants_get_room_to_reason(self):
        """At 8 tokens they were cut off mid-<think> and never reached an answer."""
        from undertone import adapters

        for key in adapters.list_adapters():
            a = adapters.get_adapter(key)
            if a.strip_reasoning:
                assert a.generation_budget >= 512, key
            else:
                assert a.generation_budget <= 32, key

    def test_a_truncated_think_block_is_unparseable_not_wrong(self):
        from undertone import scoring

        cut = "<think>\nFor this question, I need"
        assert scoring.parse_free_letter(cut, strip_reasoning=True) is None

    def test_a_completed_think_block_yields_its_answer(self):
        from undertone import scoring

        full = "<think>\nThe quiet one said five.\n</think>\nB"
        assert scoring.parse_free_letter(full, strip_reasoning=True) == "B"


class TestProcessorAudioVerification:
    def _batch(self, **keys):
        return dict(input_ids=[[1, 2]], **keys)

    def test_a_processor_that_drops_the_audio_is_a_failure(self):
        """Aero accepted `audio=` through **kwargs and discarded it, then
        returned identical logits for different clips."""
        import numpy as np
        import pytest

        from undertone.adapters.base import call_processor

        def swallowing(text=None, sampling_rate=None, return_tensors=None, **kw):
            return {"input_ids": [[1, 2]]}      # no audio features

        with pytest.raises(TypeError, match="never ingested the audio"):
            call_processor(swallowing, "q", np.zeros(16), 16000)

    def test_a_processor_that_encodes_audio_succeeds(self):
        import numpy as np

        from undertone.adapters.base import call_processor

        def good(text=None, sampling_rate=None, return_tensors=None, **kw):
            assert "audios" in kw or "audio" in kw
            return {"input_ids": [[1, 2]], "input_features": [[0.0]]}

        assert "input_features" in call_processor(good, "q", np.zeros(16), 16000)

    def test_named_kwarg_is_not_second_guessed(self):
        import numpy as np

        from undertone.adapters.base import call_processor

        seen = []

        def picky(text=None, sampling_rate=None, return_tensors=None, **kw):
            seen.append(sorted(kw))
            return {"input_ids": [[1]], "audio_data": [[0.0]]}

        call_processor(picky, "q", np.zeros(16), 16000, audio_kwarg="audios")
        assert seen == [["audios"]]

    def test_aero_names_its_kwarg(self):
        import inspect

        from undertone.adapters import aero

        assert 'audio_kwarg="audios"' in inspect.getsource(aero.Aero1Audio.build_inputs)


class TestLadderCostSampleGuard:
    def _rows(self, n, role="correct", cond="L1"):
        return [{"role_chosen": role, "correct_role": "correct", "is_null": False,
                 "condition": cond, "recording_id": "r0"} for _ in range(n)]

    def test_a_cost_from_a_handful_of_cells_is_withheld(self):
        """Phi-4 reported 0.422 off an acc_L3 that survived on a few rows after
        162 errored - indistinguishable in a table from a solid measurement."""
        import math

        from undertone import scoring

        out = scoring.ladder_costs({"L1": self._rows(80),
                                    "L3": self._rows(3, "salience", "L3")})
        assert math.isnan(out["RetrievalCost"])
        assert out["n_L3"] == 3

    def test_a_well_sampled_cost_is_reported(self):
        from undertone import scoring

        out = scoring.ladder_costs({"L1": self._rows(80),
                                    "L3": self._rows(40, "salience", "L3")})
        assert out["RetrievalCost"] == 1.0
        assert out["n_L1"] == 80 and out["n_L3"] == 40

    def test_per_condition_n_is_always_reported(self):
        from undertone import scoring

        out = scoring.ladder_costs({"L1": self._rows(5)})
        assert {"n_L1", "n_L2", "n_L3", "n_L4"} <= set(out)


class TestPackFingerprint:
    def _pack(self, n=3, band=300):
        from undertone.items import ItemPack

        return ItemPack([MCQItem(
            item_id=f"it_{i}", recording_id="r0", lang="en", category="P3",
            sector="meetings", audio_path="a.flac", duration_band=band,
            needle_start=10.0, needle_end=14.0, question="q",
            options={"correct": f"v{i}", "salience": "b", "recency": "c",
                     "absent": "x"}) for i in range(n)])

    def test_same_content_same_fingerprint(self):
        assert self._pack().fingerprint == self._pack().fingerprint

    def test_a_different_band_changes_it(self):
        """The stale-pack failure that wasted a sweep: same items, 10-minute
        bands instead of 5, and nothing anywhere noticed."""
        assert self._pack(band=300).fingerprint != self._pack(band=600).fingerprint

    def test_different_items_change_it(self):
        assert self._pack(3).fingerprint != self._pack(4).fingerprint

    def test_it_is_written_into_the_saved_pack(self, tmp_path):
        from undertone.items import ItemPack

        pack = self._pack()
        pack.save(tmp_path / "p.jsonl")
        back = ItemPack.load(tmp_path / "p.jsonl")
        assert back.meta["fingerprint"] == pack.fingerprint
        assert back.meta["n_items"] == 3
