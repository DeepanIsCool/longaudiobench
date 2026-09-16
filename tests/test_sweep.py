"""The prominence gain sweep.

The only experiment here that manipulates prominence rather than observing it,
so its controls matter more than most: if the competitor is not audible, or the
control level is missing, the curve measures something other than the trap.
"""

from __future__ import annotations

import numpy as np
import pytest

from undertone import sweep
from undertone.items import MCQItem
from undertone.ladder import Window


def item(needle=(100.0, 104.0), band=300):
    return MCQItem(
        item_id="it_1", recording_id="r0", lang="en", category="P3",
        sector="meetings", audio_path="a.flac", duration_band=band,
        needle_start=needle[0], needle_end=needle[1],
        question="Which dose did she mention in passing?",
        options={"correct": "five milligrams", "salience": "fifty milligrams",
                 "recency": "fifteen milligrams", "absent": "x"})


class StubModel:
    """Answers 'salience' once the answer is quieter than `threshold_db`."""

    key = "stub"

    class _HW:
        signature = "cuda/float16"
    hardware = _HW()

    def __init__(self, threshold_db=-6.0):
        self.threshold_db = threshold_db

    def score_letters(self, audio, prompt, sr=16000):
        # Read the level of the target span itself. Measuring the first second
        # would miss the manipulation entirely - the target sits at 10-12 s.
        span = audio[10 * sr:12 * sr]
        base = audio[30 * sr:32 * sr]
        rel = 20 * np.log10(
            max(float(np.sqrt(np.mean(span ** 2))), 1e-9)
            / max(float(np.sqrt(np.mean(base ** 2))), 1e-9))
        want = "salience" if rel < self.threshold_db else "correct"
        target = {"correct": "five milligrams", "salience": "fifty milligrams",
                  "recency": "fifteen milligrams"}[want]
        for line in prompt.splitlines():
            if line[1:3] == ". " and line[3:] == target:
                return {L: (5.0 if L == line[0] else 0.0) for L in "ABCD"}
        return {L: 0.0 for L in "ABCD"}


class TestContrastWindow:
    def test_window_holds_both_mentions(self):
        """A salience trap cannot operate if the salient thing is out of
        earshot - an L1 window centred on the needle usually excludes it."""
        w = sweep.contrast_window(item(), 60.0, 62.0)
        assert w.start <= 60.0 and w.end >= 104.0

    def test_window_is_padded_for_context(self):
        w = sweep.contrast_window(item(), 90.0, 92.0)
        assert w.start < 90.0 and w.end > 104.0

    def test_window_stays_inside_the_recording(self):
        w = sweep.contrast_window(item(needle=(2.0, 4.0)), 0.5, 1.0)
        assert w.start >= 0.0
        assert w.end <= 300.0

    def test_distant_mentions_fall_back_to_a_capped_window(self):
        w = sweep.contrast_window(item(needle=(250.0, 252.0)), 5.0, 7.0)
        assert w.seconds <= sweep.MAX_CONTRAST_WINDOW + 1


class TestSweep:
    def _audio(self, seconds=40.0, sr=16000):
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.25, int(seconds * sr)).astype(np.float32)

    def test_control_level_is_included(self):
        """Level 0 is the unedited recording. Without it the curve starts from
        an assumption instead of from what the audio actually did."""
        assert 0.0 in sweep.DEFAULT_LEVELS
        assert sweep.DEFAULT_LEVELS[0] == 0.0

    def test_one_row_per_level(self):
        it = item(needle=(10.0, 12.0))
        rows = sweep.sweep_item(StubModel(), it, self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)])
        assert len(rows) == len(sweep.DEFAULT_LEVELS)
        assert [r["level_db"] for r in rows] == list(sweep.DEFAULT_LEVELS)

    def test_a_model_flips_as_the_answer_gets_quieter(self):
        it = item(needle=(10.0, 12.0))
        rows = sweep.sweep_item(StubModel(threshold_db=-9.0), it, self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)])
        roles = [r["role_chosen"] for r in rows]
        assert roles[0] == "correct", roles
        assert "salience" in roles, roles

    def test_threshold_is_the_first_level_that_flips(self):
        rows = [{"level_db": 0.0, "role_chosen": "correct"},
                {"level_db": -3.0, "role_chosen": "correct"},
                {"level_db": -6.0, "role_chosen": "salience"},
                {"level_db": -9.0, "role_chosen": "salience"}]
        assert sweep.flip_threshold(rows) == -6.0

    def test_never_flipping_is_nan_not_zero(self):
        """A model that resists the whole sweep is a finding; reporting it as
        0 dB would say the opposite."""
        import math

        rows = [{"level_db": l, "role_chosen": "correct"}
                for l in sweep.DEFAULT_LEVELS]
        assert math.isnan(sweep.flip_threshold(rows))

    def test_a_failed_level_does_not_kill_the_curve(self):
        class Broken(StubModel):
            def score_letters(self, *a, **k):
                raise RuntimeError("boom")

        rows = sweep.sweep_item(Broken(), item(needle=(10.0, 12.0)), self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)])
        assert len(rows) == len(sweep.DEFAULT_LEVELS)
        assert all(r["error"] for r in rows)


class TestCurve:
    def test_curve_reports_rate_per_level(self):
        rows = ([{"level_db": 0.0, "role_chosen": "correct", "correct_role": "correct",
                  "is_null": False, "achieved_contrast_db": 0.0}] * 4
                + [{"level_db": -9.0, "role_chosen": "salience", "correct_role": "correct",
                    "is_null": False, "achieved_contrast_db": 9.0}] * 4)
        out = {c["level_db"]: c for c in sweep.curve(rows, (0.0, -9.0))}
        assert out[0.0]["accuracy"] == 1.0
        assert out[-9.0]["salience_rate"] == 1.0

    def test_empty_levels_are_nan_not_zero(self):
        import math

        out = sweep.curve([], (0.0, -6.0))
        assert all(math.isnan(c["accuracy"]) for c in out)


class TestSweepIsActuallyWired:
    """sweep.py existed for days with no caller outside its own tests: every
    results row was L1-L4 and not one carried a SWEEP condition. The causal
    experiment was written, tested, and never run."""

    def test_every_model_notebook_runs_the_sweep(self):
        import json
        import pathlib

        from undertone.adapters.base import _REGISTRY

        infra = {"00", "01", "02", "03", "04", "05", "90"}   # 03-05 build packs, not models
        for path in sorted(pathlib.Path("notebooks").glob("*.ipynb")):
            if path.stem.split("_")[0] in infra:
                continue
            src = "".join("".join(c["source"])
                          for c in json.loads(path.read_text())["cells"])
            assert "sweep.run_sweep" in src, f"{path.name} never runs the sweep"

    def test_competitor_spans_come_from_provenance(self):
        from undertone.sweep import competitor_spans

        it = item()
        it.provenance = {"salience_at": 40.0}
        spans = competitor_spans(it)
        assert len(spans) == 1
        assert spans[0][0] < 40.0 < spans[0][1]

        it.provenance = {}
        assert competitor_spans(it) == [], (
            "an item with no recorded competitor must be skipped, not swept - "
            "a salience trap cannot operate when the salient thing is absent")


class TestRemovalControl:
    """The first real sweep moved 3 of 70 answers over 12 dB with the gain edit
    verifiably landing, and reported a 0.0 dB flip threshold for a model that
    never flipped. Both were artefacts of the level set and the threshold rule.
    """

    def test_the_range_reaches_past_twelve_db(self):
        from undertone.sweep import DEFAULT_LEVELS

        assert min(l for l in DEFAULT_LEVELS if l > -60) <= -18.0, (
            "-12 dB moved almost nothing; the curve needs range to find a "
            "threshold")

    def test_removal_control_is_present_and_off_curve(self):
        from undertone.sweep import DEFAULT_LEVELS, NEEDLE_REMOVED_DB

        assert NEEDLE_REMOVED_DB in DEFAULT_LEVELS
        assert NEEDLE_REMOVED_DB <= -60.0, "the needle must be inaudible"

    def test_already_lost_is_not_a_zero_db_flip(self):
        from undertone.sweep import flip_threshold

        rows = [{"level_db": 0.0, "role_chosen": "salience"},
                {"level_db": -3.0, "role_chosen": "salience"}]
        assert flip_threshold(rows) != 0.0, (
            "an item answering salience before any attenuation was never "
            "flipped by the sweep and must not report a 0 dB threshold")

    def test_needle_necessity_measures_the_drop(self):
        from undertone.sweep import NEEDLE_REMOVED_DB, needle_necessity

        rows = [{"level_db": 0.0, "role_chosen": "correct", "correct_role": "correct"},
                {"level_db": NEEDLE_REMOVED_DB, "role_chosen": "absent",
                 "correct_role": "correct"}]
        out = needle_necessity(rows)
        assert out["acc_intact"] == 1.0 and out["acc_removed"] == 0.0
        assert out["audio_dependence"] == 1.0


class TestQuestionOnlyBaseline:
    """Plan section 13 lists this against "priors solve it". The -60 dB level
    removes the needle but keeps the meeting; this removes the audio entirely.
    Accuracy near chance is the result that makes every other number mean
    something.
    """

    def test_it_feeds_silence_not_audio(self):
        import inspect

        from undertone.sweep import question_only

        src = inspect.getsource(question_only)
        assert "np.zeros" in src, "the control must pass silence, not the recording"
        assert '"QUESTION_ONLY"' in src

    def test_rows_carry_provenance(self):
        import inspect

        from undertone.sweep import question_only

        src = inspect.getsource(question_only)
        for field in ("pack_fingerprint", "code_sha", "correct_role"):
            assert field in src, f"question-only rows must carry {field}"


class RelativeStub(StubModel):
    """Answers by the needle-vs-competitor *ratio*, not the needle alone.

    The competitor arm exists to tell these two accounts apart. This stub is
    the relative-prominence account: quieten the competitor and it recovers the
    answer even though the needle never changed.
    """

    def score_letters(self, audio, prompt, sr=16000):
        needle = audio[10 * sr:12 * sr]
        comp = audio[20 * sr:22 * sr]
        rel = 20 * np.log10(
            max(float(np.sqrt(np.mean(needle ** 2))), 1e-9)
            / max(float(np.sqrt(np.mean(comp ** 2))), 1e-9))
        want = "salience" if rel < self.threshold_db else "correct"
        target = {"correct": "five milligrams", "salience": "fifty milligrams",
                  "recency": "fifteen milligrams"}[want]
        for line in prompt.splitlines():
            if line[1:3] == ". " and line[3:] == target:
                return {L: (5.0 if L == line[0] else 0.0) for L in "ABCD"}
        return {L: 0.0 for L in "ABCD"}


class TestEditTarget:
    """The 2x2: which span the gain lands on, and what that does to contrast."""

    def _audio(self, seconds=40.0, sr=16000):
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.25, int(seconds * sr)).astype(np.float32)

    def test_default_is_the_needle_and_rows_say_so(self):
        rows = sweep.sweep_item(StubModel(), item(needle=(10.0, 12.0)), self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)])
        assert {r["edit_target"] for r in rows} == {"needle"}

    def test_competitor_edit_leaves_the_needle_untouched(self):
        """StubModel reads only the needle's level. Attenuating the competitor
        must not move it - otherwise the arm is editing the wrong span."""
        it = item(needle=(10.0, 12.0))
        rows = sweep.sweep_item(StubModel(threshold_db=-6.0), it, self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)],
                                levels=(0.0, -12.0, -24.0), edit_target="competitor")
        assert [r["role_chosen"] for r in rows] == ["correct"] * 3
        assert {r["edit_target"] for r in rows} == {"competitor"}

    def test_competitor_edit_lowers_measured_contrast(self):
        """achieved_contrast_db is competitor-minus-needle: positive means the
        trap is set. Quietening the competitor must read as it *falling* by
        about the requested amount - the mirror image of the needle arm."""
        it = item(needle=(10.0, 12.0))
        rows = sweep.sweep_item(StubModel(), it, self._audio(),
                                Window(0.0, 40.0, False), [(20.0, 22.0)],
                                levels=(0.0, -12.0), edit_target="competitor")
        drop = rows[0]["achieved_contrast_db"] - rows[1]["achieved_contrast_db"]
        assert 10 < drop < 13

    def test_the_two_arms_separate_the_two_accounts(self):
        """A needle-only model does not recover when the competitor is
        quietened; a relative-prominence model does. That difference is the
        point of running both arms."""
        it = item(needle=(10.0, 12.0))
        audio = self._audio()
        # Make the needle already quiet, so both stubs start on 'salience'.
        sr = 16000
        audio[10 * sr:12 * sr] *= 10 ** (-9 / 20)
        args = (it, audio, Window(0.0, 40.0, False), [(20.0, 22.0)])
        absolute = sweep.sweep_item(StubModel(-6.0), *args,
                                    levels=(0.0, -18.0), edit_target="competitor")
        relative = sweep.sweep_item(RelativeStub(-6.0), *args,
                                    levels=(0.0, -18.0), edit_target="competitor")
        assert [r["role_chosen"] for r in absolute] == ["salience", "salience"]
        assert [r["role_chosen"] for r in relative] == ["salience", "correct"]

    def test_boost_recovers_a_lost_answer(self):
        """The needle arm in the other direction: if raising the needle brings
        the answer back, prominence was what was missing."""
        it = item(needle=(10.0, 12.0))
        audio = self._audio()
        sr = 16000
        audio[10 * sr:12 * sr] *= 10 ** (-9 / 20)
        rows = sweep.sweep_item(StubModel(-6.0), it, audio,
                                Window(0.0, 40.0, False), [(20.0, 22.0)],
                                levels=sweep.BOOST_LEVELS)
        roles = [r["role_chosen"] for r in rows]
        assert roles[0] == "salience" and roles[-1] == "correct"

    def test_boost_does_not_clip(self):
        """Clip protection is per span. A +9 dB edit on loud audio must not
        rescale anything outside the edited span."""
        from undertone.harvest.construct import GainEdit, apply_gain_edits
        sr = 16000
        audio = np.full(5 * sr, 0.8, dtype=np.float32)
        out = apply_gain_edits(audio, [GainEdit(1.0, 2.0, 9.0)], sr)
        assert float(np.max(np.abs(out))) <= 1.0
        assert np.array_equal(out[:sr], audio[:sr])          # untouched before
        assert np.array_equal(out[3 * sr:], audio[3 * sr:])  # untouched after

    def test_unknown_target_is_refused(self):
        with pytest.raises(ValueError, match="edit_target"):
            sweep.sweep_item(StubModel(), item(needle=(10.0, 12.0)), self._audio(),
                             Window(0.0, 40.0, False), [(20.0, 22.0)],
                             edit_target="both")

    def test_competitor_arm_needs_a_competitor(self):
        with pytest.raises(ValueError, match="no competitor"):
            sweep.sweep_item(StubModel(), item(needle=(10.0, 12.0)), self._audio(),
                             Window(0.0, 40.0, False), [], edit_target="competitor")

    def test_calibrated_levels_stay_in_the_exact_dose_region(self):
        """Table 3b: shortfall <= 0.5 dB down to -12, 34 dB at -60. The fine
        sweep must not wander past where the dose is honest."""
        assert min(sweep.CALIBRATED_LEVELS) >= -12.0
        assert sweep.CALIBRATED_LEVELS[0] == 0.0
        assert sweep.BOOST_LEVELS[0] == 0.0


class TestResumeAcrossArms:
    def test_arms_do_not_mark_each_other_done(self, tmp_path):
        """Needle and competitor rows can share a file. A finished needle arm
        must not make run_sweep skip the competitor arm for the same item."""
        import json
        from undertone.items import ItemPack

        it = MCQItem.from_dict({**item(needle=(10.0, 12.0)).to_dict(),
                                "provenance": {"salience_at": 21.0}})
        pack = ItemPack([it])
        out = tmp_path / "sweep.jsonl"
        out.write_text(json.dumps({
            "item_id": "it_1", "level_db": 0.0, "edit_target": "needle",
            "pack_fingerprint": pack.fingerprint, "error": None,
            "role_chosen": "correct"}) + "\n")
        # Monkeypatch audio loading so no file is needed.
        import undertone.adapters.base as base
        sr = 16000
        base_load = base.load_audio
        base.load_audio = lambda p: np.random.default_rng(0).normal(
            0, 0.25, 40 * sr).astype(np.float32)
        try:
            sweep.run_sweep(StubModel(), pack, out, levels=(0.0,),
                            edit_target="competitor", progress=False)
        finally:
            base.load_audio = base_load
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert [(r["level_db"], r["edit_target"]) for r in rows] == \
            [(0.0, "needle"), (0.0, "competitor")]

    def test_legacy_rows_count_as_the_needle_arm(self, tmp_path):
        """Rows written before edit_target existed have no such field. They
        are needle rows and must resume as such, not be re-run."""
        import json
        from undertone.items import ItemPack

        it = MCQItem.from_dict({**item(needle=(10.0, 12.0)).to_dict(),
                                "provenance": {"salience_at": 21.0}})
        pack = ItemPack([it])
        out = tmp_path / "sweep.jsonl"
        out.write_text(json.dumps({
            "item_id": "it_1", "level_db": 0.0,
            "pack_fingerprint": pack.fingerprint, "error": None,
            "role_chosen": "correct"}) + "\n")
        import undertone.adapters.base as base
        base_load = base.load_audio
        base.load_audio = lambda p: (_ for _ in ()).throw(AssertionError("should not load"))
        try:
            sweep.run_sweep(StubModel(), pack, out, levels=(0.0,), progress=False)
        finally:
            base.load_audio = base_load
        assert len(out.read_text().splitlines()) == 1
