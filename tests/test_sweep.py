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
