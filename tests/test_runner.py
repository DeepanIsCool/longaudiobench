"""End-to-end runner tests with a stub adapter -- no torch, no audio files.

Covers the machinery that decides what lands in a results row: resume, the
truncation flag, role mapping, primary-scorer selection, and error containment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from undertone import runner
from undertone.adapters.base import ModelAdapter
from undertone.items import ItemPack, MCQItem
from undertone.ladder import CONDITIONS


class StubAdapter(ModelAdapter):
    """Always prefers the letter currently holding a chosen role."""

    key = "stub"
    model_id = "stub/stub"
    max_audio_s = 30.0
    primary = "logits"
    notes = "test double"

    def __init__(self, prefer: str = "correct", fail_on: set[str] | None = None,
                 max_audio_s: float | None = None) -> None:
        super().__init__()
        self.prefer = prefer
        self.fail_on = fail_on or set()
        self.seen_seconds: list[float] = []
        if max_audio_s is not None:
            self.max_audio_s = max_audio_s

    def load(self) -> None:
        self.model = object()

    def build_inputs(self, audio, prompt, sr=16000):  # pragma: no cover - unused
        return {}

    def _letter_for(self, prompt: str) -> str:
        target = {"correct": "five milligrams", "salience": "fifty milligrams",
                  "recency": "fifteen milligrams"}[self.prefer]
        for line in prompt.splitlines():
            if line[1:3] == ". " and line[3:] == target:
                return line[0]
        raise AssertionError(f"option {target!r} not in prompt")

    def score_letters(self, audio, prompt, sr=16000):
        self.seen_seconds.append(len(audio) / sr)
        if self.prefer in self.fail_on:
            raise RuntimeError("boom")
        chosen = self._letter_for(prompt)
        return {L: (5.0 if L == chosen else 0.0) for L in "ABCD"}

    def generate(self, audio, prompt, sr=16000, max_new_tokens=8):
        return self._letter_for(prompt)

    def unload(self) -> None:
        self.model = None


def make_pack(n: int = 3, band: int = 300) -> ItemPack:
    return ItemPack([
        MCQItem(
            item_id=f"it_{i:03d}",
            recording_id=f"rec_{i // 2}",
            lang="en",
            category="P3",
            sector="meetings",
            audio_path=f"audio/rec_{i // 2}.flac",
            duration_band=band,
            needle_start=100.0 + i,
            needle_end=104.0 + i,
            question="Which dose did she mention in passing?",
            options={"correct": "five milligrams", "salience": "fifty milligrams",
                     "recency": "fifteen milligrams", "absent": "placeholder"},
        )
        for i in range(n)
    ])


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch):
    """No files on disk: return silence of exactly the requested length."""
    def _load(path, start=0.0, end=None, sr=16000):
        seconds = (end - start) if end is not None else 1.0
        return np.zeros(int(seconds * sr), dtype=np.float32)

    monkeypatch.setattr(runner, "load_audio", _load)


class TestSweep:
    def test_writes_one_row_per_cell(self, tmp_path):
        pack = make_pack(3)
        out = runner.run_model(StubAdapter(), pack, tmp_path / "r.jsonl", progress=False)
        rows = runner.load_rows(out)
        assert len(rows) == 3 * len(CONDITIONS)
        assert {r["condition"] for r in rows} == set(CONDITIONS)
        assert all(r["error"] is None for r in rows)

    def test_preferring_the_correct_role_scores_one(self, tmp_path):
        out = runner.run_model(StubAdapter("correct"), make_pack(2),
                               tmp_path / "r.jsonl", progress=False)
        rows = runner.load_rows(out)
        assert all(r["role_chosen"] == "correct" for r in rows)
        assert all(r["correct"] for r in rows)

    def test_preferring_the_loud_mention_is_a_salience_trap(self, tmp_path):
        """The headline diagnostic: chose the louder competing mention."""
        from undertone import scoring

        out = runner.run_model(StubAdapter("salience"), make_pack(4),
                               tmp_path / "r.jsonl", progress=False)
        rows = runner.scorable(runner.load_rows(out))
        summary = scoring.summarize(rows)
        assert summary["accuracy"] == 0.0
        assert summary["salience_trap_rate"] == 1.0

    def test_letter_choice_varies_while_role_does_not(self, tmp_path):
        """Roles are the measurement; letters are shuffled per item."""
        out = runner.run_model(StubAdapter("salience"), make_pack(8),
                               tmp_path / "r.jsonl", progress=False)
        rows = runner.load_rows(out)
        assert len({r["letter_chosen"] for r in rows}) > 1
        assert {r["role_chosen"] for r in rows} == {"salience"}

    def test_both_scorers_are_recorded(self, tmp_path):
        out = runner.run_model(StubAdapter(), make_pack(1), tmp_path / "r.jsonl",
                               progress=False)
        row = runner.load_rows(out)[0]
        assert row["scorer"] == "logits"
        assert row["logit_letter"] == row["letter_chosen"]
        assert row["gen_letter"] == row["logit_letter"]   # gap is measurable, not assumed


class TestTruncation:
    def test_over_long_windows_are_flagged_not_zeroed(self, tmp_path):
        adapter = StubAdapter("correct", max_audio_s=30.0)
        out = runner.run_model(adapter, make_pack(2), tmp_path / "r.jsonl", progress=False)
        rows = runner.load_rows(out)

        l1 = [r for r in rows if r["condition"] == "L1"]
        l3 = [r for r in rows if r["condition"] == "L3"]
        assert not any(r["truncated"] for r in l1)      # 20 s window fits a 30 s cap
        assert all(r["truncated"] for r in l3)          # 300 s band does not
        assert all(r["seconds_seen"] == 30.0 for r in l3)
        assert all(r["seconds_offered"] == 300.0 for r in l3)
        # Truncated cells are still scored, just excluded from the accuracy table.
        assert all(r["role_chosen"] is not None for r in l3)

        # A 30 s model keeps only L1: the 120 s L2 window already overruns it.
        # That is the roster fact for Qwen2-Audio and both Gemma-3n models --
        # they are perception-ceiling instruments, not full-ladder models.
        assert {r["condition"] for r in runner.scorable(rows)} == {"L1"}
        assert {r["condition"] for r in rows if r["truncated"]} == {"L2", "L3", "L4"}

    def test_a_long_ceiling_truncates_nothing(self, tmp_path):
        adapter = StubAdapter("correct", max_audio_s=1800.0)
        out = runner.run_model(adapter, make_pack(2), tmp_path / "r.jsonl", progress=False)
        rows = runner.load_rows(out)
        assert not any(r["truncated"] for r in rows)
        assert len(runner.scorable(rows)) == len(rows)

    def test_adapter_never_sees_more_than_its_ceiling(self, tmp_path):
        adapter = StubAdapter("correct", max_audio_s=30.0)
        runner.run_model(adapter, make_pack(2), tmp_path / "r.jsonl", progress=False)
        assert max(adapter.seen_seconds) <= 30.0


class TestResume:
    def test_second_run_skips_completed_cells(self, tmp_path):
        path = tmp_path / "r.jsonl"
        pack = make_pack(2)
        runner.run_model(StubAdapter(), pack, path, conditions=["L1"], progress=False)
        assert len(runner.load_rows(path)) == 2

        adapter = StubAdapter()
        runner.run_model(adapter, pack, path, conditions=["L1", "L2"], progress=False)
        rows = runner.load_rows(path)
        assert len(rows) == 4                       # 2 kept, 2 added
        assert len(adapter.seen_seconds) == 2       # only the new cells ran

    def test_a_half_written_final_line_does_not_break_resume(self, tmp_path):
        """A session killed mid-write leaves a truncated JSON line."""
        path = tmp_path / "r.jsonl"
        runner.run_model(StubAdapter(), make_pack(1), path, conditions=["L1"],
                         progress=False)
        with path.open("a") as fh:
            fh.write('{"item_id": "it_000", "condi')
        assert runner.completed_keys(path) == {"it_000|L1"}

    def test_errored_cells_are_retried(self, tmp_path):
        path = tmp_path / "r.jsonl"
        runner.run_model(StubAdapter("salience", fail_on={"salience"}), make_pack(1),
                         path, conditions=["L1"], progress=False)
        assert runner.load_rows(path)[0]["error"].startswith("RuntimeError")
        assert runner.completed_keys(path) == set()   # failure is not "done"


class TestErrorContainment:
    def test_a_failing_cell_records_and_does_not_raise(self, tmp_path):
        out = runner.run_model(
            StubAdapter("salience", fail_on={"salience"}), make_pack(1),
            tmp_path / "r.jsonl", conditions=["L1"], progress=False,
            max_consecutive_errors=99)
        row = runner.load_rows(out)[0]
        assert row["role_chosen"] is None
        assert "traceback" in row
        assert row not in runner.scorable([row])

    def test_repeated_failure_aborts_rather_than_burning_quota(self, tmp_path):
        with pytest.raises(RuntimeError, match="consecutive failures"):
            runner.run_model(
                StubAdapter("salience", fail_on={"salience"}), make_pack(10),
                tmp_path / "r.jsonl", progress=False, max_consecutive_errors=3)


class TestFreegenPrimary:
    def test_thinking_style_adapter_scores_from_generation(self, tmp_path):
        adapter = StubAdapter("recency")
        adapter.primary = "freegen"
        out = runner.run_model(adapter, make_pack(1), tmp_path / "r.jsonl",
                               conditions=["L1"], progress=False)
        row = runner.load_rows(out)[0]
        assert row["scorer"] == "freegen"
        assert row["letter_chosen"] == row["gen_letter"]
        assert row["role_chosen"] == "recency"


class TestRowSchema:
    def test_row_carries_everything_the_analysis_needs(self, tmp_path):
        out = runner.run_model(StubAdapter(), make_pack(1), tmp_path / "r.jsonl",
                               conditions=["L4"], progress=False)
        row = runner.load_rows(out)[0]
        for field in ("run_id", "seed", "model_key", "model_id", "item_id",
                      "recording_id", "lang", "category", "sector", "duration_band",
                      "condition", "window_start", "window_end", "is_null",
                      "correct_role", "letter_to_role", "seconds_offered",
                      "seconds_seen", "truncated", "logit_scores", "logit_letter",
                      "logit_role", "logit_degenerate", "role_chosen",
                      "letter_chosen", "scorer", "correct", "latency_ms"):
            assert field in row, field
        assert json.dumps(row)          # must round-trip
