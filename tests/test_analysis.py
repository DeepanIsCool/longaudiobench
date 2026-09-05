"""Analysis tests: what may and may not enter a results table."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from undertone import analysis

REPO = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(model="m", category="P3", condition="L3", role="salience", *, lang="en",
        truncated=False, error=None, null=False, recording="r0", letter="A", **kw):
    base = {
        "model_key": model, "category": category, "condition": condition,
        "lang": lang, "recording_id": recording, "is_null": null,
        "correct_role": "absent" if null else "correct",
        "role_chosen": role, "letter_chosen": letter, "logit_role": role,
        "logit_letter": letter, "gen_role": role, "gen_letter": letter,
        "truncated": truncated, "error": error, "seconds_seen": 300.0,
    }
    base.update(kw)
    return base


class TestUsable:
    def test_truncated_rows_never_reach_an_accuracy_table(self):
        """"Could not hear it" and "heard it and was wrong" are different findings."""
        rows = [row(), row(truncated=True)]
        assert len(analysis.usable(rows)) == 1

    def test_errored_and_unscored_rows_are_dropped(self):
        rows = [row(), row(error="boom", role=None), row(role=None)]
        assert len(analysis.usable(rows)) == 1

    def test_unverified_rows_are_excluded_by_default(self):
        rows = [row(verified=True), row(verified=False)]
        assert len(analysis.usable(rows)) == 1
        assert len(analysis.usable(rows, require_verified=False)) == 2


class TestTables:
    def test_main_table_reports_the_trap_not_just_the_miss(self):
        rows = [row(role="salience") for _ in range(8)] + [row(role="correct") for _ in range(2)]
        cell = analysis.table1_main(rows, ci=False)[0]
        assert cell["accuracy"] == 0.2
        assert cell["salience_trap"] == 0.8

    def test_ladder_costs_are_measured_against_l1(self):
        rows = ([row(condition="L1", role="correct") for _ in range(10)]
                + [row(condition="L3", role="salience") for _ in range(10)]
                + [row(condition="L4", role="correct") for _ in range(5)]
                + [row(condition="L4", role="salience") for _ in range(5)])
        cell = analysis.table2_ladder(rows)[0]
        assert cell["RetrievalCost"] == 1.0
        assert cell["LongContextCost"] == 0.5

    def test_truncation_table_reports_coverage_not_a_score(self):
        rows = ([row(condition="L1") for _ in range(4)]
                + [row(condition="L3", truncated=True) for _ in range(4)])
        cell = analysis.table4_truncation(rows, adapters={"m": 30.0})[0]
        assert cell["max_audio_s"] == 30.0
        assert cell["coverage"] == 0.5
        assert cell["coverage_L1"] == 1.0
        assert cell["coverage_L3"] == 0.0
        assert "accuracy" not in cell

    def test_null_accuracy_is_reported_separately(self):
        rows = [row(null=True, role="absent") for _ in range(3)] + [row(null=True, role="correct")]
        assert analysis.table1_nulls(rows)[0]["null_accuracy"] == 0.75

    def test_language_table_carries_the_f3prime_cells(self):
        rows = ([row(lang="en", role="correct") for _ in range(4)]
                + [row(lang="hi", role="salience") for _ in range(4)]
                + [row(lang="bn", role="salience") for _ in range(4)])
        table = {r["lang"]: r for r in analysis.table_language(rows)}
        assert table["en"]["salience_trap"] == 0.0
        assert table["hi"]["salience_trap"] == table["bn"]["salience_trap"] == 1.0

    def test_no_table_emits_a_composite_score(self):
        """The paper plan says so, and composites hide which term was zero."""
        rows = [row() for _ in range(4)]
        for table in (analysis.table1_main(rows, ci=False), analysis.table2_ladder(rows),
                      analysis.table4_truncation(rows), analysis.table_language(rows)):
            for cell in table:
                assert not any("composite" in k for k in cell)


class TestScorerGap:
    def test_disagreement_between_scorers_is_surfaced(self):
        rows = [row(logit_letter="A", gen_letter="A") for _ in range(6)] + \
               [row(logit_letter="A", gen_letter="B") for _ in range(4)]
        gap = analysis.scorer_gap(rows)[0]
        assert gap["agreement"] == 0.6

    def test_unparseable_generation_is_a_rate_not_a_wrong_answer(self):
        rows = [row() for _ in range(3)] + [row(gen_letter=None, gen_role=None)]
        gap = analysis.scorer_gap(rows)[0]
        assert gap["unparseable_rate"] == 0.25
        assert gap["n"] == 3


class TestSanityChecks:
    def test_clean_rows_report_nothing(self):
        rows = [row(letter=L, role=r) for L, r in
                zip("ABCDABCD", ["correct", "salience", "recency", "absent"] * 2)]
        assert analysis.sanity_checks(rows) == []

    def test_degenerate_logits_are_flagged(self):
        rows = [row(logit_degenerate=True) for _ in range(3)] + [row()]
        assert any("logits equal" in p for p in analysis.sanity_checks(rows))

    def test_letter_position_bias_is_flagged(self):
        rows = [row(letter="A") for _ in range(9)] + [row(letter="B")]
        assert any("letter-position bias" in p for p in analysis.sanity_checks(rows))

    def test_no_usable_rows_is_itself_the_finding(self):
        assert analysis.sanity_checks([row(truncated=True)]) == ["no usable rows at all"]


@pytest.fixture(scope="module")
def script():
    return load_script("build_item_pack")


class TestBuildScript:
    def test_band_is_the_largest_that_fits(self, script):
        assert script.band_for(1900) == 1800
        assert script.band_for(700) == 600
        assert script.band_for(100) == 300      # floor: shortest band

    def test_category_shares_match_the_paper_plan(self, script):
        assert script.CATEGORY_SHARE == {"P1": 0.22, "P2": 0.22, "P3": 0.22,
                                         "P4": 0.22, "C1": 0.12}
        assert sum(script.CATEGORY_SHARE.values()) == pytest.approx(1.0)

    def test_balance_spreads_across_recordings(self, script):
        """All items from one meeting would make the clustered CIs meaningless."""
        import random

        from undertone.items import MCQItem

        items = [
            MCQItem(item_id=f"i{r}_{k}", recording_id=f"rec{r}", lang="en",
                    category="P3", sector="meetings", audio_path="a.flac",
                    duration_band=300, needle_start=10.0, needle_end=14.0,
                    question="q", options={"correct": "1", "salience": "2",
                                           "recency": "3", "absent": "x"})
            for r in range(4) for k in range(20)
        ]
        picked = script.balance(items, target_total=40, rng=random.Random(0))
        per_recording = {}
        for item in picked:
            per_recording[item.recording_id] = per_recording.get(item.recording_id, 0) + 1
        assert len(per_recording) == 4
        assert max(per_recording.values()) - min(per_recording.values()) <= 1


class TestFigures:
    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        pytest.importorskip("matplotlib")

    def _rows(self):
        out = []
        for lang in ("en", "hi", "bn"):
            for category in ("P1", "P2", "P3", "P4", "C1"):
                for condition in ("L1", "L2", "L3", "L4"):
                    for i in range(4):
                        role = "correct" if condition == "L1" else (
                            "salience" if category != "C1" else "recency")
                        out.append(row(model="m1", category=category, condition=condition,
                                       lang=lang, role=role, recording=f"r{i % 2}",
                                       letter="ABCD"[i % 4], duration_band=300))
        return out

    def test_every_figure_renders(self, tmp_path):
        from undertone.analysis import figures

        paths = figures.all_figures(self._rows(), tmp_path)
        # fig5 needs repetition_count, which these rows do not carry.
        names = {p.stem for p in paths}
        assert {"fig2_signature", "fig3_fingerprint", "fig4_ladder"} <= names
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    def test_figures_on_empty_input_return_nothing(self, tmp_path):
        from undertone.analysis import figures

        assert figures.all_figures([], tmp_path) == []

    def test_figures_exclude_truncated_rows(self, tmp_path):
        """A figure must never show a cell its table excluded."""
        from undertone.analysis import figures

        assert figures.fig4_ladder([row(truncated=True)] * 4, tmp_path) == []

    def test_fig5_needs_repetition_count(self, tmp_path):
        from undertone.analysis import figures

        assert figures.fig5_repetition(self._rows(), tmp_path) == []
        with_counts = [{**r, "repetition_count": 1 + (i % 3)}
                       for i, r in enumerate(self._rows())]
        assert figures.fig5_repetition(with_counts, tmp_path)


@pytest.fixture(scope="module")
def push():
    return load_script("push_kernels")


class TestPushKernels:
    def test_kernels_are_created_private(self, push):
        """Unverified numbers must not go public under the user's name."""
        meta = push.metadata("someone", Path("notebooks/16_aero_1_audio.ipynb"), True)
        assert meta["is_private"] is True

    def test_gpu_only_where_a_model_runs(self, push):
        assert push.metadata("u", Path("10_qwen2_audio_7b.ipynb"), True)["enable_gpu"]
        # 01 harvests on CPU but then transcribes and runs the leak filter.
        assert push.metadata("u", Path("01_build_item_pack.ipynb"), False)["enable_gpu"]
        assert push.metadata("u", Path("02_cascaded_control.ipynb"), True)["enable_gpu"]
        assert not push.metadata("u", Path("90_analysis.ipynb"), True)["enable_gpu"]

    def test_item_pack_is_attached_only_where_it_exists(self, push):
        """01 builds the pack, so it cannot depend on it."""
        assert push.metadata("u", Path("16_aero_1_audio.ipynb"), True)["dataset_sources"]
        assert not push.metadata("u", Path("01_build_item_pack.ipynb"), False)["dataset_sources"]

    def test_gpu_kernels_pin_the_t4_shape(self, push):
        """Every ceiling and VRAM figure in the roster assumes 2xT4 / sm75.
        A P100 from the shared pool would silently be a different experiment."""
        gpu = push.metadata("u", Path("16_aero_1_audio.ipynb"), True)
        assert gpu["machine_shape"] == "NvidiaTeslaT4"
        cpu = push.metadata("u", Path("90_analysis.ipynb"), True)
        assert "machine_shape" not in cpu

    def test_title_fits_kaggle_limit(self, push):
        for stem in ("00_smoke_test", "21_moss_audio_8b_thinking", "22_audio_flamingo_next"):
            assert len(push.metadata("u", Path(f"{stem}.ipynb"), True)["title"]) <= 50


class TestCostStatus:
    def test_ok_when_both_conditions_present(self):
        rows = {"L1": [row(condition="L1")], "L3": [row()]}
        assert analysis.cost_status(rows) == "ok"

    def test_truncated_l3_is_undefined_not_missing(self):
        """A 30 s-capped model cannot reach L3; NaN would read as missing data."""
        status = analysis.cost_status({"L1": [row(condition="L1")], "L3": []}, 30.0)
        assert "undefined" in status and "30s" in status

    def test_no_l1_means_nothing_is_interpretable(self):
        status = analysis.cost_status({"L1": [], "L3": [row()]})
        assert "perception ceiling is unmeasured" in status


class TestL3FailureReason:
    def test_truncated_and_errored_are_different_findings(self):
        """Truncated is architectural; OOM is this GPU. They must not merge."""
        trunc = [row(condition="L3", truncated=True) for _ in range(4)]
        assert analysis.l3_failure_reason(trunc)["reason"].startswith("architecture")

        oom = [row(condition="L3", error="OutOfMemoryError: CUDA out of memory")
               for _ in range(4)]
        r = analysis.l3_failure_reason(oom)
        assert r["reason"].startswith("hardware")
        assert r["of_which_oom"] == 4

    def test_counts_are_exact(self):
        rows = ([row(condition="L3") for _ in range(3)]
                + [row(condition="L3", truncated=True)]
                + [row(condition="L3", error="OutOfMemoryError: x")])
        r = analysis.l3_failure_reason(rows)
        assert (r["n"], r["scored"], r["truncated"], r["errored"]) == (5, 3, 1, 1)

    def test_empty_is_not_an_error(self):
        assert analysis.l3_failure_reason([])["n"] == 0


class TestBandCap:
    def test_bands_are_capped_to_what_a_t4_can_run(self, script):
        """At 1800 s every model errors at L3 - quadratic attention over ~45k
        audio tokens. A band nobody can run measures nothing."""
        assert script.MAX_RUNNABLE_BAND == 600
        assert script.band_for(3600) == 600
        assert script.band_for(700) == 600
        assert script.band_for(400) == 300

    def test_the_cap_is_overridable_for_bigger_hardware(self, script):
        assert script.band_for(3600, cap=1800) == 1800
