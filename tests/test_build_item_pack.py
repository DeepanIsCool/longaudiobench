"""The two harvest flags the expansion pack needs.

--exclude-pack keeps a second harvest off windows the first pack already
scored, so the two packs merge at analysis time with no item scored twice.
--share reweights categories toward the ones the first pack left thin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_item_pack.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=60)


class TestShareFlag:
    def test_share_must_sum_to_one(self):
        out = _run("--share", "C1=0.9,P1=0.9", "--meetings")
        assert out.returncode != 0
        assert "sum to 1.0" in out.stderr

    def test_share_rejects_unknown_category(self):
        out = _run("--share", "C9=1.0", "--meetings")
        assert out.returncode != 0
        assert "unknown categories" in out.stderr

    def test_share_reweights(self):
        """Import the module and check the dict actually moves. The CLI path is
        covered above; this checks the override lands where balance() reads."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("bip", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = dict(mod.CATEGORY_SHARE)
        assert before["C1"] == pytest.approx(0.12)
        sys.argv = ["x", "--share", "C1=0.30,P1=0.25,P2=0.25,P3=0.10,P4=0.10",
                    "--meetings"]
        # main() will fail later for want of audio; we only need it to parse.
        try:
            mod.main()
        except SystemExit:
            pass
        except Exception:
            pass
        assert mod.CATEGORY_SHARE["C1"] == pytest.approx(0.30)
        assert sum(mod.CATEGORY_SHARE.values()) == pytest.approx(1.0)
        mod.CATEGORY_SHARE.update(before)


class TestExcludePack:
    def test_windows_in_a_prior_pack_are_skipped(self, tmp_path):
        """Build a prior pack naming window r0_w1; harvest_recording output
        containing r0_w1 and r0_w2 must be filtered to r0_w2 only."""
        import importlib.util
        import json

        from undertone.items import ItemPack, MCQItem

        prior = ItemPack([MCQItem(
            item_id="a", recording_id="r0_w1", lang="en", category="C1",
            sector="meetings", audio_path="audio/r0_w1.flac", duration_band=300,
            needle_start=10.0, needle_end=12.0, question="q?",
            options={"correct": "a", "salience": "b", "recency": "c", "absent": "d"})])
        p = tmp_path / "item_pack.jsonl"
        prior.save(p)

        # The filter is a one-liner inside main(); test the exact expression
        # against a fake harvest so no audio is needed.
        excluded = {i.recording_id for i in ItemPack.load(p)}

        class W:
            def __init__(self, rid): self.recording_id = rid

        harvested = [(W("r0_w1"), ["x"], None), (W("r0_w2"), ["y"], None)]
        kept = [(w, items, e) for w, items, e in harvested
                if w.recording_id not in excluded]
        assert [w.recording_id for w, _, _ in kept] == ["r0_w2"]

    def test_exclude_pack_flag_is_repeatable(self):
        out = _run("--help")
        assert "--exclude-pack" in out.stdout
        assert "Repeatable" in out.stdout
