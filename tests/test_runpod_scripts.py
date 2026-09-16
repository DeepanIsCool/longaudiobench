"""The pod-side scripts must parse and self-describe without torch present.

They run in a venv per model on the pod, so the laptop cannot execute them
end to end. What it can check: every script parses, `--help` works, the
sweep spec grammar is what the experiment drivers use, and no script in the
tree carries a literal secret.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNPOD = ROOT / "scripts" / "runpod"
EXPERIMENTS = ROOT / "experiments"


def _py(script, *args):
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, timeout=60)


class TestHelp:
    @pytest.mark.parametrize("script", ["rp.py", "run_model.py", "pull.py"])
    def test_help_needs_no_torch_and_no_key(self, script, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        out = _py(RUNPOD / script, "--help")
        assert out.returncode == 0, out.stderr
        assert "usage" in out.stdout.lower()

    def test_rp_refuses_without_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        out = _py(RUNPOD / "rp.py", "status")
        assert out.returncode != 0
        assert "RUNPOD_API_KEY" in out.stderr

    def test_run_model_sweep_spec_grammar(self):
        """Every --sweep in the experiment drivers must be arm:levels with
        names run_model.py knows, or the pod fails after the venv is built."""
        from undertone import sweep

        levels = {"default", "calibrated", "boost"}
        specs = re.findall(r"--sweep\s+(\S+)", "\n".join(
            p.read_text() for p in EXPERIMENTS.glob("*.sh")))
        assert specs, "no --sweep in any driver"
        for spec in specs:
            arm, _, lv = spec.partition(":")
            assert arm in sweep.EDIT_TARGETS, spec
            assert (lv or "default") in levels, spec


class TestBashSyntax:
    @pytest.mark.parametrize("script", sorted(p.name for p in EXPERIMENTS.glob("*.sh")))
    def test_driver_parses(self, script):
        out = subprocess.run(["bash", "-n", str(EXPERIMENTS / script)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_bootstrap_parses(self):
        out = subprocess.run(["bash", "-n", str(RUNPOD / "bootstrap.sh")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_every_driver_sources_bootstrap_and_finishes(self):
        for p in EXPERIMENTS.glob("*.sh"):
            text = p.read_text()
            assert "bootstrap.sh" in text, p.name
            assert text.rstrip().endswith("finish"), f"{p.name} must end with finish"
            assert "set -e" not in text, f"{p.name}: set -e makes Runpod restart the pod"


class TestNoSecrets:
    PATTERNS = [r"hf_[A-Za-z0-9]{30,}", r"rpa_[A-Za-z0-9]{30,}",
                r'"key"\s*:\s*"[0-9a-f]{32}"']

    def test_no_literal_tokens_in_tracked_scripts(self):
        files = list(RUNPOD.glob("*")) + list(EXPERIMENTS.glob("*")) + \
            list((ROOT / "scripts").glob("*.py"))
        for f in files:
            if f.is_dir():
                continue
            text = f.read_text(errors="ignore")
            for pat in self.PATTERNS:
                assert not re.search(pat, text), f"{f}: matches {pat}"

    def test_launch_reads_secrets_from_env_only(self):
        src = (RUNPOD / "rp.py").read_text()
        assert 'os.environ.get("HF_TOKEN"' in src
        assert 'os.environ.get("KAGGLE_JSON"' in src
        assert ".hf_token" not in src.replace("cat .hf_token", "")  # docs only
