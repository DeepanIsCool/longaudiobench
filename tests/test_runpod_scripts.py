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
        assert '"KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key' in src
        assert "KAGGLE_JSON" not in src.split('"env": {')[1].split("}")[0], \
            "the JSON blob must not reach the pod env; it did not survive the trip"
        assert ".hf_token" not in src.replace("cat .hf_token", "")  # docs only


class TestWatchdogLogic:
    """The watchdog's decision function, exercised on synthetic logs.

    Every branch here is a real failure that billed money: a fast restart
    loop looks alive because run.log grows each cycle; a slow model looks
    dead if only completions are counted. The state() parser is imported
    with the network stubbed out.
    """

    def _state(self, monkeypatch, log, index="", model_dirs=()):
        import importlib.util
        import sys

        monkeypatch.setenv("RUNPOD_API_KEY", "x")
        monkeypatch.setattr(sys, "argv", ["watchdog.py", "pod123", "3", "1"])
        spec = importlib.util.spec_from_file_location("wd", RUNPOD / "watchdog.py")
        src = (RUNPOD / "watchdog.py").read_text()
        # Load only the definitions: cut the file at the polling loop.
        head = src[: src.index("last_sig, last_change")]
        ns = {"__name__": "wd"}
        exec(compile(head, str(RUNPOD / "watchdog.py"), "exec"), ns)

        def fake_curl(args, t=40):
            url = args[-1]
            if url.endswith("/run.log"):
                return log
            if url.endswith("-8000.proxy.runpod.net/"):
                return "".join(f'<a href="{m}/">{m}/</a>' for m in model_dirs)
            for m, files in model_dirs.items() if isinstance(model_dirs, dict) else []:
                if url.endswith(f"/{m}/"):
                    return "".join(f'<a href="{f}">{f}</a>' for f in files)
                for f, body in files.items():
                    if url.endswith(f"/{m}/{f}"):
                        return body
            return ""
        ns["curl"] = fake_curl
        return ns["state"]()

    def test_two_banners_is_a_restart(self, monkeypatch):
        log = "=== RUN START x.sh on A40 ===\npip...\n=== RUN START x.sh on A40 ===\n"
        cells, size, ok, complete, starts = self._state(monkeypatch, log)
        assert starts == 2

    def test_one_banner_is_normal(self, monkeypatch):
        cells, size, ok, complete, starts = self._state(
            monkeypatch, "=== RUN START x.sh on A40 ===\nmodel_a_OK\n")
        assert starts == 1 and ok == 1 and not complete

    def test_complete_marker(self, monkeypatch):
        *_, complete, starts = self._state(
            monkeypatch, "=== RUN START x.sh ===\nRUN_COMPLETE\n")
        assert complete

    def test_cells_counted_in_any_jsonl(self, monkeypatch):
        dirs = {"m1": {"results.jsonl": '{"item_id":1}\n{"item_id":2}\n',
                       "sweep_competitor_default.jsonl": '{"item_id":1}\n'}}
        cells, *_ = self._state(monkeypatch, "=== RUN START ===\n", model_dirs=dirs)
        assert cells == 3

    def test_bootstrap_prints_the_banner_the_watchdog_counts(self):
        assert 'echo "=== RUN START ' in (RUNPOD / "bootstrap.sh").read_text()
        assert 'log.count("=== RUN START ")' in (RUNPOD / "watchdog.py").read_text()

    def test_bootstrap_has_no_set_e_or_set_u(self):
        src = (RUNPOD / "bootstrap.sh").read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "set -e" not in code and "set -u" not in code


class TestDatasetSlugs:
    def test_every_pack_dataset_is_under_the_account_the_cli_uses(self):
        """The first two launches 403'd on a slug copied from an old script
        under a different Kaggle username. Six cents. Every slug in the tree
        must match the username in kaggle.json."""
        import json
        kj = ROOT / "kaggle.json"
        if not kj.exists():
            pytest.skip("no kaggle.json at repo root")
        user = json.loads(kj.read_text())["username"]
        for f in list(RUNPOD.glob("*.sh")) + list(EXPERIMENTS.glob("*.sh")):
            for m in re.finditer(r"([a-z0-9]+)/undertone-item-pack", f.read_text()):
                assert m.group(1) in (user, "<user>"), f"{f.name}: {m.group(0)} is not under {user}"


class TestPackCheck:
    """The pack-present test in bootstrap.sh, run against both layouts Kaggle
    has produced: flat (v1: item_pack.jsonl at the root) and nested
    (item_pack/item_pack.jsonl). The third twins launch aborted a good
    download because `ls` with an unmatched glob exits non-zero."""

    def _check(self, tmp_path, rel):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
        src = (RUNPOD / "bootstrap.sh").read_text()
        line = next(l for l in src.splitlines() if "-name item_pack.jsonl" in l and "if [" in l)
        cond = line.strip()[3:].rstrip("; then").strip()
        out = subprocess.run(["bash", "-c", f'PACK="{tmp_path}"; if {cond}; then echo MISSING; else echo FOUND; fi'],
                             capture_output=True, text=True)
        return out.stdout.strip()

    def test_flat_layout_is_found(self, tmp_path):
        assert self._check(tmp_path, "item_pack.jsonl") == "FOUND"

    def test_nested_layout_is_found(self, tmp_path):
        assert self._check(tmp_path, "item_pack/item_pack.jsonl") == "FOUND"

    def test_empty_is_missing(self, tmp_path):
        src = (RUNPOD / "bootstrap.sh").read_text()
        line = next(l for l in src.splitlines() if "-name item_pack.jsonl" in l and "if [" in l)
        cond = line.strip()[3:].rstrip("; then").strip()
        out = subprocess.run(["bash", "-c", f'PACK="{tmp_path}"; if {cond}; then echo MISSING; else echo FOUND; fi'],
                             capture_output=True, text=True)
        assert out.stdout.strip() == "MISSING"


class TestPinsFile:
    def test_pins_txt_matches_the_generator(self):
        """The pod reads pins.txt; the generator is the source of truth. If
        META changes and this file is not regenerated, the pod runs stale pins."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("pins", RUNPOD / "pins.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        assert (RUNPOD / "pins.txt").read_text() == mod.render(mod.generate()), \
            "run: python scripts/runpod/pins.py --write"

    def test_every_experiment_model_has_pins(self):
        pins = {l.split("|")[0] for l in (RUNPOD / "pins.txt").read_text().splitlines()
                if l and not l.startswith("#")}
        for f in EXPERIMENTS.glob("*.sh"):
            for key in re.findall(r"\b([a-z0-9]+_[a-z0-9_]+)\b", f.read_text()):
                if key.startswith(("cascaded_", "qwen", "moss", "gemma", "phi4", "voxtral", "aero", "audio_flamingo")):
                    assert key in pins, f"{f.name}: {key} has no pins"

    def test_bootstrap_reads_pins_with_grep_not_python(self):
        src = (RUNPOD / "bootstrap.sh").read_text()
        assert 'grep "^${KEY}|"' in src
        assert "python scripts/runpod/pins.py" not in src
