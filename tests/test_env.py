"""Credential resolution and the mixed-hardware guard."""

from __future__ import annotations

import pytest

from undertone import env
from undertone.analysis import tables


class TestToken:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        env.hf_token.cache_clear()
        yield
        env.hf_token.cache_clear()

    def test_environment_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "  from_env  ")
        assert env.hf_token() == "from_env"

    def test_falls_back_to_the_gitignored_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        (home / env.TOKEN_FILE).write_text("from_file\n")
        monkeypatch.setattr(env.Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(env, "__file__", str(tmp_path / "nowhere" / "env.py"))
        assert env.hf_token() == "from_file"

    def test_missing_token_is_an_explicit_error_not_a_silent_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(env.Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(env, "__file__", str(tmp_path / "nowhere" / "env.py"))
        assert env.hf_token() is None
        env.hf_token.cache_clear()
        with pytest.raises(RuntimeError, match="Gemma-3n"):
            env.hf_token(required=True)

    def test_token_is_never_read_from_tracked_source(self):
        """The repo is public and its notebooks clone it at run time."""
        import pathlib

        root = pathlib.Path(env.__file__).resolve().parent.parent
        gitignore = (root / ".gitignore").read_text()
        assert env.TOKEN_FILE in gitignore
        for path in list((root / "undertone").rglob("*.py")) + \
                list((root / "scripts").glob("*.py")):
            assert "hf_" + "m" not in path.read_text(), f"literal token in {path}"


class TestHardware:
    def test_resolves_without_torch(self):
        """Item packs and tables need neither torch nor a GPU."""
        hardware = env.resolve_hardware()
        assert hardware.backend in {"cuda", "mps", "cpu"}
        assert hardware.dtype in {"float16", "float32"}
        assert hardware.signature == f"{hardware.backend}/{hardware.dtype}"

    def test_cpu_uses_fp32(self):
        """torch leaves fp16 CPU kernels unimplemented; they fail mid-forward."""
        cpu = env.Hardware("cpu", "float32", None, "CPU", 0.0, False)
        assert cpu.dtype == "float32"
        assert cpu.device_map is None

    def test_fits_leaves_headroom_for_the_os(self):
        mac = env.Hardware("mps", "float16", None, "Apple MPS", 8.0, False)
        assert env.fits(3.6, mac)        # Aero-1-Audio, marginally
        assert not env.fits(9.5, mac)    # Voxtral exceeds total system RAM
        assert not env.fits(7.0, mac)    # inside 8 GB but not inside the headroom

    def test_versions_are_recorded_even_when_absent(self):
        recorded = env.versions()
        assert set(recorded) >= {"torch", "transformers"}
        assert all(isinstance(v, str) for v in recorded.values())


def row(signature="cuda/float16", **kw):
    base = {"model_key": "m", "category": "P3", "condition": "L3", "lang": "en",
            "recording_id": "r0", "is_null": False, "correct_role": "correct",
            "role_chosen": "correct", "letter_chosen": "A", "truncated": False,
            "error": None, "signature": signature}
    base.update(kw)
    return base


class TestMixedHardwareGuard:
    def test_one_backend_is_fine(self):
        assert len(tables.usable([row(), row()])) == 2

    def test_two_backends_refuse_to_share_a_table(self):
        """One model on a Mac and twelve on a T4 compares kernels, not models."""
        rows = [row(model_key="aero", signature="mps/float16"),
                row(model_key="voxtral", signature="cuda/float16")]
        with pytest.raises(tables.MixedHardware, match="more than one backend"):
            tables.usable(rows)

    def test_the_error_names_which_models_are_the_odd_ones_out(self):
        rows = [row(model_key="aero", signature="mps/float16")] + \
               [row(model_key=f"m{i}") for i in range(3)]
        with pytest.raises(tables.MixedHardware, match="aero"):
            tables.usable(rows)

    def test_mixing_can_be_allowed_explicitly(self):
        rows = [row(signature="mps/float16"), row(signature="cuda/float16")]
        assert len(tables.usable(rows, allow_mixed_hardware=True)) == 2

    def test_sanity_checks_report_it_rather_than_raising(self):
        rows = [row(signature="mps/float16"), row(signature="cuda/float16")]
        problems = tables.sanity_checks(rows)
        assert any("different backends" in p for p in problems)


class TestSingleDevicePreference:
    def test_moss_4b_is_not_sharded(self):
        """masked_scatter_ needs source and target co-resident; accelerate
        splitting a 10.4 GB model across two T4s broke it."""
        from undertone.adapters.base import _REGISTRY

        assert _REGISTRY["moss_audio_4b_instruct"].prefers_single_device
        assert _REGISTRY["moss_audio_4b_thinking"].prefers_single_device

    def test_models_that_need_both_gpus_still_shard(self):
        from undertone.adapters.base import _REGISTRY

        for key in ("qwen2_audio_7b", "moss_audio_8b_instruct", "audio_flamingo_next"):
            assert not _REGISTRY[key].prefers_single_device, key

    def test_single_device_only_applies_on_cuda(self):
        from undertone.adapters.base import get_adapter

        a = get_adapter("moss_audio_4b_instruct")
        a._hardware = env.Hardware("mps", "float16", None, "Apple MPS", 8.0, False)
        assert "device_map" not in a.load_kwargs()
