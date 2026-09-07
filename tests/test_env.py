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
        """load_kwargs resolves a real torch dtype, so this needs torch."""
        pytest.importorskip("torch")
        from undertone.adapters.base import get_adapter

        a = get_adapter("moss_audio_4b_instruct")
        a._hardware = env.Hardware("mps", "float16", None, "Apple MPS", 8.0, False)
        assert "device_map" not in a.load_kwargs()


class TestGemmaDevicePlacement:
    def test_e2b_is_pinned_to_one_gpu(self):
        """~11 GB fits a T4; splitting it produced a cuda:0/cuda:1 mismatch."""
        from undertone.adapters.base import _REGISTRY

        assert _REGISTRY["gemma3n_e2b"].prefers_single_device

    def test_e4b_overflows_to_cpu_rather_than_a_second_gpu(self):
        """~16 GB does not fit one T4, and two GPUs break the same way."""
        from undertone.adapters.base import _REGISTRY

        cls = _REGISTRY["gemma3n_e4b"]
        assert cls.single_gpu_with_cpu_overflow
        assert not cls.prefers_single_device

    def test_cpu_overflow_keeps_gpu_tensors_on_one_device(self):
        pytest.importorskip("torch")
        from undertone.adapters.base import get_adapter

        a = get_adapter("gemma3n_e4b")
        a._hardware = env.Hardware("cuda", "float16", "auto", "2x T4", 31.0, False)
        kw = a.load_kwargs()
        assert set(kw["max_memory"]) == {0, "cpu"}


class TestMoss8BDevicePlacement:
    def test_8b_overflows_to_cpu_like_gemma_e4b(self):
        """18 GB will not fit a T4, and masked_scatter_ breaks across two."""
        from undertone.adapters.base import _REGISTRY

        for key in ("moss_audio_8b_instruct", "moss_audio_8b_thinking"):
            cls = _REGISTRY[key]
            assert cls.single_gpu_with_cpu_overflow, key
            assert not cls.prefers_single_device, key

    def test_every_model_over_15gb_avoids_a_two_gpu_split(self):
        """The device-mismatch class of bug hit MOSS-4B, Gemma-E2B and
        Gemma-E4B in turn; models that cannot fit one card must say how."""
        from undertone.adapters.base import _REGISTRY

        for key in ("moss_audio_8b_instruct", "moss_audio_8b_thinking",
                    "gemma3n_e4b"):
            cls = _REGISTRY[key]
            assert cls.single_gpu_with_cpu_overflow or cls.prefers_single_device, key


class TestHardwareBlocked:
    def test_gemma_e4b_is_marked_blocked_not_broken(self):
        """16 GB fp16 against a 15.6 GB T4: splitting gives a device mismatch,
        CPU overflow gives one inside clamp. Absent-because-too-big is a
        different fact from ran-and-scored-badly."""
        from undertone.adapters.base import _REGISTRY

        assert _REGISTRY["gemma3n_e4b"].hardware_blocked

    def test_models_that_fit_are_not_marked(self):
        from undertone.adapters.base import _REGISTRY

        for key in ("aero_1_audio", "voxtral_mini_3b", "gemma3n_e2b",
                    "moss_audio_4b_instruct"):
            assert _REGISTRY[key].hardware_blocked is None, key

    def test_the_flag_reaches_the_results(self):
        from undertone.adapters.base import get_adapter

        assert "hardware_blocked" in get_adapter("gemma3n_e4b").describe()

    def test_cpu_overflow_leaves_room_for_activations(self):
        """MOSS-8B passed smoke on a 20 s clip and OOM'd mid-sweep with 88 MiB
        free - 14 GiB of weights on a 14.56 GiB card."""
        pytest.importorskip("torch")
        from undertone.adapters.base import get_adapter

        a = get_adapter("moss_audio_8b_instruct")
        a._hardware = env.Hardware("cuda", "float16", "auto", "2x T4", 31.0, False)
        assert a.load_kwargs()["max_memory"][0] == "11GiB"


class TestAttentionBackend:
    # Models whose own code refuses sdpa. Each entry is a load failure, not a
    # preference: phi4's remote code wants eager, and Gemma-3n's vision tower is
    # a TimmWrapperModel that raises "does not support an attention
    # implementation through torch.nn.functional.scaled_dot_product_attention".
    SDPA_EXEMPT = {"phi4_multimodal": None,
                   "gemma3n_e2b": "eager", "gemma3n_e4b": "eager"}

    def test_every_model_asks_for_sdpa_unless_its_code_refuses(self):
        """The math kernel materialises the full attention matrix - a 60 GiB
        allocation over ~45k audio tokens, which is what made L3 unreachable.
        Voxtral, MOSS and AF-Next were all defaulting to it."""
        from undertone import adapters

        for key in adapters.list_adapters():
            a = adapters.get_adapter(key)
            if key in self.SDPA_EXEMPT:
                assert a.attn_implementation == self.SDPA_EXEMPT[key], key
            else:
                assert a.attn_implementation == "sdpa", key

    def test_the_preference_is_reported_not_assumed(self):
        """Returns what was actually enabled - silently failing to switch
        backends would leave the memory wall exactly where it was."""
        result = env.prefer_memory_efficient_attention()
        assert result in {"torch absent", "cpu", "default",
                          "mem_efficient (math fallback)",
                          "mem_efficient (legacy toggles)"}

    def test_flash_is_not_requested_on_sm75(self):
        import inspect

        src = inspect.getsource(env.prefer_memory_efficient_attention)
        assert "enable_flash_sdp(False)" in src


class TestNoDuplicateAttnKwarg:
    def test_no_adapter_passes_attn_implementation_explicitly(self):
        """base.load_kwargs supplies it; passing it in the from_pretrained call
        too is a duplicate keyword and the load fails outright - which is how
        the attention fix broke every adapter that had set it by hand."""
        import pathlib
        import re

        for f in pathlib.Path("undertone/adapters").glob("*.py"):
            if f.name == "base.py":
                continue
            src = f.read_text()
            for m in re.finditer(r"from_pretrained\([^)]*\)", src, re.S):
                call = m.group(0)
                if "load_kwargs()" not in call:
                    continue
                # Exactly the kwarg base supplies. Phi-4's remote code takes
                # `_attn_implementation`, a different name, and it opts out of
                # the base one by setting attn_implementation = None.
                assert not re.search(r"(?<!_)\battn_implementation\s*=", call), \
                    f"{f.name}: {call[:80]}"


class TestWeightBudget:
    """The 140-error pattern: MOSS and Omni-7B failed at exactly L3 and L4 -
    the conditions carrying 5 minutes of audio - because weights had taken the
    whole card and left nothing for activations."""

    def test_budget_covers_the_worst_observed_activation(self):
        from undertone.adapters.base import WEIGHT_BUDGET_GIB

        card_gib = 14.56          # a Kaggle T4 as reported by torch
        worst_activation = 6.07   # Qwen2.5-Omni-7B, measured
        assert card_gib - WEIGHT_BUDGET_GIB > worst_activation, (
            "weights leave less headroom than the largest allocation seen")

    def test_moss_would_now_have_room(self):
        """It missed by 0.3 GiB: 2.15 GiB needed, 1.83 GiB free."""
        from undertone.adapters.base import WEIGHT_BUDGET_GIB

        assert 14.56 - WEIGHT_BUDGET_GIB > 2.15

    def test_every_offloaded_model_is_capped(self):
        pytest.importorskip("torch")
        from undertone.adapters.base import _REGISTRY, get_adapter

        for key, cls in _REGISTRY.items():
            if not (cls.prefers_single_device or cls.single_gpu_with_cpu_overflow
                    or cls.needs_balancing):
                continue
            a = get_adapter(key)
            a._hardware = env.Hardware("cuda", "float16", "auto", "2x T4", 31.0, False)
            assert a.load_kwargs().get("max_memory"), key

    def test_pinning_to_one_device_still_caps_it(self):
        """prefers_single_device used to mean 'cuda:0, uncapped', which is how
        12.73 GiB of weights ended up on a 14.56 GiB card."""
        import inspect

        from undertone.adapters import base

        src = inspect.getsource(base.ModelAdapter.load_kwargs)
        assert 'kwargs["device_map"] = "cuda:0"' not in src


class TestBalancedSplitStaysOnTheGpus:
    """Qwen2.5-Omni-7B cloned the weight-budget fix and still OOM'd 140 cells.

    Its log carried "Some parameters are on the meta device because they were
    offloaded to the cpu": a "cpu" entry in max_memory made accelerate offload a
    model that fits across two T4s, and staging those layers back through GPU 0
    left 428 MiB free where the uncapped run had left 4.66 GiB.
    """

    def test_balancing_offers_no_cpu(self):
        """Runs without a GPU: the CUDA hardware is stubbed in, so the branch is
        exercised on any machine rather than skipped where it matters least."""
        import undertone.env as env
        from undertone.adapters.base import _REGISTRY
        from undertone.env import Hardware

        cuda = Hardware(backend="cuda", dtype="float16", device_map="auto",
                        detail="2x Tesla T4 (sm75)", total_memory_gb=14.56,
                        supports_bf16=False)
        balanced = [k for k, c in _REGISTRY.items() if c.needs_balancing]
        assert balanced, "no adapter uses the balanced split; test is vacuous"
        real_dtype = env.torch_dtype
        env.torch_dtype = lambda *a, **k: "float16"   # the only line needing torch
        try:
            maps = {}
            for key in balanced:
                adapter = _REGISTRY[key].__new__(_REGISTRY[key])
                adapter._hardware = cuda
                maps[key] = adapter.load_kwargs().get("max_memory", {})
        finally:
            env.torch_dtype = real_dtype
        for key, max_memory in maps.items():
            assert "cpu" not in max_memory, (
                f"{key} offers CPU; accelerate will offload and stage through "
                "GPU 0 instead of splitting across the two cards")
            assert 1 in max_memory, f"{key} must be given the second card"

    def test_two_capped_gpus_hold_the_real_weights(self):
        """Sizes read off the Hub, not the adapter notes.

        The Omni-7B note claimed "~14GB fp16" - that is the Thinker alone. The
        checkpoint is 22.4 GB (thinker + talker + token2wav), which is why an
        8+8 GiB split offloaded to disk and kept failing.
        """
        from undertone.adapters.base import (FIRST_GPU_BUDGET_GIB,
                                             SECOND_GPU_BUDGET_GIB)

        budget = FIRST_GPU_BUDGET_GIB + SECOND_GPU_BUDGET_GIB
        audio_flamingo_gib = 16.5 / 1.074
        omni_thinker_gib = 22.4 * (1346 / 2448) / 1.074   # thinker's tensor share
        assert budget > audio_flamingo_gib, "Audio-Flamingo will offload again"
        assert budget > omni_thinker_gib, "Omni-7B Thinker will offload again"

    def test_the_two_long_context_failures_are_hardware_not_config(self):
        """No split of a 14.56 GiB card fits these two at L3.

        A lopsided 5/13 split was tried and did free 3.41 GiB on cuda:0 for
        Audio-Flamingo, which still needed 17.31 GiB there; Omni-7B just moved
        its OOM to cuda:1 and lost L2. Both shortfalls are positive, so the
        remaining failures are a hardware limit and not a budget to re-tune.
        """
        from undertone.adapters.base import (AF_NEXT_L3_SHORTFALL_GIB,
                                             OMNI_7B_L3_SHORTFALL_GIB)

        assert AF_NEXT_L3_SHORTFALL_GIB > 0
        assert OMNI_7B_L3_SHORTFALL_GIB > 0


class TestAttentionOverrides:
    """sdpa is the class default because the math kernel materialises the full
    attention matrix. Not every model accepts it."""

    def test_gemma3n_does_not_request_sdpa(self):
        from undertone.adapters.base import _REGISTRY

        for key, cls in _REGISTRY.items():
            if key.startswith("gemma3n"):
                assert cls.attn_implementation == "eager", (
                    f"{key} requests {cls.attn_implementation}; its TimmWrapper "
                    "vision tower rejects sdpa and the adapter fails to load")
