"""The generated notebooks must be valid, complete and internally consistent.

Regenerating is a one-line command, so the only way these drift is if nobody
checks. This checks.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def notebooks(tmp_path_factory) -> dict[str, dict]:
    out = tmp_path_factory.mktemp("nb")
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_notebooks.py"),
                    "--out", str(out)], check=True, cwd=REPO)
    return {p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out.glob("*.ipynb"))}


def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_one_notebook_per_model_plus_four_shared(notebooks):
    """13 models, plus smoke test, item-pack build, cascaded control, analysis."""
    from undertone import adapters

    assert len(notebooks) == len(adapters.list_adapters()) + 4 == 17
    for shared in ("00_smoke_test.ipynb", "01_build_item_pack.ipynb",
                   "02_cascaded_control.ipynb", "90_analysis.ipynb"):
        assert shared in notebooks


def test_cascaded_control_has_no_model_notebook_of_its_own_number(notebooks):
    """It is a validity check, not the fourteenth model under test."""
    from undertone import adapters

    assert "cascaded_whisper_llm" not in adapters.list_adapters()
    joined = "\n".join(code_cells(notebooks["02_cascaded_control.ipynb"]))
    assert "CascadedWhisperLLM" in joined
    assert "faster-whisper" in joined


def test_every_code_cell_is_valid_python(notebooks):
    """Template brace escaping is easy to get wrong and silent until runtime."""
    for name, nb in notebooks.items():
        for i, src in enumerate(code_cells(nb)):
            stripped = "\n".join(
                "pass" if re.match(r"\s*[%!]", line) else line for line in src.split("\n"))
            try:
                ast.parse(stripped)
            except SyntaxError as exc:
                pytest.fail(f"{name} code cell {i}: {exc}")


def test_every_model_notebook_names_its_own_adapter(notebooks):
    from undertone import adapters

    for key in adapters.list_adapters():
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        assert f'ADAPTER_KEY = "{key}"' in "\n".join(code_cells(notebooks[name]))


def test_every_model_notebook_states_its_ceiling(notebooks):
    from undertone import adapters
    from undertone.adapters.base import get_adapter

    for key in adapters.list_adapters():
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        header = "".join(notebooks[name]["cells"][0]["source"])
        assert f"{get_adapter(key).max_audio_s:.0f} s" in header, key


def test_gated_models_hard_fail_without_a_token(notebooks):
    """Token resolution is shared; only the *requirement* differs."""
    for key in ("gemma3n_e2b", "gemma3n_e4b"):
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        assert "GATED = True" in "\n".join(code_cells(notebooks[name])), key
    for key in ("aero_1_audio", "voxtral_mini_3b"):
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        assert "GATED = True" not in "\n".join(code_cells(notebooks[name])), key


def test_no_notebook_contains_a_literal_token(notebooks):
    """These get pushed to Kaggle and the repo they clone from is public."""
    for name, nb in notebooks.items():
        joined = "\n".join(code_cells(nb))
        assert "hf_" + "m" not in joined, name
        assert "hf_" + "read" not in joined, name


def test_every_notebook_records_its_hardware_signature(notebooks):
    """A results table mixing backends compares machines, not models."""
    for name, nb in notebooks.items():
        assert "signature" in "\n".join(code_cells(nb)), name


def test_models_needing_a_different_pin_get_one(notebooks):
    """No single transformers release loads the whole roster."""
    expected = {
        "phi4_multimodal": "transformers==4.48.2",      # version-sensitive remote code
        "aero_1_audio": "transformers==4.52.4",         # needs video_utils AND Qwen2AudioFlashAttention2
        "audio_flamingo_next": "transformers>=5.0.0",   # musicflamingo not in 4.x
    }
    for key, pin in expected.items():
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        assert pin in "\n".join(code_cells(notebooks[name])), key


def test_every_model_notebook_smoke_checks_itself(notebooks):
    """The shared smoke test cannot cover a roster needing three pins."""
    from undertone import adapters

    for key in adapters.list_adapters():
        name = next(n for n in notebooks if n.endswith(f"{key}.ipynb"))
        joined = "\n".join(code_cells(notebooks[name]))
        assert "smoke_adapter" in joined, key
        assert "assert report.get(\"ok\")" in joined, key


def test_phi4_gets_its_own_transformers_pin(notebooks):
    """Its remote code is version-sensitive; sharing the roster pin breaks it."""
    name = next(n for n in notebooks if n.endswith("phi4_multimodal.ipynb"))
    assert "transformers==4.48.2" in "\n".join(code_cells(notebooks[name]))
    other = next(n for n in notebooks if n.endswith("voxtral_mini_3b.ipynb"))
    assert "transformers==4.48.2" not in "\n".join(code_cells(notebooks[other]))


def test_voxtral_installs_mistral_common(notebooks):
    name = next(n for n in notebooks if n.endswith("voxtral_mini_3b.ipynb"))
    assert "mistral-common" in "\n".join(code_cells(notebooks[name]))


def test_weights_are_cached_outside_the_output_quota(notebooks):
    """A 16-18 GB checkpoint in /kaggle/working fails the 20 GB commit."""
    for name, nb in notebooks.items():
        joined = "\n".join(code_cells(nb))
        assert '"HF_HOME", "/kaggle/temp/hf"' in joined, name


def test_every_notebook_is_gpu_and_seeded(notebooks):
    for name, nb in notebooks.items():
        assert nb["metadata"]["accelerator"] == "GPU", name
        assert "SEED = 20260904" in "\n".join(code_cells(nb)), name


def test_build_notebook_gates_on_verification(notebooks):
    """An item nobody has listened to is a proposal, not evidence."""
    joined = "\n".join(code_cells(notebooks["01_build_item_pack.ipynb"]))
    assert "leakfilter" in joined
    assert "question_only" in joined          # the ~25% floor check
    assert "VERDICTS" in joined               # the listening pass


def test_analysis_notebook_reports_truncation_separately(notebooks):
    joined = "\n".join(code_cells(notebooks["90_analysis.ipynb"]))
    assert "table4_truncation" in joined
    assert "sanity_checks" in joined
    assert "composite" not in joined.lower()
