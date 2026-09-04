"""Credentials and hardware resolution.

Two jobs, both about making runs reproducible without leaking anything.

**Tokens.** Resolved from a precedence chain, never from source. The repo is
public and its notebooks clone it at run time, so a token written into a tracked
file is published to anyone who visits -- and scraped from GitHub within
minutes. ``.hf_token`` is gitignored; Kaggle Secrets is the path on Kaggle.

**Devices.** Every model in the roster loads in fp16 wherever fp16 is real, so
the only thing that varies across machines is the backend. That variation is
recorded in every result row and the analysis refuses to put two backends in one
table: CUDA and MPS do not produce identical logits, and a benchmark whose rows
came from different kernels is not comparing models, it is comparing machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TOKEN_FILE = ".hf_token"


@lru_cache(maxsize=4)
def hf_token(required: bool = False) -> str | None:
    """HF token, in precedence order. Never read from tracked source.

    1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the environment
    2. Kaggle Secrets (the path the gated-model notebooks use)
    3. ``.hf_token`` at the repo root, or ``~/.hf_token`` -- both gitignored
    4. ``~/.cache/huggingface/token``, written by ``huggingface-cli login``
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret("HF_TOKEN")
        if value:
            return value.strip()
    except Exception:  # noqa: BLE001 - not on Kaggle, or the secret is unset
        pass

    for candidate in (Path(__file__).resolve().parent.parent / TOKEN_FILE,
                      Path.home() / TOKEN_FILE,
                      Path.home() / ".cache/huggingface/token"):
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue

    if required:
        raise RuntimeError(
            "no Hugging Face token found. Set HF_TOKEN, add it as a Kaggle secret "
            f"named HF_TOKEN, or write it to {TOKEN_FILE} at the repo root "
            "(gitignored). Gated repos -- Gemma-3n E2B and E4B -- cannot load "
            "without one.")
    return None


def export_hf_token() -> bool:
    """Put the resolved token in the environment for transformers to pick up."""
    token = hf_token()
    if not token:
        return False
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return True


@dataclass(frozen=True)
class Hardware:
    backend: str          # "cuda" | "mps" | "cpu"
    dtype: str            # "float16" | "float32"
    device_map: str | None
    detail: str
    total_memory_gb: float
    supports_bf16: bool

    @property
    def signature(self) -> str:
        """What must match across every row in one results table."""
        return f"{self.backend}/{self.dtype}"


def resolve_hardware() -> Hardware:
    """What this machine can actually run, and in what precision.

    fp16 on CUDA and MPS; **fp32 on CPU**, because torch leaves many fp16 CPU
    kernels unimplemented and a half-supported path fails deep inside a forward
    pass rather than at load. The cost is that CPU rows are not comparable with
    GPU rows, which is exactly why the signature is recorded.

    Degrades to a torchless CPU descriptor rather than raising: building an item
    pack and rendering tables need neither torch nor a GPU, and a hard import
    here would make those paths unrunnable on a plain machine.
    """
    try:
        import torch
    except ImportError:
        return Hardware("cpu", "float32", None, "CPU (torch not installed)", 0.0, False)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = sum(torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count())) / 1e9
        return Hardware(
            backend="cuda",
            dtype="float16",
            device_map="auto",
            detail=f"{torch.cuda.device_count()}x {props.name} "
                   f"(sm{props.major}{props.minor})",
            total_memory_gb=round(total, 1),
            # sm80+ has bf16, but we stay on fp16 everywhere so a run on an A100
            # is comparable with a run on a T4.
            supports_bf16=props.major >= 8,
        )

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        import subprocess

        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9
        except Exception:  # noqa: BLE001
            total = 0.0
        return Hardware("mps", "float16", None, "Apple MPS (unified memory)",
                        round(total, 1), False)

    return Hardware("cpu", "float32", None, "CPU", 0.0, False)


def torch_dtype(hardware: Hardware | None = None):
    import torch

    return getattr(torch, (hardware or resolve_hardware()).dtype)


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is not installed. Model sweeps need it; building an item pack "
            "and rendering tables do not.") from exc


def fits(required_gb: float, hardware: Hardware | None = None,
         headroom: float = 0.80) -> bool:
    """Whether a model of ``required_gb`` plausibly fits.

    On MPS the budget is unified memory shared with the OS, so the headroom is
    not optional -- an 8 GB Mac does not have 8 GB for a model.
    """
    hardware = hardware or resolve_hardware()
    if hardware.total_memory_gb <= 0:
        return False
    return required_gb <= hardware.total_memory_gb * headroom


def versions() -> dict[str, str]:
    """Recorded in every row so a results table can prove it is homogeneous."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for package in ("torch", "transformers", "accelerate", "librosa"):
        try:
            out[package] = md.version(package)
        except md.PackageNotFoundError:
            out[package] = "absent"
    return out
