#!/usr/bin/env python3
"""Push the generated notebooks to Kaggle as kernels.

    python scripts/push_kernels.py --user <kaggle-username>            # all
    python scripts/push_kernels.py --user <u> --only 00 16             # a subset
    python scripts/push_kernels.py --user <u> --dry-run                # just write metadata

``kaggle kernels push`` wants a directory holding one notebook and its
``kernel-metadata.json``, so each notebook is staged into its own temp directory
rather than reorganising ``notebooks/`` into thirteen subdirectories.

Kernels are created **private**. Publishing a benchmark's model notebooks before
the items are verified would put unverified numbers under your name; flip
``is_private`` yourself when the results are ready to stand behind.
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ITEM_PACK_SLUG = "undertone-item-pack"

# Kaggle's free GPU pool is a single P100 or a dual T4. Every ceiling, VRAM
# figure and dtype choice in the roster assumes 2xT4 / 32 GB / sm75, so the
# shape is pinned rather than left to whatever the pool hands out -- a P100 run
# would silently be a different experiment.
MACHINE_SHAPE = "NvidiaTeslaT4"

# 01 is mixed: harvesting is CPU, but the same notebook then runs Whisper over
# every recording and a 7B text model for the leak filter, so it needs a GPU.
# Only the analysis is genuinely CPU-only.
CPU_ONLY = {"90_analysis"}


def metadata(user: str, notebook: Path, attach_pack: bool) -> dict:
    stem = notebook.stem
    return {
        "id": f"{user}/undertone-{stem.replace('_', '-')}",
        "title": f"UNDERTONE {stem}"[:50],
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": stem not in CPU_ONLY,
        **({} if stem in CPU_ONLY else {"machine_shape": MACHINE_SHAPE}),
        "enable_internet": True,
        "dataset_sources": [f"{user}/{ITEM_PACK_SLUG}"] if attach_pack else [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _inject_token(notebook: Path) -> None:
    """Paste the HF token into a staged notebook copy.

    Only ever the staged copy under a temp directory - never the file in the
    repo, which is committed to a public GitHub repository that the notebooks
    themselves clone at run time. A token pushed there is scraped within
    minutes.

    Even so this leaves the token readable in the Kaggle kernel's source, so the
    version should be deleted once the run finishes.
    """
    from undertone import env

    token = env.hf_token(required=True)
    data = json.loads(notebook.read_text(encoding="utf-8"))
    cell = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# Injected at push time, not present in the repository.\n",
            "# DELETE THIS KERNEL VERSION once the run finishes - the token is\n",
            "# readable in the source of any notebook that carries it.\n",
            "import os\n",
            f'os.environ["HF_TOKEN"] = "{token}"\n',
            'os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]\n',
            'print("HF token set inline")\n',
        ],
    }
    # Before everything, so the pip cell can already authenticate downloads.
    data["cells"].insert(0, cell)
    notebook.write_text(json.dumps(data, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="your Kaggle username")
    ap.add_argument("--notebooks", type=Path, default=Path("notebooks"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="numeric prefixes to push, e.g. 00 16 17")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inject-hf-token", action="store_true",
                    help="paste the token from .hf_token into the STAGED notebook "
                         "copy (never the repo one). For gated models when a "
                         "Kaggle secret is not set. Remove the kernel version "
                         "afterwards - the token is visible in its source.")
    args = ap.parse_args()

    paths = sorted(args.notebooks.glob("*.ipynb"))
    if args.only:
        paths = [p for p in paths if p.stem.split("_")[0] in set(args.only)]
    if not paths:
        print("no notebooks matched; run scripts/make_notebooks.py first")
        return 1

    for path in paths:
        # 01 builds the pack, so it cannot depend on it; 00 needs no items.
        attach = path.stem not in {"00_smoke_test", "01_build_item_pack"}
        meta = metadata(args.user, path, attach)

        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            shutil.copy(path, staged / path.name)
            if args.inject_hf_token:
                _inject_token(staged / path.name)
            (staged / "kernel-metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")

            if args.dry_run:
                print(f"{meta['id']}  gpu={meta['enable_gpu']}  "
                      f"shape={meta.get('machine_shape', '-')}  "
                      f"datasets={meta['dataset_sources']}")
                continue

            print(f"pushing {meta['id']} ...")
            cmd = ["kaggle", "kernels", "push", "-p", str(staged)]
            if meta["enable_gpu"]:
                cmd += ["--accelerator", MACHINE_SHAPE]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print((result.stdout or result.stderr).strip())
            if result.returncode != 0:
                print(f"  failed: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
