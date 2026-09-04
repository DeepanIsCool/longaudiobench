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
import shutil
import subprocess
import tempfile
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="your Kaggle username")
    ap.add_argument("--notebooks", type=Path, default=Path("notebooks"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="numeric prefixes to push, e.g. 00 16 17")
    ap.add_argument("--dry-run", action="store_true")
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
