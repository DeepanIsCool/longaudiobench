#!/usr/bin/env python3
"""Fetch finished Kaggle runs and file them under results/<model>/.

Selection is by content, never by directory name: a run is kept only if its
item-id set hashes to the canonical pack. Four incompatible packs (70, 75, 77
and 83 items) reached Kaggle before the fingerprint guard worked, and no item is
shared across all of them, so a name-based rule would silently mix them.

    python scripts/collect_results.py --user <u> --config <dir> [--only 12 22]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CANONICAL_ITEM_SET = "512c3c690559"
PACK_FINGERPRINT = "56bd324cf6a3"
RESULTS = Path("results")
CODE_PIN = "paper-run-1"


def pinned_sha() -> str:
    """The commit the paper table must be scored by, or "" outside a checkout."""
    done = subprocess.run(["git", "rev-list", "-n", "1", CODE_PIN],
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def item_set_id(rows: list[dict]) -> str:
    """Identify the pack a run scored against.

    Prefer the fingerprint stamped on the row: an arm holding only L3 and L4 has
    no L1 rows, so deriving the set from L1 item ids hashes the empty string and
    rejects a perfectly canonical run.
    """
    stamped = {r.get("pack_fingerprint") for r in rows if r.get("pack_fingerprint")}
    if len(stamped) == 1:
        return CANONICAL_ITEM_SET if stamped == {PACK_FINGERPRINT} else stamped.pop()
    ids = sorted({r["item_id"] for r in rows if r.get("condition") == "L1"})
    return hashlib.sha256("".join(ids).encode()).hexdigest()[:12]


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written final line from a killed session
        if row.get("condition") and row.get("model_key"):
            rows.append(row)
    return rows


def usable(rows: list[dict]) -> int:
    return len([r for r in rows if not r.get("error") and r.get("role_chosen")])


def fetch(user: str, slug: str, config: str, dest: Path) -> bool:
    env = dict(os.environ)
    if config:
        env["KAGGLE_CONFIG_DIR"] = config
    done = subprocess.run(
        ["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dest)],
        capture_output=True, text=True, timeout=900, env=env)
    return done.returncode == 0


def collect(user: str, config: str, slugs: list[str]) -> int:
    RESULTS.mkdir(exist_ok=True)
    filed = 0
    for slug in slugs:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if not fetch(user, slug, config, tmp):
                print(f"{slug}: fetch failed")
                continue
            # Not rglob over everything: the kernel output also contains the
            # repo it cloned, and results/ is committed there, so a plain walk
            # finds ten other models' files and would file them as this run's.
            for path in sorted(tmp.rglob("*.jsonl")):
                if "longaudiobench" in path.parts:
                    continue
                rows = read_rows(path)
                if not rows:
                    continue
                model = rows[0]["model_key"]
                found = item_set_id(rows)
                if found != CANONICAL_ITEM_SET:
                    print(f"{slug}: {model} ran pack {found}, not "
                          f"{CANONICAL_ITEM_SET} - not filed")
                    continue
                out = RESULTS / model
                out.mkdir(exist_ok=True)
                existing = out / "results.jsonl"
                if existing.exists():
                    old_rows = read_rows(existing)
                    # A run that records the commit it cloned always wins over
                    # one that does not, whatever the cell counts say. The old
                    # files predate code_sha and some carry duplicate cells from
                    # a double run, so "more usable cells" would keep an
                    # unattributable 332-row file over a clean pinned 280.
                    # Match the CURRENT pin, not merely "has some sha". The pin
                    # moved once to pick up the sweep wiring, and treating any
                    # recorded sha as pinned made three runs at the old commit
                    # block their own replacements on cell count.
                    pin = pinned_sha()
                    incoming_pinned = bool(pin) and any(
                        r.get("code_sha") == pin for r in rows)
                    existing_pinned = bool(pin) and any(
                        r.get("code_sha") == pin for r in old_rows)
                    if existing_pinned and not incoming_pinned:
                        print(f"{slug}: {model} is unpinned, keeping the "
                              "pinned run already on disk")
                        continue
                    if (existing_pinned == incoming_pinned
                            and usable(old_rows) >= usable(rows)):
                        print(f"{slug}: {model} {usable(rows)} cells, "
                              f"keeping the better run already on disk")
                        continue
                shutil.copy2(path, existing)
                for extra in tmp.rglob("*summary*.json"):
                    shutil.copy2(extra, out / "summary.json")
                    break
                for log in tmp.rglob("*.log"):
                    shutil.copy2(log, out / "kernel.log")
                    break
                print(f"{slug}: filed {model} with {usable(rows)}/{len(rows)} cells")
                filed += 1
    return filed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--config", default="", help="KAGGLE_CONFIG_DIR for this account")
    ap.add_argument("--only", nargs="*", default=[], help="notebook number prefixes")
    args = ap.parse_args()

    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    slugs = []
    for nb in notebooks:
        number = nb.stem.split("_")[0]
        if args.only and number not in args.only:
            continue
        # 02 is the cascaded control - not a model, but a scored arm the paper
        # needs, so it is collected like one. 00 smoke-tests, 01 builds the
        # pack and 90 only reads what the others wrote.
        if number in {"00", "01", "90"}:
            continue
        slugs.append(f"undertone-{nb.stem.replace('_', '-')}")
    print(f"checking {len(slugs)} kernels on {args.user}")
    print(f"filed {collect(args.user, args.config, slugs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
