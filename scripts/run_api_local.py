#!/usr/bin/env python3
"""Run an API model (OpenAI / Gemini adapters) on the laptop.

    python scripts/run_api_local.py gpt_audio_mini --pack v1 --conditions L1 L3 --limit 3
    python scripts/run_api_local.py gpt_audio_mini --pack v1 v2 --conditions L1 L3 --subset 140

Rows go to results/<family>/<key>/<fingerprint>/results.jsonl, the same
place the Kaggle kernels bank to, and resume cell-by-cell. The rows keep the
pack's real fingerprint even for a subset, so consolidation treats them
like any other run - just with fewer items.

--subset N draws a stratified sample across the packs given, proportional
by (pack, category) with a fixed seed, so a 140-item run is a faithful
miniature of the 408. --limit N takes the first N items of each pack: for
smoke tests only.
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from undertone import ItemPack, adapters, runner  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKS = {"v1": ROOT / "UNDERTONE_report/data/item_packs/v1.jsonl",
         "v2": ROOT / "UNDERTONE_report/data/item_packs/v2.jsonl",
         "600": ROOT / "UNDERTONE_report/data/item_packs/600.jsonl"}
AUDIO_ROOT = {"v1": ROOT / "data/item_pack",
              "v2": ROOT / "data/item_pack_v2", "600": ROOT / "data/item_pack_600"}


class Subset:
    """The items to run, wearing the parent pack's fingerprint."""

    def __init__(self, pack: ItemPack, items):
        self.items, self.fingerprint = list(items), pack.fingerprint

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def stratified(packs: dict[str, ItemPack], n: int, seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    strata = collections.defaultdict(list)
    for name, pack in packs.items():
        for it in pack:
            strata[(name, it.category)].append(it)
    total = sum(len(v) for v in strata.values())
    picked = {name: [] for name in packs}
    # Largest-remainder apportionment so the strata sum to exactly n.
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    base = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - base[k], reverse=True)[: n - sum(base.values())]:
        base[k] += 1
    for (name, _), items in strata.items():
        take = rng.sample(items, min(base[(name, _)], len(items)))
        picked[name].extend(take)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("--pack", nargs="+", default=["v1", "v2"], choices=list(PACKS))
    ap.add_argument("--conditions", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--subset", type=int, help="stratified sample size across the packs")
    ap.add_argument("--limit", type=int, help="first N items per pack (smoke test)")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--budget", type=float, help="OPENAI_BUDGET_USD override")
    args = ap.parse_args()

    adapter = adapters.get_adapter(args.key)
    family = "openai" if args.key.startswith("gpt_") else "gemini"
    out_root = ROOT / "results" / family / args.key
    out_root.mkdir(parents=True, exist_ok=True)
    keyfile = ROOT / (".openai_key" if family == "openai" else ".gemini_key")
    env = "OPENAI_API_KEY" if family == "openai" else "GEMINI_API_KEY"
    if not os.environ.get(env) and keyfile.exists():
        os.environ[env] = keyfile.read_text().strip()
    if args.budget is not None:
        os.environ["OPENAI_BUDGET_USD"] = str(args.budget)
    os.environ.setdefault("OPENAI_SPEND_FILE", str(out_root / "spend.json"))
    os.environ.setdefault("GEMINI_UPLOAD_CACHE", str(out_root / "uploads.json"))
    adapter.load()

    packs = {p: ItemPack.load(PACKS[p]) for p in args.pack}
    if args.subset:
        chosen = stratified(packs, args.subset, args.seed)
    else:
        chosen = {p: list(pack)[: args.limit] if args.limit else list(pack) for p, pack in packs.items()}

    for name, pack in packs.items():
        sub = Subset(pack, chosen[name])
        out = out_root / pack.fingerprint / "results.jsonl"
        print(f"\n=== {name} ({len(sub)} items x {args.conditions}) -> {out}", flush=True)
        runner.run_model(adapter, sub, out, conditions=args.conditions,
                         audio_root=AUDIO_ROOT[name], progress=True)
        rows = [r for r in runner.load_rows(out) if r["item_id"] in {i.item_id for i in sub}]
        ok = [r for r in rows if not r.get("error") and r.get("role_chosen")]
        print(f"rows={len(rows)} valid={len(ok)} errors={len(rows) - len(ok)} "
              f"calls={adapter.calls} spend=${getattr(adapter, 'spend_usd', 0.0):.3f}")
        for c in args.conditions:
            s = [r for r in ok if r["condition"] == c]
            if s:
                print(f"  {c} n={len(s)} acc={sum(r['correct'] for r in s) / len(s):.3f} "
                      f"sal={sum(r['role_chosen'] == 'salience' for r in s) / len(s):.3f} "
                      f"abs={sum(r['role_chosen'] == 'absent' for r in s) / len(s):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
