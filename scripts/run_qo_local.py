#!/usr/bin/env python3
"""Question-only control for an API model, on the laptop, resumable.

    python scripts/run_qo_local.py gpt_audio_mini --budget 15
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from undertone import ItemPack, adapters, runner, sweep  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("--budget", type=float)
ap.add_argument("--pack", nargs="+", default=["v1", "v2"]); a = ap.parse_args()
family = "openai" if a.key.startswith("gpt_") else "gemini"
out_root = ROOT / "results" / family / a.key
keyfile = ROOT / (".openai_key" if family == "openai" else ".gemini_key")
env = "OPENAI_API_KEY" if family == "openai" else "GEMINI_API_KEY"
os.environ.setdefault(env, keyfile.read_text().strip())
if a.budget is not None:
    os.environ["OPENAI_BUDGET_USD"] = str(a.budget)
os.environ.setdefault("OPENAI_SPEND_FILE", str(out_root / "spend.json"))
adapter = adapters.get_adapter(a.key); adapter.load()
for p in a.pack:
    pack = ItemPack.load(ROOT / f"UNDERTONE_report/data/item_packs/{p}.jsonl")
    out = out_root / pack.fingerprint / "question_only.jsonl"
    sweep.question_only(adapter, pack, out)
    rows = [r for r in runner.load_rows(out) if r.get("error") is None and not r.get("logit_degenerate")]
    print(f"{p}: QO n={len(rows)} acc={sum(r['correct'] for r in rows) / max(1, len(rows)):.3f} "
          f"spend=${getattr(adapter, 'spend_usd', 0):.3f}", flush=True)
print("QO DONE", flush=True)
