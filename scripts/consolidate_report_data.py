#!/usr/bin/env python3
"""Build UNDERTONE_report/data/ - one folder per task, one file per model -
from the per-pack banks in results/banked/.

    data/item_packs/{v1,v2,600}.jsonl
    data/<task>/<model>.jsonl        v1 + v2 pooled: the 408-item main band
    data/band_600/<task>/<model>.jsonl   the 85-item 600 s band, a separate factor

Every row keeps its pack_fingerprint, so the pack is still a column. Rows are
de-duplicated on their cell key (last valid row wins) and the two packs are
asserted disjoint. Re-run after any new bank; the tree is rebuilt from scratch.
"""
import glob, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "results", "banked")
OUT = os.path.join(ROOT, "UNDERTONE_report", "data")
FP = {"v1": "56bd324cf6a3", "v2": "0b14538c9854", "600": "7e4fd5c5c30c"}
TASKS = ["ladder", "sweep", "sweep_boost", "sweep_competitor", "sweep_calibrated", "question_only"]
PACKS = {"v1": os.path.join(BANK, "item_pack_v1", "item_pack.jsonl"),
         "v2": os.path.join(ROOT, "data", "item_pack_v2", "item_pack.jsonl"),
         "600": os.path.join(ROOT, "data", "item_pack_600", "item_pack.jsonl")}


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r["item_id"], r.get("condition"), r.get("level_db"), r.get("edit_target"))


def sources(pack, task):
    """{model: [files]} for one pack and task. v1 lives in two places: the
    first report's layout (11 models, twins under cascaded/) and the later
    bank (Aero, AF-Next, 2x2 sweeps). Both are offered; merge() keeps the
    better row per cell."""
    out = {}
    for f in glob.glob(os.path.join(BANK, pack, task, "*.jsonl")):
        out.setdefault(os.path.basename(f)[:-6], []).append(f)
    if pack == "v1":
        for d in ([task, "cascaded"] if task == "ladder" else [task]):
            for f in glob.glob(os.path.join(BANK, "v1_orig", d, "*.jsonl")):
                out.setdefault(os.path.basename(f)[:-6], []).append(f)
    if task == "ladder":  # API models bank under results/gemini/<key>/<fp>/
        for f in glob.glob(os.path.join(ROOT, "results", "gemini", "*", FP[pack], "results.jsonl")):
            out.setdefault(f.split(os.sep)[-3], []).append(f)
    return out


def merge(files, fp):
    best = {}
    for f in files:
        for r in rows(f):
            if r.get("pack_fingerprint") != fp:
                sys.exit(f"{f}: row carries fingerprint {r.get('pack_fingerprint')} not {fp}")
            k = key(r)
            if k not in best or (best[k].get("error") and not r.get("error")):
                best[k] = r
    return best


def write(path, cells):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in cells.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "item_packs"))
    for pack, p in PACKS.items():
        shutil.copy(p, os.path.join(OUT, "item_packs", f"{pack}.jsonl"))
    report = []
    for task in TASKS:
        s1, s2 = sources("v1", task), sources("v2", task)
        for model in sorted(set(s1) | set(s2)):
            a = merge(s1.get(model, []), FP["v1"]); b = merge(s2.get(model, []), FP["v2"])
            if set(a) & set(b):
                sys.exit(f"{task}/{model}: packs overlap on {len(set(a) & set(b))} cells")
            cells = {**a, **b}
            write(os.path.join(OUT, task, f"{model}.jsonl"), cells)
            err = sum(1 for r in cells.values() if r.get("error"))
            report.append((task, model, len(a), len(b), err))
        for model, files in sources("600", task).items():
            cells = merge(files, FP["600"])
            write(os.path.join(OUT, "band_600", task, f"{model}.jsonl"), cells)
            report.append((f"band_600/{task}", model, 0, len(cells), sum(1 for r in cells.values() if r.get("error"))))
    for task, model, a, b, err in report:
        print(f"{task:26s} {model:28s} v1={a:4d} v2={b:5d} total={a+b:5d}" + (f"  errors={err}" if err else ""))
    print(f"{len(report)} files written under {OUT}")


if __name__ == "__main__":
    main()
