#!/usr/bin/env bash
# Bank a pulled results directory into the report package, with checks.
#
#   scripts/runpod/bank.sh results/exp03_all v2
#
# Copies every <model>/<task>.jsonl to UNDERTONE_report/data/<pack>/<task>/
# <model>.jsonl, refuses rows whose pack_fingerprint disagrees with the
# pack, prints row counts, and pushes the restore dataset so the tree is
# also off-laptop. Idempotent: run it after every model, every step.
set -u
SRC=$1; PACK=$2
cd "$(dirname "$0")/../.."
case "$PACK" in
  v1)  FP=56bd324cf6a3;; v2) FP=0b14538c9854;; 600) FP=7e4fd5c5c30c;;
  *) echo "pack must be v1|v2|600"; exit 1;;
esac
DEST=UNDERTONE_report/data/$PACK
python3 - "$SRC" "$DEST" "$FP" <<'PY'
import json, os, sys, glob, shutil
src, dest, fp = sys.argv[1:4]
task_of = {"results.jsonl": "ladder", "question_only.jsonl": "question_only",
           "sweep_needle_default.jsonl": "sweep", "sweep_needle_boost.jsonl": "sweep_boost",
           "sweep_competitor_coarse.jsonl": "sweep_competitor",
           "sweep_needle_calibrated.jsonl": "sweep_calibrated"}
banked = 0
for mdir in sorted(glob.glob(os.path.join(src, "*", ""))):
    model = os.path.basename(mdir.rstrip("/"))
    for fname, task in task_of.items():
        f = os.path.join(mdir, fname)
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        bad = [r for r in rows if r.get("pack_fingerprint") != fp]
        if bad:
            print(f"  REFUSED {model}/{fname}: {len(bad)} rows carry fingerprint "
                  f"{bad[0].get('pack_fingerprint')} not {fp}")
            continue
        out = os.path.join(dest, task); os.makedirs(out, exist_ok=True)
        shutil.copy2(f, os.path.join(out, f"{model}.jsonl"))
        ok = sum(1 for r in rows if not r.get("error") and r.get("role_chosen"))
        print(f"  {model:28s} {task:18s} {len(rows):5d} rows  {ok:5d} valid  "
              f"{'DONE' if os.path.exists(os.path.join(mdir, 'DONE')) else ''}")
        banked += 1
print(f"banked {banked} files into {dest}")
PY
scripts/runpod/push_restore.sh "$SRC" 2>&1 | tail -1
