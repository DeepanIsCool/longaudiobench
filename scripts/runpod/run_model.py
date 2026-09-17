#!/usr/bin/env python3
"""Score one model in a fresh process. What to score is on the command line.

A fresh process per model is not tidiness. Four models pin incompatible
transformers versions (4.48.2, 4.52.4, 4.57.1, >=5.15.1) and a reinstall
does not affect already-imported modules, so one long-lived process would
silently run some models against the wrong library. bootstrap.sh makes a
venv per model and calls this once.

    python run_model.py qwen2_5_omni_7b --pack /workspace/pack --out /workspace/out \\
        --ladder \\
        --sweep needle:default --sweep competitor:default --sweep needle:boost \\
        --sweep needle:calibrated

    python run_model.py gemma3n_e2b --pack ... --out ... --qo

Every task appends to its own file under <out>/<key>/ and resumes from what
is there, so re-running after a crash costs only the cells that were lost.
Writes <key>_OK or <key>_FAILED to the log so the watchdog can count.
"""
import argparse
import glob
import json
import os
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("key")
ap.add_argument("--pack", required=True, help="directory holding item_pack.jsonl (searched recursively)")
ap.add_argument("--out", required=True)
ap.add_argument("--fingerprint", help="refuse to run against any other pack")
ap.add_argument("--repo", default="/workspace/repo")
ap.add_argument("--ladder", nargs="*", metavar="COND",
                help="run the ladder; optionally restrict to these conditions")
ap.add_argument("--sweep", action="append", default=[], metavar="ARM:LEVELS",
                help="needle|competitor : default|calibrated|boost. Repeatable.")
ap.add_argument("--qo", action="store_true", help="question-only baseline")
ap.add_argument("--skip-preflight", action="store_true")
a = ap.parse_args()

OUT = os.path.join(a.out, a.key)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, a.repo)
log = open(os.path.join(OUT, "log.txt"), "a", buffering=1)


def say(*parts):
    line = " ".join(str(x) for x in parts)
    print(line, flush=True)
    log.write(line + "\n")


def summarise(rows, by):
    for c in sorted({r[by] for r in rows}):
        ok = [r for r in rows if r[by] == c and not r.get("error") and r.get("role_chosen")]
        if ok:
            n = len(ok)
            say(f"  {by}={c} n={n} "
                f"acc={sum(r['role_chosen'] == r['correct_role'] for r in ok) / n:.3f} "
                f"sal={sum(r['role_chosen'] == 'salience' for r in ok) / n:.3f} "
                f"abs={sum(r['role_chosen'] == 'absent' for r in ok) / n:.3f}")


try:
    import numpy as np
    import torch

    from undertone import ItemPack, adapters, env, runner, sweep

    cand = glob.glob(os.path.join(a.pack, "**", "item_pack.jsonl"), recursive=True)
    assert cand, f"no item_pack.jsonl under {a.pack}"
    pack = ItemPack.load(cand[0])
    pack_dir = os.path.dirname(cand[0])
    if a.fingerprint:
        assert pack.fingerprint == a.fingerprint, \
            f"wrong pack {pack.fingerprint}, expected {a.fingerprint}"
    say(f"pack {len(pack)} items fp={pack.fingerprint} sha={os.environ.get('UNDERTONE_CODE_SHA')}")

    p = torch.cuda.get_device_properties(0)
    say(f"gpu {p.name} {p.total_memory / 2**30:.1f}GiB sm{p.major}{p.minor} "
        f"devices={torch.cuda.device_count()}")
    say("attention:", env.prefer_memory_efficient_attention())

    adapter = adapters.get_adapter(a.key)
    kw = adapter.load_kwargs()
    say("load_kwargs:", kw)
    if p.total_memory / 2**30 >= 40:
        # 48 GiB per device must mean no cap. A cap here pushes most of a
        # loadable model to CPU and quietly runs at a crawl.
        assert "max_memory" not in kw, "capped on a big card - headroom check failed"
    adapter.load()
    free, total = torch.cuda.mem_get_info()
    say(f"WEIGHTS {(total - free) / 2**30:.2f}GiB of {total / 2**30:.2f}")

    if not a.skip_preflight and not getattr(adapter, "is_control", False):
        # hears_audio: two different clips must not give identical logits.
        # This caught Aero scoring a text prior through a processor that
        # discarded the audio - a complete, plausible, meaningless table.
        sr, rng = 16000, np.random.default_rng(0)
        q = ("Listen and answer.\n\nWhat dose was mentioned?\nA) five milligrams\n"
             "B) fifty milligrams\nC) fifteen milligrams\nD) not stated\n\nOne letter.")
        s1 = adapter.score_letters((0.05 * rng.standard_normal(20 * sr)).astype("float32"), q, sr)
        s2 = adapter.score_letters((0.05 * rng.standard_normal(20 * sr)).astype("float32"), q, sr)
        finite = all(np.isfinite(list(s1.values())))
        distinct = len({round(float(v), 4) for v in s1.values()}) > 1
        hears = s1 != s2
        say(f"checks finite={finite} distinct={distinct} hears_audio={hears}")
        assert finite and distinct and hears, \
            f"pre-flight failed: finite={finite} distinct={distinct} hears={hears}"

    if a.ladder is not None:
        conds = a.ladder or ["L1", "L2", "L3", "L4"]
        path = os.path.join(OUT, "results.jsonl")
        runner.run_model(adapter, pack, path, conditions=conds,
                         audio_root=pack_dir, progress=True)
        rows = runner.load_rows(path)
        say(f"LADDER rows={len(rows)} errors={sum(1 for r in rows if r.get('error'))}")
        summarise(rows, "condition")

    LEVELS = {"default": sweep.DEFAULT_LEVELS, "calibrated": sweep.CALIBRATED_LEVELS,
              "boost": sweep.BOOST_LEVELS, "coarse": sweep.COMPETITOR_LEVELS}
    for spec in a.sweep:
        arm, _, lv = spec.partition(":")
        levels = LEVELS[lv or "default"]
        path = os.path.join(OUT, f"sweep_{arm}_{lv or 'default'}.jsonl")
        sweep.run_sweep(adapter, pack, path, levels=levels, audio_root=pack_dir,
                        edit_target=arm)
        rows = [r for r in runner.load_rows(path) if not r.get("error")]
        say(f"SWEEP[{arm}:{lv or 'default'}] rows={len(rows)}")
        summarise(rows, "level_db")
        if arm == "needle" and sweep.NEEDLE_REMOVED_DB in levels:
            nec = sweep.needle_necessity(rows)
            say(f"  audio_dependence {nec['audio_dependence']:+.3f}")

    if a.qo:
        path = os.path.join(OUT, "question_only.jsonl")
        sweep.question_only(adapter, pack, path)
        rows = [r for r in runner.load_rows(path) if r.get("role_chosen") and not r.get("error")]
        n = len(rows)
        say(f"QUESTION_ONLY n={n} acc={sum(r['role_chosen'] == r['correct_role'] for r in rows) / n:.3f}")

    json.dump({"model": adapter.describe()}, open(os.path.join(OUT, "summary.json"), "w"),
              indent=2, default=str)
    open(os.path.join(OUT, "DONE"), "w").write("ok\n")
    say(f"{a.key}_OK")
except Exception:
    say(f"{a.key}_FAILED")
    say(traceback.format_exc())
    open(os.path.join(OUT, "FAILED"), "w").write("see log.txt\n")
