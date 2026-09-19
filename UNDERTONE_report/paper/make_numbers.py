#!/usr/bin/env python3
"""Every number in the paper, computed from UNDERTONE_report/data/ into
numbers.tex as LaTeX macros. Re-run after any new bank; the prose never
states a count, so only this file changes.

    python UNDERTONE_report/paper/make_numbers.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import random
import re
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
PACKS = os.path.join(os.path.dirname(HERE), "item_packs")
OUT = os.path.join(HERE, "numbers.tex")

# key, display name, family, nominal params (B, from the model card; None = not stated), ceiling s, scorer
MODELS = [
    ("qwen2_audio_7b", "Qwen2-Audio-7B", "Qwen", 7.0, 30, "logits"),
    ("qwen2_5_omni_3b", "Qwen2.5-Omni-3B", "Qwen", 3.0, 1260, "logits"),
    ("qwen2_5_omni_7b", "Qwen2.5-Omni-7B", "Qwen", 7.0, 1260, "logits"),
    ("phi4_multimodal", "Phi-4-multimodal", "Microsoft", 5.6, 1800, "logits"),
    ("gemma3n_e2b", "Gemma-3n-E2B", "Google", 2.0, 30, "logits"),
    ("gemma3n_e4b", "Gemma-3n-E4B", "Google", 4.0, 30, "logits"),
    ("aero_1_audio", "Aero-1-Audio", "LMMS", 1.5, 900, "logits"),
    ("voxtral_mini_3b", "Voxtral-Mini-3B", "Mistral", 3.0, 600, "logits"),
    ("moss_audio_4b_instruct", "MOSS-Audio-4B-Instruct", "OpenMOSS", 4.0, 1800, "logits"),
    ("moss_audio_4b_thinking", "MOSS-Audio-4B-Thinking", "OpenMOSS", 4.0, 1800, "gen"),
    ("moss_audio_8b_instruct", "MOSS-Audio-8B-Instruct", "OpenMOSS", 8.0, 1800, "logits"),
    ("moss_audio_8b_thinking", "MOSS-Audio-8B-Thinking", "OpenMOSS", 8.0, 1800, "gen"),
    ("audio_flamingo_next", "Audio-Flamingo-Next", "NVIDIA", None, 1800, "logits"),
]
FULLCOV = [m[0] for m in MODELS if m[4] >= 300]
TRUNC = [m[0] for m in MODELS if m[4] < 300]
TWINS = [("cascaded_whisper_llm", "Whisper$\\rightarrow$Qwen2.5-7B"),
         ("cascaded_whisper_llama31_8b", "Whisper$\\rightarrow$Llama-3.1-8B"),
         ("cascaded_whisper_mistral_7b", "Whisper$\\rightarrow$Mistral-7B"),
         ("cascaded_whisper_gemma2_9b", "Whisper$\\rightarrow$Gemma-2-9B")]
CLOSED = [("gemini_3_5_flash", "Gemini 3.5 Flash"), ("gemini_3_1_flash_lite", "Gemini 3.1 Flash-Lite"),
          ("gpt_audio_mini", "GPT-audio-mini")]
QO_MODELS = ["gemma3n_e2b", "moss_audio_4b_instruct", "phi4_multimodal", "qwen2_5_omni_7b", "gpt_audio_mini"]
CATS = ["P1", "P2", "P3", "P4", "C1"]
ROLES = ["correct", "salience", "recency", "absent"]

M = {}


_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
          "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro(name, value):
    """LaTeX macro names take letters only: digits become words, underscores go."""
    name = "".join(_WORDS.get(ch, ch) for ch in name.replace("_", ""))
    M[name] = value


def fmt(x, d=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}\\%"


def rows(task, model, band=None):
    p = os.path.join(DATA, band or "", task, f"{model}.jsonl") if band else os.path.join(DATA, task, f"{model}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def valid(rs):
    return [r for r in rs if r.get("error") is None and r.get("role_chosen")]


def acc(s):
    return sum(r["correct"] for r in s) / len(s) if s else float("nan")


def share(s, role):
    return sum(r["role_chosen"] == role for r in s) / len(s) if s else float("nan")


def cond(s, c):
    return [r for r in s if r["condition"] == c]


def nonnull(s):
    return [r for r in s if not r.get("is_null")]


def null(s):
    return [r for r in s if r.get("is_null")]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def ptex(p):
    if p < 1e-4:
        e = int(math.floor(math.log10(p)))
        return f"<10^{{{e}}}"
    return f"={p:.3f}" if p >= 0.001 else f"={p:.1e}"


def lv(s, level):
    return [r for r in s if r.get("level_db") == level]


# ---------------------------------------------------------------- items
items = [json.loads(l) for p in ("v1", "v2") for l in open(os.path.join(PACKS, f"{p}.jsonl")) if '"item_id"' in l]
item_by = {i["item_id"]: i for i in items}
win = lambda i: re.sub(r"_(P\d|C1)_\d+(_null)?$", "", i)
macro("Nitems", f"{len(items):,}")
macro("Nnonnull", f"{sum(1 for i in items if not i['is_null']):,}")
macro("Nnull", str(sum(1 for i in items if i["is_null"])))
macro("Nwindows", str(len({i["recording_id"] for i in items})))
macro("Nmeetings", str(len({i["recording_id"].split("_")[0] for i in items})))
macro("Nconstructed", str(sum(1 for i in items if i.get("provenance", {}).get("constructed"))))
cats = collections.Counter(i["category"] for i in items)
for c in CATS:
    macro(f"Ncat{c}", str(cats[c]))
mt = collections.Counter(i["recording_id"][:2] for i in items)
macro("Nscenario", str(sum(v for k, v in mt.items() if k in ("ES", "IS", "TS"))))
macro("Nnonscenario", str(sum(v for k, v in mt.items() if k in ("EN", "IB", "IN"))))
band_items = [json.loads(l) for l in open(os.path.join(PACKS, "600.jsonl")) if '"item_id"' in l]
macro("NitemsBand", str(len(band_items)))
macro("Nmodels", str(len(MODELS)))
macro("Nfullcov", str(len(FULLCOV)))
macro("Ntrunc", str(len(TRUNC)))
macro("Nfamilies", str(len({m[2] for m in MODELS})))
macro("Ntwins", str(len(TWINS)))

# ---------------------------------------------------------------- ladder per model
L = {m[0]: valid(rows("ladder", m[0])) for m in MODELS}
S = {m[0]: valid(rows("sweep", m[0])) for m in MODELS}
B = {m[0]: valid(rows("sweep_boost", m[0])) for m in MODELS}
C = {m[0]: valid(rows("sweep_competitor", m[0])) for m in MODELS}
K = {m[0]: valid(rows("sweep_calibrated", m[0])) for m in MODELS}

lines = []
per = {}
for key, name, fam, params, ceil, scorer in MODELS:
    l = nonnull(L[key])
    a = {c: acc(cond(l, c)) for c in ("L1", "L2", "L3", "L4")}
    l3 = cond(l, "L3")
    dec = share(l3, "salience"); den = share(l3, "absent")
    Sshare = dec / (1 - a["L3"]) if a["L3"] < 1 else float("nan")
    audio_dep = acc(lv(S[key], 0)) - acc(lv(S[key], -60))
    per[key] = dict(a=a, dec=dec, den=den, S=Sshare, dep=audio_dep, params=params, ceil=ceil,
                    dec1=share(cond(l, "L1"), "salience"), den1=share(cond(l, "L1"), "absent"))
    covnote = "\\ctr{30\\,s}" if ceil < 300 else ("600\\,s" if ceil == 600 else "$\\geq$\\,600\\,s")
    lines.append(f"{name} & {fam} & {fmt(params,1) if params else '--'} & {covnote} & {'gen' if scorer=='gen' else 'logit'} & "
                 f"{fmt(a['L1'])} & {fmt(a['L2'])} & {fmt(a['L3'])} & {fmt(a['L4'])} & "
                 f"{fmt(a['L1']-a['L3'])} & {fmt(dec)} & {fmt(den)} & {fmt(Sshare)} & {fmt(audio_dep, 2)} \\\\")
macro("ladderBody", "\n".join(lines))

# pooled over full-coverage models, non-null items
pool = {c: [r for m in FULLCOV for r in cond(nonnull(L[m]), c)] for c in ("L1", "L2", "L3", "L4")}
for c in ("L1", "L2", "L3", "L4"):
    for role in ROLES:
        macro(f"role{role.capitalize()}{c}", fmt(share(pool[c], role)))
macro("retrievalCost", fmt(acc(pool["L1"]) - acc(pool["L3"])))
macro("coordsRoles", " ".join(f"({c},{share(pool[c], r):.3f})" for c in ("L1", "L2", "L3", "L4") for r in ["correct"]))
for role in ROLES:
    macro(f"coords{role.capitalize()}", " ".join(f"({c},{share(pool[c], role):.3f})" for c in ("L1", "L2", "L3", "L4")))
macro("Sfull", fmt(share(pool["L3"], "salience") / (1 - acc(pool["L3"]))))
macro("Siso", fmt(share(pool["L1"], "salience") / (1 - acc(pool["L1"]))))

# item-level paired tests (model x item cells), full-coverage models
def paired(role_or_correct, c1, c2):
    w = l = 0
    for m in FULLCOV:
        by = collections.defaultdict(dict)
        for r in nonnull(L[m]):
            by[r["item_id"]][r["condition"]] = r
        for it, d in by.items():
            if c1 in d and c2 in d:
                if role_or_correct == "correct":
                    x, y = d[c1]["correct"], d[c2]["correct"]
                else:
                    x, y = d[c1]["role_chosen"] == role_or_correct, d[c2]["role_chosen"] == role_or_correct
                if x and not y: w += 1
                elif y and not x: l += 1
    return w, l


for tag, what in (("correct", "correct"), ("salience", "salience"), ("absent", "absent"), ("recency", "recency")):
    w, l = paired(what, "L1", "L3")
    macro(f"pair{tag.capitalize()}Down", f"{w:,}"); macro(f"pair{tag.capitalize()}Up", f"{l:,}")
    macro(f"pair{tag.capitalize()}P", ptex(sign_p(w, l)))
w, l = paired("absent", "L3", "L4"); macro("pairAbsentOracleDown", f"{w:,}"); macro("pairAbsentOracleUp", f"{l:,}"); macro("pairAbsentOracleP", ptex(sign_p(w, l)))
w, l = paired("correct", "L3", "L4"); macro("pairCorrectOracleDown", f"{w:,}"); macro("pairCorrectOracleUp", f"{l:,}"); macro("pairCorrectOracleP", ptex(sign_p(w, l)))

# model-level sign counts (all 13 and full-coverage)
def count(models, f):
    return sum(1 for m in models if f(m))
macro("signL3belowL1all", f"{count([m[0] for m in MODELS], lambda m: per[m]['a']['L3'] < per[m]['a']['L1'])}/{len(MODELS)}")
macro("signL3belowL1", f"{count(FULLCOV, lambda m: per[m]['a']['L3'] < per[m]['a']['L1'])}/{len(FULLCOV)}")
macro("signDecoyUp", f"{count(FULLCOV, lambda m: per[m]['dec'] > per[m]['dec1'])}/{len(FULLCOV)}")
macro("signDenialUp", f"{count(FULLCOV, lambda m: per[m]['den'] > per[m]['den1'])}/{len(FULLCOV)}")
macro("signAudioDep", f"{count([m[0] for m in MODELS], lambda m: per[m]['dep'] > 0)}/{len(MODELS)}")
macro("signP", ptex(sign_p(len(FULLCOV), 0)))
macro("signPall", ptex(sign_p(len(MODELS), 0)))
macro("meanRetrievalCost", fmt(st.mean(per[m]["a"]["L1"] - per[m]["a"]["L3"] for m in FULLCOV)))
macro("minRetrievalCost", fmt(min(per[m]["a"]["L1"] - per[m]["a"]["L3"] for m in FULLCOV)))
macro("maxRetrievalCost", fmt(max(per[m]["a"]["L1"] - per[m]["a"]["L3"] for m in FULLCOV)))
macro("oracleGain", fmt(acc(pool["L4"]) - acc(pool["L3"])))
macro("oracleDenialDrop", fmt(share(pool["L3"], "absent") - share(pool["L4"], "absent")))

# ---------------------------------------------------------------- search-load curve (within item): L1 20 s, L2 120 s, L3 300 s; band 600
Lb = {m: valid(rows("ladder", m, band="band_600")) for m in FULLCOV}
band_models = [m for m in FULLCOV if Lb[m]]
poolb = [r for m in band_models for r in cond(nonnull(Lb[m]), "L3")]
poolb1 = [r for m in band_models for r in cond(nonnull(Lb[m]), "L1")]
macro("NbandModels", str(len(band_models)))
curve = [("20\\,s", pool["L1"]), ("120\\,s", pool["L2"]), ("300\\,s", pool["L3"]), ("600\\,s", poolb)]
macro("lenCorrect", " ".join(f"({n},{acc(s):.3f})" for n, s in curve))
macro("lenSalience", " ".join(f"({n},{share(s,'salience'):.3f})" for n, s in curve))
macro("lenRecency", " ".join(f"({n},{share(s,'recency'):.3f})" for n, s in curve))
macro("lenAbsent", " ".join(f"({n},{share(s,'absent'):.3f})" for n, s in curve))
BAND = "600\\,s"
lenlines = []
for n, s_ in curve:
    models_n = len(band_models) if n == BAND else len(FULLCOV)
    kind = "separate band" if n == BAND else "within item"
    lenlines.append(f"{n} & {kind} & {len(s_) // max(1, models_n):,} & {models_n} & {fmt(acc(s_))} & "
                    f"{fmt(share(s_, 'salience'))} & {fmt(share(s_, 'recency'))} & {fmt(share(s_, 'absent'))} \\\\")
macro("lenBody", "\n".join(lenlines))
macro("bandCorrIso", fmt(acc(poolb1))); macro("bandCorrFull", fmt(acc(poolb))); macro("bandDecoy", fmt(share(poolb, "salience"))); macro("bandDenial", fmt(share(poolb, "absent")))
macro("lenSalFirst", fmt(share(pool["L1"], "salience"))); macro("lenSalLast", fmt(share(poolb, "salience")))
macro("lenCorrFirst", fmt(acc(pool["L1"]))); macro("lenCorrLast", fmt(acc(poolb)))

# ---------------------------------------------------------------- causal arm (pooled, all 13 and full-coverage)
levels = [0, -3, -6, -9, -12, -18, -24, -60]
Spool = [r for m in FULLCOV for r in nonnull(S[m])]
macro("sweepCorrect", " ".join(f"({l},{acc(lv(Spool,l)):.3f})" for l in levels[:-1]))
macro("sweepSalience", " ".join(f"({l},{share(lv(Spool,l),'salience'):.3f})" for l in levels[:-1]))
macro("sweepAbsent", " ".join(f"({l},{share(lv(Spool,l),'absent'):.3f})" for l in levels[:-1]))
macro("sweepRecency", " ".join(f"({l},{share(lv(Spool,l),'recency'):.3f})" for l in levels[:-1]))
LTAG = {0: "z", -3: "mThree", -6: "mSix", -9: "mNine", -12: "mTwelve", -18: "mEighteen", -24: "mTwentyfour", -60: "mSixty"}
for l in levels:
    tag = LTAG[l]
    macro(f"sweepCorr{tag}", fmt(acc(lv(Spool, l)))); macro(f"sweepSal{tag}", fmt(share(lv(Spool, l), "salience")))
    macro(f"sweepAbs{tag}", fmt(share(lv(Spool, l), "absent"))); macro(f"sweepRec{tag}", fmt(share(lv(Spool, l), "recency")))
macro("sweepBody", "\n".join(
    f"{l:+d} & {fmt(acc(lv(Spool,l)))} & {fmt(share(lv(Spool,l),'salience'))} & {fmt(share(lv(Spool,l),'recency'))} & {fmt(share(lv(Spool,l),'absent'))} \\\\"
    for l in levels))
MODELS_ = [m[0] for m in MODELS]
macro("audioDepMean", fmt(st.mean(per[m]["dep"] for m in MODELS_)))
macro("audioDepMin", fmt(min(per[m]["dep"] for m in MODELS_))); macro("audioDepMax", fmt(max(per[m]["dep"] for m in MODELS_)))
# per-model sweep: monotone? accuracy falls, decoy rises (model-level)
def mono_fall(m):
    return acc(lv(S[m], 0)) > acc(lv(S[m], -60))
macro("signSweepFall", f"{count(MODELS_, mono_fall)}/{len(MODELS_)}")
macro("signSweepDecoyUp", f"{count(MODELS_, lambda m: share(lv(S[m],-60),'salience') > share(lv(S[m],0),'salience'))}/{len(MODELS_)}")
macro("signSweepDenialUp", f"{count(MODELS_, lambda m: share(lv(S[m],-60),'absent') > share(lv(S[m],0),'absent'))}/{len(MODELS_)}")
# removed level: what do models pick when the answer is gone (non-null items: answer removed -> sentinel now "correct" in spirit)
rm = lv(Spool, -60)
macro("sweeprmSalience", fmt(share(rm, "salience"))); macro("sweeprmAbsent", fmt(share(rm, "absent"))); macro("sweeprmCorrect", fmt(share(rm, "correct"))); macro("sweeprmRecency", fmt(share(rm, "recency")))
# boost / competitor / calibrated
def eff(D, m, hi, lo=0):
    return acc(lv(D[m], hi)) - acc(lv(D[m], lo))
boost = [eff(B, m, 9) for m in MODELS_]; comp = [eff(C, m, -60) for m in MODELS_]; cal = [eff(K, m, -12) for m in MODELS_]
macro("boostMean", f"{st.mean(boost):+.3f}"); macro("boostMin", f"{min(boost):+.3f}"); macro("boostMax", f"{max(boost):+.3f}")
macro("signBoost", f"{sum(1 for v in boost if v > 0)}/{len(boost)}")
macro("compMean", f"{st.mean(comp):+.3f}"); macro("signComp", f"{sum(1 for v in comp if v > 0)}/{len(comp)}")
macro("calMean", f"{st.mean(cal):+.3f}"); macro("signCal", f"{sum(1 for v in cal if v < 0)}/{len(cal)}")
Bpool = [r for m in FULLCOV for r in nonnull(B[m])]; Cpool = [r for m in FULLCOV for r in nonnull(C[m])]; Kpool = [r for m in FULLCOV for r in nonnull(K[m])]
macro("boostCurve", " ".join(f"({l},{acc(lv(Bpool,l)):.3f})" for l in (0, 3, 6, 9)))
macro("boostDecoy", " ".join(f"({l},{share(lv(Bpool,l),'salience'):.3f})" for l in (0, 3, 6, 9)))
macro("compCurve", " ".join(f"({l},{acc(lv(Cpool,l)):.3f})" for l in (0, -6, -12, -24, -60)))
macro("compDecoy", " ".join(f"({l},{share(lv(Cpool,l),'salience'):.3f})" for l in (0, -6, -12, -24, -60)))
macro("calCurve", " ".join(f"({l},{acc(lv(Kpool,l)):.3f})" for l in (0, -2, -4, -6, -8, -10, -12)))
macro("compZero", fmt(acc(lv(Cpool, 0)))); macro("compRemoved", fmt(acc(lv(Cpool, -60))))
macro("compDecoyZero", fmt(share(lv(Cpool, 0), "salience"))); macro("compDecoyRemoved", fmt(share(lv(Cpool, -60), "salience")))
macro("boostZero", fmt(acc(lv(Bpool, 0)))); macro("boostNine", fmt(acc(lv(Bpool, 9))))
macro("twoByTwoBody", "\n".join(
    f"{name} & {fmt(per[key]['dep'],2)} & {fmt(eff(B,key,9),3)} & {fmt(eff(C,key,-60),3)} & {fmt(eff(K,key,-12),3)} \\\\"
    for key, name, *_ in MODELS))
# competitor cells per model (items with competitor in window)
ncomp = len({r["item_id"] for r in C[FULLCOV[0]]}); macro("NcompItems", str(ncomp))

# ---------------------------------------------------------------- categories (full-coverage pool)
catlines = []
for c in CATS:
    s1 = [r for r in pool["L1"] if r["category"] == c]; s3 = [r for r in pool["L3"] if r["category"] == c]
    ni = len({r["item_id"] for r in s3})
    catlines.append(f"{c} & {ni} & {fmt(acc(s1))} & {fmt(acc(s3))} & {fmt(acc(s1)-acc(s3))} & {fmt(share(s3,'salience'))} & {fmt(share(s3,'recency'))} & {fmt(share(s3,'absent'))} \\\\")
    macro(f"cat{c}Corr", fmt(acc(s3))); macro(f"cat{c}Dec", fmt(share(s3, "salience"))); macro(f"cat{c}Den", fmt(share(s3, "absent"))); macro(f"cat{c}R", fmt(acc(s1) - acc(s3)))
macro("catBody", "\n".join(catlines))
# orderings for the prose, so the text stays right after a rerun
R_ = {c: acc([r for r in pool["L1"] if r["category"] == c]) - acc([r for r in pool["L3"] if r["category"] == c]) for c in CATS}
D_ = {c: share([r for r in pool["L3"] if r["category"] == c], "salience") for c in CATS}
mx = max(R_, key=R_.get); mn = min(R_, key=R_.get)
macro("catRmaxName", mx); macro("catRmaxVal", fmt(R_[mx])); macro("catRminName", mn); macro("catRminVal", fmt(R_[mn]))
dmx = max(D_, key=D_.get); dmn = min(D_, key=D_.get)
macro("catDecMaxName", dmx); macro("catDecMaxVal", fmt(D_[dmx])); macro("catDecMinName", dmn); macro("catDecMinVal", fmt(D_[dmn]))
# scenario vs non-scenario
def mtype(r): return "scenario" if r["recording_id"][:2] in ("ES", "IS", "TS") else "non-scenario"
for t in ("scenario", "non-scenario"):
    s1 = [r for r in pool["L1"] if mtype(r) == t]; s3 = [r for r in pool["L3"] if mtype(r) == t]
    k = "Scen" if t == "scenario" else "Nonscen"
    macro(f"{k}Corr1", fmt(acc(s1))); macro(f"{k}Corr3", fmt(acc(s3))); macro(f"{k}Dec", fmt(share(s3, "salience"))); macro(f"{k}Den", fmt(share(s3, "absent")))
    macro(f"{k}N", str(len({r['item_id'] for r in s3})))

# ---------------------------------------------------------------- null items and calibration
nullpool = {c: [r for m in FULLCOV for r in cond(null(L[m]), c)] for c in ("L1", "L3", "L4")}
for c in ("L1", "L3", "L4"):
    macro(f"nullSentinel{c}", fmt(share(nullpool[c], "absent")))
    macro(f"nullDecoy{c}", fmt(share(nullpool[c], "salience")))
macro("calibGapL3", fmt(share(nullpool["L3"], "absent") - share(pool["L3"], "absent")))
# question-only
qlines = []
for m in QO_MODELS:
    q = [r for r in rows("question_only", m) if r.get("error") is None and not r.get("logit_degenerate")]
    name = dict((k, n) for k, n, *_ in MODELS).get(m) or dict(CLOSED)[m]
    qn = null(q); qnn = nonnull(q)
    qlines.append(f"{name} & {len(q)} & {fmt(acc(qnn))} & {fmt(share(qnn,'absent'))} & {fmt(share(qn,'absent'))} \\\\")
    macro(f"qo{m}", fmt(acc(qnn)))
macro("qoBody", "\n".join(qlines))
qall = [r for m in QO_MODELS if not m.startswith("gpt") for r in rows("question_only", m) if r.get("error") is None and not r.get("logit_degenerate")]
macro("qoPoolAcc", fmt(acc(nonnull(qall)))); macro("qoPoolSentinel", fmt(share(nonnull(qall), "absent"))); macro("qoNullSentinel", fmt(share(null(qall), "absent")))

# ---------------------------------------------------------------- twins vs audio
tw = {k: valid(rows("ladder", k)) for k, _ in TWINS}
twlines = []
for k, name in TWINS:
    t = nonnull(tw[k]); a1, a3 = acc(cond(t, "L1")), acc(cond(t, "L3"))
    twlines.append(f"{name} & {fmt(a1)} & {fmt(a3)} & {fmt(a1-a3)} & {fmt(share(cond(t,'L3'),'salience'))} & {fmt(share(cond(t,'L3'),'absent'))} \\\\")
macro("twinBody", "\n".join(twlines))
twpool3 = [r for k, _ in TWINS for r in cond(nonnull(tw[k]), "L3")]; twpool1 = [r for k, _ in TWINS for r in cond(nonnull(tw[k]), "L1")]
macro("twinCorr1", fmt(acc(twpool1))); macro("twinCorr3", fmt(acc(twpool3))); macro("twinDecoy3", fmt(share(twpool3, "salience"))); macro("twinDenial3", fmt(share(twpool3, "absent")))
macro("twinDenial1", fmt(share(twpool1, "absent")))
# per category twin vs audio at L3
twcat = []
for c in CATS:
    a = [r for r in pool["L3"] if r["category"] == c]; t = [r for r in twpool3 if r["category"] == c]
    twcat.append(f"{c} & {fmt(acc(a))} & {fmt(acc(t))} & {fmt(share(a,'absent'))} & {fmt(share(t,'absent'))} \\\\")
macro("twinCatBody", "\n".join(twcat))
# paired audio-right/text-wrong at L3 (best audio model vs its twin: omni-7b vs whisper->qwen)
by_a = {(r["item_id"]): r["correct"] for r in cond(nonnull(L["qwen2_5_omni_7b"]), "L3")}
by_t = {(r["item_id"]): r["correct"] for r in cond(nonnull(tw["cascaded_whisper_llm"]), "L3")}
w = sum(1 for i in by_a if i in by_t and by_a[i] and not by_t[i]); l = sum(1 for i in by_a if i in by_t and by_t[i] and not by_a[i])
macro("twinPairAudioRight", str(w)); macro("twinPairTextRight", str(l)); macro("twinPairP", ptex(sign_p(w, l)))

# ---------------------------------------------------------------- closed models
cl = []
for k, name in CLOSED:
    r_ = nonnull(valid(rows("ladder", k)))
    a = {c: acc(cond(r_, c)) for c in ("L1", "L2", "L3", "L4")}
    n1 = len(cond(r_, "L1")); n3 = len(cond(r_, "L3"))
    allrows = [r for r in rows("ladder", k) if not r.get("is_null")]
    refus = sum(1 for r in cond(allrows, "L3") if r.get("error") is None and not r.get("role_chosen"))
    l3 = cond(r_, "L3")
    cl.append(f"{name} & {n1} & {fmt(a['L1'])} & {fmt(a['L2'])} & {fmt(a['L3'])} & {fmt(a['L4'])} & {fmt(a['L1']-a['L3'])} & {fmt(share(l3,'salience'))} & {fmt(share(l3,'absent'))} & {refus} \\\\")
    tag = {"gemini_3_5_flash": "geminiflash", "gemini_3_1_flash_lite": "geminilite", "gpt_audio_mini": "gptaudiomini"}[k]
    macro(f"closed{tag}Corr1", fmt(a["L1"])); macro(f"closed{tag}Corr3", fmt(a["L3"])); macro(f"closed{tag}Dec3", fmt(share(l3, "salience"))); macro(f"closed{tag}Den3", fmt(share(l3, "absent")))
    macro(f"closed{tag}Refusals", str(refus)); macro(f"closed{tag}N", str(n1))
    # paired
    by = collections.defaultdict(dict)
    for r in r_: by[r["item_id"]][r["condition"]] = r["correct"]
    w = sum(1 for d in by.values() if "L1" in d and "L3" in d and d["L1"] and not d["L3"]); l = sum(1 for d in by.values() if "L1" in d and "L3" in d and d["L3"] and not d["L1"])
    macro(f"closed{tag}PairDown", str(w)); macro(f"closed{tag}PairUp", str(l)); macro(f"closed{tag}PairP", ptex(sign_p(w, l)))
macro("closedBody", "\n".join(cl))

# ---------------------------------------------------------------- scale invariance: S vs params (models with a stated size)
xs = [per[m]["params"] for m in MODELS_ if per[m]["params"]]; ys = [per[m]["S"] for m in MODELS_ if per[m]["params"]]
accs = [per[m]["a"]["L3"] for m in MODELS_ if per[m]["params"]]
def corr(x, y):
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else float("nan")
def corr_p(r, n):
    t = r * math.sqrt((n-2) / max(1e-9, 1-r*r)); # two-sided t approx via normal for small n is rough; use t distribution
    from statistics import NormalDist
    return 2 * (1 - NormalDist().cdf(abs(t)))
lx = [math.log(v) for v in xs]
rS = corr(lx, ys); rA = corr(lx, accs)
macro("rSalErrSize", fmt(rS)); macro("pSalErrSize", fmt(corr_p(rS, len(xs)), 2)); macro("rAccSize", fmt(rA)); macro("pAccSize", fmt(corr_p(rA, len(xs)), 2))
macro("salErrLo", fmt(min(ys))); macro("salErrHi", fmt(max(ys))); macro("NscaleModels", str(len(xs)))
macro("scaleCoords", " ".join(f"({x},{y:.3f})" for x, y in zip(xs, ys)))
macro("scaleAccCoords", " ".join(f"({x},{y:.3f})" for x, y in zip(xs, accs)))
# thinking vs instruct
for sz in ("4b", "8b"):
    i, t = per[f"moss_audio_{sz}_instruct"], per[f"moss_audio_{sz}_thinking"]
    sz_ = {"4b": "Fourb", "8b": "Eightb"}[sz]
    macro(f"moss{sz_}InstDen", fmt(i["den"])); macro(f"moss{sz_}ThinkDen", fmt(t["den"])); macro(f"moss{sz_}InstCorr", fmt(i["a"]["L3"])); macro(f"moss{sz_}ThinkCorr", fmt(t["a"]["L3"]))
    macro(f"moss{sz_}InstDec", fmt(i["dec"])); macro(f"moss{sz_}ThinkDec", fmt(t["dec"])); macro(f"moss{sz_}InstS", fmt(i["S"])); macro(f"moss{sz_}ThinkS", fmt(t["S"]))

# ---------------------------------------------------------------- artefacts: letters, truncation, scorer agreement
letters = collections.Counter(r["letter_chosen"] for r in pool["L3"])
tot = sum(letters.values())
macro("letterBars", " ".join(f"({L},{letters[L]/tot:.3f})" for L in "ABCD"))
macro("letterMax", fmt(max(letters.values()) / tot)); macro("letterMin", fmt(min(letters.values()) / tot))
sent_by_letter = collections.Counter(); sent_tot = collections.Counter()
for r in pool["L3"]:
    Lr = [k for k, v in r["letter_to_role"].items() if v == "absent"][0]
    sent_tot[Lr] += 1; sent_by_letter[Lr] += r["role_chosen"] == "absent"
macro("sentinelByLetter", " ".join(f"({L},{sent_by_letter[L]/max(1,sent_tot[L]):.3f})" for L in "ABCD"))
macro("sentinelLetterMax", fmt(max(sent_by_letter[L]/max(1,sent_tot[L]) for L in "ABCD"))); macro("sentinelLetterMin", fmt(min(sent_by_letter[L]/max(1,sent_tot[L]) for L in "ABCD")))
# truncated models: L1 vs L3 (their L3 is the first 30 s)
tp = {c: [r for m in TRUNC for r in cond(nonnull(L[m]), c)] for c in ("L1", "L3")}
macro("truncCorr1", fmt(acc(tp["L1"]))); macro("truncCorr3", fmt(acc(tp["L3"]))); macro("truncDen3", fmt(share(tp["L3"], "absent"))); macro("truncDec3", fmt(share(tp["L3"], "salience")))
# needle inside first 30 s for truncated models?
inside = [r for r in tp["L3"] if item_by[r["item_id"]]["needle_start"] - r["window_start"] < 30]
macro("truncInsideN", str(len({r['item_id'] for r in inside}))); macro("truncInsideCorr", fmt(acc(inside)))
outside = [r for r in tp["L3"] if item_by[r["item_id"]]["needle_start"] - r["window_start"] >= 30]
macro("truncOutsideCorr", fmt(acc(outside))); macro("truncOutsideDen", fmt(share(outside, "absent")))
# scorer agreement on logit-scored models: gen_letter vs logit_letter at L3
agree = both = 0
for m in FULLCOV:
    for r in cond(nonnull(L[m]), "L3"):
        if r.get("gen_letter") and r.get("logit_letter"):
            both += 1; agree += r["gen_letter"] == r["logit_letter"]
macro("scorerAgree", fmt(agree / both if both else float("nan"))); macro("scorerAgreeN", f"{both:,}")

# ---------------------------------------------------------------- subsample stability (appendix)
W = collections.defaultdict(list)
for i in items: W[win(i["item_id"])].append(i["item_id"])
ws = list(W); rng = random.Random(1)
def agg(keep):
    keep = set(keep); d = []; sal = []
    for m in FULLCOV:
        l = [r for r in nonnull(L[m]) if r["item_id"] in keep]
        d.append(acc(cond(l, "L3")) - acc(cond(l, "L1"))); sal.append(share(cond(l, "L3"), "salience"))
    return st.mean(d), sum(v < 0 for v in d), st.mean(sal)
res = []
for _ in range(200):
    rng.shuffle(ws); keep = []
    for w_ in ws:
        if len(keep) >= len(items) // 2: break
        keep += W[w_]
    res.append(agg(keep))
full = agg([i["item_id"] for i in items])
qd = sorted(r[0] for r in res); qs = sorted(r[2] for r in res)
macro("subDeltaFull", fmt(full[0])); macro("subDeltaLo", fmt(qd[5])); macro("subDeltaHi", fmt(qd[-6]))
macro("subSalFull", fmt(full[2])); macro("subSalLo", fmt(qs[5])); macro("subSalHi", fmt(qs[-6]))
macro("subSignMin", str(min(r[1] for r in res))); macro("subN", str(len(items) // 2))
# cluster bootstrap CI on retrieval cost (windows)
boot = []
for _ in range(300):
    keep = [i for w_ in rng.choices(ws, k=len(ws)) for i in W[w_]]
    boot.append(agg(keep)[0])
macro("ciRetrievalCost", fmt(1.96 * st.pstdev(boot)))

# ---------------------------------------------------------------- compute
cells = 0
for root, _, files in os.walk(DATA):
    for f in files:
        if f.endswith(".jsonl"):
            cells += sum(1 for _ in open(os.path.join(root, f)))
macro("Ncells", f"{cells:,}")

with open(OUT, "w") as fh:
    fh.write("% generated by make_numbers.py - do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(M)} macros -> {OUT}")
