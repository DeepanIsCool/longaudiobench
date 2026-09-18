"""Recompute every table in UNDERTONE_findings.pdf from data/.

Run from the report root:  python3 analysis/regenerate.py > analysis/verified_tables.txt

Conventions, applied consistently and stated in the PDF:
  * a row counts only if error is null and role_chosen is set
  * `correct` is used as-is; on the 10 null items it is true when role_chosen == "absent"
  * abstention rates, the sweep tables and the predictor correlations use non-null items
    only, because "absent" is the correct answer on null items and would entangle them
  * the oracle-window decomposition is paired by (model, item), not compared as marginals
  * headline significance is a model-level sign test, not a cell-level z: items are
    clustered (70 items, 46 windows, 31 meetings) so cell-level tests overstate
"""
import json, glob, os, math, collections, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The report's tables are the original 70-item pack. data/ now pools v1+v2
# (408 items) per task; filter to the v1 fingerprint so every number here
# still reproduces the PDF. The pooled analysis lives in analysis/pooled.py.
V1 = "56bd324cf6a3"
def rows(p): return [json.loads(l) for l in open(p) if l.strip()]
def load(d): return {os.path.basename(f)[:-6]: [x for x in rows(f) if x.get("pack_fingerprint") == V1]
                     for f in sorted(glob.glob(os.path.join(ROOT, "data", d, "*.jsonl")))
                     if not os.path.basename(f).startswith(("cascaded_", "gemini_", "gpt_"))}
def valid(r): return [x for x in r if not x.get("error") and x.get("role_chosen")]
def share(s, role): return sum(1 for x in s if x["role_chosen"] == role) / len(s)
def acc(s): return sum(x["correct"] for x in s) / len(s)

L, S, Q = load("ladder"), load("sweep"), load("question_only")
C = valid([x for x in rows(os.path.join(ROOT, "data/ladder/cascaded_whisper_llm.jsonl")) if x.get("pack_fingerprint") == V1])
PACK = {x["item_id"]: x for x in rows(os.path.join(ROOT, "item_packs/v1.jsonl"))[1:]}
# The PDF's 12-model tables predate the Audio-Flamingo chunk-padding fix;
# it is excluded here to keep those tables reproducible, not because its
# (now complete) rows are bad.
M12 = sorted(set(L) - {"audio_flamingo_next"})
def cells(models, cond=None, nonnull=False):
    out = [x for m in models for x in valid(L[m])]
    if cond: out = [x for x in out if x["condition"] == cond]
    if nonnull: out = [x for x in out if not x["is_null"]]
    return out

def sign_test(pairs):
    up = sum(1 for a, b in pairs if b > a); dn = sum(1 for a, b in pairs if b < a); n = up + dn
    if not n: return up, dn, 1.0
    p = sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) * 2 / 2 ** n
    return up, dn, min(1.0, p)

def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den
def tstat(r, n): return r * math.sqrt((n - 2) / (1 - r * r))

def h(t): print("\n" + t + "\n" + "=" * len(t))

h("TABLE 1  Ladder accuracy, 12 models with a complete ladder")
print(f"{'model':28s} {'L1':>6s} {'L2':>6s} {'L3':>6s} {'L4':>6s} {'Retr':>7s} {'LC':>7s}")
rc, lc = [], []
for m in sorted(M12, key=lambda m: -acc([x for x in valid(L[m]) if x["condition"] == "L1"])):
    a = {c: acc([x for x in valid(L[m]) if x["condition"] == c]) for c in ("L1", "L2", "L3", "L4")}
    rc.append(a["L1"] - a["L3"]); lc.append(a["L1"] - a["L4"])
    print(f"{m:28s} {a['L1']:6.3f} {a['L2']:6.3f} {a['L3']:6.3f} {a['L4']:6.3f} "
          f"{a['L1']-a['L3']:+7.3f} {a['L1']-a['L4']:+7.3f}")
print(f"{'mean':28s} {'':6s} {'':6s} {'':6s} {'':6s} {st.mean(rc):+7.3f} {st.mean(lc):+7.3f}")
v = valid(L["audio_flamingo_next"])
print(f"\naudio_flamingo_next (L1/L2 only; all 140 L3+L4 cells are errors): "
      f"L1 {acc([x for x in v if x['condition']=='L1']):.3f}  L2 {acc([x for x in v if x['condition']=='L2']):.3f}")

h("TABLE 2  Role shares by condition, pooled over 12 models")
print(f"{'cond':5s} {'n':>5s} {'correct':>9s} {'salience':>9s} {'recency':>9s} {'absent':>9s}")
for c in ("L1", "L2", "L3", "L4"):
    s = cells(M12, c)
    print(f"{c:5s} {len(s):5d} {share(s,'correct'):9.3f} {share(s,'salience'):9.3f} "
          f"{share(s,'recency'):9.3f} {share(s,'absent'):9.3f}")

h("TABLE 3  Sweep, non-null items only, 11 models (n=660 per level)")
sw = [x for r in S.values() for x in valid(r) if not x["is_null"]]
print(f"{'level':>8s} {'n':>5s} {'accuracy':>9s} {'salience':>9s} {'absent':>9s} {'recency':>9s}")
for lv in sorted({x["level_db"] for x in sw}, reverse=True):
    s = [x for x in sw if x["level_db"] == lv]
    print(f"{lv:8.0f} {len(s):5d} {acc(s):9.3f} {share(s,'salience'):9.3f} "
          f"{share(s,'absent'):9.3f} {share(s,'recency'):9.3f}")

h("TABLE 3b  Requested attenuation vs measured contrast (stimulus audit)")
ref = {(x["item_id"], x["level_db"]): x["achieved_contrast_db"] for x in S["qwen2_5_omni_7b"]}
assert all({(x["item_id"], x["level_db"]): x["achieved_contrast_db"] for x in r} == ref
           for r in S.values()), "achieved_contrast must be identical across models"
isnan = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
base = {}
for (i, lv), v in ref.items():
    if not isnan(v): base.setdefault(i, {})[lv] = v
print(f"{'requested':>10s} {'measured':>10s} {'increase':>10s} {'sd':>7s} {'shortfall':>10s}")
for lv in sorted({l for d in base.values() for l in d}, reverse=True):
    d = [b[lv] - b[0.0] for b in base.values()]
    print(f"{lv:10.0f} {st.mean([b[lv] for b in base.values()]):10.2f} {st.mean(d):10.2f} "
          f"{st.pstdev(d):7.2f} {-lv-st.mean(d):10.2f}")
print(f"({len(base)} items have a competitor; {70-len(base)} do not and have no defined contrast)")

h("TABLE 4  C1 inversion: salience-grab at L3+L4")
inv = 0
for m in sorted(M12, key=lambda m: -share([x for x in valid(L[m])
                if x["condition"] in ("L3","L4") and x["category"]=="C1"], "salience")):
    s = [x for x in valid(L[m]) if x["condition"] in ("L3", "L4")]
    c1 = share([x for x in s if x["category"] == "C1"], "salience")
    pp = share([x for x in s if x["category"] != "C1"], "salience")
    inv += c1 > pp
    print(f"{m:28s} C1 {c1:.3f}  P1-P4 {pp:.3f}  {'inverted' if c1>pp else '--'}")
print(f"inverted in {inv}/{len(M12)} models")

h("TABLE 5  Accuracy / salience-grab by category and condition")
print(f"{'cat':4s} " + " ".join(f"{c:>13s}" for c in ("L1", "L2", "L3", "L4")))
for cat in ("P1", "P2", "P3", "P4", "C1"):
    out = []
    for c in ("L1", "L2", "L3", "L4"):
        s = [x for x in cells(M12, c) if x["category"] == cat]
        out.append(f"{acc(s):.2f} / {share(s,'salience'):.2f}")
    print(f"{cat:4s} " + " ".join(f"{o:>13s}" for o in out))

h("TABLE 6  Oracle window, paired by (model, item)")
tr = collections.Counter(); n = 0
for m in M12:
    v = valid(L[m])
    d3 = {x["item_id"]: x for x in v if x["condition"] == "L3"}
    d4 = {x["item_id"]: x for x in v if x["condition"] == "L4"}
    for i in set(d3) & set(d4):
        n += 1; tr[(d3[i]["role_chosen"], d4[i]["role_chosen"])] += 1
rel = sum(c for (a, b), c in tr.items() if a == "absent" and b != "absent")
a3 = sum(c for (a, b), c in tr.items() if a == "absent") / n
a4 = sum(c for (a, b), c in tr.items() if b == "absent") / n
print(f"paired cells {n};  abstention {a3:.3f} -> {a4:.3f}  ({a4-a3:+.3f})")
for b in ("correct", "salience", "recency"):
    print(f"  absent -> {b:9s} {tr[('absent',b)]:4d}  {tr[('absent',b)]/rel:.0%} of released  {tr[('absent',b)]/n:+.3f} overall")
g = acc(cells(M12, "L1")) - acc(cells(M12, "L3"))
print(f"L1-L3 gap {g:.3f};  L4 recovers {(acc(cells(M12,'L4'))-acc(cells(M12,'L3')))/g:.0%}")

h("TABLE 7  Question-only baseline (4 models)")
qn = [x for r in Q.values() for x in valid(r) if not x["is_null"]]
qz = [x for r in Q.values() for x in valid(r) if x["is_null"]]
print(f"models: {', '.join(sorted(Q))}")
print(f"no audio, non-null  n={len(qn):3d}  acc {acc(qn):.3f}  picks absent {share(qn,'absent'):.3f}")
print(f"no audio, null      n={len(qz):3d}  acc {acc(qz):.3f}")
print(f"full 5 min (L3)     n={len(cells(M12,'L3')):3d}  acc {acc(cells(M12,'L3')):.3f}  "
      f"picks absent {share(cells(M12,'L3'),'absent'):.3f}")

h("TABLE 7b  Cascade vs its matched twin (non-null; Qwen2.5-7B-Instruct vs Qwen2.5-Omni-7B)")
O = [x for x in valid(L["qwen2_5_omni_7b"]) if not x["is_null"]]
CN = [x for x in C if not x["is_null"]]
print(f"{'cond':5s} {'text acc':>9s} {'audio acc':>10s} {'text sal':>9s} {'audio sal':>10s} {'text abs':>9s} {'audio abs':>10s}")
for c in ("L1", "L2", "L3", "L4"):
    t = [x for x in CN if x["condition"] == c]; a = [x for x in O if x["condition"] == c]
    print(f"{c:5s} {acc(t):9.3f} {acc(a):10.3f} {share(t,'salience'):9.3f} "
          f"{share(a,'salience'):10.3f} {share(t,'absent'):9.3f} {share(a,'absent'):10.3f}")
for lab, src in (("text", CN), ("audio", O)):
    s = [x for x in src if x["condition"] in ("L3","L4") and x["category"] == "C1"]
    p = [x for x in src if x["condition"] in ("L3","L4") and x["category"] != "C1"]
    print(f"  {lab:5s} L3+L4  C1 salience {share(s,'salience'):.3f}   P1-P4 {share(p,'salience'):.3f}")
c3 = {x["item_id"]: x for x in CN if x["condition"] == "L3"}
o3 = {x["item_id"]: x for x in O if x["condition"] == "L3"}
both = set(c3) & set(o3)
print(f"  L3 paired: audio right/text wrong {sum(1 for i in both if o3[i]['correct'] and not c3[i]['correct'])}, "
      f"reverse {sum(1 for i in both if c3[i]['correct'] and not o3[i]['correct'])}")

h("TABLE 8  Direction consistency across models (two-sided sign test)")
def per_model(models, src, sel_a, sel_b, role):
    out = []
    for m in models:
        v = valid(src[m])
        out.append((share(sel_a(v), role), share(sel_b(v), role)))
    return out
cond = lambda c: (lambda v: [x for x in v if x["condition"] == c])
cat_is = lambda c: (lambda v: [x for x in v if x["condition"] in ("L3","L4") and x["category"] == c])
cat_not = lambda c: (lambda v: [x for x in v if x["condition"] in ("L3","L4") and x["category"] != c])
lvl = lambda l: (lambda v: [x for x in v if x["level_db"] == l and not x["is_null"]])
TESTS = [
    ("accuracy falls L1->L3",        per_model(M12, L, cond("L3"), cond("L1"), "correct")),
    ("salience rises L1->L3",        per_model(M12, L, cond("L1"), cond("L3"), "salience")),
    ("abstention rises L1->L3",      per_model(M12, L, cond("L1"), cond("L3"), "absent")),
    ("recency changes L1->L3",       per_model(M12, L, cond("L1"), cond("L3"), "recency")),
    ("C1 salience > degraded",       per_model(M12, L, cat_not("C1"), cat_is("C1"), "salience")),
    ("degraded abstention > C1",     per_model(M12, L, cat_is("C1"), cat_not("C1"), "absent")),
    ("P1 abstention > C1",           per_model(M12, L, cat_is("C1"), cat_is("P1"), "absent")),
    ("oracle: abstention falls",     per_model(M12, L, cond("L4"), cond("L3"), "absent")),
    ("oracle: salience rises",       per_model(M12, L, cond("L3"), cond("L4"), "salience")),
    ("oracle: accuracy rises",       per_model(M12, L, cond("L3"), cond("L4"), "correct")),
    ("sweep: accuracy falls",        per_model(sorted(S), S, lvl(-60.0), lvl(0.0), "correct")),
    ("sweep: salience rises",        per_model(sorted(S), S, lvl(0.0), lvl(-60.0), "salience")),
    ("sweep: abstention rises",      per_model(sorted(S), S, lvl(0.0), lvl(-60.0), "absent")),
    ("sweep: recency changes",       per_model(sorted(S), S, lvl(0.0), lvl(-60.0), "recency")),
]
print(f"{'claim':34s} {'up':>3s}/{'n':<3s} {'p':>8s}   verdict")
for lab, pairs in TESTS:
    up, dn, p = sign_test(pairs)
    v = "null as predicted" if p > .5 else ("HOLDS" if p < .01 else "holds" if p < .05 else "NOT ESTABLISHED")
    print(f"{lab:34s} {up:3d}/{up+dn:<3d} {p:8.4f}   {v}")

h("TABLE 9  P3 natural vs constructed")
for lab, keep in (("natural", False), ("constructed", True)):
    sel = [x for x in cells(M12) if x["category"] == "P3"
           and PACK[x["item_id"]]["provenance"]["constructed"] is keep]
    it = len({x["item_id"] for x in sel})
    for c in ("L1", "L3"):
        s = [x for x in sel if x["condition"] == c]
        print(f"P3 {lab:12s} ({it:2d} items) {c}  acc {acc(s):.3f}  sal {share(s,'salience'):.3f}  abs {share(s,'absent'):.3f}")

h("TABLE 10  Abstention, accuracy, conditional accuracy (non-null)")
PARAMS = {'qwen2_5_omni_7b':10.7,'qwen2_5_omni_3b':5.4,'qwen2_audio_7b':8.4,'phi4_multimodal':5.6,
          'gemma3n_e2b':5.4,'gemma3n_e4b':7.9,'voxtral_mini_3b':3.0,'moss_audio_4b_instruct':5.2,
          'moss_audio_4b_thinking':5.2,'moss_audio_8b_instruct':9.1,'moss_audio_8b_thinking':9.1}
ab, ac, co, pa, pac = [], [], [], [], []
print(f"{'model':28s} {'abst':>6s} {'acc':>6s} {'cond':>6s} {'n':>5s}")
for m in sorted(M12, key=lambda m: -acc([x for x in valid(L[m]) if not x["is_null"]
                                         and x["role_chosen"] != "absent"])):
    s = [x for x in valid(L[m]) if not x["is_null"]]
    c = [x for x in s if x["role_chosen"] != "absent"]
    ab.append(share(s, "absent")); ac.append(acc(s)); co.append(acc(c))
    if m in PARAMS: pa.append(PARAMS[m]); pac.append(acc(s))
    print(f"{m:28s} {share(s,'absent'):6.3f} {acc(s):6.3f} {acc(c):6.3f} {len(c):5d}")
n = len(ab)
for lab, r, k in (("abstention ~ accuracy", pearson(ab, ac), n),
                  ("abstention ~ conditional accuracy", pearson(ab, co), n),
                  ("parameters ~ accuracy (11 verified)", pearson(pa, pac), len(pa))):
    print(f"  {lab:36s} r={r:+.3f}  r2={r*r:.3f}  t={tstat(r,k):+.2f}  n={k}")

h("TABLE 11  Matched instruct / thinking pairs")
for m in ("moss_audio_4b_instruct","moss_audio_4b_thinking","moss_audio_8b_instruct","moss_audio_8b_thinking"):
    v = valid(L[m]); s = [x for x in v if not x["is_null"]]
    c = [x for x in s if x["role_chosen"] != "absent"]
    print(f"{m:28s} L1 {acc([x for x in v if x['condition']=='L1']):.3f}  "
          f"L3 {acc([x for x in v if x['condition']=='L3']):.3f}  "
          f"abst {share(s,'absent'):.3f}  cond {acc(c):.3f}")

h("PROVENANCE")
tot = valid_n = 0
for d in ("ladder", "sweep", "question_only", "cascaded"):
    n_ = v_ = 0
    for f in glob.glob(os.path.join(ROOT, "data", d, "*.jsonl")):
        r = rows(f); n_ += len(r); v_ += len(valid(r))
    print(f"  {d:14s} rows {n_:5d}  valid {v_:5d}")
    tot += n_; valid_n += v_
print(f"  {'TOTAL':14s} rows {tot:5d}  valid {valid_n:5d}")
fp = {x["pack_fingerprint"] for r in list(L.values())+list(S.values())+list(Q.values()) for x in r} | {x["pack_fingerprint"] for x in C}
sha = {x.get("code_sha") for r in list(L.values())+list(S.values())+list(Q.values()) for x in r} | {x.get("code_sha") for x in C}
print(f"  pack fingerprints: {sorted(fp)}")
print(f"  code commits:      {sorted(s[:8] for s in sha if s)}")
print(f"  items: {len(PACK)}; windows {len({x['recording_id'] for x in PACK.values()})}; "
      f"meetings {len({x['provenance']['source_recording'] for x in PACK.values()})}; "
      f"human-verified {sum(1 for x in PACK.values() if x['provenance']['verified'])}")
