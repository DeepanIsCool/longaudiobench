# Experiments: the $20 plan

What to run, in order, to take the UNDERTONE paper from "one prominence
experiment on 70 items" to "a 2x2 causal design with a duration axis, a
general cascade result, and enough items for the weak claims to resolve".
Ordered cheapest-and-safest first so that a blown run late costs nothing
already banked. Costs are for one A40 on Runpod community cloud at ~$0.40/h,
scaled from the first paper run (12 models, ladder + sweep, ~$3.06).

| # | script | what it adds | runtime | cost | running |
| --- | --- | --- | --- | --- | --- |
| 0 | *(none — CPU, free)* | Expansion pack: C1 9→40, P1/P2 →30, natural P3 →25. `build_item_pack.py --exclude-pack --share`. | hours, local | $0 | $0 |
| 1 | `01_text_twins.sh` | Three more text LLMs behind the same Whisper. Generalises the salience-prior result beyond one model pair. | ~40 min | $1 | $1 |
| 2 | `02_prominence_2x2.sh` | Boost the needle; attenuate the competitor; fine sweep in the calibrated region. Turns "prominence breaks it" into "prominence is relative, has a threshold in dB, and raising it repairs the failure". | ~5 h | $4 | $5 |
| 3 | `03_new_items.sh` | Ladder on the expansion pack only. Resolves the two claims that failed the sign test (abstention rises with context, 9/12 p=.15; quiet items abstain more, 8/11 p=.23). | ~9 h | $4 | $9 |
| 4 | `04_band_600.sh` | Ladder at 600 s. Makes "needle type explains more variance than duration" testable, against every benchmark that varies only length. | ~11 h | $6 | $15 |
| — | reserve | untouched | | $5 | $20 |

After step 2 the paper already has a new contribution for $5. If step 3's
pack is large and eats into step 4, keep step 3 and run step 4 on six models
(`MODELS=...` env, default list in the script is ordered for this).

## Cut from the plan, and why

- **Aero's sweep, Audio-Flamingo's L3/L4** ($1). Tidiness. The sweep is
  11/11 and the ladder 12/12 on everything that holds.
- **A 30B open model** ($3). Parameter count is already not significant
  (r=+0.43, p≈.19); a 13th model of the same kind adds nothing and a 30B one
  is defensive rather than novel.
- **Closed models** (~$8, API not Runpod). Worth doing; not this budget.
- **LoRA probe on P3**. Needs ≥25 natural P3 items to train on. Next budget,
  after step 0 exists.
- **1200 s band**. H100 territory.

## What each script assumes

- `experiments/*.sh` are run *on the pod* by `rp.py launch`; they source
  `scripts/runpod/bootstrap.sh` for everything shared.
- The item pack arrives as a Kaggle dataset (`PACK_DATASET`), which is how
  the first run did it. Steps 3 and 4 need new datasets; the header comment
  of each script has the build command and the launch line.
- Results land in `results/expNN_*/<model>/`, one file per task
  (`results.jsonl`, `sweep_<arm>_<levels>.jsonl`, `question_only.jsonl`).
  Every row carries `pack_fingerprint` and `code_sha`; the analysis merges
  packs by concatenating rows.
- Analysis for the 2x2 is `undertone.analysis.prominence_2x2` and
  `arm_direction`; the headline test for every claim is
  `undertone.analysis.sign_test`, not a pooled z.

## What is not code

Step 0 needs a human to listen to a test split. Every item is still
`verified: false`, and no compute changes that.
