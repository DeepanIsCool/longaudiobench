# Experiments: the $20 plan

What to run, in order, to take the UNDERTONE paper from "one prominence
experiment on 70 items" to "a 2x2 causal design with a duration axis, a
general cascade result, and enough items for the weak claims to resolve".
Ordered cheapest-and-safest first so that a blown run late costs nothing
already banked. Costs are for one A40 or A6000 on Runpod community cloud at $0.33-0.35/h (the live price on 2026-09-17; the table was costed at $0.40, so each step should come in ~12% under),
scaled from the first paper run (12 models, ladder + sweep, ~$3.06).

Every table the paper already has, on the same items for every model — then the
new 2×2 last, with whatever is left. Costed at the secure-cloud A40 price this
pod actually runs at, $0.49/h. One pod (`--name undertone`) carries every step
by stop/resume.

Every task on every AMI item for every model that can run it — 408 items at
300 s, 85 at 600 s — costed at this pod's $0.51/h. One pod (`--name undertone`)
carries every step; a lost pod restores from `undertone-restore` and resumes.

| # | script | what it does | cost | running |
| --- | --- | --- | --- | --- |
| 1 | `01_text_twins.sh` | Four text twins on the 70. Done. | $0.37 | $0.37 |
| 2 | `03_all_items.sh` | Ladder + main sweep + question-only on the 338 new items, one load per model. 12 models. **Running.** | ~$10.30 | $10.70 |
| 3 | `03b_twins_all_items.sh` | Four text twins on the 338. | $1.20 | $11.90 |
| 4 | `04_band_600.sh` | 600 s band: ladder + main sweep, 9 models; then the four twins. 85 items. | $5.40 | $17.30 |
| 5 | `02_prominence_2x2.sh` ×3 packs | Boost (+3/+6/+9), coarse competitor (−6/−12/−24/removed) and the fine 0…−12 sweep on **all 493 items**, 11 models. Runs last; resumable. | $7.70 | $25.00 |

That is the whole protocol, old and new, on the same items for every model,
nothing abbreviated.

## Not in this budget, and why

- **Aero's sweep, Audio-Flamingo's L3/L4** ($1). Aero has no recorded
  competitor spans, so it cannot be swept; AF-Next needs 17 GiB on one device
  and its own transformers pin.
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
  the first run did it. Step 3's pack is built and sits in `data/item_pack_v2/`
  with its `dataset-metadata.json`; upload with
  `kaggle datasets create -p data/item_pack_v2 --dir-mode zip`. Step 4's
  600 s pack is built and uploaded: `deepansadhukhanjeet/undertone-item-pack-600`.
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
