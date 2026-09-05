# UNDERTONE

**Multilingual needle-in-a-haystack for prominence-conditioned retrieval in long-form audio**

Needle-in-a-haystack evaluation assumes one reason a needle is hard to find: the haystack is long.
Real listening fails for a different reason. The needle was muttered, talked over, thrown away as an
aside, or corrected in a quieter voice while a louder wrong answer was repeated three times.

**Central claim:** audio-language models retrieve by *acoustic prominence*, not by *relevance to the
query*.

---

## Status

| Piece | State |
|---|---|
| `undertone/` package — items, ladder, protocol, scoring, runner, smoke | done, 177 tests |
| 13 model adapters | written against each model's own docs; **unrun on GPU** |
| 17 Kaggle notebooks + push script | generated and validated |
| Harvest — English (AMI) | done |
| Harvest — Hindi and Bengali (YODAS2) | done; no diarisation, so **no P2** in those languages |
| Analysis — tables and figures | done |
| Item pack | pipeline validated end-to-end on real AMI audio; **not built at scale**, nothing verified or leak-filtered |
| ASR transcripts + cascaded control | done — `harvest/asr.py`, `adapters/cascaded.py`, notebook 02 |
| Any results | **none** |

Nothing here has produced a number yet. The first two things to run are both kill points.

---

## The five categories

Four share one mechanism. The fifth is the discriminant.

| | category | acoustic cause | share |
|---|---|---|---|
| P1 | quiet | low energy relative to this recording, *and* a real drop below its median | 22% |
| P2 | masked | overlapped by another speaker — gold, from segment timings | 22% |
| P3 | backgrounded | prosodically flat aside while a competing value is loud and repeated | 22% |
| P4 | corrected | self-repair; the corrected value is the answer | 22% |
| C1 | delivered | carried by hesitation and strain | 12% |

If P1–P4 fail via prominence-selection and C1 fails differently, the mechanism is specific rather
than "hard things are hard".

## The four options

Every distractor is a **real competing mention of the same quantity kind from the same recording**,
never a fabrication. That is what makes a wrong answer diagnostic rather than merely wrong.

| role | content | diagnoses |
|---|---|---|
| `correct` | the right answer | — |
| `salience` | the loudest and/or most repeated competing value | **salience prior** |
| `recency` | the latest competing value | recency bias |
| `absent` | "not mentioned in the recording" | fabrication; correct on the 10% null items |

Letters are redrawn per item per run, so preferring "A" cannot beat chance. What is scored is the
**role** the model picked.

## The ladder

`L1` isolated (~20 s) · `L2` local (2 min) · `L3` full band · `L4` full band, told where to look.

`RetrievalCost = Acc(L1) − Acc(L3)` and `LongContextCost = Acc(L1) − Acc(L4)`.

Without L1 a 20% L3 score is uninterpretable. That is what makes every finding falsifiable, and it is
why models capped at 30 s are still useful: they anchor the perception ceiling.

## Scoring

**Primary is the next-token distribution over A/B/C/D** — one forward pass, no generation, no regex.
Deterministic, and it makes the salience-trap rate exactly `P(salience)`.

Free generation runs alongside with a strict single-letter parse, and `unparseable` is its own
reported bucket — never a wrong answer. The gap between the two scorers separates "does not know"
from "will not emit a bare letter".

The MOSS `*-Thinking` variants invert this: their first generated token is a thought, so free
generation is primary there and letter logits are a diagnostic.

---

## Model roster

Ceilings are from the model card, `config.json`, or AudioMarathon Table 8.

| Model | fp16 | Max audio | GPU | Note |
|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct | 16.8 GB | **30 s** | 2×T4 | fixed 3000-frame encoder; L1 only |
| Qwen2.5-Omni-3B / 7B | 11 / 21 GB | ~21 min | 1× / 2×T4 | `disable_talker()` at load |
| Phi-4-multimodal-instruct | 11 GB | 30 min | 1×T4 | own transformers pin (4.48.2), eager attn |
| Gemma-3n-E2B / E4B | 11 / 16 GB | **30 s** | 1× / 2×T4 | **gated** |
| Aero-1-Audio | ~4 GB | 15 min | 1×T4 | sdpa, not flash-attn-2 |
| Voxtral-Mini-3B-2507 | 9.5 GB | **40 min** | 1×T4 | longest documented window |
| MOSS-Audio 4B / 8B × {Instruct, Thinking} | 10.4 / 18 GB | ~45 min* | 1× / 2×T4 | bf16 mel forced to fp16 |
| Audio-Flamingo Next | 16.5 GB | 30 min | 2×T4 | card uses bf16; fp16 here |

\* config-derived, not documented. `undertone.smoke.measure_ceiling` measures the real one.

Cells that exceed a model's ceiling run **truncated and flagged**, and are excluded from the accuracy
table. They are never scored as zero.

Three constraints every adapter inherits from Kaggle's 2×T4 (sm75): no bf16 compute, no
flash-attention-2, and weights cached in `/kaggle/temp` so an 18 GB checkpoint does not blow the
20 GB `/kaggle/working` cap.

---

## Layout

```
undertone/
  items.py protocol.py ladder.py scoring.py runner.py smoke.py
  adapters/     the only per-model code — 13 of them
  harvest/      sources, mentions, features, build, leakfilter
  analysis/     tables; no composite scores, by design
scripts/
  build_item_pack.py    CPU; emits item_pack.jsonl + FLAC clips
  make_notebooks.py     regenerates all 16 notebooks
notebooks/
  00_smoke_test  01_build_item_pack  02_cascaded_control  10..22 (13 models)  90_analysis
```

Everything above `adapters/` is shared. That is what lets thirteen notebooks each be written against
its own model's documentation and still produce numbers that compare.

## Running it

```bash
pip install -e .
python -m pytest tests/ --ignore=tests/test_core.py
python scripts/make_notebooks.py --out notebooks
```

Then, in order:

1. **`00_smoke_test.ipynb`** — ~15 min of quota. Every adapter is checked for load, finite logits,
   non-degenerate letter discrimination, parseable generation, determinism, and loud truncation.
   Run this before spending quota on a sweep.
2. **`01_build_item_pack.ipynb`** — build, leak-filter, floor-check, verify. Upload the result as the
   `undertone-item-pack` Kaggle dataset.
3. **`02_cascaded_control.ipynb`** — the ASR+LLM validity control, same protocol.
4. **`10_*.ipynb` … `22_*.ipynb`** — one sweep per model, resumable across the 12 h session limit.
5. **`90_analysis.ipynb`** — Tables 1, 2, 4, the language table and Figures 2–5.

## The two kill points

Both are cheap and both can end the project before it scales.

1. **Leak filter on ~40 English items.** If a text model solves them from the transcript, the
   audio-necessity claim is gone and the item design changes before anything else happens.
2. **B-choice on P3 in English.** If the salience trap does not exceed chance, F2 is wrong and the
   paper is reframed before any Hindi or Bengali work starts.

---

## Honest limitations

**F3 became F3′.** The paper plan's F3 needs a tone language, where pitch is unavailable for
prominence. English, Hindi and Bengali contain none. The testable replacement: English marks
prominence with lexically free stress-accent, while Hindi and Bengali have weak or fixed word stress,
phrase-level pitch accent, and grammaticalised focus (`hī`/`to`/`bhī`; `-i`/`-o`; word order). So:

> **F3′** — Models inherit an English stress-accent prominence prior. Salience-trap rate on P3 is
> higher in Hindi and Bengali, where the prominence a competent listener uses is carried by particles
> and word order that an acoustic salience prior cannot see.

This is weaker than F3, and Hindi and Bengali are both Indo-Aryan, so the design is 3 languages / 2
families rather than 5 languages / 4 prosodic systems.

**Audio-necessity is measured, not asserted.** `needle_recovery` reports the share of items whose
correct answer never appears in the Whisper transcript near the needle. If the words were never
written down, no cascaded system could have answered — evidence that does not depend on how good the
language model behind the ASR happens to be.

**The leak filter is two-tier, and one tier does not gate.** Gold transcripts gate P4 and C1, whose
answers genuinely are not in the words. ASR transcripts gate P1/P2/P3, which are claims about
cascaded systems — gating those on a perfect transcript would reject the whole category and prove
nothing. The gold-leak rate for P1/P2/P3 is reported anyway.

**The IHM/SDM microphone pair is not the main arm.** It is a channel manipulation and the paper plan
excludes channel/room confounds, which need an expensive codec control. It stays as a validity
ablation.

**Scale.** 200 items, not 3 750–4 500. Kaggle free tier is 30 GPU-hours/week; the plan's full grid
budgets 500–1 000 A100-hours. This is the plan's own week-1/week-2 pilot, executed properly.

**Yield is the binding constraint, and it is measured.** On AMI, one meeting gives ~104 mentions,
which collapse to ~51 attribute groups, of which roughly **one** has the three distinct competing
values an item needs — about **3.5 usable proposals per meeting**. Reaching 180 items takes ~50
meetings (~25 h of audio, ~7 GB of Mix-Headset wav, several hours of CPU). This is the paper plan's
"P3 natural yield below usable rate" risk with a number attached.

**C1 currently asks the wrong question.** The paper plan's C1 asks about *delivery* ("was there a
point where he sounded unsure?"). The implemented C1 asks the same value question with a
hesitancy-marked target, which is a weaker instrument. Items carry
`question_type: "value_not_delivery"` in provenance so this cannot be forgotten. Redesigning C1 is
open work.

**Annotation.** One person verifying, not five native-speaker teams. Per-language, per-category
agreement cannot be reported.

**P2 is English-only.** AMI has gold speaker turns, so overlap is exact. YODAS2 has no diarisation,
so Hindi and Bengali contribute P1/P3/P4/C1 and no masked items. The item pack records which
categories each source could express at all, so an empty P2 cell for `hi` is distinguishable from a
cell where nothing was found.

---

## Retired: LongAudioBench

The previous four-task suite has been **removed** from the tree. Three of its
tasks had ground truth that was never derived from the audio, so its outputs
could not be evidence for anything:

| Task | Defect |
|---|---|
| Narrative Coherence | `verdict = random.choice(["Validates","Debunks"])`; clues B and C were never mixed into any audio |
| Speaker Drift | `target_speaker_id` fixed at 2, first appearance a constant `01:00` - always answering "Speaker 3, 01:00" scored 1.0 |
| Soundscape | event timeline generated randomly *before* the audio |
| ANiH | audio-derived timestamp, but synthetic damped-sine needles and a hardcoded `preceding_sound` label |

Its code is recoverable from git history (removed in the commit that references
this section). Its result files are archived in `archive/` and gitignored.

## License

MIT. Source corpora carry their own: AMI is CC-BY-4.0, YODAS2 CC-BY-3.0, Shrutilipi CC-BY-4.0.
