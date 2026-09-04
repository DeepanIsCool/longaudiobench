#!/usr/bin/env python3
"""Generate the UNDERTONE Kaggle notebooks.

One notebook per model, each written against that model's own documentation --
but every cell except the adapter view is generated from a single template, so
the shared 90% cannot drift between thirteen files.  That is what makes
thirteen tailored notebooks still produce comparable numbers.

    python scripts/make_notebooks.py --out notebooks

Regenerate after touching the template or the roster; never hand-edit the
output, it will be overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from undertone import adapters  # noqa: E402
from undertone.adapters.base import _REGISTRY  # noqa: E402

REPO_URL = "https://github.com/DeepanIsCool/longaudiobench.git"
REPO_REF = "undertone"     # pin to a commit sha before the run that goes in the paper
ITEM_PACK_DATASET = "undertone-item-pack"

# transformers floor for the roster.  Freeze to the exact resolved version
# (printed by every notebook's environment cell) before the paper run.
TRANSFORMERS = "transformers>=4.57.1"
BASE_PIP = [
    TRANSFORMERS,
    "accelerate>=1.0.0",
    "librosa>=0.10.2",
    "soundfile>=0.12.1",
]

# Human-facing facts for the header cell.  Everything mechanical (ceiling,
# primary scorer, caveat text) is read off the adapter class instead.
META = {
    "qwen2_audio_7b": dict(
        n=10, params="8.40B", vram="16.8 GB", gpu="2xT4",
        doc="https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct",
        facts=[
            "Whisper-large-v3 encoder fixed at 3000 mel frames = **30 s hard cap**.",
            "Audio past 30 s is dropped by the feature extractor with no warning.",
            "L1 is the only untruncated condition; L2/L3/L4 rows carry `truncated: true`.",
        ]),
    "qwen2_5_omni_3b": dict(
        n=11, params="5.44B", vram="~11 GB", gpu="1xT4",
        doc="https://huggingface.co/Qwen/Qwen2.5-Omni-3B",
        pip=[TRANSFORMERS, "accelerate>=1.0.0", "librosa>=0.10.2", "soundfile>=0.12.1"],
        facts=[
            "32k context at 25 audio tokens/s; AudioMarathon Table 8 gives ~21 min.",
            "Thinker-Talker architecture: `disable_talker()` at load, no speech output wanted.",
            "Logits come off `model.thinker`, not the top-level wrapper.",
        ]),
    "qwen2_5_omni_7b": dict(
        n=12, params="10.73B", vram="~21 GB (~14 GB after disable_talker)", gpu="2xT4",
        doc="https://huggingface.co/Qwen/Qwen2.5-Omni-7B",
        facts=[
            "Same ceiling and plumbing as the 3B; only the shard count differs.",
            "Talker disabled at load, which is what brings it inside 2xT4.",
        ]),
    "phi4_multimodal": dict(
        n=13, params="5.57B", vram="~11 GB", gpu="1xT4",
        doc="https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
        pip=["transformers==4.48.2", "accelerate>=0.34.0", "librosa>=0.10.2",
             "soundfile>=0.12.1", "peft>=0.13.2", "backoff", "scipy"],
        facts=[
            "**Own transformers pin (4.48.2)** -- its remote code is version-sensitive.",
            "`_attn_implementation='eager'`; the card specifies this for pre-Ampere GPUs.",
            "Prompt is literal: `<|user|><|audio_1|>{q}<|end|><|assistant|>`.",
            "Audio goes in as `audios=[(array, sr)]` -- a tuple, unlike every other model here.",
            "Card suggests **40 s for QA**, 30 min only for summarization: long bands "
            "exceed the documented QA guidance and are flagged in the results.",
        ]),
    "gemma3n_e2b": dict(
        n=14, params="5.44B raw", vram="~11 GB", gpu="1xT4", gated=True,
        doc="https://huggingface.co/google/gemma-3n-E2B-it",
        facts=[
            "**Gated.** Accept the licence on the Hub, then add `HF_TOKEN` as a Kaggle secret.",
            "USM audio encoder ships configured for **30 s** at ~6.25 tokens/s (160 ms/token).",
            "Google note the encoder is streaming and not fundamentally capped; the "
            "*released* implementation is, so 30 s is what we report.",
            "'E2B' is the effective size -- 5.44B raw params, and VRAM follows the raw count.",
        ]),
    "gemma3n_e4b": dict(
        n=15, params="7.85B raw", vram="~16 GB", gpu="2xT4", gated=True,
        doc="https://huggingface.co/google/gemma-3n-E4B-it",
        facts=[
            "**Gated.** Same licence step as E2B.",
            "Same 30 s ceiling; 7.85B raw params, so it shards over both T4s.",
        ]),
    "aero_1_audio": dict(
        n=16, params="~1.5B + encoder", vram="~4 GB", gpu="1xT4",
        doc="https://huggingface.co/lmms-lab/Aero-1-Audio",
        facts=[
            "Cheapest full-ladder run in the roster: ~4 GB leaves real headroom on one T4.",
            "Card recommends `flash_attention_2`; **T4 is sm75 so we use `sdpa`**.",
            "`eos_token_id=151645` must be passed to `generate` explicitly.",
            "Message type is `audio_url` with the literal string `\"placeholder\"`.",
            "Card snippet says `Aero-1-Audio-1.5B`; the repo is `Aero-1-Audio`. Both are tried.",
        ]),
    "voxtral_mini_3b": dict(
        n=17, params="4.68B", vram="~9.5 GB", gpu="1xT4",
        doc="https://huggingface.co/mistralai/Voxtral-Mini-3B-2507",
        pip=BASE_PIP + ["mistral-common[audio]>=1.8.1"],
        facts=[
            "Longest documented window in the roster: 30 min ASR, **40 min understanding**.",
            "Mistral ship it for vLLM; we use the Transformers path so letter logits "
            "are readable, which is what keeps scoring uniform across all thirteen.",
            "Chat template takes audio by path, so each window is written to a temp wav.",
        ]),
    "moss_audio_4b_instruct": dict(
        n=18, params="~5.2B", vram="10.4 GB", gpu="1xT4",
        doc="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        facts=[
            "`processor_config.json` sets `mel_dtype: bfloat16`; **T4 has no bf16 compute**, "
            "so the adapter forces fp16 before the first processor call.",
            "`MossAudioProcessor.__call__` neither chunks nor truncates -- the ceiling is the "
            "40 960-token LLM context at 12.5 audio tokens/s plus 2-second time markers.",
            "`config.json` maps AutoConfig and AutoProcessor but **not** AutoModel; the "
            "adapter tries each auto-class and records which one worked.",
            "Declared ceiling is config-derived. Run `measure_ceiling` to get a real number.",
        ]),
    "moss_audio_4b_thinking": dict(
        n=19, params="~5.2B", vram="10.4 GB", gpu="1xT4",
        doc="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        facts=[
            "Trained to reason first, so the **first generated token is a thought, not an "
            "answer** -- free generation with the `<think>` block stripped is primary here, "
            "and letter logits are kept only as a diagnostic.",
            "Same fp16 mel override and context arithmetic as the Instruct variant.",
        ]),
    "moss_audio_8b_instruct": dict(
        n=20, params="9.05B", vram="~18 GB", gpu="2xT4",
        doc="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        facts=["Same plumbing as the 4B Instruct; 9.05B shards over both T4s."]),
    "moss_audio_8b_thinking": dict(
        n=21, params="9.05B", vram="~18 GB", gpu="2xT4",
        doc="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking",
        facts=["Thinking variant on 2xT4: free-gen primary, logits diagnostic."]),
    "audio_flamingo_next": dict(
        n=22, params="8.27B", vram="~16.5 GB", gpu="2xT4",
        doc="https://huggingface.co/nvidia/audio-flamingo-next-hf",
        facts=[
            "Only model whose documented ceiling reaches our longest band exactly: the "
            "released processor is configured for **1800 s**, in 30 s internal windows.",
            "Card loads in bfloat16; **fp16 here for sm75**, with `input_features` cast "
            "to the model dtype explicitly -- the processor does not do it.",
            "Conversation is a list **of lists** (batched), unlike the rest of the roster.",
            "Architecture string is `musicflamingo`; `AutoModel` is the documented entry point.",
        ]),
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.rstrip().splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CELL_PIP = """\
# Pinned for this model. If `load()` fails, this cell is the first thing to change.
{pips}
print("--- resolved versions (freeze these before the paper run) ---")
import importlib.metadata as md
for pkg in ["transformers", "accelerate", "torch", "librosa"]:
    try:
        print(f"{{pkg:14s}} {{md.version(pkg)}}")
    except md.PackageNotFoundError:
        print(f"{{pkg:14s}} not installed")
"""

CELL_ENV = """\
import os, random, sys, json
import numpy as np, torch

SEED = 20260904
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Weights go to /kaggle/temp: scratch, and it does NOT count against the 20 GB
# /kaggle/working output cap. A 16-18 GB checkpoint in /kaggle/working would
# fail the commit at the end of the session.
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
{token_block}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{{i}}  {{p.name}}  {{p.total_memory/1e9:.1f}} GB  sm{{p.major}}{{p.minor}}")
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
    print("\\nsm < 80: no bf16 compute and no flash-attention-2. "
          "Every adapter loads in fp16 for this reason.")
"""

TOKEN_BLOCK_GATED = """
# Gated model. The token is resolved after the repo is cloned (next cell), from:
#   HF_TOKEN in the environment -> Kaggle secret named HF_TOKEN -> .hf_token at
#   the repo root (gitignored - the repo is public, so a token in tracked source
#   would be scraped from GitHub within minutes).
# Accept the licence on the Hub with the account that owns the token first.
GATED = True
"""

CELL_REPO = """\
REPO_URL = "{repo_url}"
REPO_REF = "{repo_ref}"   # pin to a commit sha before the paper run

import subprocess, shutil, os, sys
if os.path.exists("/kaggle/working/longaudiobench"):
    shutil.rmtree("/kaggle/working/longaudiobench")
for attempt in range(3):
    rc = subprocess.call(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                          REPO_URL, "/kaggle/working/longaudiobench"])
    if rc == 0:
        break
else:
    raise RuntimeError("could not clone the benchmark repo")

sys.path.insert(0, "/kaggle/working/longaudiobench")
import importlib; importlib.invalidate_caches()

from undertone import ItemPack, adapters, env, runner, scoring
print("adapters registered:", len(adapters.list_adapters()))

if env.export_hf_token():
    print("HF token resolved")
elif globals().get("GATED"):
    raise RuntimeError(
        "this model is gated and no token was found. Add a Kaggle secret named "
        "HF_TOKEN, or write the token to .hf_token at the repo root.")

hw = env.resolve_hardware()
print(f"hardware: {{hw.detail}}  dtype={{hw.dtype}}  signature={{hw.signature}}")
print(f"versions: {{env.versions()}}")
# Every result row is stamped with this signature. The analysis refuses to put
# two signatures in one table -- a benchmark whose rows came from different
# backends compares machines, not models.
"""

CELL_ADAPTER = """\
# The adapter is the ONLY per-model code in this project. Everything above it --
# prompts, option shuffling, ladder windows, scoring, checkpointing -- is shared,
# which is what lets thirteen tailored notebooks produce comparable numbers.
import inspect
from undertone.adapters.base import get_adapter

ADAPTER_KEY = "{key}"
adapter = get_adapter(ADAPTER_KEY)

print(json.dumps(adapter.describe(), indent=2))
print("\\n" + "=" * 72 + "\\n")
print(inspect.getsource(type(adapter)))
"""

CELL_PACK = """\
# The item pack is built once on CPU (notebook 01) and attached as a Kaggle
# Dataset, so a model sweep never re-harvests audio.
PACK_DIR = "/kaggle/input/{pack}"
if not os.path.isdir(PACK_DIR):
    raise FileNotFoundError(
        f"attach the '{pack}' dataset to this notebook "
        "(Add Input -> Datasets), or run notebooks/01_build_item_pack.ipynb first")

pack = ItemPack.load(os.path.join(PACK_DIR, "item_pack.jsonl"))
print(f"{{len(pack)}} items from {{len({{i.recording_id for i in pack}})}} recordings")
for key, n in sorted(pack.counts("lang", "category").items()):
    print(f"  {{key[0]}}  {{key[1]}}  n={{n}}")

# What this model can and cannot ingest, stated before the run rather than
# discovered from a table of zeros afterwards.
from undertone.ladder import CONDITIONS, window_for
print(f"\\ndocumented ceiling: {{adapter.max_audio_s:.0f}} s")
for cond in CONDITIONS:
    over = sum(1 for i in pack if window_for(i, cond).seconds > adapter.max_audio_s)
    print(f"  {{cond}}: {{over}}/{{len(pack)}} cells exceed it -> truncated, not scored as 0")
"""

CELL_RUN = """\
OUT = f"/kaggle/working/results/{{ADAPTER_KEY}}.jsonl"

# Resumable: a killed 12 h session picks up where it stopped. Rerun this cell
# after a restart rather than starting the sweep over.
runner.run_model(
    adapter,
    pack,
    out_path=OUT,
    conditions=CONDITIONS,
    seed=SEED,
    run_id="pilot",
    audio_root=PACK_DIR,
)
adapter.unload()
print("done ->", OUT)
"""

CELL_SUMMARY = """\
rows = runner.load_rows(OUT)
usable = runner.scorable(rows)          # drops errors and truncated cells
print(f"{{len(rows)}} rows, {{len(usable)}} scorable, "
      f"{{sum(1 for r in rows if r.get('truncated'))}} truncated, "
      f"{{sum(1 for r in rows if r.get('error'))}} errored")

by_cond = {{c: [r for r in usable if r["condition"] == c] for c in CONDITIONS}}
costs = scoring.ladder_costs(by_cond)
print("\\nladder:", json.dumps({{k: round(v, 3) for k, v in costs.items()}}, indent=2))

print("\\nper category (L3), salience trap is the headline diagnostic:")
for cat in ["P1", "P2", "P3", "P4", "C1"]:
    subset = [r for r in by_cond["L3"] if r["category"] == cat]
    if subset:
        s = scoring.summarize(subset)
        print(f"  {{cat}}  n={{s['n']}}  acc={{s['accuracy']:.3f}}  "
              f"salience={{s['salience_trap_rate']:.3f}}  "
              f"recency={{s['recency_trap_rate']:.3f}}")

degenerate = [r for r in usable if r.get("logit_degenerate")]
if degenerate:
    print(f"\\nWARNING: {{len(degenerate)}} cells had all four letter logits equal. "
          "That is not a 25% baseline, it is a broken measurement -- check the adapter.")

with open(f"/kaggle/working/results/{{ADAPTER_KEY}}_summary.json", "w") as fh:
    json.dump({{"model": adapter.describe(), "ladder": costs,
               "overall": scoring.summarize(usable)}}, fh, indent=2, default=str)

import subprocess
subprocess.run(["tar", "-czf", f"/kaggle/working/{{ADAPTER_KEY}}_results.tar.gz",
                "-C", "/kaggle/working", "results"], check=True)
print("packaged")
"""


def build_model_notebook(key: str) -> dict:
    cls = _REGISTRY[key]
    meta = META[key]
    adapter = cls()
    pips = "\n".join(f'%pip install -q "{p}"' for p in meta.get("pip", BASE_PIP))

    facts = "\n".join(f"- {f}" for f in meta["facts"])
    header = f"""# UNDERTONE - `{key}`

**{cls.model_id}** &nbsp;|&nbsp; {meta['params']} &nbsp;|&nbsp; {meta['vram']} fp16 &nbsp;|&nbsp; {meta['gpu']}
&nbsp;|&nbsp; documented ceiling **{cls.max_audio_s:.0f} s** &nbsp;|&nbsp; primary scorer **{cls.primary}**

[model card]({meta['doc']})

## What this model's documentation says

{facts}

## What is shared with the other twelve notebooks

Prompt construction, per-run option shuffling, ladder windows (L1/L2/L3/L4),
letter-log-likelihood scoring, checkpointing and the output schema all come from
the `undertone` package. Only the adapter below is model-specific. That is the
whole design: tailored where the model demands it, identical everywhere else, so
the numbers compare.

## Protocol

Four options per item carrying **roles**, not fixed letters (letters are redrawn
per item per run, so preferring "A" cannot beat chance):

| role | content | diagnoses |
|---|---|---|
| `correct` | the right answer | - |
| `salience` | a louder / stressed / repeated competing mention | **salience prior** |
| `recency` | the most recent mention of the topic | recency bias |
| `absent` | "not mentioned in the recording" | fabrication; correct on null items |

Primary metric is the next-token distribution over A/B/C/D -- one forward pass,
no generation, no regex. Free generation runs alongside and `unparseable` is its
own bucket, never a wrong answer.

Cells whose window exceeds the ceiling above run **truncated and flagged**, and
are excluded from the accuracy table. They are never scored as zero.
"""

    token_block = TOKEN_BLOCK_GATED if meta.get("gated") else ""
    return notebook([
        md(header),
        code(CELL_PIP.format(pips=pips)),
        code(CELL_ENV.format(token_block=token_block)),
        code(CELL_REPO.format(repo_url=REPO_URL, repo_ref=REPO_REF)),
        code(CELL_ADAPTER.format(key=key)),
        code(CELL_PACK.format(pack=ITEM_PACK_DATASET)),
        code(CELL_RUN),
        code(CELL_SUMMARY),
    ])


SMOKE_HEADER = """# UNDERTONE - smoke test

Run this **before** any sweep. It costs ~15 minutes of the weekly 30 GPU-hour
quota and it is the difference between finding a broken adapter now and finding
it in six hours of zeros.

Each adapter is checked for:

| check | failure it catches |
|---|---|
| `loads` | gated repo, wrong auto-class, OOM across 2xT4 |
| `finite` | bf16-trained weights overflowing in fp16 on sm75 -- a NaN row argmaxes to "A" |
| `discriminates` | all four letter logits equal: wrong token ids, or a prompt the model never sees as a question |
| `parses` | free generation that never emits a bare letter |
| `deterministic` | sampling left on. The retired suite's "same" config gave 0.14 and 0.28 on two runs |
| `truncates_loudly` | over-long audio setting the flag instead of vanishing silently |

Models are loaded one at a time and unloaded immediately, so peak VRAM is one
model's worth, not thirteen.
"""

SMOKE_RUN = """\
# One at a time: 13 models will not co-reside on 32 GB.
# Start with the two that bracket the roster -- Aero (~4 GB, 15 min ceiling) and
# Qwen2-Audio (16.8 GB, 30 s ceiling) -- then widen once both are green.
# Smallest first, so a disk or driver problem surfaces in minutes rather than
# after a 17 GB download. Gated models last: they fail fast without a token and
# that failure should not sit in front of twelve models that would have passed.
KEYS = [
    "aero_1_audio",            # ~4 GB, brackets the light end
    "voxtral_mini_3b",         # 9.5 GB
    "moss_audio_4b_instruct",  # 10.4 GB, bf16 mel override
    "moss_audio_4b_thinking",  # free-gen primary
    "phi4_multimodal",         # 11 GB, own transformers pin
    "qwen2_5_omni_3b",         # 11 GB, disable_talker
    "audio_flamingo_next",     # 16.5 GB, 2xT4
    "qwen2_audio_7b",          # 16.8 GB, 30 s cap - brackets the heavy end
    "moss_audio_8b_instruct",  # 18 GB, 2xT4
    "moss_audio_8b_thinking",
    "qwen2_5_omni_7b",         # 21 GB
    "gemma3n_e2b",             # gated
    "gemma3n_e4b",             # gated
]

from undertone.smoke import disk_free_gb, purge_cache, smoke_adapter

# ~130 GB of checkpoints against ~60 GB of Kaggle disk: each model's weights are
# deleted after it is checked, or the run dies on a download partway through
# rather than on anything worth knowing.
PURGE_AFTER_EACH = True

reports = []
for key in KEYS:
    print(f"\\n{'=' * 72}\\n{key}   (disk free: {disk_free_gb()} GB)\\n{'=' * 72}")
    adapter = get_adapter(key)
    report = smoke_adapter(adapter)
    reports.append(report)
    for name, result in report["checks"].items():
        print(f"  {'PASS' if result['ok'] else 'FAIL'}  {name}: {result['detail']}")
    if report.get("traceback"):
        print(report["traceback"])
    if PURGE_AFTER_EACH:
        print(f"  freed {purge_cache(adapter.model_id)} GB")

with open("/kaggle/working/smoke_report.json", "w") as fh:
    json.dump(reports, fh, indent=2, default=str)
"""

SMOKE_VERDICT = """\
bad = [r for r in reports if not r.get("ok")]
for r in reports:
    print(f"{'ok  ' if r.get('ok') else 'FAIL'} {r['key']:26s} "
          f"{r.get('peak_vram_gb', '?')} GB  {r.get('seconds', '?')}s  "
          f"{'failed: ' + ', '.join(r['failures']) if r['failures'] else ''}")

assert not bad, f"fix these adapters before spending quota on a sweep: {[r['key'] for r in bad]}"
print("\\nall green - safe to run a sweep")
"""

SMOKE_CEILING = """\
# Optional and expensive. MOSS-Audio's ceiling is inferred from config.json
# (12.5 tokens/s against a 40 960 context) and is not documented by its authors,
# so this is how the truncation table gets a measured number.
from undertone.smoke import measure_ceiling

MEASURE = []          # e.g. ["moss_audio_4b_instruct"]
for key in MEASURE:
    print(json.dumps(measure_ceiling(get_adapter(key)), indent=2))
"""


def build_smoke_notebook() -> dict:
    return notebook([
        md(SMOKE_HEADER),
        code(CELL_PIP.format(pips="\n".join(f'%pip install -q "{p}"' for p in BASE_PIP))),
        code(CELL_ENV.format(token_block=TOKEN_BLOCK_GATED)),
        code(CELL_REPO.format(repo_url=REPO_URL, repo_ref=REPO_REF)),
        code("from undertone.adapters.base import get_adapter\n"
             "for key in adapters.list_adapters():\n"
             "    a = get_adapter(key)\n"
             "    print(f\"{key:26s} {a.max_audio_s:7.0f}s  {a.primary:8s} {a.model_id}\")"),
        code(SMOKE_RUN),
        code(SMOKE_VERDICT),
        code(SMOKE_CEILING),
    ])


# --------------------------------------------------------------------------
# the two notebooks that are not per-model
# --------------------------------------------------------------------------

BUILD_HEADER = """# UNDERTONE - build the item pack

Run this **once**, on CPU, then upload `data/item_pack/` to Kaggle as the
`undertone-item-pack` dataset. The thirteen model notebooks attach it, so a
sweep never re-harvests audio.

## What an item is

One four-option question about **one span of one real recording**. Nothing is
inserted, spliced or synthesised - the prominence differences the categories
name are differences the speakers themselves produced. That is the whole
difference from the retired LongAudioBench tasks, whose labels were drawn with
`random.choice` before the audio existed.

| category | acoustic cause | share |
|---|---|---|
| P1 quiet | low energy relative to this recording, and a real drop below its median | 22% |
| P2 masked | overlapped by another speaker (gold, from segment timings) | 22% |
| P3 backgrounded | prosodically flat aside, while a competing value is loud and repeated | 22% |
| P4 corrected | self-repair; the corrected value is the answer | 22% |
| C1 delivered | carried by hesitation and strain - the discriminant | 12% |

## Where the options come from

Every distractor is a **real competing mention of the same quantity kind from
the same recording**, never a fabrication. That is what makes a wrong answer
diagnostic rather than merely wrong:

- `salience` - the loudest and/or most repeated competing value
- `recency` - the latest competing value
- `absent` - "not mentioned in the recording"; correct on the 10% null items

## Source

AMI Meeting Corpus (CC-BY-4.0), Mix-Headset channel. Real 30-60 min meetings
with genuine crosstalk, asides and self-repairs.

The IHM/SDM microphone pair is **not** used for the main arm. It is a channel
manipulation, and the paper plan excludes channel/room confounds explicitly
(they need an expensive codec control). It stays available as a validity-control
ablation.

## This notebook does not produce usable items

Everything it writes is `verified: false`. It becomes evidence after the leak
filter runs and after you have listened to the clips. The analysis refuses to
report unverified cells.
"""

CELL_BUILD = """\
# ~150 MB of audio per meeting. Fifteen meetings is roughly one hour of CPU.
# English from AMI (multi-party, gold speaker turns -> the only source that
# can express P2). Hindi and Bengali from YODAS2: long-form and CC-BY, but
# no diarisation, so those languages contribute P1/P3/P4/C1 and no P2.
!python /kaggle/working/longaudiobench/scripts/build_item_pack.py --out /kaggle/working/item_pack --audio-cache /kaggle/temp/source_audio --langs en hi bn --per-lang 10 --target 180
"""

CELL_LEAK = """\
# The audio-necessity gate. Two tiers, because the categories make different
# claims:
#
#   gold transcript -> gates P4 and C1. Their answers genuinely are not in the
#                      words: ASR normalises a repair away and never records
#                      hesitancy. If a text model solves one, it is not an item.
#   ASR transcript  -> gates P1, P2 and P3. These are claims about *cascaded*
#                      systems. A perfect transcript of a muttered utterance
#                      contains the answer by construction, so gating them on
#                      gold would reject the whole category and prove nothing.
#
# Both rates are reported for every category regardless of which one gates. The
# gold-leak rate we do not reject on is a limitation, and it goes in the paper
# rather than in a drawer.
from undertone import ItemPack
from undertone.harvest import asr, leakfilter

PACK_DIR = "/kaggle/working/item_pack"
pack = ItemPack.load(f"{PACK_DIR}/item_pack.jsonl")
GOLD = json.load(open(f"{PACK_DIR}/gold_transcripts.json"))

# Whisper over every recording, cached: a 30-minute transcription is minutes of
# GPU and a notebook restart must not redo it.
transcripts = asr.transcripts_for(pack, audio_root=PACK_DIR,
                                  cache_dir="/kaggle/working/asr_cache")
ASR = asr.as_text(transcripts)
print(f"transcribed {len(ASR)} recordings")
"""

CELL_RECOVERY = """\
# Sharper than "did a text model get it right": did the ASR write the answer
# down at all? If Whisper never transcribed the muttered words, no language
# model behind it could have found them -- audio-necessity as a fact rather than
# an inference from one model's score. This table goes in the paper.
recovery = asr.needle_recovery(transcripts, list(pack))
for row in recovery:
    print(f"{row['category']}  n={row['n']:3d}  "
          f"ASR recovered the answer in {row['recovery_rate']:.0%} of items  "
          f"-> {row['unrecoverable_rate']:.0%} unanswerable by any cascaded system")

with open("/kaggle/working/item_pack/needle_recovery.json", "w") as fh:
    json.dump(recovery, fh, indent=2)
"""

CELL_LEAKRUN = """\
# Plug in whatever text model you have. A stronger solver is a stronger claim.
def make_solver(model_id="Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto").eval()
    ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}

    def solve(prompt: str):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        inputs = tok(chat, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            logits = llm(**inputs).logits[0, -1]
        return max(ids, key=lambda L: float(logits[ids[L]]))
    return solve

solver = make_solver()
report = leakfilter.run_filter(list(pack), GOLD, ASR, solvers=[solver])
print(json.dumps(report.table(list(pack)), indent=2))

kept = leakfilter.apply_filter(list(pack), report)
ItemPack(kept, meta={**pack.meta, "leak_filtered": True}).save(
    f"{PACK_DIR}/item_pack.jsonl")
print(f"\\n{len(pack)} -> {len(kept)} items survived the filter")
"""

CELL_QONLY = """\
# The floor check. No audio, question and options only: this must sit at ~25%.
# If it does not, the distractors leak the answer and every downstream number is
# about the option text rather than about the audio.
from undertone.protocol import question_only
from undertone.scoring import parse_free_letter

hits = 0
for item in pack:
    rendered = question_only(item, seed=SEED)
    letter = solver(rendered.prompt)
    hits += rendered.letter_to_role.get(letter) == item.correct_role
print(f"question-only accuracy: {hits}/{len(pack)} = {hits/len(pack):.3f}  (want ~0.25)")
"""

CELL_VERIFY = """\
# Verification: listen, confirm the key, mark it. Nothing is evidence until this
# has happened. Run it over the pack in batches; it is the slow step and there is
# no way around it.
import IPython.display as ipd, librosa

TO_REVIEW = list(pack)[:20]
for item in TO_REVIEW:
    path = os.path.join("/kaggle/working/item_pack", item.audio_path)
    audio, sr = librosa.load(path, sr=16000, mono=True,
                             offset=max(0, item.needle_start - 3),
                             duration=(item.needle_end - item.needle_start) + 6)
    print(f"\\n{item.item_id}  [{item.category}]  {item.question}")
    for role in ("correct", "salience", "recency"):
        print(f"    {role:9s} {item.options[role]}")
    print(f"    provenance: {item.provenance['why']}")
    ipd.display(ipd.Audio(audio, rate=sr))

# After listening, record the verdicts and re-save:
#   VERDICTS = {"ES2002a_P1_000": True, "ES2002a_P3_004": False, ...}
VERDICTS = {}
if VERDICTS:
    kept = []
    for item in pack:
        if VERDICTS.get(item.item_id) is False:
            continue
        data = item.to_dict()
        data["provenance"] = {**item.provenance,
                              "verified": bool(VERDICTS.get(item.item_id, False))}
        kept.append(type(item).from_dict(data))
    ItemPack(kept, meta={**pack.meta, "verified": True}).save(
        "/kaggle/working/item_pack/item_pack.jsonl")
    print(f"{sum(1 for v in VERDICTS.values() if v)} verified, "
          f"{sum(1 for v in VERDICTS.values() if not v)} rejected")
"""

ANALYSIS_HEADER = """# UNDERTONE - analysis

Reads every `results/<model>.jsonl` the thirteen sweeps produced and builds the
paper tables.

Three rules the retired analysis broke and this one does not:

1. **Truncated cells never enter an accuracy table.** "This model cannot ingest
   the audio" and "this model heard it and got it wrong" are different findings.
   Truncation gets its own table, reported as coverage.
2. **Unverified items are excluded and counted.** A proposal nobody has listened
   to is not evidence.
3. **No composite score.** The paper plan says so, and the retired suite's
   composites hid which term was zero - every reported `0.000` there meant
   "nothing parsed", not "no error".

## The findings these tables test

| | claim | where to look |
|---|---|---|
| F1 | type dominates duration | spread across Category at fixed band vs spread across band at fixed Category |
| F2 | the salience prior is the mechanism | `salience_trap` on P3 vs chance (0.25) |
| F3' | the prior is English-shaped | `salience_trap` on P3, hi and bn vs en |
| F4 | perception is intact, retrieval is not | `RetrievalCost` large while `acc_L1` is high |
| F5 | the mechanism is specific | P1-P4 share the trap signature; C1 does not |

F3' is weaker than the paper plan's F3, which needed a tone language. en/hi/bn
has none: English marks prominence with lexically free stress-accent, while
Hindi and Bengali have weak or fixed word stress plus grammaticalised focus
(`hi`/`to`/`bhi`; `-i`/`-o`; word order). Both the substitution and the fact that
Hindi and Bengali are the same family belong in Limitations.
"""

CELL_LOAD_RESULTS = """\
import glob
from undertone import analysis, runner
from undertone.adapters.base import get_adapter

rows = []
for path in sorted(glob.glob("/kaggle/input/*/results/*.jsonl")
                   + glob.glob("/kaggle/working/results/*.jsonl")):
    rows.extend(runner.load_rows(path))
print(f"{len(rows)} rows from {len({r['model_key'] for r in rows})} models")

problems = analysis.sanity_checks(rows)
for p in problems:
    print("SANITY:", p)
if not problems:
    print("sanity checks clean")
"""

CELL_TABLES = """\
import pandas as pd

limits = {k: get_adapter(k).max_audio_s for k in {r["model_key"] for r in rows}}

t4 = pd.DataFrame(analysis.table4_truncation(rows, limits))
print("Table 4 - per-model input limits and truncation coverage")
print(t4.to_string(index=False))

t1 = pd.DataFrame(analysis.table1_main(rows))
print("\\nTable 1 - main results at L3 (salience_trap is the headline)")
print(t1.to_string(index=False))

t2 = pd.DataFrame(analysis.table2_ladder(rows))
print("\\nTable 2 - ladder decomposition")
print(t2.to_string(index=False))

print("\\nNull items")
print(pd.DataFrame(analysis.table1_nulls(rows)).to_string(index=False))

recovery_path = "/kaggle/input/undertone-item-pack/needle_recovery.json"
if os.path.exists(recovery_path):
    print("\\nAudio necessity - share of items whose answer ASR never wrote down")
    print(pd.DataFrame(json.load(open(recovery_path))).to_string(index=False))

print("\\nScorer gap: letter logits vs free generation")
print(pd.DataFrame(analysis.scorer_gap(rows)).to_string(index=False))

from undertone.analysis import figures
os.makedirs("/kaggle/working/analysis", exist_ok=True)
made = figures.all_figures(rows, "/kaggle/working/analysis")
print("figures:", [p.name for p in made])

for name, frame in [("table1_main", t1), ("table2_ladder", t2),
                    ("table4_truncation", t4),
                    ("table_language", pd.DataFrame(analysis.table_language(rows)))]:
    frame.to_csv(f"/kaggle/working/analysis/{name}.csv", index=False)
    with open(f"/kaggle/working/analysis/{name}.tex", "w") as fh:
        fh.write(frame.to_latex(index=False))
print("\\nwrote /kaggle/working/analysis/")
"""

CELL_FINDINGS = """\
# F2: does the salience trap exist at all? This is the paper plan's kill point.
p3 = [r for r in analysis.usable(rows) if r["category"] == "P3" and r["condition"] == "L3"]
for model in sorted({r["model_key"] for r in p3}):
    cell = [r for r in p3 if r["model_key"] == model]
    rate = scoring.salience_trap(cell)
    lo, hi = scoring.cluster_bootstrap_ci(cell, scoring.salience_trap)
    verdict = "above chance" if lo > 0.25 else "NOT above chance"
    print(f"{model:26s} P3 salience trap {rate:.3f} [{lo:.3f}, {hi:.3f}]  {verdict}")

# F5: the mechanism has to be specific. P1-P4 share the signature; C1 must not.
print()
for category in ["P1", "P2", "P3", "P4", "C1"]:
    cell = [r for r in analysis.usable(rows)
            if r["category"] == category and r["condition"] == "L3"]
    if cell:
        print(f"{category}  salience {scoring.salience_trap(cell):.3f}  "
              f"acc {scoring.accuracy(cell):.3f}  n={len(cell)}")

# F3': is the prior English-shaped? Weaker than the plan's F3 - no tone language.
print()
lang_table = analysis.table_language(rows)
for lang in ("en", "hi", "bn"):
    cells = [r for r in lang_table if r["lang"] == lang and r["category"] == "P3"]
    if cells:
        mean = sum(c["salience_trap"] for c in cells) / len(cells)
        print(f"{lang}  P3 salience trap {mean:.3f}  ({len(cells)} model cells)")
"""


CASCADED_HEADER = """# UNDERTONE - cascaded ASR+LLM control

**Not one of the thirteen.** This is the validity check that preempts the
obvious objection: *isn't this just ASR plus a language model?*

It runs exactly that system through exactly the same protocol - same items, same
option shuffling, same ladder windows, same letter-logit scoring - so any gap
between it and the audio models is the modality, not the harness. That is why it
is a `ModelAdapter` and not a separate script.

## What its failure means, per category

| | why the cascade cannot answer |
|---|---|
| P1 / P2 | Whisper never wrote the muttered or overlapped words down. No language model behind it could have found them. |
| P4 | ASR normalises a self-repair away - "twenty twelve milligrams" loses the seam that says which value survived. |
| C1 | Hesitancy is not in the words at all. |

`01_build_item_pack` measures this directly with `needle_recovery`: the share of
items whose answer never appears in the ASR transcript. That is stronger evidence
than this model's score, because it does not depend on how good the language
model is.

## One detail that matters

VAD filtering is **off**. Voice-activity detection would drop exactly the quiet
spans P1 is about, which would flatter the cascade by never asking it the hard
question.

Each language is swept separately so Whisper is given the right language code
rather than guessing - the most favourable setting for the control.
"""

CELL_CASCADED = """\
from undertone.adapters.cascaded import CascadedWhisperLLM
from undertone.ladder import CONDITIONS

OUT = "/kaggle/working/results/cascaded_whisper_llm.jsonl"

for lang in ("en", "hi", "bn"):
    subset = pack.filter(lang=lang)
    if not len(subset):
        continue
    print(f"\\n=== {lang}: {len(subset)} items ===")
    control = CascadedWhisperLLM(lang=lang)
    runner.run_model(control, subset, out_path=OUT, conditions=CONDITIONS,
                     seed=SEED, run_id="pilot", audio_root=PACK_DIR)
    control.unload()

rows = runner.load_rows(OUT)
usable = runner.scorable(rows)
print(f"\\n{len(usable)} scorable rows")
for cat in ["P1", "P2", "P3", "P4", "C1"]:
    cell = [r for r in usable if r["category"] == cat and r["condition"] == "L3"]
    if cell:
        s = scoring.summarize(cell)
        print(f"  {cat}  n={s['n']:3d}  acc={s['accuracy']:.3f}  "
              f"salience={s['salience_trap_rate']:.3f}")
"""


def build_cascaded_notebook() -> dict:
    return notebook([
        md(CASCADED_HEADER),
        code(CELL_PIP.format(pips="\n".join(
            f'%pip install -q "{p}"' for p in BASE_PIP + ["faster-whisper>=1.0.0"]))),
        code(CELL_ENV.format(token_block="")),
        code(CELL_REPO.format(repo_url=REPO_URL, repo_ref=REPO_REF)),
        code(CELL_PACK.format(pack=ITEM_PACK_DATASET)
             .replace("adapter.max_audio_s", "7200.0")
             .replace("print(json.dumps(adapter.describe(), indent=2))", "")),
        code(CELL_CASCADED),
    ])


def build_item_pack_notebook() -> dict:
    return notebook([
        md(BUILD_HEADER),
        code(CELL_PIP.format(pips="\n".join(
            f'%pip install -q "{p}"' for p in BASE_PIP + ["datasets>=2.19.0"]))),
        code(CELL_ENV.format(token_block="")),
        code(CELL_REPO.format(repo_url=REPO_URL, repo_ref=REPO_REF)),
        code(CELL_BUILD),
        code(CELL_LEAK),
        code(CELL_RECOVERY),
        code(CELL_LEAKRUN),
        code(CELL_QONLY),
        code(CELL_VERIFY),
    ])


def build_analysis_notebook() -> dict:
    return notebook([
        md(ANALYSIS_HEADER),
        code(CELL_PIP.format(pips="\n".join(
            f'%pip install -q "{p}"' for p in ["pandas>=2.1.0", "matplotlib>=3.8.0"]))),
        code(CELL_ENV.format(token_block="")),
        code(CELL_REPO.format(repo_url=REPO_URL, repo_ref=REPO_REF)),
        code(CELL_LOAD_RESULTS),
        code(CELL_TABLES),
        code("from undertone import scoring\n" + CELL_FINDINGS),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="notebooks", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = adapters.list_adapters()
    missing = [k for k in keys if k not in META]
    if missing:
        raise SystemExit(f"no notebook metadata for adapters: {missing}")

    written = []
    for name, builder in (("00_smoke_test", build_smoke_notebook),
                          ("01_build_item_pack", build_item_pack_notebook),
                          ("02_cascaded_control", build_cascaded_notebook),
                          ("90_analysis", build_analysis_notebook)):
        path = args.out / f"{name}.ipynb"
        path.write_text(json.dumps(builder(), indent=1), encoding="utf-8")
        written.append(path)

    for key in keys:
        path = args.out / f"{META[key]['n']}_{key}.ipynb"
        path.write_text(json.dumps(build_model_notebook(key), indent=1), encoding="utf-8")
        written.append(path)

    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written)} notebooks ({len(keys)} models + smoke test, item-pack build, cascaded control and analysis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
