#!/usr/bin/env python3
"""Build an UNDERTONE item pack from real recordings.

    python scripts/build_item_pack.py --out data/item_pack --meetings ES2002a ES2003a

Emits a directory that uploads to Kaggle as the ``undertone-item-pack`` dataset:

    item_pack.jsonl        one MCQItem per line, plus a metadata header line
    audio/<rec>.flac       band-length 16 kHz mono audio, referenced by the items

Everything it writes is marked ``verified: false``. An unverified proposal is
not an item -- it becomes one when a person has listened to the clip and
confirmed the key, and when the leak filter has run. The analysis refuses to
report unverified cells for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from undertone.harvest import build, features, sources  # noqa: E402
from undertone.items import BANDS, CATEGORIES, ItemPack, MCQItem  # noqa: E402

# Paper plan section 4: P1-P4 share one mechanism, C1 is the discriminant.
CATEGORY_SHARE = {"P1": 0.22, "P2": 0.22, "P3": 0.22, "P4": 0.22, "C1": 0.12}
NULL_SHARE = 0.10
SAMPLE_RATE = 16000


# Largest band that a 15 GB GPU can actually hold. At 1800 s every model in the
# roster errors at L3: ~45k audio tokens, quadratic attention, one model tried
# to allocate 60.78 GiB. A band nobody can run measures nothing.
#
# This is also the version the paper plan argues for. Section 7 chose the bands
# "so the claim reads 'fails at 8 minutes' rather than 'fails at 90', which
# removes duration as an explanation", and F1 is that type dominates duration --
# demonstrated by failure at SHORT durations, not long ones.
MAX_RUNNABLE_BAND = 600


def band_for(duration: float, cap: int = MAX_RUNNABLE_BAND) -> int:
    """Largest runnable band a recording can fill. Bands are prefixes."""
    usable = [b for b in BANDS if b <= duration and b <= cap]
    return usable[-1] if usable else BANDS[0]


def harvest_recording(recording: sources.Recording, audio, band: int,
                      with_f0: bool) -> list[tuple[sources.Recording, list[MCQItem]]]:
    """Every band-length window of a recording, each harvested separately.

    Returns (window, items) pairs so the caller can write one audio file per
    window. A 40-minute meeting is four 10-minute haystacks, not one.
    """
    out = []
    for window in recording.windows(band):
        offset = int(window.meta["window_start"] * SAMPLE_RATE)
        chunk = audio[offset: offset + band * SAMPLE_RATE]
        if len(chunk) < band * SAMPLE_RATE * 0.9:
            continue
        feats = features.segment_features(window, chunk, with_f0=with_f0)
        candidates = build.propose(window, feats, band)
        if candidates:
            out.append((window, [build.to_item(window, c, band, i)
                                 for i, c in enumerate(candidates)]))
    return out


def balance(items: list[MCQItem], target_total: int, rng: random.Random) -> list[MCQItem]:
    """Sample to the category shares, spreading across recordings.

    Round-robin over recordings rather than taking the first N: eight items from
    one meeting and none from the other fourteen would make the
    recording-clustered CIs meaningless, and every model would be scored on one
    room's acoustics.
    """
    by_category: dict[str, dict[str, list[MCQItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_category[item.category][item.recording_id].append(item)

    chosen: list[MCQItem] = []
    for category in CATEGORIES:
        want = round(target_total * CATEGORY_SHARE[category])
        buckets = by_category.get(category, {})
        order = sorted(buckets)
        rng.shuffle(order)
        for bucket in buckets.values():
            rng.shuffle(bucket)

        picked, cursor = [], 0
        while len(picked) < want and any(buckets[r] for r in order):
            recording = order[cursor % len(order)]
            if buckets[recording]:
                picked.append(buckets[recording].pop())
            cursor += 1
        chosen.extend(picked)
    return chosen


def add_nulls(items: list[MCQItem], kinds_by_recording: dict[str, set[str]],
              rng: random.Random) -> list[MCQItem]:
    want = max(1, round(len(items) * NULL_SHARE / (1 - NULL_SHARE)))
    pool = [i for i in items if build.absent_kinds(kinds_by_recording.get(i.recording_id, set()))]
    rng.shuffle(pool)
    nulls: list[MCQItem] = []
    for item in pool[:want]:
        absent = build.absent_kinds(kinds_by_recording[item.recording_id])
        context = item.question.split('"')[1] if '"' in item.question else "the discussion"
        nulls.append(build.make_null(item, context, rng.choice(absent)))
    return nulls


def write_audio(path_in: str, path_out: Path, band: int,
               offset: float = 0.0) -> None:
    import librosa
    import soundfile as sf

    path_out.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(path_in, sr=SAMPLE_RATE, mono=True,
                            offset=offset, duration=band)
    sf.write(path_out, audio, SAMPLE_RATE, format="FLAC")


def collect_recordings(langs: list[str], meetings: list[str], per_lang: int,
                       audio_cache: Path) -> list[tuple[sources.Recording, str]]:
    """(recording, local wav/flac path) pairs across every requested language.

    English comes from AMI: real multi-party meetings with gold speaker turns,
    which is the only source here that can express P2 at all. Hindi and Bengali
    come from YODAS2, which is long-form and CC-BY but carries no diarisation --
    so those languages contribute P1/P3/P4/C1 and no P2. That asymmetry is a
    property of what is openly available, and it belongs in Limitations rather
    than in a footnote about an empty cell.
    """
    out: list[tuple[sources.Recording, str]] = []

    if "en" in langs:
        if not meetings:
            raise SystemExit(
                "--langs includes 'en' but --meetings is empty. Silently "
                "harvesting zero meetings is how a shell-quoting mistake becomes "
                "an empty item pack three cells later.")
        print(f"fetching AMI annotations (~30 MB) for {len(meetings)} meetings...")
        ann_dir = sources.download_ami_annotations(str(audio_cache / "ami_annotations"))
        tables = sources.ami_segments_from_annotations(meetings, ann_dir)
        print(f"  transcripts for {len(tables)} meetings")
        for meeting, segments in sorted(tables.items()):
            wav = sources.download_ami_audio(meeting, str(audio_cache / "ami"))
            out.append((sources.Recording(
                recording_id=meeting, audio_path="", lang="en", sector="meetings",
                duration=0.0, segments=segments), wav))

    for lang in [l for l in langs if l != "en"]:
        print(f"streaming YODAS2 for {lang}...")
        found = sources.yodas_recordings(
            lang, limit=per_lang, audio_dir=str(audio_cache / f"yodas_{lang}"))
        print(f"  {len(found)} long-form videos")
        out.extend((rec, rec.audio_path) for rec in found)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/item_pack"))
    ap.add_argument("--langs", nargs="+", default=["en"], choices=["en", "hi", "bn"])
    ap.add_argument("--meetings", nargs="*", default=list(sources.AMI_DEFAULT_MEETINGS))
    ap.add_argument("--per-lang", type=int, default=10,
                    help="recordings per non-English language")
    ap.add_argument("--audio-cache", type=Path, default=Path("data/source_audio"))
    ap.add_argument("--target", type=int, default=180,
                    help="primary items before null items are added")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--band-cap", type=int, default=MAX_RUNNABLE_BAND,
                    help="largest duration band to emit; 1800 needs a GPU that "
                         "can hold ~45k audio tokens of attention")
    ap.add_argument("--no-f0", action="store_true",
                    help="skip the pitch pass (faster; disables P3 detection)")
    args = ap.parse_args()

    import librosa

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    proposals: list[MCQItem] = []
    kinds_by_recording: dict[str, set[str]] = {}
    expressible: dict[str, set[str]] = {}
    # The leak filter's gold tier needs these, and regenerating them later would
    # mean re-parsing every annotation file.
    gold_transcripts: dict[str, str] = {}
    total_windows = 0

    for recording, source_path in collect_recordings(
            args.langs, args.meetings, args.per_lang, args.audio_cache):
        duration = recording.duration or librosa.get_duration(path=source_path)
        band = band_for(duration, args.band_cap)
        if duration < BANDS[0]:
            print(f"{recording.recording_id}: {duration:.0f}s is under the "
                  f"{BANDS[0]}s floor, skipping")
            continue

        # collect_recordings builds AMI entries with duration 0.0 and leaves the
        # real value to be measured here. windows() needs it: at 0.0 the loop
        # condition start + band <= duration is never true and every meeting
        # silently yields no windows.
        recording.duration = duration

        # Load the whole recording, not just the first band: every window of it
        # is a haystack.
        audio, _ = librosa.load(source_path, sr=SAMPLE_RATE, mono=True)

        harvested = harvest_recording(recording, audio, band, with_f0=not args.no_f0)
        total_windows += len(harvested)
        n = sum(len(items) for _, items in harvested)
        print(f"{recording.recording_id} [{recording.lang}]: {duration / 60:.0f} min "
              f"-> {len(harvested)} x {band}s windows -> {n} proposals")

        for window, items in harvested:
            rel = f"audio/{window.recording_id}.flac"
            items = [MCQItem.from_dict({**i.to_dict(), "audio_path": rel})
                     for i in items]
            write_audio(source_path, args.out / rel, band,
                        offset=window.meta["window_start"])
            proposals.extend(items)
            key = window.recording_id
            gold_transcripts[key] = " ".join(seg.text for seg in window.segments)
            kinds_by_recording[key] = {i.provenance["quantity_kind"] for i in items}
            expressible[key] = sources.available_categories(window)

    if not proposals:
        if total_windows == 0:
            print("\nZERO windows from every recording. That is a bug, not a yield "
                  "problem: check that each Recording carries its real duration "
                  "before windows() is called.")
        else:
            print("\nno proposals. Widen --meetings/--per-lang, or drop --no-f0 for P3.")
        return 1

    primary = balance(proposals, args.target, rng)
    pack_items = primary + add_nulls(primary, kinds_by_recording, rng)

    pack = ItemPack(pack_items, meta={
        "build_seed": args.seed,
        "langs": args.langs,
        "sources": {"en": "AMI Meeting Corpus (CC-BY-4.0), Mix-Headset",
                    "hi": "YODAS2 (CC-BY-3.0), long-form",
                    "bn": "YODAS2 (CC-BY-3.0), long-form"},
        "proposals": len(proposals),
        "kept": len(primary),
        # Paper plan section 11: report this so findings cannot be blamed on
        # writing items until models fail.
        "rejection_rate": round(1 - len(primary) / len(proposals), 3),
        # Which categories each source could express at all, so an empty cell is
        # distinguishable from a cell the source could never have filled.
        "expressible_categories": {k: sorted(v) for k, v in expressible.items()},
        "verified": False,
        "leak_filtered": False,
    })
    pack.save(args.out / "item_pack.jsonl")
    (args.out / "gold_transcripts.json").write_text(
        json.dumps(gold_transcripts, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{len(proposals)} proposals -> {len(primary)} balanced "
          f"+ {len(pack_items) - len(primary)} null = {len(pack_items)} items")
    for key, n in sorted(pack.counts("lang", "category").items()):
        print(f"  {key[0]}  {key[1]}: {n}")
    print(f"  recordings: {len({i.recording_id for i in pack})}")
    print(f"\nwrote {args.out}/item_pack.jsonl")
    print("\nNOT YET USABLE: run the leak filter and listen to the clips. "
          "Every item is marked verified=false until you do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
