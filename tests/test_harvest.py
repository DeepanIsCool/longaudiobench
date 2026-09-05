"""Tests for the harvest pipeline.

This is where ground truth is created, so it is where the retired suite failed:
labels were drawn with random.choice before the audio existed. Every test here
checks that an item's key traces back to a real span of a real recording.
"""

from __future__ import annotations

import numpy as np
import pytest

from undertone.harvest import build, features, leakfilter, mentions
from undertone.harvest.sources import Recording, Segment
from undertone.items import ROLES

SR = 16000


# ---------------------------------------------------------------- mentions

class TestMentions:
    def test_finds_value_and_unit(self):
        found = mentions.find_mentions("we settled on twelve milligrams for now",
                                       "en", 0, 10.0, 14.0)
        assert len(found) == 1
        assert found[0].value == 12.0
        assert found[0].kind == "mass"
        assert "twelve milligrams" in found[0].surface

    def test_digit_forms(self):
        found = mentions.find_mentions("about 250 mg", "en", 0, 0.0, 2.0)
        assert found[0].value == 250.0

    @pytest.mark.parametrize("lang,text,value", [
        ("hi", "पंद्रह मिलीग्राम की खुराक", 15.0),
        ("bn", "পাঁচ মিলিগ্রাম করে দিন", 5.0),
    ])
    def test_indic_number_words(self, lang, text, value):
        found = mentions.find_mentions(text, lang, 0, 0.0, 3.0)
        assert found and found[0].value == value

    def test_indic_digits_are_normalised(self):
        found = mentions.find_mentions("२० मिलीग्राम", "hi", 0, 0.0, 2.0)
        assert found and found[0].value == 20.0

    def test_context_excludes_the_number_itself(self):
        """Otherwise the question would contain its own answer."""
        found = mentions.find_mentions(
            "for the trial arm we used twelve milligrams every morning",
            "en", 0, 0.0, 5.0)
        assert "twelve" not in found[0].context
        assert "trial" in found[0].context

    def test_timing_is_inside_the_segment(self):
        found = mentions.find_mentions("later we agreed on 40 percent", "en", 0, 100.0, 106.0)
        assert 100.0 <= found[0].start <= 106.0

    def test_grouping_is_by_quantity_kind(self):
        found = (mentions.find_mentions("20 mg", "en", 0, 0, 1)
                 + mentions.find_mentions("30 percent", "en", 1, 1, 2)
                 + mentions.find_mentions("40 mg", "en", 2, 2, 3))
        groups = mentions.group_by_kind(found)
        assert set(groups) == {"mass", "percent"}
        assert mentions.distinct_values(groups["mass"]) == 2

    def test_no_unit_no_mention(self):
        """A bare number has no comparable competitors, so it makes no item."""
        assert mentions.find_mentions("there were 47 of them", "en", 0, 0, 2) == []


# ---------------------------------------------------------------- features

def toy_recording(texts_and_times, lang="en", duration=300.0) -> Recording:
    segs = [Segment(start=s, end=e, speaker=spk, text=t)
            for s, e, spk, t in texts_and_times]
    return Recording("rec_test", "audio/rec_test.flac", lang, "meetings", duration, segs)


class TestOverlap:
    def test_overlap_is_gold_from_timings(self):
        a = Segment(10.0, 20.0, "A", "x")
        b = Segment(15.0, 25.0, "B", "y")
        assert features.overlap_ratio(a, [a, b]) == pytest.approx(0.5)

    def test_same_speaker_does_not_count_as_masking(self):
        a = Segment(10.0, 20.0, "A", "x")
        b = Segment(12.0, 18.0, "A", "y")
        assert features.overlap_ratio(a, [a, b]) == 0.0

    def test_no_overlap(self):
        a = Segment(0.0, 5.0, "A", "x")
        b = Segment(6.0, 9.0, "B", "y")
        assert features.overlap_ratio(a, [a, b]) == 0.0


class TestSegmentFeatures:
    def test_quiet_is_relative_to_this_recording(self):
        """An absolute dB cut would only ever find quiet recordings."""
        rec = toy_recording([(0, 2, "A", "a"), (2, 4, "B", "b"),
                             (4, 6, "C", "c"), (6, 8, "D", "d")], duration=10)
        audio = np.zeros(SR * 10, dtype=np.float32)
        for i, gain in enumerate([0.5, 0.4, 0.3, 0.01]):   # last one is the mutter
            audio[i * 2 * SR:(i + 1) * 2 * SR] = gain
        feats = features.segment_features(rec, audio, with_f0=False)
        assert feats[3].is_quiet
        assert not feats[0].is_quiet
        assert feats[3].rms_db < feats[0].rms_db

    def test_uniform_loudness_yields_no_quiet_segments(self):
        """Rank alone always fires: the bottom quartile of a flat recording is
        still a quartile. Without a real dB drop, P1 fills with segments that
        are not quiet and the category becomes noise."""
        rec = toy_recording([(i * 2, i * 2 + 2, chr(65 + i), "x") for i in range(8)],
                            duration=20)
        audio = np.full(SR * 20, 0.3, dtype=np.float32)
        feats = features.segment_features(rec, audio, with_f0=False)
        assert not any(f.is_quiet for f in feats)
        assert any(f.energy_percentile <= 0.25 for f in feats)   # rank did fire

    def test_masked_beats_quiet_when_overlapped(self):
        rec = toy_recording([(0, 4, "A", "a"), (2, 6, "B", "b")], duration=10)
        audio = np.full(SR * 10, 0.2, dtype=np.float32)
        feats = features.segment_features(rec, audio, with_f0=False)
        assert feats[0].is_masked
        assert not feats[0].is_quiet     # masking and quietness are exclusive

    def test_unmeasured_f0_is_not_flat(self):
        """0.0 means 'not measured'; treating it as flat would invent P3 items."""
        rec = toy_recording([(0, 2, "A", "a")], duration=5)
        feats = features.segment_features(rec, np.zeros(SR * 5, dtype=np.float32),
                                          with_f0=False)
        assert feats[0].f0_range_semitones == 0.0
        assert not feats[0].is_flat


class TestMarkers:
    def test_repair_and_hesitation_markers(self):
        assert features.has_marker("no wait, twelve", "en", features.REPAIR_MARKERS)
        assert features.has_marker("um, I think so", "en", features.HESITATION_MARKERS)
        assert not features.has_marker("twelve milligrams", "en", features.REPAIR_MARKERS)

    def test_markers_exist_for_all_three_languages(self):
        for table in (features.REPAIR_MARKERS, features.HESITATION_MARKERS):
            assert set(table) == {"en", "hi", "bn"}
            assert all(table[lang] for lang in table)


# ---------------------------------------------------------------- build

def proposed(rec: Recording, audio: np.ndarray, band: int = 300, with_f0: bool = False):
    feats = features.segment_features(rec, audio, with_f0=with_f0)
    return build.propose(rec, feats, band), feats


class TestProposal:
    def test_needs_three_distinct_values_to_make_an_item(self):
        """Two values cannot fill correct + salience + recency."""
        rec = toy_recording([(0, 4, "A", "twenty milligrams"),
                             (10, 14, "B", "thirty milligrams")])
        cands, _ = proposed(rec, np.full(SR * 300, 0.2, dtype=np.float32))
        assert cands == []

    def test_distractors_are_real_competing_mentions(self):
        rec = toy_recording([
            (0, 4, "A", "the dose is fifty milligrams"),
            (10, 14, "A", "again, fifty milligrams, everyone"),
            (100, 104, "B", "actually we used five milligrams"),
            (200, 204, "C", "fifteen milligrams was the other option"),
        ])
        audio = np.full(SR * 300, 0.3, dtype=np.float32)
        audio[100 * SR:104 * SR] = 0.01          # the quiet one
        cands, _ = proposed(rec, audio)
        assert cands, "expected at least one candidate"
        for c in cands:
            assert c.salience.value != c.target.value
            assert c.recency.value != c.target.value
            assert c.salience.value != c.recency.value
            # every option came out of this recording, none was fabricated
            for m in (c.target, c.salience, c.recency):
                assert 0 <= m.start <= rec.duration

    def test_salience_prefers_the_loud_repeated_value(self):
        rec = toy_recording([
            (0, 4, "A", "the agreed dose is fifty milligrams per day"),
            (10, 14, "A", "again, fifty milligrams for the main arm"),
            (100, 104, "B", "we used five milligrams in the trial arm"),
            (200, 204, "C", "fifteen milligrams was the other option"),
        ])
        audio = np.full(SR * 300, 0.3, dtype=np.float32)
        audio[100 * SR:104 * SR] = 0.01
        cands, _ = proposed(rec, audio)
        quiet = [c for c in cands if c.target.value == 5.0]
        assert quiet and quiet[0].salience.value == 50.0

    def test_recency_is_the_latest_competing_mention(self):
        rec = toy_recording([
            (0, 4, "A", "the agreed dose is fifty milligrams per day"),
            (10, 14, "A", "again, fifty milligrams for the main arm"),
            (100, 104, "B", "we used five milligrams in the trial arm"),
            (250, 254, "C", "fifteen milligrams was the other option"),
        ])
        audio = np.full(SR * 300, 0.3, dtype=np.float32)
        audio[100 * SR:104 * SR] = 0.01
        cands, _ = proposed(rec, audio)
        quiet = [c for c in cands if c.target.value == 5.0]
        assert quiet and quiet[0].recency.value == 15.0

    def test_self_repair_is_categorised_p4_over_its_acoustics(self):
        rec = toy_recording([
            (0, 4, "A", "the starting dose is twenty milligrams"),
            (4, 8, "A", "make it twelve milligrams for the trial"),  # same speaker, 4 s later
            (100, 104, "B", "fifty milligrams in the other arm"),
            (200, 204, "C", "fifteen milligrams was also discussed"),
        ])
        cands, _ = proposed(rec, np.full(SR * 300, 0.2, dtype=np.float32))
        repairs = [c for c in cands if c.category == "P4"]
        assert repairs
        assert any(c.target.value == 12.0 for c in repairs)
        assert all("repaired_from" in c.why for c in repairs)

    def test_lexically_marked_repairs_are_flagged_for_the_leak_filter(self):
        rec = toy_recording([
            (0, 4, "A", "the starting dose is twenty milligrams"),
            (4, 8, "A", "sorry, twelve milligrams for the trial"),
            (100, 104, "B", "fifty milligrams in the other arm"),
            (200, 204, "C", "fifteen milligrams was also discussed"),
        ])
        cands, _ = proposed(rec, np.full(SR * 300, 0.2, dtype=np.float32))
        marked = [c for c in cands if c.category == "P4" and c.why["lexically_marked"]]
        assert marked, "a text model will solve these; they must be flagged not hidden"

    def test_overlapped_mention_is_p2(self):
        rec = toy_recording([
            (0, 10, "A", "we used five milligrams in the trial arm"),
            (2, 12, "B", "talking over the top of that"),
            (100, 104, "C", "fifty milligrams in the other arm"),
            (200, 204, "D", "fifteen milligrams was also discussed"),
        ])
        cands, _ = proposed(rec, np.full(SR * 300, 0.2, dtype=np.float32))
        assert any(c.category == "P2" and c.target.value == 5.0 for c in cands)

    def test_hesitation_is_c1(self):
        rec = toy_recording([
            (0, 4, "A", "um, maybe five milligrams for that arm, not sure"),
            (100, 104, "B", "fifty milligrams in the other arm"),
            (200, 204, "C", "fifteen milligrams was also discussed"),
        ])
        cands, _ = proposed(rec, np.full(SR * 300, 0.2, dtype=np.float32))
        assert any(c.category == "C1" for c in cands)


class TestItemAssembly:
    def _one(self):
        rec = toy_recording([
            (0, 4, "A", "the dose is fifty milligrams"),
            (10, 14, "A", "fifty milligrams again"),
            (100, 104, "B", "we used five milligrams in the trial arm"),
            (200, 204, "C", "fifteen milligrams"),
        ])
        audio = np.full(SR * 300, 0.3, dtype=np.float32)
        audio[100 * SR:104 * SR] = 0.01
        cands, _ = proposed(rec, audio)
        target = next(c for c in cands if c.target.value == 5.0)
        return rec, build.to_item(rec, target, 300, 0)

    def test_item_validates_and_keys_to_the_target(self):
        _, item = self._one()
        assert item.options["correct"].startswith("five")
        assert set(item.options) == set(ROLES)
        assert item.correct_role == "correct"

    def test_needle_span_brackets_the_mention_and_fits_the_band(self):
        rec, item = self._one()
        assert 0.0 <= item.needle_start < item.needle_end <= item.duration_band
        assert item.needle_end - item.needle_start >= 2.0
        assert 99.0 <= item.needle_mid <= 106.0

    def test_question_does_not_leak_the_prominence(self):
        """Naming the manipulation would hand the model the answer."""
        _, item = self._one()
        lowered = item.question.lower()
        for word in ("quiet", "quietly", "muttered", "overlap", "background",
                     "aside", "loud", "stressed"):
            assert word not in lowered

    def test_question_does_not_contain_the_answer(self):
        _, item = self._one()
        assert "five milligrams" not in item.question

    def test_question_does_not_name_its_own_distractors(self):
        """The first real harvest produced "options from titanium, rubber,
        plastic" with the answer "wood" - answerable by elimination, with the
        loud competitor handed over in the prompt."""
        _, item = self._one()
        lowered = item.question.casefold()
        for role in ("correct", "salience", "recency"):
            assert item.options[role].casefold() not in lowered, role


    def test_item_is_marked_unverified(self):
        """Nothing is a real item until a human has listened to it."""
        _, item = self._one()
        assert item.provenance["verified"] is False
        assert item.provenance["leak_checked"] is False

    def test_provenance_records_where_the_distractors_came_from(self):
        _, item = self._one()
        assert item.provenance["salience_at"] == pytest.approx(0.0, abs=15.0)
        assert item.provenance["recency_at"] > 190.0
        assert item.provenance["quantity_kind"] == "mass"


class TestContextGuard:
    def test_context_naming_an_option_is_rejected(self):
        from undertone.harvest.build import context_is_clean

        target = mentions.find_mentions("we picked wood for the case", "en", 0, 0, 3)
        salience = mentions.find_mentions("rubber is cheaper though", "en", 1, 3, 6)
        assert target and salience
        assert not context_is_clean("options from titanium , rubber , plastic",
                                    target[0], salience[0])
        assert context_is_clean("for the outer case of the unit",
                                target[0], salience[0])

    def test_a_context_too_short_to_locate_anything_is_rejected(self):
        from undertone.harvest.build import context_is_clean

        m = mentions.find_mentions("five milligrams", "en", 0, 0, 2)
        assert not context_is_clean("so", *m)
        assert not context_is_clean("", *m)


class TestP3Ordering:
    def test_p3_requires_a_repeated_competitor_not_just_flatness(self):
        """Checking flatness last produced zero P3 on the first real harvest:
        AMI is ~50% overlapped, so every flat aside was claimed as P2 first."""
        import inspect

        from undertone.harvest import build

        source = inspect.getsource(build._categorize)
        p3 = source.index('"P3"')
        p2 = source.index('"P2"')
        assert p3 < p2, "P3 must be tested before the other acoustic causes"
        assert "loudest_repeat" in source
        assert "competitor_repeats" in source


# ---------------------------------------------------------------- leak filter

def solver_always(letter: str):
    return lambda prompt: letter


def solver_reads_key(items_by_prompt_marker: str):
    """Solves an item iff its correct option text appears in the transcript."""
    def solve(prompt: str) -> str | None:
        transcript = prompt.split("TRANSCRIPT:", 1)[1].split("\n\n", 1)[0]
        for line in prompt.splitlines():
            if len(line) > 3 and line[1:3] == ". " and line[3:] in transcript:
                return line[0]
        return None
    return solve


class TestLeakFilter:
    def _items(self):
        rec = toy_recording([
            (0, 4, "A", "the dose is fifty milligrams"),
            (10, 14, "A", "fifty milligrams again"),
            (100, 104, "B", "we used five milligrams in the trial arm"),
            (200, 204, "C", "fifteen milligrams"),
        ])
        audio = np.full(SR * 300, 0.3, dtype=np.float32)
        audio[100 * SR:104 * SR] = 0.01
        cands, _ = proposed(rec, audio)
        return rec, [build.to_item(rec, c, 300, i) for i, c in enumerate(cands)]

    def test_gate_assignment_matches_the_claim_each_category_makes(self):
        assert leakfilter.GATE == {"P1": "asr", "P2": "asr", "P3": "asr",
                                   "P4": "gold", "C1": "gold"}

    def test_asr_gated_item_survives_a_gold_only_leak(self):
        """P1 is a claim about cascaded systems, not about perfect transcripts."""
        rec, items = self._items()
        item = next(i for i in items
                    if i.category == "P1" and i.options["correct"].startswith("five"))
        report = leakfilter.run_filter(
            [item],
            gold_transcripts={rec.recording_id: " ".join(s.text for s in rec.segments)},
            asr_transcripts={rec.recording_id: "the dose is fifty milligrams"},
            solvers=[solver_reads_key("")],
        )
        assert report.per_item[item.item_id]["gold"] is True      # reported...
        assert report.per_item[item.item_id]["asr"] is False      # ...but not gating
        assert item.item_id not in report.rejected

    def test_asr_gated_item_is_rejected_when_asr_solves_it(self):
        rec, items = self._items()
        item = next(i for i in items
                    if i.category == "P1" and i.options["correct"].startswith("five"))
        full = " ".join(s.text for s in rec.segments)
        report = leakfilter.run_filter(
            [item], {rec.recording_id: full}, {rec.recording_id: full},
            solvers=[solver_reads_key("")])
        assert item.item_id in report.rejected

    def test_majority_vote_ignores_a_single_lucky_solver(self):
        rec, items = self._items()
        item = items[0]
        full = " ".join(s.text for s in rec.segments)
        # One solver always says "A", two never answer: below majority.
        report = leakfilter.run_filter(
            [item], {rec.recording_id: full}, {rec.recording_id: full},
            solvers=[solver_always("A"), lambda p: None, lambda p: None])
        assert not report.per_item[item.item_id]["gold"]

    def test_apply_filter_records_verdicts_and_drops_rejects(self):
        rec, items = self._items()
        full = " ".join(s.text for s in rec.segments)
        report = leakfilter.run_filter(
            items, {rec.recording_id: full}, {rec.recording_id: full},
            solvers=[solver_reads_key("")])
        kept = leakfilter.apply_filter(items, report)
        assert len(kept) == len(items) - len(report.rejected)
        assert all(i.provenance["leak_checked"] for i in kept)
        assert all(i.item_id not in report.rejected for i in kept)

    def test_report_table_shows_both_tiers(self):
        rec, items = self._items()
        full = " ".join(s.text for s in rec.segments)
        report = leakfilter.run_filter(
            items, {rec.recording_id: full}, {rec.recording_id: "nothing useful"},
            solvers=[solver_reads_key("")])
        rows = report.table(items)
        assert rows
        for row in rows:
            assert {"category", "n", "gate", "gold_leak", "asr_leak", "rejected"} <= set(row)

    def test_options_are_shuffled_the_same_way_as_the_audio_run(self):
        """A text model must not face a different option order to the real run."""
        from undertone.protocol import assign_letters

        _, items = self._items()
        item = items[0]
        _, letter_to_role = leakfilter.render_text_prompt(item, "transcript", seed=7)
        assert {v: k for k, v in letter_to_role.items()} == assign_letters(item, 7)

    def test_no_solver_is_an_error_not_a_silent_pass(self):
        _, items = self._items()
        with pytest.raises(ValueError, match="at least one text solver"):
            leakfilter.run_filter(items, {}, {}, solvers=[])


# ---------------------------------------------------------------- yodas

class TestYodasSegments:
    def test_dict_of_lists_shape(self):
        """YODAS2 revisions differ; guessing one shape fails silently as 'no speech'."""
        from undertone.harvest.sources import _yodas_segments

        segs = _yodas_segments({"utterances": {
            "start": [0.0, 5.0], "end": [4.0, 9.0], "text": ["hello", "world"]}})
        assert [s.text for s in segs] == ["hello", "world"]
        assert segs[0].start == 0.0 and segs[0].end == 4.0

    def test_list_of_dicts_shape(self):
        from undertone.harvest.sources import _yodas_segments

        segs = _yodas_segments({"utterances": [
            {"start": 1.0, "end": 3.0, "text": "one"},
            {"begin_time": 4.0, "end_time": 6.0, "text": "two"}]})
        assert len(segs) == 2

    def test_empty_and_malformed_rows_are_dropped(self):
        from undertone.harvest.sources import _yodas_segments

        assert _yodas_segments({}) == []
        assert _yodas_segments({"utterances": {
            "start": [0.0, 5.0, 9.0], "end": [4.0, 4.0, 12.0],
            "text": ["ok", "zero length", "   "]}}) == [
            __import__("undertone.harvest.sources", fromlist=["Segment"])
            .Segment(0.0, 4.0, "spk", "ok")]

    def test_single_speaker_source_cannot_express_p2(self):
        """A source with no diarisation must not report an empty P2 cell as
        'no items found' -- it could never have produced one."""
        from undertone.harvest.sources import available_categories

        rec = toy_recording([(0, 4, "spk", "a"), (10, 14, "spk", "b")])
        assert "P2" not in available_categories(rec)
        multi = toy_recording([(0, 4, "A", "a"), (10, 14, "B", "b")])
        assert "P2" in available_categories(multi)

    def test_configs_cover_the_three_languages(self):
        from undertone.harvest.sources import YODAS2_CONFIGS

        assert set(YODAS2_CONFIGS) == {"en", "hi", "bn"}
        assert all(YODAS2_CONFIGS[lang] for lang in YODAS2_CONFIGS)


class TestWindowing:
    def test_a_long_meeting_yields_several_haystacks(self):
        """Taking only a prefix threw away three quarters of every meeting:
        60 meetings at a 10-minute prefix produced fewer items than 25 did at
        30 minutes."""
        segs = [(i * 10.0, i * 10.0 + 8.0, "A" if i % 2 else "B", f"turn {i} of it")
                for i in range(240)]                       # 40 minutes of speech
        rec = toy_recording(segs, duration=2400.0)
        windows = rec.windows(600)
        assert len(windows) == 4
        assert [w.meta["window_index"] for w in windows] == [0, 1, 2, 3]

    def test_windows_do_not_overlap(self):
        segs = [(i * 10.0, i * 10.0 + 8.0, "A", f"turn {i} here") for i in range(240)]
        rec = toy_recording(segs, duration=2400.0)
        seen = set()
        for w in rec.windows(600):
            for s in w.segments:
                absolute = round(s.start + w.meta["window_start"], 2)
                assert absolute not in seen, "a needle would appear in two haystacks"
                seen.add(absolute)

    def test_segment_times_are_relative_to_the_window(self):
        segs = [(i * 10.0, i * 10.0 + 8.0, "A", f"turn {i} here") for i in range(240)]
        w = toy_recording(segs, duration=2400.0).windows(600)[2]
        assert all(0 <= s.start < 600 for s in w.segments)
        assert w.meta["window_start"] == 1200.0

    def test_a_sparse_window_is_dropped(self):
        rec = toy_recording([(0, 4, "A", "only one turn")], duration=1200.0)
        assert rec.windows(600) == []

    def test_a_short_recording_yields_one_window_or_none(self):
        segs = [(i * 10.0, i * 10.0 + 8.0, "A", f"turn {i} here") for i in range(50)]
        rec = toy_recording(segs, duration=600.0)
        assert len(rec.windows(600)) == 1
        assert rec.windows(1200) == []


class TestWindowRequiresRealDuration:
    def test_a_zero_duration_recording_yields_nothing(self):
        """The failure mode that produced '0 x 600s windows' for 60 meetings:
        the Recording carried duration 0.0, so the loop condition was never
        true and every window was silently skipped."""
        segs = [(i * 10.0, i * 10.0 + 8.0, "A", f"turn {i} here") for i in range(240)]
        rec = toy_recording(segs, duration=0.0)
        assert rec.windows(600) == []

    def test_setting_the_real_duration_recovers_them(self):
        segs = [(i * 10.0, i * 10.0 + 8.0, "A", f"turn {i} here") for i in range(240)]
        rec = toy_recording(segs, duration=0.0)
        rec.duration = 2400.0
        assert len(rec.windows(600)) == 4


# ---------------------------------------------------------------- P3-constructed

class TestConstructedArm:
    def _audio(self, seconds=30.0, sr=16000):
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.2, int(seconds * sr)).astype(np.float32)

    def test_the_contrast_is_actually_created(self):
        """P3 is defined by a prominence contrast. On AMI it yields ~10 items
        and its trap rate came out lowest of five categories - the paper plan
        names this as its top risk and prescribes exactly this arm."""
        from undertone.harvest import construct

        audio = self._audio()
        before = construct.measure_contrast(audio, (10.0, 11.0), (20.0, 21.0))
        edited, edits = construct.construct_p3(audio, 10.0, 11.0, [(20.0, 21.0)])
        after = construct.measure_contrast(edited, (10.0, 11.0), (20.0, 21.0))
        assert after - before > 8.0, (before, after)
        assert len(edits) == 2

    def test_nothing_is_cut_or_inserted(self):
        """Only relative loudness changes - every word stays where it was."""
        from undertone.harvest import construct

        audio = self._audio()
        edited, _ = construct.construct_p3(audio, 10.0, 11.0, [(20.0, 21.0)])
        assert len(edited) == len(audio)
        untouched = slice(int(25 * 16000), int(28 * 16000))
        assert np.allclose(edited[untouched], audio[untouched])

    def test_edits_are_recorded_for_audit(self):
        from undertone.harvest import construct

        _, edits = construct.construct_p3(self._audio(), 10.0, 11.0,
                                          [(20.0, 21.0), (24.0, 25.0)])
        assert [e["gain_db"] for e in edits] == [
            construct.TARGET_ATTENUATION_DB,
            construct.COMPETITOR_BOOST_DB,
            construct.COMPETITOR_BOOST_DB,
        ]
        assert all({"start", "end", "gain_db"} == set(e) for e in edits)

    def test_the_attenuated_span_is_still_audible(self):
        """Beyond roughly 12 dB the aside stops being recoverable and the item
        becomes a perception test rather than a retrieval one."""
        from undertone.harvest import construct

        assert construct.TARGET_ATTENUATION_DB > -12.0

    def test_gain_is_ramped_not_stepped(self):
        from undertone.harvest import construct

        audio = np.ones(16000 * 4, dtype=np.float32)
        edited = construct.apply_gain_edits(
            audio, [construct.GainEdit(1.0, 3.0, -9.0)])
        ramp = edited[16000:16000 + int(construct.RAMP_SECONDS * 16000)]
        assert ramp[0] > ramp[-1], "gain should ease in, not step"
        assert len(set(np.round(ramp, 3))) > 3, "a step would click"

    def test_output_never_clips(self):
        from undertone.harvest import construct

        loud = np.full(16000, 0.9, dtype=np.float32)
        out = construct.apply_gain_edits(loud, [construct.GainEdit(0.0, 1.0, 12.0)])
        assert float(np.max(np.abs(out))) <= 1.0
