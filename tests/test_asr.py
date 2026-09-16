"""ASR transcripts: the cascaded control and the audio-necessity evidence.

No faster-whisper needed -- everything tested here is the logic that turns a
transcript into a claim.
"""

from __future__ import annotations

import json

import pytest

from undertone.harvest import asr
from undertone.items import MCQItem


def transcript(segments, recording_id="r0", lang="en"):
    return asr.Transcript(
        recording_id=recording_id, model="test", lang=lang,
        text=" ".join(t for _, _, t in segments),
        segments=[asr.ASRSegment(s, e, t) for s, e, t in segments])


def item(correct="five milligrams", start=100.0, end=104.0, category="P1"):
    return MCQItem(
        item_id="it_1", recording_id="r0", lang="en", category=category,
        sector="meetings", audio_path="a.flac", duration_band=300,
        needle_start=start, needle_end=end, question="q",
        options={"correct": correct, "salience": "fifty milligrams",
                 "recency": "fifteen milligrams", "absent": "x"})


class TestWindow:
    def test_returns_only_overlapping_segments(self):
        t = transcript([(0, 10, "start"), (98, 106, "we used five milligrams"),
                        (200, 210, "end")])
        assert t.between(100, 104, pad=0) == "we used five milligrams"

    def test_pad_catches_a_needle_on_a_segment_boundary(self):
        """ASR boundaries drift; without slack a split needle looks missing."""
        t = transcript([(90, 99.5, "we used five"), (104.5, 112, "milligrams daily")])
        assert t.between(100, 104, pad=0) == ""
        assert "five" in t.between(100, 104, pad=2.0)

    def test_empty_window(self):
        assert transcript([(0, 5, "hello")]).between(100, 104) == ""


class TestRecovered:
    def test_exact_words_recovered(self):
        t = transcript([(98, 106, "we used five milligrams in the trial")])
        assert asr.recovered(t, item())

    def test_punctuation_and_case_do_not_matter(self):
        t = transcript([(98, 106, "We used FIVE, Milligrams.")])
        assert asr.recovered(t, item())

    def test_not_recovered_when_asr_missed_the_mutter(self):
        """This is the audio-necessity finding, not a bug."""
        t = transcript([(98, 106, "we used [inaudible] in the trial")])
        assert not asr.recovered(t, item())

    def test_answer_elsewhere_in_the_recording_does_not_count(self):
        """The competing loud mention is not the needle."""
        t = transcript([(10, 20, "five milligrams was mentioned early"),
                        (98, 106, "we used something in the trial")])
        assert not asr.recovered(t, item())

    def test_empty_transcript_is_not_recovery(self):
        assert not asr.recovered(transcript([]), item())


class TestNeedleRecovery:
    def test_rate_is_reported_per_category(self):
        transcripts = {
            "r0": transcript([(98, 106, "we used five milligrams")]),
            "r1": transcript([(98, 106, "we used mumble"), ], recording_id="r1"),
        }
        items = [item(), item(category="P2")]
        items[1].__dict__["recording_id"] = "r1"
        rows = {r["category"]: r for r in asr.needle_recovery(transcripts, items)}
        assert rows["P1"]["recovery_rate"] == 1.0
        assert rows["P2"]["recovery_rate"] == 0.0
        assert rows["P2"]["unrecoverable_rate"] == 1.0

    def test_categories_with_no_transcript_are_skipped_not_zeroed(self):
        rows = asr.needle_recovery({}, [item()])
        assert rows == []


class TestTranscriptRoundTrip:
    def test_json_round_trip_preserves_segments(self):
        t = transcript([(0, 4, "a"), (4, 8, "b")])
        back = asr.Transcript.from_dict(json.loads(json.dumps(t.to_dict())))
        assert back.recording_id == t.recording_id
        assert [s.text for s in back.segments] == ["a", "b"]

    def test_as_text_shape_matches_the_leak_filter(self):
        out = asr.as_text({"r0": transcript([(0, 4, "hello there")])})
        assert out == {"r0": "hello there"}


class TestCascadedControl:
    def test_it_is_a_control_not_one_of_the_thirteen(self):
        from undertone import adapters

        assert "cascaded_whisper_llm" not in adapters.list_adapters()
        assert "cascaded_whisper_llm" in adapters.list_adapters(include_controls=True)
        assert "cascaded_whisper_llm" in adapters.list_controls()
        assert len(adapters.list_adapters()) == 13


class TestTextTwins:
    """Three more text models behind the identical Whisper front end."""

    TWINS = ("cascaded_whisper_llama31_8b", "cascaded_whisper_mistral_7b",
             "cascaded_whisper_gemma2_9b")

    def test_twins_are_controls_not_models(self):
        from undertone import adapters

        for key in self.TWINS:
            assert key not in adapters.list_adapters()
            assert key in adapters.list_controls()
        assert len(adapters.list_adapters()) == 13

    def test_twins_share_everything_but_the_text_model(self):
        """If anything but ``text_model`` differed, a gap between twins would
        be partly a harness difference."""
        from undertone.adapters.base import _REGISTRY
        from undertone.adapters.cascaded import CascadedWhisperLLM

        base = _REGISTRY["cascaded_whisper_llm"]
        seen = {base.text_model}
        for key in self.TWINS:
            cls = _REGISTRY[key]
            assert issubclass(cls, CascadedWhisperLLM)
            assert cls.asr_model == base.asr_model
            assert cls.primary == base.primary
            assert cls.build_inputs is base.build_inputs
            assert cls.transcribe_window is base.transcribe_window
            assert cls.text_model not in seen, "two twins with one text model"
            seen.add(cls.text_model)
            assert cls.text_model in cls.model_id

    def test_transcript_cache_is_keyed_on_audio_and_asr_not_text_model(self, tmp_path, monkeypatch):
        """The whole point of the cache: twin #2 must reuse twin #1's
        transcripts. A key that included the text model would defeat it."""
        import numpy as np

        from undertone.adapters import cascaded

        monkeypatch.setenv(cascaded.ASR_CACHE_ENV, str(tmp_path))
        calls = []

        class FakeSeg:
            text = " hello "

        class FakeEngine:
            def transcribe(self, path, **kw):
                calls.append(path)
                return [FakeSeg()], None

        monkeypatch.setattr("undertone.harvest.asr._engine", lambda m: FakeEngine())
        audio = np.zeros(16000, dtype=np.float32)

        a = cascaded.CascadedWhisperLLM.__new__(cascaded.CascadedWhisperLLM)
        a.lang = "en"
        b = cascaded.CascadedWhisperMistral.__new__(cascaded.CascadedWhisperMistral)
        b.lang = "en"
        assert a.transcribe_window(audio) == "hello"
        assert b.transcribe_window(audio) == "hello"
        assert len(calls) == 1, "second twin re-ran ASR instead of hitting the cache"
        assert len(list(tmp_path.glob("*.txt"))) == 1

    def test_cache_off_by_default(self, monkeypatch):
        import numpy as np

        from undertone.adapters import cascaded

        monkeypatch.delenv(cascaded.ASR_CACHE_ENV, raising=False)
        calls = []

        class FakeSeg:
            text = "x"

        class FakeEngine:
            def transcribe(self, path, **kw):
                calls.append(path)
                return [FakeSeg()], None

        monkeypatch.setattr("undertone.harvest.asr._engine", lambda m: FakeEngine())
        a = cascaded.CascadedWhisperLLM.__new__(cascaded.CascadedWhisperLLM)
        a.lang = "en"
        a.transcribe_window(np.zeros(16000, dtype=np.float32))
        a.transcribe_window(np.zeros(16000, dtype=np.float32))
        assert len(calls) == 2

    def test_its_ceiling_does_not_constrain_the_grid(self):
        """Whisper chunks internally; the limit is the text model's context."""
        from undertone.adapters.base import _REGISTRY

        assert _REGISTRY["cascaded_whisper_llm"].max_audio_s >= 1800

    def test_prompts_exist_for_all_three_languages(self):
        from undertone.adapters.cascaded import PROMPT

        assert set(PROMPT) == {"en", "hi", "bn"}
        for template in PROMPT.values():
            assert "{transcript}" in template and "{question}" in template

    def test_vad_is_off(self):
        """VAD filtering would drop exactly the quiet needles P1 is about."""
        import inspect

        from undertone.adapters import cascaded

        source = inspect.getsource(cascaded.CascadedWhisperLLM.transcribe_window)
        assert "vad_filter=False" in source


class TestAudioSensitivityCheck:
    def test_smoke_suite_verifies_the_model_hears_the_audio(self):
        """An adapter that builds a valid prompt but never attaches the audio
        loads, discriminates and parses -- and measures a text prior."""
        import inspect

        from undertone import smoke

        source = inspect.getsource(smoke.smoke_adapter)
        assert "hears_audio" in source
        assert "score_letters" in source

    def test_the_check_is_part_of_the_reported_failures(self):
        from undertone import smoke

        assert "hears_audio" in inspect_source(smoke)


def inspect_source(module):
    import inspect

    return inspect.getsource(module)


class TestGenerationHead:
    def test_af_next_prefers_the_conditional_generation_class(self):
        """AutoModel returns the base MusicFlamingoModel, which loads and scores
        logits fine and then dies on .generate() at inference."""
        import inspect

        from undertone.adapters import audio_flamingo

        # Check the search tuple itself, not the prose above it - the comment
        # explaining the bug also contains the word "AutoModel".
        import re

        src = inspect.getsource(audio_flamingo.AudioFlamingoNext.load)
        start = src.index("for name in (")
        tuple_src = src[start:src.index("):", start)]
        names = re.findall(r'"([A-Za-z]+)"', tuple_src)
        assert names[0] == "MusicFlamingoForConditionalGeneration", names
        assert names[-1] == "AutoModel", names

    def test_a_model_without_generate_is_rejected_at_load(self):
        import inspect

        from undertone.adapters import audio_flamingo

        src = inspect.getsource(audio_flamingo.AudioFlamingoNext.load)
        assert 'hasattr(model, "generate")' in src
