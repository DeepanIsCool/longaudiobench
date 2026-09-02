"""
Tests for LongAudioBench core functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from longaudiobench.tasks import get_task, list_tasks
from longaudiobench.tasks.base import TaskInstance, generate_instance_id
from longaudiobench.tasks.anih.task import ANIHTask, create_anih_config
from longaudiobench.tasks.speaker_drift.task import SpeakerDriftTask
from longaudiobench.tasks.soundscape.task import SoundscapeTask
from longaudiobench.tasks.narrative_coherence.task import NarrativeCoherenceTask
from longaudiobench.metrics import compute_task_metrics, bootstrap_confidence_interval
from longaudiobench.data.audio_mixing import mix_audio_at_timestamp
import numpy as np


class TestBase:
    def test_task_registry(self):
        tasks = list_tasks()
        assert "anih" in tasks
        assert "speaker_drift" in tasks
        assert "soundscape" in tasks
        assert "narrative_coherence" in tasks
        assert len(tasks) == 4
    
    def test_get_task(self):
        task = get_task("anih")
        assert isinstance(task, ANIHTask)
        assert task.task_name == "anih"
    
    def test_generate_instance_id(self):
        id1 = generate_instance_id("anih")
        id2 = generate_instance_id("anih")
        assert id1 != id2
        assert id1.startswith("anih_")
    
    def test_task_instance_creation(self):
        inst = TaskInstance(
            instance_id="test_001",
            task_name="anih",
            audio_path="/path/to/audio.wav",
            duration=3600.0,
            ground_truth={"needle_timestamp": "12:34"},
            prompt="Test prompt",
        )
        assert inst.instance_id == "test_001"
        assert inst.task_name == "anih"
        d = inst.to_dict()
        assert d["instance_id"] == "test_001"


class TestANIH:
    def test_config_creation(self):
        config = create_anih_config(background_duration=1800)
        assert config["background_duration"] == 1800
    
    def test_prompt_template(self):
        task = ANIHTask()
        prompt = task.get_prompt_template()
        assert "needle_type" in prompt
        assert "timestamp" in prompt.lower()
    
    def test_parse_response(self):
        task = ANIHTask()
        response = "Timestamp: 12:34, Preceding sound: pages turning"
        parsed = task.parse_model_response(response)
        assert parsed["timestamp"] == "12:34"
        assert "pages turning" in parsed["preceding_sound"]
    
    def test_evaluate(self):
        task = ANIHTask()
        pred = {"timestamp": "12:34", "preceding_sound": "pages turning"}
        gt = {"needle_timestamp": "12:34", "preceding_sound": "pages turning"}
        metrics = task.evaluate(pred, gt)
        assert metrics["timestamp_hit_5s"] == 1.0
        assert metrics["preceding_sound_iou"] == 1.0
        assert metrics["composite_score"] == 1.0
    
    def test_evaluate_with_error(self):
        task = ANIHTask()
        pred = {"timestamp": "13:00", "preceding_sound": "pages turning"}
        gt = {"needle_timestamp": "12:34", "preceding_sound": "pages turning"}
        metrics = task.evaluate(pred, gt)
        assert metrics["timestamp_error_seconds"] == 26.0
        assert metrics["timestamp_hit_5s"] == 0.0


class TestSpeakerDrift:
    def test_prompt_template(self):
        task = SpeakerDriftTask()
        prompt = task.get_prompt_template()
        assert "speaker" in prompt.lower()
        assert "first appearance" in prompt.lower()
    
    def test_parse_response(self):
        task = SpeakerDriftTask()
        response = "Speaker: 3, First appearance: 01:30"
        parsed = task.parse_model_response(response)
        assert parsed["predicted_speaker"] == 3
        assert parsed["first_appearance"] == "01:30"
    
    def test_evaluate(self):
        task = SpeakerDriftTask()
        pred = {"predicted_speaker": 3, "first_appearance": "01:30"}
        gt = {"target_speaker_id": 2, "first_appearance_timestamp": "01:30"}  # 0-indexed internally, 1-indexed for display
        metrics = task.evaluate(pred, gt)
        assert metrics["speaker_accuracy"] == 1.0
        assert metrics["appearance_hit_5s"] == 1.0
        assert metrics["joint_accuracy"] == 1.0


class TestSoundscape:
    def test_prompt_template(self):
        task = SoundscapeTask()
        prompt = task.get_prompt_template()
        assert "timeline" in prompt.lower()
        assert "event" in prompt.lower()
    
    def test_evaluate_matching(self):
        task = SoundscapeTask()
        pred = {"events": [
            {"event_type": "rain_start", "start": "00:00", "end": "05:00", "overlaps": ""},
            {"event_type": "wind_howling", "start": "02:00", "end": "10:00", "overlaps": "rain_start"},
        ]}
        gt = {"events": [
            {"event_type": "rain_start", "start": "00:00", "end": "05:00", "overlaps": ""},
            {"event_type": "wind_howling", "start": "02:00", "end": "10:00", "overlaps": "rain_start"},
        ]}
        metrics = task.evaluate(pred, gt)
        assert metrics["event_f1"] == 1.0
        assert metrics["timeline_iou"] == 1.0
        assert metrics["ordering_accuracy"] == 1.0


class TestNarrativeCoherence:
    def test_prompt_template(self):
        task = NarrativeCoherenceTask()
        prompt = task.get_prompt_template()
        assert "clue" in prompt.lower()
        assert "verdict" in prompt.lower()
    
    def test_parse_response(self):
        task = NarrativeCoherenceTask()
        response = "Verdict: Debunks. Step 1: Clock chimes at 05:00. Step 2: Character mentions at 20:00. Step 3: Alibi contradicts."
        parsed = task.parse_model_response(response)
        assert parsed["verdict"] == "Debunks"
        assert len(parsed["reasoning_steps"]) == 3
        assert "05:00" in parsed["cited_timestamps"]
    
    def test_evaluate(self):
        task = NarrativeCoherenceTask()
        pred = {
            "verdict": "Debunks",
            "reasoning_steps": ["Identify clock chime at 05:00", "Character references clock chime at 20:00", "Alibi contradicts clock chime evidence"],
            "cited_timestamps": ["05:00", "20:00", "35:00"],
        }
        gt = {
            "correct_verdict": "Debunks",
            "required_reasoning_steps": ["Identify clock chime", "Note character reference", "Compare alibi"],
            "key_timestamps": ["05:00", "20:00", "35:00"],
        }
        metrics = task.evaluate(pred, gt)
        assert metrics["verdict_accuracy"] == 1.0
        assert metrics["reasoning_completeness"] == 1.0
        assert metrics["timestamp_citation_f1"] == 1.0


class TestMetrics:
    def test_bootstrap_ci(self):
        values = [0.5, 0.6, 0.55, 0.58, 0.62, 0.48, 0.52, 0.59, 0.61, 0.57]
        lower, upper = bootstrap_confidence_interval(values, n_bootstrap=1000)
        assert 0.4 <= lower <= 0.6
        assert 0.5 <= upper <= 0.7
        assert lower < upper
    
    def test_compute_task_metrics(self):
        from longaudiobench.tasks.anih.task import ANIHTask
        task = ANIHTask()
        
        predictions = [
            {"timestamp": "12:34", "preceding_sound": "pages turning"},
            {"timestamp": "13:00", "preceding_sound": "keyboard typing"},
        ]
        ground_truths = [
            {"needle_timestamp": "12:34", "preceding_sound": "pages turning"},
            {"needle_timestamp": "12:58", "preceding_sound": "keyboard typing"},
        ]
        
        metrics = compute_task_metrics("anih", predictions, ground_truths, task.evaluate)
        assert "timestamp_hit_5s" in metrics
        assert "mean" in metrics["timestamp_hit_5s"]


class TestAudioMixing:
    def test_mix_audio_at_timestamp(self):
        sr = 16000
        bg = np.random.randn(sr * 10) * 0.1  # 10 seconds background
        fg = np.random.randn(sr * 1) * 0.5   # 1 second foreground
        
        mixed = mix_audio_at_timestamp(bg, fg, 5.0, sr, snr_db=10.0)
        assert len(mixed) == len(bg)
        
        # Check that foreground was added around 5 seconds
        start = int(5.0 * sr)
        end = start + len(fg)
        assert np.max(np.abs(mixed[start:end])) > np.max(np.abs(bg[start:end]))
    
    def test_mix_handles_short_background(self):
        sr = 16000
        bg = np.random.randn(sr * 2) * 0.1  # 2 seconds
        fg = np.random.randn(sr * 1) * 0.5  # 1 second
        
        mixed = mix_audio_at_timestamp(bg, fg, 1.5, sr, snr_db=10.0)
        assert len(mixed) >= int(2.5 * sr)  # Extended to fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])