"""
Task 1: Acoustic Needle-in-Haystack (ANiH)

Tests long-range acoustic retrieval without text cheating.
Injects a short non-verbal acoustic artifact into long background audio.
"""

import random
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..base import BaseTask, TaskInstance, AudioSegment, generate_instance_id


@dataclass
class ANIHConfig:
    """Configuration for ANiH task generation."""
    background_duration: float = 3600.0  # 1 hour default
    needle_duration: float = 0.5  # 500ms artifact
    needle_types: List[str] = None  # e.g., ["glass_shatter", "pen_click", "phone_notification"]
    insertion_depths: List[float] = None  # seconds from start
    snr_db: float = 10.0  # signal-to-noise ratio for injection
    background_sources: List[str] = None  # ["library", "office", "lecture"]
    
    def __post_init__(self):
        if self.needle_types is None:
            self.needle_types = ["glass_shatter", "pen_click", "phone_notification", "door_slam", "keyboard_click"]
        if self.insertion_depths is None:
            # Default: random positions
            self.insertion_depths = []
        if self.background_sources is None:
            self.background_sources = ["library", "office", "lecture"]


class ANIHTask(BaseTask):
    """Acoustic Needle-in-Haystack task implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = ANIHConfig().__dict__
        if config:
            default_config.update(config)
        super().__init__("anih", default_config)
        self.config_obj = ANIHConfig(**default_config)
    
    def get_prompt_template(self) -> str:
        return (
            "Listen carefully to the entire audio recording. "
            "At what exact timestamp (in MM:SS format) did the {needle_type} occur, "
            "and what environmental sound immediately preceded it (within 5 seconds)? "
            "Provide your answer as: 'Timestamp: MM:SS, Preceding sound: <description>'"
        )
    
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model response for timestamp and preceding sound."""
        result = {"timestamp": None, "preceding_sound": None, "raw_response": response}
        
        # Try to extract timestamp in MM:SS format
        import re
        timestamp_match = re.search(r'(\d{1,2}:\d{2}(?:\.\d+)?)', response)
        if timestamp_match:
            result["timestamp"] = timestamp_match.group(1)
        
        # Try to extract preceding sound. Bounded to stop at a newline as
        # well as a period - otherwise a multi-line response gets the rest
        # of the message folded into "preceding_sound" (same class of bug
        # as speaker_drift's first-appearance regex).
        preceding_match = re.search(r'[Pp]receding\s+sound[:\s]+([^.\n]+)', response)
        if preceding_match:
            result["preceding_sound"] = preceding_match.group(1).strip()
        else:
            # Fallback: look for "sound:" or "was"
            sound_match = re.search(r'(?:sound|was)[:\s]+([^.\n]+)', response, re.IGNORECASE)
            if sound_match:
                result["preceding_sound"] = sound_match.group(1).strip()
        
        return result
    
    def evaluate(self, 
                 prediction: Dict[str, Any], 
                 ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate ANiH prediction."""
        metrics = {}
        
        # Timestamp accuracy (edit distance in seconds)
        pred_ts = prediction.get("timestamp")
        true_ts = ground_truth.get("needle_timestamp")
        
        if pred_ts and true_ts:
            pred_seconds = self._timestamp_to_seconds(pred_ts)
            true_seconds = self._timestamp_to_seconds(true_ts)
            metrics["timestamp_error_seconds"] = abs(pred_seconds - true_seconds)
            metrics["timestamp_hit_1s"] = 1.0 if abs(pred_seconds - true_seconds) <= 1.0 else 0.0
            metrics["timestamp_hit_5s"] = 1.0 if abs(pred_seconds - true_seconds) <= 5.0 else 0.0
        else:
            metrics["timestamp_error_seconds"] = float('inf')
            metrics["timestamp_hit_1s"] = 0.0
            metrics["timestamp_hit_5s"] = 0.0
        
        # Preceding sound match (semantic similarity - simplified to keyword overlap)
        # `.get(key, "")` only falls back when the key is missing - but
        # parse_model_response() explicitly stores None there when nothing
        # matched, so `or ""` is required to actually catch that case.
        pred_sound = (prediction.get("preceding_sound") or "").lower()
        true_sound = (ground_truth.get("preceding_sound") or "").lower()
        
        if pred_sound and true_sound:
            pred_words = set(pred_sound.split())
            true_words = set(true_sound.split())
            overlap = len(pred_words & true_words)
            total = len(pred_words | true_words)
            metrics["preceding_sound_iou"] = overlap / total if total > 0 else 0.0
        else:
            metrics["preceding_sound_iou"] = 0.0
        
        # Composite score
        metrics["composite_score"] = (
            0.7 * metrics["timestamp_hit_5s"] + 
            0.3 * metrics["preceding_sound_iou"]
        )
        
        return metrics
    
    def _timestamp_to_seconds(self, ts: str) -> float:
        """Convert MM:SS or MM:SS.mmm to seconds."""
        parts = ts.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    def generate_instances(self, 
                          source_data: Dict[str, Any], 
                          num_instances: int,
                          split: str = "test") -> List[TaskInstance]:
        """
        Generate ANiH instances.
        
        source_data should contain:
        - background_audio_paths: Dict[str, List[str]] mapping source_type -> list of audio file paths
        - needle_audio_paths: Dict[str, str] mapping needle_type -> audio file path
        - metadata about each audio file (duration, sample_rate, etc.)
        """
        instances = []
        
        background_paths = source_data.get("background_audio_paths", {})
        needle_paths = source_data.get("needle_audio_paths", {})
        
        for i in range(num_instances):
            # Select background
            bg_source = random.choice(self.config_obj.background_sources)
            bg_path = random.choice(background_paths.get(bg_source, []))
            
            # Select needle
            needle_type = random.choice(self.config_obj.needle_types)
            needle_path = needle_paths.get(needle_type)
            
            if not bg_path or not needle_path:
                continue
            
            # Select insertion depth
            if self.config_obj.insertion_depths:
                insertion_time = random.choice(self.config_obj.insertion_depths)
            else:
                # Random position, avoiding first/last 30 seconds
                max_pos = self.config_obj.background_duration - 30
                insertion_time = random.uniform(30, max_pos)
            
            # Create instance
            instance_id = generate_instance_id("anih", f"{bg_source}_{needle_type}_{i}")
            
            # In practice, you'd actually mix the audio here
            # For now, we create the metadata for the mixed audio
            mixed_audio_path = f"data/anih/{instance_id}.wav"
            
            ground_truth = {
                "needle_type": needle_type,
                "needle_timestamp": self._seconds_to_timestamp(insertion_time),
                "needle_timestamp_seconds": insertion_time,
                "preceding_sound": self._get_preceding_sound_label(bg_source, insertion_time),
                "background_source": bg_source,
                "snr_db": self.config_obj.snr_db,
            }
            
            prompt = self.get_prompt_template().format(needle_type=needle_type.replace("_", " "))
            
            instance = TaskInstance(
                instance_id=instance_id,
                task_name="anih",
                audio_path=mixed_audio_path,
                duration=self.config_obj.background_duration,
                ground_truth=ground_truth,
                prompt=prompt,
                metadata={
                    "background_source": bg_source,
                    "needle_type": needle_type,
                    "insertion_time": insertion_time,
                    "bg_audio_path": bg_path,
                    "needle_audio_path": needle_path,
                }
            )
            
            instances.append(instance)
        
        return instances
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _get_preceding_sound_label(self, bg_source: str, insertion_time: float) -> str:
        """Get the expected preceding sound label for a background source at a given time.
        
        In practice, this would analyze the actual audio. For generation, we return
        a template based on the source type.
        """
        labels = {
            "library": "pages turning",
            "office": "keyboard typing",
            "lecture": "professor speaking",
        }
        return labels.get(bg_source, "ambient noise")


def create_anih_config(**kwargs) -> Dict[str, Any]:
    """Factory function to create ANiH config."""
    config = ANIHConfig(**kwargs)
    return config.__dict__