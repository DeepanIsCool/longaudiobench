"""
Task 3: Speaker Diarization Drift

Tests long-term tracking of speaker identities over massive timelines.
Speaker speaks briefly at start, goes silent for 40+ minutes, returns at end.
"""

import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..base import BaseTask, TaskInstance, generate_instance_id


@dataclass
class SpeakerDriftConfig:
    """Configuration for Speaker Diarization Drift task."""
    total_duration: float = 3600.0  # 60 minutes
    num_speakers: int = 5
    target_speaker_id: int = 2  # 0-indexed, the one who disappears and returns
    initial_speech_duration: float = 120.0  # first 2 minutes
    silence_duration: float = 2400.0  # 40 minutes of silence
    final_speech_time: float = 2595.0  # 43:15 mark
    final_speech_duration: float = 10.0  # single crucial sentence
    speaker_similarity: str = "high"  # "high" = similar registers/accents
    dataset_sources: List[str] = None  # ["voxceleb2", "ami", "librispeech"]
    
    def __post_init__(self):
        if self.dataset_sources is None:
            self.dataset_sources = ["voxceleb2", "ami", "librispeech"]


class SpeakerDriftTask(BaseTask):
    """Speaker Diarization Drift task implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = SpeakerDriftConfig().__dict__
        if config:
            default_config.update(config)
        super().__init__("speaker_drift", default_config)
        self.config_obj = SpeakerDriftConfig(**default_config)
    
    def get_prompt_template(self) -> str:
        return (
            "This is a {duration_minutes}-minute panel discussion with {num_speakers} speakers "
            "who have very similar voices. Listen to the entire recording carefully. "
            "At timestamp {target_timestamp}, a speaker says a single sentence. "
            "Identify which speaker this is (Speaker 1-{num_speakers}). "
            "Has this speaker spoken previously in the audio? If yes, provide the timestamp "
            "of their first appearance in MM:SS format. "
            "Answer format: 'Speaker: X, First appearance: MM:SS (or \"Never\")'"
        )
    
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model response for speaker ID and first appearance."""
        result = {
            "predicted_speaker": None,
            "first_appearance": None,
            "raw_response": response
        }
        
        import re
        
        # Extract speaker number
        speaker_match = re.search(r'[Ss]peaker[:\s]+(\d+)', response)
        if speaker_match:
            result["predicted_speaker"] = int(speaker_match.group(1))
        
        # Extract first appearance
        appearance_match = re.search(r'[Ff]irst\s+appearance[:\s]+([^\.\n]+)', response)
        if appearance_match:
            appearance_text = appearance_match.group(1).strip()
            if appearance_text.lower() not in ["never", "none", "n/a"]:
                result["first_appearance"] = appearance_text
        
        return result
    
    def evaluate(self, 
                 prediction: Dict[str, Any], 
                 ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate Speaker Drift prediction."""
        metrics = {}
        
        # Speaker identification accuracy
        pred_speaker = prediction.get("predicted_speaker")
        true_speaker = ground_truth.get("target_speaker_id", 0) + 1  # 1-indexed for display
        
        metrics["speaker_accuracy"] = 1.0 if pred_speaker == true_speaker else 0.0
        
        # First appearance detection
        pred_appearance = prediction.get("first_appearance")
        true_appearance = ground_truth.get("first_appearance_timestamp")
        
        if pred_appearance and true_appearance:
            try:
                pred_seconds = self._timestamp_to_seconds(pred_appearance)
                true_seconds = self._timestamp_to_seconds(true_appearance)
                metrics["appearance_error_seconds"] = abs(pred_seconds - true_seconds)
                metrics["appearance_hit_5s"] = 1.0 if abs(pred_seconds - true_seconds) <= 5.0 else 0.0
            except (ValueError, IndexError):
                # Free-text model output doesn't always come back as a clean
                # MM:SS timestamp - treat anything unparseable as a miss
                # rather than crashing the whole evaluation run.
                metrics["appearance_error_seconds"] = float('inf')
                metrics["appearance_hit_5s"] = 0.0
        elif pred_appearance is None and true_appearance is None:
            metrics["appearance_error_seconds"] = 0.0
            metrics["appearance_hit_5s"] = 1.0
        else:
            metrics["appearance_error_seconds"] = float('inf')
            metrics["appearance_hit_5s"] = 0.0
        
        # Joint score (both correct)
        metrics["joint_accuracy"] = (
            1.0 if (metrics["speaker_accuracy"] == 1.0 and metrics["appearance_hit_5s"] == 1.0) else 0.0
        )
        
        return metrics
    
    def _timestamp_to_seconds(self, ts: str) -> float:
        """Convert MM:SS to seconds."""
        parts = ts.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def generate_instances(self, 
                          source_data: Dict[str, Any], 
                          num_instances: int,
                          split: str = "test") -> List[TaskInstance]:
        """
        Generate Speaker Drift instances.
        
        source_data should contain:
        - speaker_audio_paths: Dict[str, List[Dict]] mapping dataset -> list of 
          {"speaker_id": str, "audio_path": str, "duration": float, "gender": str, "accent": str}
        """
        instances = []
        
        speaker_data = source_data.get("speaker_audio_paths", {})
        
        for i in range(num_instances):
            # Select dataset source
            dataset = random.choice(self.config_obj.dataset_sources)
            available_speakers = speaker_data.get(dataset, [])
            
            if len(available_speakers) < self.config_obj.num_speakers:
                continue
            
            # Select speakers with similar characteristics (same gender, similar accent)
            # For simplicity, we just pick random speakers; in practice, filter by similarity
            selected_speakers = random.sample(available_speakers, self.config_obj.num_speakers)
            
            # Designate target speaker
            target_idx = self.config_obj.target_speaker_id
            target_speaker = selected_speakers[target_idx]
            
            # Create instance metadata
            instance_id = generate_instance_id("speaker_drift", f"{dataset}_{i}")
            
            # Build the timeline
            timeline = self._build_timeline(selected_speakers, target_idx)
            
            mixed_audio_path = f"data/speaker_drift/{instance_id}.wav"
            
            ground_truth = {
                "target_speaker_id": target_idx + 1,  # 1-indexed
                "target_speaker_metadata": target_speaker,
                "first_appearance_timestamp": self._seconds_to_timestamp(
                    self.config_obj.initial_speech_duration / 2  # middle of first speech
                ),
                "target_return_timestamp": self._seconds_to_timestamp(
                    self.config_obj.final_speech_time
                ),
                "num_speakers": self.config_obj.num_speakers,
                "total_duration": self.config_obj.total_duration,
                "speaker_order": [s["speaker_id"] for s in selected_speakers],
            }
            
            prompt = self.get_prompt_template().format(
                duration_minutes=int(self.config_obj.total_duration / 60),
                num_speakers=self.config_obj.num_speakers,
                target_timestamp=self._seconds_to_timestamp(self.config_obj.final_speech_time)
            )
            
            instance = TaskInstance(
                instance_id=instance_id,
                task_name="speaker_drift",
                audio_path=mixed_audio_path,
                duration=self.config_obj.total_duration,
                ground_truth=ground_truth,
                prompt=prompt,
                metadata={
                    "dataset": dataset,
                    "selected_speakers": selected_speakers,
                    "target_speaker_idx": target_idx,
                    "timeline": timeline,
                }
            )
            
            instances.append(instance)
        
        return instances
    
    def _build_timeline(self, speakers: List[Dict], target_idx: int) -> List[Dict]:
        """Build the speaking timeline for the panel discussion."""
        timeline = []
        
        # First 2 minutes: all speakers participate
        current_time = 0.0
        segment_duration = self.config_obj.initial_speech_duration / self.config_obj.num_speakers
        
        for idx, speaker in enumerate(speakers):
            timeline.append({
                "speaker_idx": idx,
                "speaker_id": speaker["speaker_id"],
                "start_time": current_time,
                "end_time": current_time + segment_duration,
                "type": "initial"
            })
            current_time += segment_duration
        
        # 40 minutes: only non-target speakers
        non_target = [i for i in range(self.config_obj.num_speakers) if i != target_idx]
        remaining_time = self.config_obj.final_speech_time - current_time
        
        # Distribute among non-target speakers
        turns = max(10, int(remaining_time / 60))  # ~1 turn per minute
        turn_duration = remaining_time / turns
        
        for turn in range(turns):
            speaker_idx = random.choice(non_target)
            timeline.append({
                "speaker_idx": speaker_idx,
                "speaker_id": speakers[speaker_idx]["speaker_id"],
                "start_time": current_time,
                "end_time": current_time + turn_duration,
                "type": "middle"
            })
            current_time += turn_duration
        
        # Final sentence from target speaker
        timeline.append({
            "speaker_idx": target_idx,
            "speaker_id": speakers[target_idx]["speaker_id"],
            "start_time": self.config_obj.final_speech_time,
            "end_time": self.config_obj.final_speech_time + self.config_obj.final_speech_duration,
            "type": "target_return"
        })
        
        return timeline


def create_speaker_drift_config(**kwargs) -> Dict[str, Any]:
    """Factory function to create Speaker Drift config."""
    config = SpeakerDriftConfig(**kwargs)
    return config.__dict__