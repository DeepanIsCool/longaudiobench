"""
Task 4: Environmental Soundscape Timeline Reconstruction

Tests non-speech auditory scene analysis and temporal ordering.
Pure environmental audio - no speech.
"""

import random
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..base import BaseTask, TaskInstance, generate_instance_id


@dataclass
class SoundscapeConfig:
    """Configuration for Environmental Soundscape Timeline task."""
    duration: float = 1800.0  # 30 minutes
    num_events: int = 8  # number of distinct environmental changes
    event_types: List[str] = None
    overlap_allowed: bool = True
    min_event_duration: float = 10.0
    max_event_duration: float = 120.0
    dataset_sources: List[str] = None
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = [
                "rain_start", "rain_stop", "rain_heavy", "rain_light",
                "wind_start", "wind_stop", "wind_howling",
                "thunder", "siren_approaching", "siren_passing", "siren_receding",
                "dog_barking", "bird_chorus", "traffic_increase", "traffic_decrease",
                "footsteps", "door_open", "door_close", "engine_start", "engine_stop",
                "crowd_cheer", "crowd_applause", "music_start", "music_stop"
            ]
        if self.dataset_sources is None:
            self.dataset_sources = ["audioset", "fsd50k", "dcase2024", "tut_acoustic_scenes"]


class SoundscapeTask(BaseTask):
    """Environmental Soundscape Timeline Reconstruction task."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = SoundscapeConfig().__dict__
        if config:
            default_config.update(config)
        super().__init__("soundscape", default_config)
        self.config_obj = SoundscapeConfig(**default_config)
    
    def get_prompt_template(self) -> str:
        return (
            "Listen to this {duration_minutes}-minute environmental recording. "
            "Construct a strict chronological timeline of all weather and environmental changes. "
            "For each change, specify: (1) the event type, (2) start timestamp (MM:SS), "
            "(3) end timestamp (MM:SS) or 'ongoing', (4) any overlapping sounds. "
            "Format as a numbered list: '1. Event: rain_start, Start: 02:15, End: 15:30, Overlaps: wind_howling (from 05:00)'"
        )
    
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model's timeline response."""
        result = {"events": [], "raw_response": response}
        
        import re
        
        # Look for numbered list items
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match pattern: "1. Event: X, Start: MM:SS, End: MM:SS, Overlaps: ..."
            event_match = re.search(
                r'Event[:\s]+([^,]+),\s*Start[:\s]+([^,]+),\s*End[:\s]+([^,]+)(?:,\s*Overlaps[:\s]+(.*))?',
                line, re.IGNORECASE
            )
            if event_match:
                result["events"].append({
                    "event_type": event_match.group(1).strip(),
                    "start": event_match.group(2).strip(),
                    "end": event_match.group(3).strip(),
                    "overlaps": event_match.group(4).strip() if event_match.group(4) else "",
                })
            else:
                # Try simpler pattern
                simple_match = re.search(
                    r'(\d+)[\.\:\)]\s*([^,]+)[,\;]\s*([^,]+)[,\;]\s*([^,]+)',
                    line
                )
                if simple_match:
                    result["events"].append({
                        "event_type": simple_match.group(2).strip(),
                        "start": simple_match.group(3).strip(),
                        "end": simple_match.group(4).strip(),
                        "overlaps": "",
                    })
        
        return result
    
    def evaluate(self, 
                 prediction: Dict[str, Any], 
                 ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate soundscape timeline prediction using IoU and event matching."""
        metrics = {}
        
        pred_events = prediction.get("events", [])
        true_events = ground_truth.get("events", [])
        
        if not true_events:
            metrics["event_f1"] = 0.0
            metrics["timeline_iou"] = 0.0
            metrics["ordering_accuracy"] = 0.0
            return metrics
        
        # Match predicted events to ground truth events
        matched_pairs = self._match_events(pred_events, true_events)
        
        # Event-level F1
        tp = len(matched_pairs)
        fp = len(pred_events) - tp
        fn = len(true_events) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["event_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Timeline IoU for matched events
        ious = []
        for pred_e, true_e in matched_pairs:
            iou = self._compute_temporal_iou(pred_e, true_e)
            ious.append(iou)
        metrics["timeline_iou"] = sum(ious) / len(ious) if ious else 0.0
        
        # Temporal ordering accuracy (for matched events in correct sequence)
        pred_order = [p["event_type"] for p, _ in matched_pairs]
        true_order = [t["event_type"] for _, t in matched_pairs]
        # Check longest common subsequence
        lcs_len = self._lcs_length(pred_order, true_order)
        metrics["ordering_accuracy"] = lcs_len / len(true_order) if true_order else 0.0
        
        # Overlap detection accuracy
        overlap_scores = []
        for pred_e, true_e in matched_pairs:
            pred_overlaps = set(pred_e.get("overlaps", "").split(", ")) if pred_e.get("overlaps") else set()
            true_overlaps = set(true_e.get("overlaps", "").split(", ")) if true_e.get("overlaps") else set()
            if true_overlaps:
                overlap_scores.append(len(pred_overlaps & true_overlaps) / len(true_overlaps))
            else:
                overlap_scores.append(1.0 if not pred_overlaps else 0.0)
        metrics["overlap_detection_f1"] = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        
        # Composite
        metrics["composite_score"] = (
            0.3 * metrics["event_f1"] + 
            0.3 * metrics["timeline_iou"] + 
            0.2 * metrics["ordering_accuracy"] + 
            0.2 * metrics["overlap_detection_f1"]
        )
        
        return metrics
    
    def _match_events(self, pred_events: List[Dict], true_events: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match predicted events to ground truth using greedy IoU matching."""
        matches = []
        used_true = set()
        
        for pred_e in pred_events:
            best_iou = 0.0
            best_idx = -1
            pred_start = self._timestamp_to_seconds(pred_e.get("start", "0:00"))
            pred_end_str = pred_e.get("end", "0:00")
            pred_end = self._timestamp_to_seconds(pred_end_str) if pred_end_str.lower() != "ongoing" else pred_start + 10
            
            for idx, true_e in enumerate(true_events):
                if idx in used_true:
                    continue
                # Type must match (fuzzy)
                if pred_e["event_type"].lower() != true_e["event_type"].lower():
                    continue
                true_start = self._timestamp_to_seconds(true_e["start"])
                true_end_str = true_e["end"]
                true_end = self._timestamp_to_seconds(true_end_str) if true_end_str.lower() != "ongoing" else true_start + 10
                
                iou = self._compute_temporal_iou(
                    {"start": pred_start, "end": pred_end},
                    {"start": true_start, "end": true_end}
                )
                if iou > best_iou and iou > 0.3:  # threshold
                    best_iou = iou
                    best_idx = idx
            
            if best_idx >= 0:
                matches.append((pred_e, true_events[best_idx]))
                used_true.add(best_idx)
        
        return matches
    
    def _compute_temporal_iou(self, e1: Dict, e2: Dict) -> float:
        """Compute IoU of two temporal segments."""
        # Handle both dict with string timestamps and dict with numeric timestamps
        if isinstance(e1.get("start"), str):
            start1 = self._timestamp_to_seconds(e1["start"])
            end1_str = e1["end"]
            end1 = self._timestamp_to_seconds(end1_str) if end1_str.lower() != "ongoing" else start1 + 10
        else:
            start1 = e1["start"]
            end1 = e1["end"]
        
        if isinstance(e2.get("start"), str):
            start2 = self._timestamp_to_seconds(e2["start"])
            end2_str = e2["end"]
            end2 = self._timestamp_to_seconds(end2_str) if end2_str.lower() != "ongoing" else start2 + 10
        else:
            start2 = e2["start"]
            end2 = e2["end"]
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        
        return intersection / union if union > 0 else 0.0
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Longest Common Subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _timestamp_to_seconds(self, ts: str) -> float:
        """Convert MM:SS to seconds."""
        try:
            parts = ts.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except:
            return 0.0
    
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
        Generate Soundscape instances.
        
        source_data should contain:
        - event_audio_paths: Dict[str, List[str]] mapping event_type -> list of audio file paths
        - background_audio_paths: List[str] of ambient background tracks
        """
        instances = []
        
        event_audio = source_data.get("event_audio_paths", {})
        background_audio = source_data.get("background_audio_paths", [])
        
        for i in range(num_instances):
            # Select background
            bg_path = random.choice(background_audio) if background_audio else None
            
            # Generate event timeline
            events = self._generate_event_timeline()
            
            instance_id = generate_instance_id("soundscape", f"{i}")
            mixed_audio_path = f"data/soundscape/{instance_id}.wav"
            
            ground_truth = {
                "events": events,
                "duration": self.config_obj.duration,
                "num_events": len(events),
            }
            
            prompt = self.get_prompt_template().format(
                duration_minutes=int(self.config_obj.duration / 60)
            )
            
            instance = TaskInstance(
                instance_id=instance_id,
                task_name="soundscape",
                audio_path=mixed_audio_path,
                duration=self.config_obj.duration,
                ground_truth=ground_truth,
                prompt=prompt,
                metadata={
                    "bg_audio_path": bg_path,
                    "events": events,
                }
            )
            
            instances.append(instance)
        
        return instances
    
    def _generate_event_timeline(self) -> List[Dict]:
        """Generate a random but plausible event timeline."""
        events = []
        current_time = 0.0
        available_events = self.config_obj.event_types.copy()
        
        # Ensure some weather events
        weather_events = [e for e in available_events if e.startswith(("rain", "wind", "thunder"))]
        other_events = [e for e in available_events if not e.startswith(("rain", "wind", "thunder"))]
        
        # Start with a weather condition
        if weather_events:
            first_event = random.choice(weather_events)
            duration = random.uniform(self.config_obj.min_event_duration, self.config_obj.max_event_duration)
            events.append({
                "event_type": first_event,
                "start": self._seconds_to_timestamp(current_time),
                "end": self._seconds_to_timestamp(current_time + duration),
                "overlaps": "",
            })
            current_time += duration * 0.3  # partial overlap for next
            available_events.remove(first_event)
        
        # Add remaining events
        num_events = random.randint(self.config_obj.num_events - 2, self.config_obj.num_events + 2)
        
        for _ in range(num_events - 1):
            if not available_events:
                break
            
            event_type = random.choice(available_events)
            duration = random.uniform(self.config_obj.min_event_duration, self.config_obj.max_event_duration)
            
            # Determine overlaps
            overlaps = []
            if self.config_obj.overlap_allowed and events and random.random() < 0.4:
                # Overlap with a recent event
                recent_event = random.choice(events[-3:])
                overlaps.append(f"{recent_event['event_type']} (from {recent_event['start']})")
            
            end_time = min(current_time + duration, self.config_obj.duration)
            
            events.append({
                "event_type": event_type,
                "start": self._seconds_to_timestamp(current_time),
                "end": self._seconds_to_timestamp(end_time) if end_time < self.config_obj.duration else "ongoing",
                "overlaps": ", ".join(overlaps),
            })
            
            current_time = end_time
            available_events.remove(event_type)
            
            if current_time >= self.config_obj.duration:
                break
        
        # Sort by start time
        events.sort(key=lambda e: self._timestamp_to_seconds(e["start"]))
        
        return events


def create_soundscape_config(**kwargs) -> Dict[str, Any]:
    """Factory function to create Soundscape config."""
    config = SoundscapeConfig(**kwargs)
    return config.__dict__