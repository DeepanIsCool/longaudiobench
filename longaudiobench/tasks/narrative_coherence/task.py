"""
Task 6: Acoustic Narrative Coherence (The Missing Link)

Tests complex multi-hop reasoning across audio timelines.
Requires cross-referencing acoustic token from deep past with semantic token from middle past.
"""

import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..base import BaseTask, TaskInstance, generate_instance_id


@dataclass
class NarrativeCoherenceConfig:
    """Configuration for Acoustic Narrative Coherence task."""
    duration: float = 2400.0  # 40 minutes
    story_sources: List[str] = None  # ["librivox", "spotify_podcasts", "gigaspeech"]
    clue_types: List[str] = None
    reasoning_depth: int = 3  # number of hops required
    
    def __post_init__(self):
        if self.story_sources is None:
            self.story_sources = ["librivox", "spotify_podcasts", "gigaspeech"]
        if self.clue_types is None:
            self.clue_types = [
                "clock_chime", "distinctive_bell", "unique_bird_call", "train_whistle",
                "church_bell", "factory_whistle", "ice_cream_truck", "emergency_alert"
            ]


class NarrativeCoherenceTask(BaseTask):
    """Acoustic Narrative Coherence task implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = NarrativeCoherenceConfig().__dict__
        if config:
            default_config.update(config)
        super().__init__("narrative_coherence", default_config)
        self.config_obj = NarrativeCoherenceConfig(**default_config)
    
    def get_prompt_template(self) -> str:
        return (
            "Listen to this {duration_minutes}-minute audio story/podcast carefully. "
            "At the beginning, a distinctive acoustic event occurs (Clue A). "
            "In the middle, a character references this event as a temporal marker (Clue B). "
            "Near the end, a suspect provides an alibi that may contradict the acoustic evidence (Clue C). "
            "Question: Does the acoustic evidence from the first half of the recording validate "
            "or debunk the suspect's alibi? Walk through the logical chain step-by-step, "
            "citing specific timestamps for each clue. "
            "Answer format: 'Verdict: Validates/Debunks/Inconclusive. Reasoning: Step 1: ... Step 2: ... Step 3: ...'"
        )
    
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model's reasoning response."""
        result = {
            "verdict": None,
            "reasoning_steps": [],
            "cited_timestamps": [],
            "raw_response": response
        }
        
        import re
        
        # Extract verdict
        verdict_match = re.search(r'[Vv]erdict[:\s]+(Validates|Debunks|Inconclusive)', response, re.IGNORECASE)
        if verdict_match:
            result["verdict"] = verdict_match.group(1).capitalize()
        
        # Extract reasoning steps - split by "Step N:" pattern
        # First find all step markers
        step_positions = [(m.start(), m.group()) for m in re.finditer(r'[Ss]tep\s+\d+[:\s]+', response)]
        
        if step_positions:
            for i, (pos, marker) in enumerate(step_positions):
                start = pos + len(marker)
                end = step_positions[i + 1][0] if i + 1 < len(step_positions) else len(response)
                step_text = response[start:end].strip().rstrip('.')
                if step_text:
                    result["reasoning_steps"].append(step_text)
        
        # Extract timestamps
        timestamps = re.findall(r'\d{1,2}:\d{2}(?:\.\d+)?', response)
        result["cited_timestamps"] = timestamps
        
        return result
    
    def evaluate(self, 
                 prediction: Dict[str, Any], 
                 ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate narrative coherence prediction."""
        metrics = {}
        
        # Verdict accuracy
        pred_verdict = prediction.get("verdict")
        true_verdict = ground_truth.get("correct_verdict")
        
        metrics["verdict_accuracy"] = 1.0 if pred_verdict == true_verdict else 0.0
        
        # Reasoning quality - check if key logical steps are present
        pred_steps = prediction.get("reasoning_steps", [])
        true_steps = ground_truth.get("required_reasoning_steps", [])
        
        step_scores = []
        for true_step in true_steps:
            # Check if any predicted step semantically matches
            matched = False
            for pred_step in pred_steps:
                if self._step_match(pred_step, true_step):
                    matched = True
                    break
            step_scores.append(1.0 if matched else 0.0)
        
        metrics["reasoning_completeness"] = sum(step_scores) / len(step_scores) if step_scores else 0.0
        
        # Timestamp citation accuracy
        pred_timestamps = set(prediction.get("cited_timestamps", []))
        true_timestamps = set(ground_truth.get("key_timestamps", []))
        
        if true_timestamps:
            metrics["timestamp_citation_precision"] = len(pred_timestamps & true_timestamps) / len(pred_timestamps) if pred_timestamps else 0.0
            metrics["timestamp_citation_recall"] = len(pred_timestamps & true_timestamps) / len(true_timestamps)
            metrics["timestamp_citation_f1"] = (
                2 * metrics["timestamp_citation_precision"] * metrics["timestamp_citation_recall"] /
                (metrics["timestamp_citation_precision"] + metrics["timestamp_citation_recall"])
                if (metrics["timestamp_citation_precision"] + metrics["timestamp_citation_recall"]) > 0 else 0.0
            )
        else:
            metrics["timestamp_citation_precision"] = 0.0
            metrics["timestamp_citation_recall"] = 0.0
            metrics["timestamp_citation_f1"] = 0.0
        
        # Composite score
        metrics["composite_score"] = (
            0.4 * metrics["verdict_accuracy"] + 
            0.4 * metrics["reasoning_completeness"] + 
            0.2 * metrics["timestamp_citation_f1"]
        )
        
        return metrics
    
    def _step_match(self, pred_step: str, true_step: str) -> bool:
        """Simple keyword overlap matching for reasoning steps."""
        pred_words = set(pred_step.lower().split())
        true_words = set(true_step.lower().split())
        # Remove stop words - only common English stop words, NOT domain-specific terms
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "was", "at", "in", "on", "to", "for", "of", "with", "by"}
        pred_words -= stop_words
        true_words -= stop_words
        
        if not true_words:
            return False
        
        overlap = len(pred_words & true_words)
        return overlap / len(true_words) >= 0.3  # 30% keyword overlap threshold
    
    def generate_instances(self, 
                          source_data: Dict[str, Any], 
                          num_instances: int,
                          split: str = "test") -> List[TaskInstance]:
        """
        Generate Narrative Coherence instances.
        
        source_data should contain:
        - story_audio_paths: List[Dict] with "path", "duration", "transcript", "metadata"
        - acoustic_clue_paths: Dict[str, str] mapping clue_type -> audio file path
        """
        instances = []
        
        stories = source_data.get("story_audio_paths", [])
        clues = source_data.get("acoustic_clue_paths", {})
        
        for i in range(num_instances):
            if not stories:
                break
            
            story = random.choice(stories)
            # Use the configured duration, not the source clip's own (much
            # shorter) metadata duration - the actual mixed audio this
            # instance points to is padded/generated to config_obj.duration
            # by create_narrative_coherence_audio, and every timestamp below
            # must stay inside that actual audio or the ground truth becomes
            # unanswerable no matter how good the model is.
            story_duration = self.config_obj.duration

            # Select clue type
            clue_type = random.choice(self.config_obj.clue_types)
            clue_path = clues.get(clue_type)

            if not clue_path:
                continue

            # Define the three key timestamps as fractions of story_duration
            # (not fixed absolute seconds) so the ordering (early -> middle ->
            # near end) holds regardless of the configured duration.
            clue_a_time = random.uniform(0.05, 0.15) * story_duration
            clue_b_time = random.uniform(0.35, 0.55) * story_duration
            clue_c_time = random.uniform(0.80, 0.95) * story_duration
            
            # Determine ground truth verdict
            # In practice, this would be derived from the actual story content
            # For generation, we create a consistent scenario
            verdict = random.choice(["Validates", "Debunks"])
            
            instance_id = generate_instance_id("narrative_coherence", f"{clue_type}_{i}")
            mixed_audio_path = f"data/narrative_coherence/{instance_id}.wav"
            
            # Required reasoning steps for this scenario
            required_steps = [
                f"Identify the {clue_type.replace('_', ' ')} at {self._seconds_to_timestamp(clue_a_time)}",
                f"Note the character's reference to the {clue_type.replace('_', ' ')} at {self._seconds_to_timestamp(clue_b_time)}",
                f"Compare suspect's alibi at {self._seconds_to_timestamp(clue_c_time)} with acoustic evidence",
            ]
            
            ground_truth = {
                "correct_verdict": verdict,
                "clue_type": clue_type,
                "clue_a_timestamp": self._seconds_to_timestamp(clue_a_time),
                "clue_a_timestamp_seconds": clue_a_time,
                "clue_b_timestamp": self._seconds_to_timestamp(clue_b_time),
                "clue_b_timestamp_seconds": clue_b_time,
                "clue_c_timestamp": self._seconds_to_timestamp(clue_c_time),
                "clue_c_timestamp_seconds": clue_c_time,
                "key_timestamps": [
                    self._seconds_to_timestamp(clue_a_time),
                    self._seconds_to_timestamp(clue_b_time),
                    self._seconds_to_timestamp(clue_c_time),
                ],
                "required_reasoning_steps": required_steps,
                "story_metadata": story,
            }
            
            prompt = self.get_prompt_template().format(
                duration_minutes=int(story_duration / 60)
            )
            
            instance = TaskInstance(
                instance_id=instance_id,
                task_name="narrative_coherence",
                audio_path=mixed_audio_path,
                duration=story_duration,
                ground_truth=ground_truth,
                prompt=prompt,
                metadata={
                    "story_path": story.get("path"),
                    "clue_type": clue_type,
                    "clue_audio_path": clue_path,
                    "clue_a_time": clue_a_time,
                    "clue_b_time": clue_b_time,
                    "clue_c_time": clue_c_time,
                    "verdict": verdict,
                }
            )
            
            instances.append(instance)
        
        return instances
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


def create_narrative_coherence_config(**kwargs) -> Dict[str, Any]:
    """Factory function to create Narrative Coherence config."""
    config = NarrativeCoherenceConfig(**kwargs)
    return config.__dict__