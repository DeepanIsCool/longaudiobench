"""
Base classes and interfaces for LongAudioBench tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    path: str
    start_time: float  # seconds
    end_time: float    # seconds
    sample_rate: int = 16000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInstance:
    """A single instance of a benchmark task."""
    instance_id: str
    task_name: str
    audio_path: str
    duration: float  # seconds
    ground_truth: Dict[str, Any]
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "task_name": self.task_name,
            "audio_path": self.audio_path,
            "duration": self.duration,
            "ground_truth": self.ground_truth,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


@dataclass
class ModelResponse:
    """Model's response to a task instance."""
    instance_id: str
    model_name: str
    response_text: str
    parsed_answer: Dict[str, Any]
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base class for all LongAudioBench tasks."""
    
    def __init__(self, task_name: str, config: Dict[str, Any]):
        self.task_name = task_name
        self.config = config
        self.instances: List[TaskInstance] = []
    
    @abstractmethod
    def generate_instances(self, 
                          source_data: Dict[str, Any], 
                          num_instances: int,
                          split: str = "test") -> List[TaskInstance]:
        """Generate task instances from source data."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Return the prompt template for this task."""
        pass
    
    @abstractmethod
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model's free-text response into structured answer."""
        pass
    
    @abstractmethod
    def evaluate(self, 
                prediction: Dict[str, Any], 
                ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single prediction against ground truth."""
        pass
    
    def save_instances(self, instances: List[TaskInstance], output_path: str):
        """Save instances to JSONL."""
        with open(output_path, 'w') as f:
            for inst in instances:
                f.write(json.dumps(inst.to_dict()) + '\n')
    
    def load_instances(self, input_path: str) -> List[TaskInstance]:
        """Load instances from JSONL."""
        instances = []
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                instances.append(TaskInstance(**data))
        return instances


def generate_instance_id(task_name: str, seed: str = "") -> str:
    """Generate a unique instance ID."""
    return f"{task_name}_{uuid.uuid4().hex[:8]}{('_' + seed) if seed else ''}"