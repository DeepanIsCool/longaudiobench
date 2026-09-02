"""
Task registry for LongAudioBench.
"""

from typing import Dict, Type, Optional
from .base import BaseTask
from .anih.task import ANIHTask
from .speaker_drift.task import SpeakerDriftTask
from .soundscape.task import SoundscapeTask
from .narrative_coherence.task import NarrativeCoherenceTask


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "anih": ANIHTask,
    "speaker_drift": SpeakerDriftTask,
    "soundscape": SoundscapeTask,
    "narrative_coherence": NarrativeCoherenceTask,
}


def get_task(task_name: str, config: Optional[Dict] = None) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](config)


def list_tasks() -> list:
    """List all available tasks."""
    return list(TASK_REGISTRY.keys())


def create_task_suite(task_names: list, configs: Optional[Dict] = None) -> Dict[str, BaseTask]:
    """Create multiple tasks at once."""
    suite = {}
    for name in task_names:
        config = configs.get(name) if configs else None
        suite[name] = get_task(name, config)
    return suite