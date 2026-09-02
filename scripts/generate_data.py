#!/usr/bin/env python3
"""
LongAudioBench Data Generation Script

Generates all task instances from public datasets.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from longaudiobench.tasks import get_task, list_tasks
from longaudiobench.configs.default import load_config


def discover_data_sources(config: dict) -> dict:
    """Discover available audio files from configured data paths."""
    data_config = config.get("data", {})
    source_data = {}
    
    # AudioSet backgrounds
    audioset_path = data_config.get("audioset_path")
    if audioset_path and os.path.exists(audioset_path):
        source_data["background_audio_paths"] = discover_audioset_backgrounds(audioset_path)
    
    # FSD50k backgrounds
    fsd50k_path = data_config.get("fsd50k_path")
    if fsd50k_path and os.path.exists(fsd50k_path):
        if "background_audio_paths" not in source_data:
            source_data["background_audio_paths"] = {}
        source_data["background_audio_paths"].update(discover_fsd50k_backgrounds(fsd50k_path))
    
    # VoxCeleb2 speakers
    voxceleb2_path = data_config.get("voxceleb2_path")
    if voxceleb2_path and os.path.exists(voxceleb2_path):
        source_data["speaker_audio_paths"] = {"voxceleb2": discover_voxceleb2_speakers(voxceleb2_path)}
    
    # AMI speakers
    ami_path = data_config.get("ami_path")
    if ami_path and os.path.exists(ami_path):
        if "speaker_audio_paths" not in source_data:
            source_data["speaker_audio_paths"] = {}
        source_data["speaker_audio_paths"]["ami"] = discover_ami_speakers(ami_path)
    
    # LibriSpeech speakers
    librispeech_path = data_config.get("librispeech_path")
    if librispeech_path and os.path.exists(librispeech_path):
        if "speaker_audio_paths" not in source_data:
            source_data["speaker_audio_paths"] = {}
        source_data["speaker_audio_paths"]["librispeech"] = discover_librispeech_speakers(librispeech_path)
    
    # Story sources
    stories = []
    for src in ["librivox_path", "spotify_podcasts_path", "gigaspeech_path"]:
        path = data_config.get(src)
        if path and os.path.exists(path):
            stories.extend(discover_stories(path, src.replace("_path", "")))
    if stories:
        source_data["story_audio_paths"] = stories
    
    # Acoustic clues
    source_data["acoustic_clue_paths"] = discover_acoustic_clues(data_config)
    
    # Event audio for soundscapes
    source_data["event_audio_paths"] = discover_event_audio(data_config)
    source_data["background_audio_paths"] = source_data.get("background_audio_paths", {})
    
    return source_data


def discover_audioset_backgrounds(path: str) -> dict:
    """Discover AudioSet background audio files by category."""
    # Placeholder - implement based on AudioSet structure
    categories = ["library", "office", "lecture"]
    result = {cat: [] for cat in categories}
    
    # In practice, you'd use AudioSet ontology to find relevant clips
    # For now, return empty - user must populate
    print(f"[INFO] AudioSet path: {path} - implement discovery based on your download structure")
    return result


def discover_fsd50k_backgrounds(path: str) -> dict:
    """Discover FSD50k background audio files."""
    categories = ["library", "office", "lecture"]
    result = {cat: [] for cat in categories}
    print(f"[INFO] FSD50k path: {path} - implement discovery")
    return result


def discover_voxceleb2_speakers(path: str) -> list:
    """Discover VoxCeleb2 speakers with metadata."""
    speakers = []
    # VoxCeleb2 structure: /idXXXXXX/YYYY.wav
    print(f"[INFO] VoxCeleb2 path: {path} - implement discovery")
    return speakers


def discover_ami_speakers(path: str) -> list:
    """Discover AMI meeting speakers."""
    print(f"[INFO] AMI path: {path} - implement discovery")
    return []


def discover_librispeech_speakers(path: str) -> list:
    """Discover LibriSpeech speakers."""
    print(f"[INFO] LibriSpeech path: {path} - implement discovery")
    return []


def discover_stories(path: str, source: str) -> list:
    """Discover long-form stories/podcasts."""
    print(f"[INFO] {source} path: {path} - implement discovery")
    return []


def discover_acoustic_clues(data_config: dict) -> dict:
    """Discover acoustic clue sounds."""
    # These would be short distinctive sounds
    clues = {}
    clue_types = [
        "clock_chime", "distinctive_bell", "unique_bird_call", "train_whistle",
        "church_bell", "factory_whistle", "ice_cream_truck", "emergency_alert",
        "glass_shatter", "pen_click", "phone_notification", "door_slam", "keyboard_click"
    ]
    
    # In practice, source from Freesound, AudioSet, or record your own
    print("[INFO] Acoustic clues - source from Freesound or record custom")
    return clues


def discover_event_audio(data_config: dict) -> dict:
    """Discover environmental event audio for soundscapes."""
    events = {}
    event_types = [
        "rain_start", "rain_stop", "rain_heavy", "rain_light",
        "wind_start", "wind_stop", "wind_howling",
        "thunder", "siren_approaching", "siren_passing", "siren_receding",
        "dog_barking", "bird_chorus", "traffic_increase", "traffic_decrease",
        "footsteps", "door_open", "door_close", "engine_start", "engine_stop",
        "crowd_cheer", "crowd_applause", "music_start", "music_stop"
    ]
    
    for et in event_types:
        events[et] = []
    
    print("[INFO] Event audio - source from AudioSet, FSD50k, DCASE")
    return events


def generate_task_data(task_name: str, config: dict, source_data: dict, output_dir: str):
    """Generate data for a single task."""
    task_config = config.get("tasks", {}).get(task_name, {})
    num_instances = task_config.get("num_instances", 10)
    
    print(f"[INFO] Generating {num_instances} instances for {task_name}...")
    
    task = get_task(task_name, task_config)
    instances = task.generate_instances(source_data, num_instances, split="test")
    
    # Save instances
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_test.jsonl")
    task.save_instances(instances, output_path)
    
    print(f"[INFO] Saved {len(instances)} instances to {output_path}")
    return instances


def main():
    parser = argparse.ArgumentParser(description="Generate LongAudioBench data")
    parser.add_argument("--config", default="longaudiobench/configs/default.yaml", help="Config file path")
    parser.add_argument("--tasks", nargs="+", choices=list_tasks() + ["all"], default=["all"], help="Tasks to generate")
    parser.add_argument("--output-dir", default="data/generated", help="Output directory")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Split to generate")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Discover data sources
    print("[INFO] Discovering data sources...")
    source_data = discover_data_sources(config)
    
    # Determine tasks to generate
    tasks_to_generate = list_tasks() if "all" in args.tasks else args.tasks
    
    # Generate each task
    for task_name in tasks_to_generate:
        try:
            generate_task_data(task_name, config, source_data, args.output_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("[INFO] Data generation complete!")


if __name__ == "__main__":
    main()