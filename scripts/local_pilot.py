#!/usr/bin/env python3
"""
Local pilot test for LongAudioBench using Whisper + rule-based "LLM"
Tests the 4 tasks with synthetic data to verify task design.
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Add longaudiobench to path
sys.path.insert(0, '/Users/deepansadhukhan/Documents/GitHub/longaudiobench')

from longaudiobench.tasks import get_task
from longaudiobench.data.audio_mixing import (
    create_anih_audio,
    create_speaker_drift_audio,
    create_soundscape_audio,
    create_narrative_coherence_audio,
)
from longaudiobench.tasks.base import TaskInstance

import whisper

DATA_DIR = Path('/Users/deepansadhukhan/Documents/GitHub/longaudiobench/data')
GEN_DIR = DATA_DIR / 'generated'
GEN_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path('/Users/deepansadhukhan/Documents/GitHub/longaudiobench/results_local')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create output directories for each task
for task_name in ["anih", "speaker_drift", "soundscape", "narrative_coherence"]:
    (GEN_DIR / task_name).mkdir(parents=True, exist_ok=True)

print("="*60)
print("LOCAL PILOT: LongAudioBench with Whisper + Rule-based LLM")
print("="*60)

# Load Whisper
print("\nLoading Whisper...")
asr_model = whisper.load_model("base", device="mps")
print("Whisper loaded")

# Generate synthetic metadata
def get_synthetic_metadata():
    """Create metadata for synthetic data."""
    return {
        "background_audio_paths": {
            "library": [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)],
            "office": [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)],
            "lecture": [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)],
            "nature": [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)],
        },
        "background_audio_paths_list": [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)],
        "needle_audio_paths": {
            "glass_shatter": str(DATA_DIR / "needle_glass.wav"),
            "pen_click": str(DATA_DIR / "needle_pen.wav"),
            "phone_notification": str(DATA_DIR / "needle_phone.wav"),
            "door_slam": str(DATA_DIR / "needle_door.wav"),
            "keyboard_click": str(DATA_DIR / "needle_keyboard.wav"),
            "clock_chime": str(DATA_DIR / "needle_glass.wav"),
            "distinctive_bell": str(DATA_DIR / "needle_door.wav"),
            "unique_bird_call": str(DATA_DIR / "needle_phone.wav"),
        },
        "speaker_audio_paths": {
            "librispeech": [{"speaker_id": f"spk_{i}", "audio_path": str(DATA_DIR / f"speaker_{i}.wav"), "duration": 10.0, "gender": "unknown"} for i in range(5)],
            "voxceleb2": [{"speaker_id": f"spk_{i}", "audio_path": str(DATA_DIR / f"speaker_{i}.wav"), "duration": 10.0, "gender": "unknown"} for i in range(5)],
        },
        "event_audio_paths": {
            "rain_start": str(DATA_DIR / "bg_0.wav"),
            "rain_stop": str(DATA_DIR / "bg_1.wav"),
            "wind_howling": str(DATA_DIR / "bg_2.wav"),
            "thunder": str(DATA_DIR / "bg_3.wav"),
            "siren_approaching": str(DATA_DIR / "bg_4.wav"),
        },
        "acoustic_clue_paths": {
            "clock_chime": str(DATA_DIR / "needle_glass.wav"),
            "distinctive_bell": str(DATA_DIR / "needle_door.wav"),
            "unique_bird_call": str(DATA_DIR / "needle_phone.wav"),
        },
        "story_audio_paths": [
            {"path": str(DATA_DIR / "bg_0.wav"), "duration": 300.0, "transcript": "Chapter one. The story begins.", "metadata": {}},
            {"path": str(DATA_DIR / "bg_1.wav"), "duration": 300.0, "transcript": "Chapter two. The plot thickens.", "metadata": {}},
        ],
    }

source_data = get_synthetic_metadata()
# Save the dict format for ANiH (it needs category-based backgrounds)
source_data["background_audio_paths_dict"] = source_data.get("background_audio_paths", {})
# Soundscape expects a list, so overwrite with list
source_data["background_audio_paths"] = [str(DATA_DIR / f"bg_{i}.wav") for i in range(5)]
# ANiH expects a dict - restore it from saved dict
source_data["background_audio_paths_for_anih"] = source_data["background_audio_paths_dict"]

# Generate instances for each task (2 per task for speed)
NUM_PER_TASK = 2
all_results = {}

# Simple rule-based "LLM" that uses transcript to answer
def rule_based_answer(transcript: str, prompt: str, task_name: str, ground_truth: dict) -> dict:
    """Simple rule-based answer using keyword matching on transcript."""
    transcript_lower = transcript.lower()
    prompt_lower = prompt.lower()
    
    if task_name == "anih":
        for needle_type in ["glass", "pen", "phone", "door", "keyboard"]:
            if needle_type in transcript_lower:
                return {"timestamp": "01:30", "preceding_sound": "ambient noise"}
        return {"timestamp": "00:00", "preceding_sound": "unknown"}
    
    elif task_name == "speaker_drift":
        # For speaker drift, we can't really detect speaker from transcript
        # Return a fixed answer - this will score 0 but tests the pipeline
        return {"predicted_speaker": 3, "first_appearance": "01:30"}
    
    elif task_name == "soundscape":
        events = []
        for ev in ["rain", "wind", "thunder", "siren", "bird"]:
            if ev in transcript_lower:
                events.append({"event_type": ev, "start": "00:00", "end": "02:00", "overlaps": ""})
        return {"events": events}
    
    elif task_name == "narrative_coherence":
        return {"verdict": "Debunks", "reasoning_steps": ["Found clue", "Character referenced it", "Alibi contradicts"], "cited_timestamps": ["05:00", "20:00", "35:00"]}
    
    return {}

def run_task(task_name: str):
    print(f"\n{'='*50}")
    print(f"TASK: {task_name}")
    print(f"{'='*50}")
    
    task = get_task(task_name)
    
    # Create task-specific source_data with correct format
    task_source_data = source_data.copy()
    if task_name == "anih":
        # ANiH needs dict format with categories
        task_source_data["background_audio_paths"] = source_data["background_audio_paths_dict"]
    elif task_name == "soundscape":
        # Soundscape needs list format (already set in source_data)
        pass
    # Other tasks use whatever format they need
    
    print(f"  Generating instances...")
    instances = task.generate_instances(task_source_data, NUM_PER_TASK, split="test")
    print(f"  Generated {len(instances)} instances")
    
    if len(instances) == 0:
        print(f"  WARNING: No instances generated!")
        return None
    
    # Create audio files
    for inst in instances:
        try:
            output_path = str(GEN_DIR / task_name / f"{inst.instance_id}.wav")
            
            if task_name == "anih":
                create_anih_audio(
                    inst.metadata["bg_audio_path"],
                    inst.metadata["needle_audio_path"],
                    output_path,
                    inst.metadata["insertion_time"],
                    bg_duration=60,
                    snr_db=10.0
                )
                inst.audio_path = output_path
            elif task_name == "speaker_drift":
                segments = []
                for seg in inst.metadata["timeline"]:
                    spk_meta = inst.metadata["selected_speakers"][seg["speaker_idx"]]
                    segments.append((spk_meta["audio_path"], seg["start_time"], seg["end_time"]))
                create_speaker_drift_audio(segments, output_path, total_duration=60)
                inst.audio_path = output_path
            elif task_name == "soundscape":
                segments = []
                for ev in inst.metadata["events"]:
                    # Use the dict format for events
                    for bg_cat, paths in source_data["background_audio_paths_dict"].items():
                        if paths:
                            start_sec = int(ev["start"].split(":")[0])*60 + int(ev["start"].split(":")[1])
                            end_sec = 60 if ev["end"] == "ongoing" else int(ev["end"].split(":")[0])*60 + int(ev["end"].split(":")[1])
                            segments.append((paths[0], start_sec, end_sec))
                            break
                # Use the list format for background
                bg_path = source_data["background_audio_paths"][0]
                create_soundscape_audio(segments, bg_path, output_path, total_duration=60)
                inst.audio_path = output_path
            elif task_name == "narrative_coherence":
                create_narrative_coherence_audio(
                    inst.metadata["story_path"],
                    inst.metadata["clue_audio_path"],
                    output_path,
                    inst.metadata["clue_a_time"],
                    story_duration=60
                )
                inst.audio_path = output_path
        except Exception as e:
            print(f"  Failed to create audio for {inst.instance_id}: {e}")
            return None
    
    # Run Whisper on each instance
    predictions = []
    latencies = []
    
    for i, instance in enumerate(instances):
        print(f"  [{i+1}/{len(instances)}] {instance.instance_id}")
        start = time.time()
        
        # Transcribe with Whisper
        result = asr_model.transcribe(instance.audio_path)
        transcript = result["text"]
        
        # Rule-based answer
        parsed = rule_based_answer(transcript, instance.prompt, task_name, instance.ground_truth)
        
        latency = (time.time() - start) * 1000
        predictions.append({
            "instance_id": instance.instance_id,
            "prediction": parsed,
            "raw_response": transcript,
            "latency_ms": latency,
        })
        latencies.append(latency)
        
        # Evaluate
        metrics = task.evaluate(parsed, instance.ground_truth)
        print(f"    Metrics: {metrics}")
    
    # Aggregate
    from longaudiobench.metrics import compute_task_metrics
    ground_truths = [inst.ground_truth for inst in instances]
    eval_results = compute_task_metrics(
        task_name,
        [p["prediction"] for p in predictions],
        ground_truths,
        task.evaluate
    )
    
    results = {
        "task": task_name,
        "model": "whisper_base_rule_based",
        "num_instances": len(instances),
        "metrics": eval_results,
        "avg_latency_ms": sum(l for l in latencies if l > 0) / len([l for l in latencies if l > 0]) if any(l > 0 for l in latencies) else 0,
        "predictions": predictions,
    }
    
    # Save
    out_path = RESULTS_DIR / f"{task_name}_whisper_rule_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results: {eval_results}")
    return results

# Run all tasks
for task_name in ["anih", "speaker_drift", "soundscape", "narrative_coherence"]:
    try:
        all_results[task_name] = run_task(task_name)
    except Exception as e:
        print(f"Task {task_name} failed: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for task_name, res in all_results.items():
    if res:
        print(f"{task_name}: {res['metrics']}")

# Save summary
with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}")