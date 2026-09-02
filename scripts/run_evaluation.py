#!/usr/bin/env python3
"""
LongAudioBench Evaluation Runner

Runs baselines on generated task instances and computes metrics.
"""

import argparse
import yaml
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from longaudiobench.tasks import get_task, list_tasks
from longaudiobench.baselines import get_baseline, ModelConfig
from longaudiobench.metrics import compute_task_metrics, bootstrap_confidence_interval, compare_models


def load_instances(task_name: str, data_dir: str, split: str = "test") -> List[Any]:
    """Load task instances from JSONL."""
    from longaudiobench.tasks.base import TaskInstance
    
    path = os.path.join(data_dir, f"{task_name}_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance file not found: {path}")
    
    instances = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            instances.append(TaskInstance(**data))
    return instances


def run_evaluation(
    task_name: str,
    model_name: str,
    model_config: ModelConfig,
    instances: List[Any],
    output_dir: str,
    save_predictions: bool = True
) -> Dict[str, Any]:
    """Run evaluation for a single task-model pair."""
    
    print(f"[INFO] Evaluating {model_name} on {task_name}...")
    
    # Get task and evaluator
    task = get_task(task_name)
    
    # Load model
    print(f"[INFO] Loading {model_name}...")
    baseline = get_baseline(model_name, model_config)
    baseline.load_model()
    
    # Run predictions
    predictions = []
    latencies = []
    
    for i, instance in enumerate(instances):
        print(f"  [{i+1}/{len(instances)}] {instance.instance_id}")
        start_time = time.time()
        
        try:
            result = baseline.predict(instance.audio_path, instance.prompt)
            latency = (time.time() - start_time) * 1000
            
            # Parse response
            parsed = task.parse_model_response(result.get("response", ""))
            
            predictions.append({
                "instance_id": instance.instance_id,
                "prediction": parsed,
                "raw_response": result.get("response", ""),
                "latency_ms": latency,
            })
            latencies.append(latency)
            
        except Exception as e:
            print(f"  [ERROR] Failed on {instance.instance_id}: {e}")
            predictions.append({
                "instance_id": instance.instance_id,
                "prediction": {},
                "raw_response": f"ERROR: {e}",
                "latency_ms": -1,
            })
            latencies.append(-1)
    
    # Evaluate
    ground_truths = [inst.ground_truth for inst in instances]
    eval_results = compute_task_metrics(
        task_name,
        [p["prediction"] for p in predictions],
        ground_truths,
        task.evaluate
    )
    
    # Add confidence intervals
    for metric_name, metric_stats in eval_results.items():
        if "mean" in metric_stats:
            # Collect individual values for bootstrap
            values = []
            for p, gt in zip(predictions, ground_truths):
                m = task.evaluate(p["prediction"], gt)
                if metric_name in m and m[metric_name] != float('inf'):
                    values.append(m[metric_name])
            
            if len(values) >= 2:
                ci_lower, ci_upper = bootstrap_confidence_interval(values)
                metric_stats["ci_95"] = [ci_lower, ci_upper]
    
    # Prepare results
    results = {
        "task": task_name,
        "model": model_name,
        "num_instances": len(instances),
        "metrics": eval_results,
        "avg_latency_ms": sum(l for l in latencies if l > 0) / len([l for l in latencies if l > 0]) if any(l > 0 for l in latencies) else 0,
        "predictions": predictions if save_predictions else [],
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_{model_name}_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to {output_path}")
    print(f"[INFO] Key metrics: {eval_results}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LongAudioBench evaluation")
    parser.add_argument("--config", default="longaudiobench/configs/default.yaml", help="Config file path")
    parser.add_argument("--tasks", nargs="+", choices=list_tasks() + ["all"], default=["all"], help="Tasks to evaluate")
    parser.add_argument("--models", nargs="+", default=["whisper_llama3", "salmonn", "qwen_audio", "ltu"], help="Models to evaluate")
    parser.add_argument("--data-dir", default="data/generated", help="Directory with generated instances")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--no-save-predictions", action="store_true", help="Don't save individual predictions")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine tasks
    tasks_to_eval = list_tasks() if "all" in args.tasks else args.tasks
    
    # Model configs
    model_configs = config.get("models", {})
    
    # Run evaluations
    all_results = {}
    
    for task_name in tasks_to_eval:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")
        
        # Load instances
        try:
            instances = load_instances(task_name, args.data_dir, args.split)
            print(f"[INFO] Loaded {len(instances)} instances")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue
        
        task_results = {}
        
        for model_name in args.models:
            if model_name not in model_configs:
                print(f"[WARNING] No config for model {model_name}, skipping")
                continue
            
            model_config_dict = model_configs[model_name].copy()
            model_config_dict["device"] = args.device
            model_config = ModelConfig(name=model_name, **model_config_dict)
            
            try:
                results = run_evaluation(
                    task_name,
                    model_name,
                    model_config,
                    instances,
                    args.output_dir,
                    save_predictions=not args.no_save_predictions
                )
                task_results[model_name] = results
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {model_name} on {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[task_name] = task_results
    
    # Cross-model comparison
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    
    for task_name, task_results in all_results.items():
        print(f"\n--- {task_name} ---")
        comparison = compare_models(task_results, "composite_score")
        print(f"  Ranking: {comparison['ranking']}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[INFO] Summary saved to {summary_path}")
    print("[INFO] Evaluation complete!")


if __name__ == "__main__":
    main()