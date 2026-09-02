"""
Unified evaluation metrics for LongAudioBench.
"""

from typing import Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsAggregator:
    """Aggregates metrics across multiple task instances."""
    
    def __init__(self):
        self.results: Dict[str, List[MetricResult]] = {}
    
    def add_result(self, task_name: str, result: MetricResult):
        if task_name not in self.results:
            self.results[task_name] = []
        self.results[task_name].append(result)
    
    def get_summary(self, task_name: str) -> Dict[str, Any]:
        """Get summary statistics for a task."""
        if task_name not in self.results:
            return {}
        
        values = [r.value for r in self.results[task_name]]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values),
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all tasks."""
        return {task: self.get_summary(task) for task in self.results}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all results."""
        return {
            task: [
                {"name": r.name, "value": r.value, "details": r.details}
                for r in results
            ]
            for task, results in self.results.items()
        }


def compute_task_metrics(
    task_name: str,
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    task_evaluator: Callable
) -> Dict[str, Any]:
    """Compute all metrics for a task across instances."""
    assert len(predictions) == len(ground_truths)
    
    all_metrics = []
    for pred, gt in zip(predictions, ground_truths):
        metrics = task_evaluator(pred, gt)
        all_metrics.append(metrics)
    
    # Aggregate
    if not all_metrics:
        return {}
    
    metric_names = all_metrics[0].keys()
    aggregated = {}
    
    for name in metric_names:
        values = [m[name] for m in all_metrics if name in m and m[name] != float('inf')]
        if values:
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }
        else:
            aggregated[name] = {"mean": 0.0, "std": 0.0, "median": 0.0}
    
    return aggregated


def bootstrap_confidence_interval(
    values: List[float], 
    n_bootstrap: int = 10000, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrapped.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrapped, alpha * 100)
    upper = np.percentile(bootstrapped, (1 - alpha) * 100)
    
    return (float(lower), float(upper))


def compare_models(
    model_results: Dict[str, Dict[str, Any]], 
    metric: str = "composite_score"
) -> Dict[str, Any]:
    """Compare multiple models on a specific metric."""
    comparison = {}
    
    for model_name, results in model_results.items():
        if metric in results:
            comparison[model_name] = results[metric].get("mean", 0.0)
    
    # Rank models
    ranked = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "metric": metric,
        "scores": comparison,
        "ranking": ranked,
        "best_model": ranked[0][0] if ranked else None,
    }


# Import at end to avoid circular dependency
from typing import Tuple