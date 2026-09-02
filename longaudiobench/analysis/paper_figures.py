"""
Analysis module for LongAudioBench paper figures and tables.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_all_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all result JSONs from directory."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.json") and fname != "summary.json":
            path = os.path.join(results_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)
                task = data["task"]
                model = data["model"]
                if task not in results:
                    results[task] = {}
                results[task][model] = data
    return results


def plot_main_results_heatmap(results: Dict, output_path: str, metric: str = "composite_score"):
    """Plot heatmap of model performance across tasks."""
    tasks = sorted(results.keys())
    models = set()
    for task_results in results.values():
        models.update(task_results.keys())
    models = sorted(models)
    
    # Primary metric per task
    primary_metrics = {
        "anih": "composite_score",
        "speaker_drift": "joint_accuracy", 
        "soundscape": "composite_score",
        "narrative_coherence": "composite_score",
    }
    
    data = []
    for task in tasks:
        row = []
        task_metric = primary_metrics.get(task, metric)
        for model in models:
            if model in results[task]:
                val = results[task][model]["metrics"].get(task_metric, {}).get("mean", 0)
            else:
                val = np.nan
            row.append(val)
        data.append(row)
    
    df = pd.DataFrame(data, index=[t.replace("_", " ").title() for t in tasks], columns=models)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5, 
                vmin=0, vmax=1, cbar_kws={'label': metric.replace('_', ' ').title()})
    plt.title(f"LongAudioBench: {metric.replace('_', ' ').title()} by Task and Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Heatmap saved to {output_path}")


def plot_radar_chart(results: Dict, output_path: str):
    """Plot radar chart comparing models across all tasks."""
    tasks = sorted(results.keys())
    models = set()
    for task_results in results.values():
        models.update(task_results.keys())
    models = sorted(models)
    
    primary_metrics = {
        "anih": "composite_score",
        "speaker_drift": "joint_accuracy",
        "soundscape": "composite_score", 
        "narrative_coherence": "composite_score",
    }
    
    # Get scores for each model
    model_scores = {}
    for model in models:
        scores = []
        for task in tasks:
            if model in results[task]:
                metric = primary_metrics.get(task, "composite_score")
                val = results[task][model]["metrics"].get(metric, {}).get("mean", 0)
            else:
                val = 0
            scores.append(val)
        model_scores[model] = scores
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    task_labels = [t.replace("_", " ").title() for t in tasks]
    
    for model, scores in model_scores.items():
        scores_loop = scores + scores[:1]
        ax.plot(angles, scores_loop, 'o-', linewidth=2, label=model, markersize=6)
        ax.fill(angles, scores_loop, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("LongAudioBench: Model Comparison Across Tasks", pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Radar chart saved to {output_path}")


def plot_depth_analysis(results: Dict, output_path: str):
    """Plot performance vs insertion depth for ANiH (if available)."""
    # This requires per-instance results with depth metadata
    # Placeholder for when depth-varied instances are generated
    pass


def plot_latency_vs_accuracy(results: Dict, output_path: str):
    """Plot latency vs accuracy tradeoff."""
    tasks = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    primary_metrics = {
        "anih": "composite_score",
        "speaker_drift": "joint_accuracy",
        "soundscape": "composite_score",
        "narrative_coherence": "composite_score",
    }
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_results = results[task]
        metric = primary_metrics.get(task, "composite_score")
        
        for model, model_results in task_results.items():
            acc = model_results["metrics"].get(metric, {}).get("mean", 0)
            latency = model_results.get("avg_latency_ms", 0) / 1000  # convert to seconds
            ax.scatter(latency, acc, s=100, label=model)
            ax.annotate(model, (latency, acc), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel("Avg Latency (s)")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(task.replace("_", " ").title())
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Latency vs Accuracy Tradeoff", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Latency vs accuracy plot saved to {output_path}")


def plot_failure_mode_breakdown(results: Dict, output_path: str):
    """Plot breakdown of failure modes per task."""
    # For each task, show which sub-metric drags down performance
    tasks = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_results = results[task]
        
        # Get all sub-metrics for the best model
        best_model = max(task_results.keys(), 
                        key=lambda m: task_results[m]["metrics"].get("composite_score", {}).get("mean", 0))
        best_results = task_results[best_model]
        
        metrics = best_results["metrics"]
        # Filter to sub-metrics (exclude composite)
        sub_metrics = {k: v for k, v in metrics.items() if k != "composite_score" and "mean" in v}
        
        if sub_metrics:
            names = list(sub_metrics.keys())
            values = [v["mean"] for v in sub_metrics.values()]
            errors = [v.get("std", 0) for v in sub_metrics.values()]
            
            bars = ax.barh(names, values, xerr=errors, capsize=3, color='steelblue', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score")
            ax.set_title(f"{task.replace('_', ' ').title()} ({best_model})")
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                       va='center', fontsize=9)
    
    plt.suptitle("Failure Mode Breakdown: Sub-metric Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Failure mode breakdown saved to {output_path}")


def generate_latex_tables(results: Dict, output_dir: str):
    """Generate LaTeX tables for paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = sorted(results.keys())
    models = set()
    for task_results in results.values():
        models.update(task_results.keys())
    models = sorted(models)
    
    primary_metrics = {
        "anih": "composite_score",
        "speaker_drift": "joint_accuracy",
        "soundscape": "composite_score",
        "narrative_coherence": "composite_score",
    }
    
    # Main table
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Main results on LongAudioBench. Primary metric per task reported with 95\\% CI.}",
        "  \\label{tab:main_results}",
        f"  \\begin{{tabular}}{{l{'c' * len(models)}}}",
        "    \\toprule",
        f"    Task & {' & '.join(models)} \\\\",
        "    \\midrule",
    ]
    
    for task in tasks:
        row = f"    {task.replace('_', ' ').title()}"
        metric = primary_metrics.get(task, "composite_score")
        for model in models:
            if model in results[task]:
                stats = results[task][model]["metrics"].get(metric, {})
                mean = stats.get("mean", 0)
                ci = stats.get("ci_95", [0, 0])
                row += f" & {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                row += " & --"
        row += " \\\\"
        lines.append(row)
    
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    
    with open(os.path.join(output_dir, "main_table.tex"), 'w') as f:
        f.write("\n".join(lines))
    
    # Detailed tables per task
    for task in tasks:
        task_results = results[task]
        all_metrics = set()
        for model_results in task_results.values():
            all_metrics.update(model_results["metrics"].keys())
        
        lines = [
            f"\\begin{{table}}[t]",
            f"  \\centering",
            f"  \\caption{{Detailed metrics for {task.replace('_', ' ').title()} task.}}",
            f"  \\label{{tab:{task}_detailed}}",
            f"  \\begin{{tabular}}{{l{'c' * len(models)}}}",
            "    \\toprule",
            f"    Metric & {' & '.join(models)} \\\\",
            "    \\midrule",
        ]
        
        for metric in sorted(all_metrics):
            row = f"    {metric.replace('_', ' ').title()}"
            for model in models:
                if model in task_results:
                    stats = task_results[model]["metrics"].get(metric, {})
                    if stats:
                        mean = stats.get("mean", 0)
                        std = stats.get("std", 0)
                        row += f" & {mean:.3f} $\\pm$ {std:.3f}"
                    else:
                        row += " & --"
                else:
                    row += " & --"
            row += " \\\\"
            lines.append(row)
        
        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ])
        
        with open(os.path.join(output_dir, f"{task}_detailed.tex"), 'w') as f:
            f.write("\n".join(lines))
    
    print(f"[INFO] LaTeX tables saved to {output_dir}")


def generate_paper_figures(results: Dict, output_dir: str):
    """Generate all paper figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    
    plot_main_results_heatmap(results, os.path.join(output_dir, "main_heatmap.pdf"))
    plot_radar_chart(results, os.path.join(output_dir, "radar_chart.pdf"))
    plot_latency_vs_accuracy(results, os.path.join(output_dir, "latency_accuracy.pdf"))
    plot_failure_mode_breakdown(results, os.path.join(output_dir, "failure_breakdown.pdf"))
    
    generate_latex_tables(results, output_dir)
    
    print(f"[INFO] All paper figures and tables generated in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="analysis/figures")
    args = parser.parse_args()
    
    results = load_all_results(args.results_dir)
    if results:
        generate_paper_figures(results, args.output_dir)
    else:
        print("[ERROR] No results found")