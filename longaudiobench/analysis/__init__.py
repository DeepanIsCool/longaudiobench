"""
Analysis module for LongAudioBench.
"""

from .paper_figures import (
    load_all_results,
    plot_main_results_heatmap,
    plot_radar_chart,
    plot_latency_vs_accuracy,
    plot_failure_mode_breakdown,
    generate_latex_tables,
    generate_paper_figures,
)

__all__ = [
    "load_all_results",
    "plot_main_results_heatmap",
    "plot_radar_chart",
    "plot_latency_vs_accuracy",
    "plot_failure_mode_breakdown",
    "generate_latex_tables",
    "generate_paper_figures",
]