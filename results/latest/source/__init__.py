"""E1: paired allocation of goods and chores."""
from .analysis import paired_differences, summarize, summary_table
from .config import ExperimentConfig, ModelConfig, ReferenceConfig, Setting
from .data import generate_instances
from .experiment import ExperimentResult, analyze_file, run_experiment, compare_saved_run
from .metrics import allocation_scores, answer_metrics, sd_scores
from .plotting import plot_inputs, plot_paired, plot_results, plot_rates, radar_table, PAPER_METRICS, EXTENDED_METRICS
from .prompts import PROMPT_VARIANTS, make_prompt, parse_allocation
from .reference import exact_reference, round_robin, minimum_social_welfare, maximum_nash_welfare

__all__ = ["ExperimentConfig", "ModelConfig", "ReferenceConfig", "Setting", "ExperimentResult",
           "run_experiment", "analyze_file", "generate_instances", "allocation_scores", "answer_metrics",
           "sd_scores", "exact_reference", "make_prompt", "parse_allocation", "summarize",
           "summary_table", "paired_differences", "plot_inputs", "plot_results", "plot_paired",
           "plot_rates", "radar_table", "PAPER_METRICS", "EXTENDED_METRICS", "round_robin",
           "minimum_social_welfare", "maximum_nash_welfare", "compare_saved_run", "PROMPT_VARIANTS"]
