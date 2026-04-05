"""
Combine results from all models and generate final evaluation charts.
Run this after train.py has completed.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from src.evaluation import (
    comparison_table, plot_metrics_comparison,
    plot_per_site_boxplot, plot_site_heatmap
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")

    results_path = os.path.join(results_dir, "results.json")

    if not os.path.exists(results_path):
        print("No results found. Run train.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    combined_aggregate = results.get("aggregate", {})
    combined_per_site = results.get("per_site", {})

    if not combined_aggregate:
        print("No aggregate results found.")
        return

    print("=" * 50)
    print("Combined Model Evaluation")
    print("=" * 50)

    # Comparison table
    comp_df = comparison_table(combined_aggregate)
    print("\nAggregate Metrics:")
    print(comp_df.to_string())
    comp_df.to_csv(os.path.join(results_dir, "combined_comparison.csv"))

    # Metrics bar chart
    plot_metrics_comparison(combined_aggregate,
                             save_path=os.path.join(results_dir, "combined_metrics.png"))

    # Per-site analysis
    if combined_per_site:
        site_results_int = {}
        for model, sites in combined_per_site.items():
            site_results_int[model] = {}
            for site_str, metrics in sites.items():
                site_results_int[model][int(site_str)] = metrics

        plot_per_site_boxplot(site_results_int, metric="RMSE",
                               save_path=os.path.join(results_dir, "combined_boxplot_rmse.png"))
        plot_per_site_boxplot(site_results_int, metric="MAE",
                               save_path=os.path.join(results_dir, "combined_boxplot_mae.png"))
        plot_site_heatmap(site_results_int, metric="RMSE",
                           save_path=os.path.join(results_dir, "combined_heatmap_rmse.png"))
        plot_site_heatmap(site_results_int, metric="R2",
                           save_path=os.path.join(results_dir, "combined_heatmap_r2.png"))

    print(f"\nCharts saved to {results_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
