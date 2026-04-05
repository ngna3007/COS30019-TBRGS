"""
Research analyses for the TBRGS report.
Generates traffic pattern analysis, route comparison, and hyperparameter insights.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_scats_data, get_site_locations, get_site_descriptions
from src.preprocessing import melt_to_timeseries, aggregate_by_site, add_time_features
from src.graph_builder import load_adjacency
from src.route_finder import find_routes
from src.traffic_conversion import flow_15min_to_hourly


def traffic_pattern_analysis(results_dir):
    """
    Research 1: Analyze daily and weekly traffic patterns.
    - Heatmap of average volume by hour and day-of-week
    - Peak hour identification per site
    - Cross-site correlation
    """
    print("  Traffic pattern analysis...")

    df = load_scats_data()
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    agg = add_time_features(agg)
    agg["hour"] = agg["datetime"].dt.hour
    agg["dow"] = agg["datetime"].dt.dayofweek
    agg["day_name"] = agg["datetime"].dt.day_name()

    # 1. Average volume heatmap (hour x day-of-week) across all sites
    pivot = agg.groupby(["dow", "hour"])["volume"].mean().reset_index()
    pivot_table = pivot.pivot(index="dow", columns="hour", values="volume")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot_table, cmap="YlOrRd", annot=False, ax=ax,
                yticklabels=day_labels)
    ax.set_title("Average Traffic Volume by Hour and Day of Week (All Sites)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_traffic_heatmap.png"), dpi=150)
    plt.close()

    # 2. Peak hour identification
    hourly_avg = agg.groupby(["scats_number", "hour"])["volume"].mean().reset_index()
    peak_hours = hourly_avg.loc[hourly_avg.groupby("scats_number")["volume"].idxmax()]
    peak_hours = peak_hours.rename(columns={"hour": "peak_hour", "volume": "peak_volume"})

    fig, ax = plt.subplots(figsize=(12, 5))
    sites = sorted(peak_hours["scats_number"].unique())
    peaks = [peak_hours[peak_hours["scats_number"] == s]["peak_hour"].values[0] for s in sites]
    ax.bar(range(len(sites)), peaks, color="steelblue")
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([str(s) for s in sites], rotation=90, fontsize=7)
    ax.set_ylabel("Peak Hour")
    ax.set_title("Peak Traffic Hour per SCATS Site")
    ax.set_ylim(0, 24)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_peak_hours.png"), dpi=150)
    plt.close()

    # 3. Weekday vs weekend comparison
    weekday = agg[agg["dow"] < 5].groupby("hour")["volume"].mean()
    weekend = agg[agg["dow"] >= 5].groupby("hour")["volume"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekday.index, weekday.values, label="Weekday", linewidth=2)
    ax.plot(weekend.index, weekend.values, label="Weekend", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Volume (vehicles/15min)")
    ax.set_title("Weekday vs Weekend Traffic Patterns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_weekday_weekend.png"), dpi=150)
    plt.close()

    # 4. Cross-site correlation matrix
    site_hourly = agg.pivot_table(index="datetime", columns="scats_number",
                                   values="volume", aggfunc="mean")
    corr = site_hourly.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False,
                xticklabels=True, yticklabels=True)
    ax.set_title("Cross-Site Traffic Volume Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_correlation.png"), dpi=150)
    plt.close()

    print("    Saved: traffic_heatmap, peak_hours, weekday_weekend, correlation")


def route_comparison_analysis(results_dir):
    """
    Research 2: Compare routes at different times of day.
    Show how the recommended route changes with traffic conditions.
    """
    print("  Route comparison analysis...")

    df = load_scats_data()
    locations = get_site_locations(df)
    descriptions = get_site_descriptions(df)
    adjacency = load_adjacency()
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    agg["hour"] = agg["datetime"].dt.hour

    # Compare routes at different hours
    origin, dest = 2000, 3002
    hours = list(range(0, 24))
    travel_times = []

    for hour in hours:
        # Get average predictions for this hour
        predictions = {}
        for site in locations:
            site_data = agg[(agg["scats_number"] == site) & (agg["hour"] == hour)]
            predictions[site] = site_data["volume"].mean() if len(site_data) > 0 else 50

        routes = find_routes(origin, dest, None, predictions, adjacency, locations, k=1)
        if routes and "error" not in routes[0]:
            travel_times.append(routes[0]["travel_time_minutes"])
        else:
            travel_times.append(None)

    fig, ax = plt.subplots(figsize=(12, 5))
    valid_hours = [h for h, t in zip(hours, travel_times) if t is not None]
    valid_times = [t for t in travel_times if t is not None]
    ax.plot(valid_hours, valid_times, "o-", linewidth=2, markersize=5, color="steelblue")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_title(f"Best Route Travel Time by Hour: {origin} -> {dest}")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)

    # Highlight peak hours
    if valid_times:
        peak_time = max(valid_times)
        peak_hour = valid_hours[valid_times.index(peak_time)]
        ax.annotate(f"Peak: {peak_time:.1f} min at {peak_hour}:00",
                    xy=(peak_hour, peak_time), xytext=(peak_hour + 2, peak_time + 1),
                    arrowprops=dict(arrowstyle="->"), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_route_by_hour.png"), dpi=150)
    plt.close()

    # ML-aware vs static routing comparison
    # Static: use average flow (free flow speed everywhere)
    static_preds = {s: 0 for s in locations}  # free flow
    peak_preds = {}
    for site in locations:
        site_data = agg[(agg["scats_number"] == site) & (agg["hour"] == 8)]
        peak_preds[site] = site_data["volume"].mean() if len(site_data) > 0 else 50

    # Test on multiple O-D pairs
    od_pairs = [
        (2000, 3002), (970, 4264), (2825, 3126), (4821, 2827), (3662, 4273),
    ]

    comparison_data = []
    for o, d in od_pairs:
        static_routes = find_routes(o, d, None, static_preds, adjacency, locations, k=1)
        peak_routes = find_routes(o, d, None, peak_preds, adjacency, locations, k=1)

        if static_routes and peak_routes and "error" not in static_routes[0] and "error" not in peak_routes[0]:
            comparison_data.append({
                "OD Pair": f"{o} -> {d}",
                "Static (min)": static_routes[0]["travel_time_minutes"],
                "Peak 8AM (min)": peak_routes[0]["travel_time_minutes"],
                "Difference (min)": peak_routes[0]["travel_time_minutes"] - static_routes[0]["travel_time_minutes"],
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(comp_df))
        width = 0.35
        ax.bar([i - width/2 for i in x], comp_df["Static (min)"], width, label="Free Flow", color="green", alpha=0.7)
        ax.bar([i + width/2 for i in x], comp_df["Peak 8AM (min)"], width, label="Peak 8AM", color="red", alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(comp_df["OD Pair"], rotation=30, ha="right")
        ax.set_ylabel("Travel Time (minutes)")
        ax.set_title("Static vs Traffic-Aware Routing")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "research_static_vs_aware.png"), dpi=150)
        plt.close()

        comp_df.to_csv(os.path.join(results_dir, "research_route_comparison.csv"), index=False)

    print("    Saved: route_by_hour, static_vs_aware, route_comparison.csv")


def flow_speed_visualization(results_dir):
    """
    Research 3: Visualize the flow-speed relationship used in the system.
    """
    print("  Flow-speed curve visualization...")
    from src.traffic_conversion import flow_to_speed

    flows = np.linspace(0, 1600, 200)
    speeds = [flow_to_speed(f) for f in flows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(flows, speeds, linewidth=2, color="steelblue")
    ax.axhline(y=60, color="blue", linestyle="--", alpha=0.5, label="Speed Limit (60 km/hr)")
    ax.axvline(x=351, color="green", linestyle="--", alpha=0.5, label="Threshold (351 veh/hr)")
    ax.axvline(x=1500, color="red", linestyle="--", alpha=0.5, label="Capacity (1500 veh/hr)")
    ax.scatter([351], [60], color="green", s=100, zorder=5)
    ax.scatter([1500], [flow_to_speed(1500)], color="red", s=100, zorder=5)
    ax.set_xlabel("Traffic Flow (vehicles/hour)")
    ax.set_ylabel("Speed (km/hr)")
    ax.set_title("Traffic Flow to Speed Conversion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 70)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "research_flow_speed_curve.png"), dpi=150)
    plt.close()

    print("    Saved: flow_speed_curve")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 50)
    print("TBRGS Research Analyses")
    print("=" * 50)

    traffic_pattern_analysis(results_dir)
    route_comparison_analysis(results_dir)
    flow_speed_visualization(results_dir)

    print("\nAll research analyses complete!")
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
