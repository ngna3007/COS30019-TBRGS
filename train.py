"""Train all ML models for traffic flow prediction."""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import yaml

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from src.data_loader import load_scats_data, get_site_locations, load_config
from src.preprocessing import (
    melt_to_timeseries, aggregate_by_site, prepare_all_sites,
    inverse_scale_volume
)
from src.models.lstm_model import LSTMTrafficModel
from src.models.gru_model import GRUTrafficModel
from src.models.rf_model import RandomForestTrafficModel
from src.evaluation import (
    evaluate_model, comparison_table, plot_actual_vs_predicted,
    plot_training_curves, plot_error_distribution, plot_metrics_comparison,
    plot_per_site_boxplot, plot_site_heatmap
)


def train_all(config_path=None, sites_to_train=None):
    """
    Train LSTM, GRU, and Random Forest on all (or selected) SCATS sites.

    Data split: train (70%) | val (10%) | test (20%)
    - Train: model fitting
    - Val: early stopping (LSTM/GRU only)
    - Test: final evaluation (never seen during training)
    """
    config = load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("TBRGS - Training ML Models")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n[1/4] Loading SCATS data...")
    scats_path = os.path.join(base_dir, config["data"]["scats_path"])
    df = load_scats_data(scats_path)
    print(f"  Loaded {len(df)} rows, {df['scats_number'].nunique()} sites")

    print("\n[2/4] Preprocessing...")
    ts_df = melt_to_timeseries(df)
    agg_df = aggregate_by_site(ts_df)
    print(f"  Time series: {len(agg_df)} records")

    prep_config = config["preprocessing"]
    all_site_data = prepare_all_sites(
        agg_df,
        window_size=prep_config["window_size"],
        horizon=prep_config["prediction_horizon"],
        train_ratio=prep_config.get("train_ratio", 0.7),
        val_ratio=prep_config.get("val_ratio", 0.1),
    )

    if sites_to_train:
        all_site_data = {s: d for s, d in all_site_data.items() if s in sites_to_train}

    print(f"  Prepared {len(all_site_data)} sites (70% train / 10% val / 20% test)")

    # Step 2: Train models
    models_dir = os.path.join(base_dir, "trained_models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_classes = [
        ("LSTM", LSTMTrafficModel, config["models"]["lstm"]),
        ("GRU", GRUTrafficModel, config["models"]["gru"]),
        ("RandomForest", RandomForestTrafficModel, config["models"].get("random_forest", {})),
    ]

    aggregate_results = {}
    site_results = {}
    all_histories = {}

    for model_name, ModelClass, model_config in model_classes:
        print(f"\n[3/4] Training {model_name}...")
        site_results[model_name] = {}
        all_preds = []
        all_trues = []
        model_start = time.time()

        for site_id, site_data in all_site_data.items():
            X_train, y_train, X_val, y_val, X_test, y_test, scaler, timestamps = site_data
            print(f"  Site {site_id}...", end=" ", flush=True)

            model = ModelClass()
            try:
                # Train: LSTM/GRU use val for early stopping; RF ignores val
                history = model.train(X_train, y_train, X_val, y_val, model_config)

                # Store training history for first site (LSTM/GRU only)
                if site_id == list(all_site_data.keys())[0] and isinstance(history, dict) and "loss" in history:
                    all_histories[model_name] = history

                # Evaluate on TEST set (never seen during training)
                y_pred_scaled = model.predict(X_test)
                y_pred_orig = inverse_scale_volume(y_pred_scaled, scaler)
                y_test_orig = inverse_scale_volume(y_test, scaler)
                y_pred_orig = np.maximum(y_pred_orig, 0)

                metrics = evaluate_model(y_test_orig, y_pred_orig)
                all_trues.append(y_test_orig)
                all_preds.append(y_pred_orig)

                # Save model
                ext = ".pkl" if model_name == "RandomForest" else ".keras"
                model_path = os.path.join(models_dir, f"{model_name}_{site_id}{ext}")
                model.save(model_path)

                site_results[model_name][site_id] = metrics
                print(f"RMSE={metrics['RMSE']:.2f}")

                # Plot actual vs predicted for selected sites
                if site_id in list(all_site_data.keys())[:3]:
                    plot_actual_vs_predicted(
                        y_test_orig, y_pred_orig, model_name, site_id,
                        save_path=os.path.join(results_dir, f"{model_name}_site{site_id}_pred.png")
                    )

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        # Aggregate metrics
        if all_trues and all_preds:
            all_true = np.concatenate(all_trues)
            all_pred = np.concatenate(all_preds)
            aggregate_results[model_name] = evaluate_model(all_true, all_pred)

        elapsed = time.time() - model_start
        print(f"  {model_name} done in {elapsed:.1f}s")

    # Step 3: Generate evaluation charts
    print("\n[4/4] Generating evaluation charts...")

    if aggregate_results:
        # Comparison table
        comp_df = comparison_table(aggregate_results)
        print("\n" + "=" * 40)
        print("Model Comparison (Aggregate)")
        print("=" * 40)
        print(comp_df.to_string())
        comp_df.to_csv(os.path.join(results_dir, "comparison_table.csv"))

        # Metrics comparison chart
        plot_metrics_comparison(aggregate_results,
                                save_path=os.path.join(results_dir, "metrics_comparison.png"))

    # Training curves
    for model_name, history in all_histories.items():
        plot_training_curves(history, model_name,
                              save_path=os.path.join(results_dir, f"{model_name}_training_curves.png"))

    # Per-site boxplot and heatmap
    if site_results:
        plot_per_site_boxplot(site_results, metric="RMSE",
                               save_path=os.path.join(results_dir, "per_site_rmse_boxplot.png"))
        plot_site_heatmap(site_results, metric="RMSE",
                           save_path=os.path.join(results_dir, "per_site_rmse_heatmap.png"))

    # Save results JSON
    results_json = {
        "aggregate": {k: {mk: float(mv) for mk, mv in v.items()}
                      for k, v in aggregate_results.items()},
        "per_site": {k: {str(sk): {mk: float(mv) for mk, mv in sv.items()}
                         for sk, sv in v.items()}
                     for k, v in site_results.items()},
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete! Results saved to results/")
    print("=" * 60)

    return aggregate_results, site_results


if __name__ == "__main__":
    train_all()
