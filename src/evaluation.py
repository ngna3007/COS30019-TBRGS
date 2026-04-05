"""Comprehensive ML evaluation: metrics, comparison tables, and charts."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (skip zero-flow intervals)."""
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_r2(y_true, y_pred):
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def evaluate_model(y_true, y_pred):
    """Compute all metrics for a model."""
    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
    }


def comparison_table(results):
    """
    Create a comparison DataFrame from results.

    Args:
        results: {model_name: {metric_name: value, ...}, ...}

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.round(4)


def plot_actual_vs_predicted(y_true, y_pred, model_name, site_id,
                              timestamps=None, save_path=None):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(y_true))
    if timestamps is not None and len(timestamps) == len(y_true):
        x = timestamps

    ax.plot(x, y_true, label="Actual", alpha=0.8, linewidth=1)
    ax.plot(x, y_pred, label="Predicted", alpha=0.8, linewidth=1)
    ax.set_title(f"{model_name} - Site {site_id}: Actual vs Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic Volume (vehicles/15min)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_training_curves(history, model_name, save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Training Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_title(f"{model_name} - Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_error_distribution(y_true, y_pred, model_name, save_path=None):
    """Plot residual distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    residuals = y_true - y_pred
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{model_name} - Error Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_metrics_comparison(results, save_path=None):
    """Bar chart comparing metrics across models."""
    df = comparison_table(results)
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", len(df))

    for i, metric in enumerate(available):
        bars = axes[i].bar(df.index, df[metric], color=colors)
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        for bar, val in zip(bars, df[metric]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_per_site_boxplot(site_results, metric="RMSE", save_path=None):
    """
    Box plot of per-site metric values across models.

    Args:
        site_results: {model_name: {site_id: {metric: value}}}
    """
    data = []
    for model_name, sites in site_results.items():
        for site_id, metrics in sites.items():
            data.append({
                "Model": model_name,
                "Site": site_id,
                metric: metrics[metric],
            })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Model", y=metric, ax=ax, palette="Set2")
    ax.set_title(f"Per-Site {metric} Distribution by Model")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_site_heatmap(site_results, metric="RMSE", save_path=None):
    """
    Heatmap of per-site performance across models.

    Args:
        site_results: {model_name: {site_id: {metric: value}}}
    """
    models = list(site_results.keys())
    sites = sorted(set(s for m in site_results.values() for s in m.keys()))

    matrix = np.zeros((len(models), len(sites)))
    for i, model in enumerate(models):
        for j, site in enumerate(sites):
            matrix[i, j] = site_results[model].get(site, {}).get(metric, np.nan)

    fig, ax = plt.subplots(figsize=(max(12, len(sites) * 0.5), 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=sites,
                yticklabels=models, cmap="YlOrRd", ax=ax)
    ax.set_title(f"Per-Site {metric} Heatmap")
    ax.set_xlabel("SCATS Site")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
