"""Feature engineering, sequence creation, and scaling for ML models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def melt_to_timeseries(df):
    """
    Convert wide format (96 V columns per row) to long format time series.

    Each row becomes (scats_number, datetime, volume).
    Volume is the raw 15-minute count from that sensor.

    Returns DataFrame with columns: scats_number, datetime, volume
    """
    volume_cols = [f"V{i:02d}" for i in range(96)]
    available_cols = [c for c in volume_cols if c in df.columns]

    records = []
    for _, row in df.iterrows():
        base_date = pd.Timestamp(row["date"])
        site = int(row["scats_number"])
        for col in available_cols:
            interval_idx = int(col[1:])  # V00->0, V01->1, ..., V95->95
            timestamp = base_date + pd.Timedelta(minutes=15 * interval_idx)
            records.append({
                "scats_number": site,
                "datetime": timestamp,
                "volume": int(row[col]),
            })

    ts_df = pd.DataFrame(records)
    ts_df = ts_df.sort_values(["scats_number", "datetime"]).reset_index(drop=True)
    return ts_df


def aggregate_by_site(ts_df):
    """
    Aggregate volume across all measurement directions per site per timestamp.

    Multiple rows in the original data can correspond to the same site + time
    (different approach directions). Sum them for total intersection flow.

    Returns DataFrame with columns: scats_number, datetime, volume
    """
    agg = ts_df.groupby(["scats_number", "datetime"], as_index=False)["volume"].sum()
    agg = agg.sort_values(["scats_number", "datetime"]).reset_index(drop=True)
    return agg


def add_time_features(df):
    """
    Add cyclical time features to the DataFrame.

    Adds: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
    """
    df = df.copy()
    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    dow = df["datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def create_sequences(series, window_size=12, horizon=1):
    """
    Create sliding window sequences for time series prediction.

    Args:
        series: numpy array of shape (n_timesteps, n_features)
        window_size: number of past timesteps as input
        horizon: number of future timesteps to predict

    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,) - the volume value to predict
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        # Predict the volume (first feature column) at the horizon
        y.append(series[i + window_size + horizon - 1, 0])
    return np.array(X), np.array(y)


def prepare_site_data(agg_df, site_id, window_size=12, horizon=1,
                      train_ratio=0.7, val_ratio=0.1):
    """
    Prepare train/val/test data for a single site.

    Temporal split: train (70%) | val (10%) | test (20%)
    - Train: used for model fitting
    - Val: used for early stopping / hyperparameter tuning
    - Test: held out, only used for final evaluation

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, timestamps_test
    """
    site_data = agg_df[agg_df["scats_number"] == site_id].copy()
    site_data = add_time_features(site_data)
    site_data = site_data.sort_values("datetime").reset_index(drop=True)

    if len(site_data) < window_size + horizon + 10:
        return None  # Not enough data

    # Feature columns: volume + time features
    feature_cols = ["volume", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    features = site_data[feature_cols].values.astype(np.float32)

    # Temporal split: train | val | test
    train_end = int(len(features) * train_ratio)
    val_end = int(len(features) * (train_ratio + val_ratio))

    train_features = features[:train_end]
    val_features = features[train_end:val_end]
    test_features = features[val_end:]

    # Scale based on training data only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, window_size, horizon)
    X_val, y_val = create_sequences(val_scaled, window_size, horizon)
    X_test, y_test = create_sequences(test_scaled, window_size, horizon)

    # Timestamps for test set (aligned with y_test)
    test_timestamps = site_data["datetime"].iloc[val_end + window_size + horizon - 1:
                                                  val_end + window_size + horizon - 1 + len(y_test)].values

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, test_timestamps


def prepare_all_sites(agg_df, window_size=12, horizon=1,
                      train_ratio=0.7, val_ratio=0.1):
    """
    Prepare train/val/test data for all sites.

    Returns dict: {site_id: (X_train, y_train, X_val, y_val, X_test, y_test, scaler, timestamps_test)}
    """
    sites = sorted(agg_df["scats_number"].unique())
    all_data = {}
    for site_id in sites:
        result = prepare_site_data(agg_df, site_id, window_size, horizon,
                                   train_ratio, val_ratio)
        if result is not None:
            all_data[site_id] = result
    return all_data


def inverse_scale_volume(y_scaled, scaler):
    """
    Inverse transform just the volume column (first feature).

    Args:
        y_scaled: scaled volume predictions (1D array)
        scaler: the MinMaxScaler used for scaling

    Returns:
        y_original: volume in original scale
    """
    # Build a dummy array with the right number of features
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, 0] = y_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]
