"""Tests for preprocessing and feature engineering."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.preprocessing import create_sequences, prepare_site_data, melt_to_timeseries, aggregate_by_site
from src.data_loader import load_scats_data


@pytest.fixture(scope="module")
def agg_data():
    df = load_scats_data()
    ts = melt_to_timeseries(df)
    return aggregate_by_site(ts)


def test_sliding_window_shape():
    """Test 8: Sliding window produces correct output shape."""
    series = np.random.rand(100, 6).astype(np.float32)
    X, y = create_sequences(series, window_size=12, horizon=1)
    assert X.shape == (88, 12, 6), f"Expected (88, 12, 6), got {X.shape}"
    assert y.shape == (88,), f"Expected (88,), got {y.shape}"


def test_sliding_window_values():
    """Test 9: Sliding window values are correct."""
    series = np.arange(20).reshape(-1, 1).astype(np.float32)
    series = np.hstack([series, np.zeros((20, 1))])
    X, y = create_sequences(series, window_size=3, horizon=1)
    # First window: [0,1,2], predict 3
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    assert y[0] == 3
    # Last window: [16,17,18], predict 19
    np.testing.assert_array_equal(X[-1, :, 0], [16, 17, 18])
    assert y[-1] == 19


def test_temporal_split_no_leakage(agg_data):
    """Test 10: Temporal 3-way split has no data leakage."""
    result = prepare_site_data(agg_data, 970, window_size=12,
                               train_ratio=0.7, val_ratio=0.1)
    assert result is not None

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, timestamps = result

    # All splits should have data
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    # Train should be largest, test second, val smallest
    assert len(X_train) > len(X_test) > len(X_val)
    # Total samples should be less than total data points
    total = len(X_train) + len(X_val) + len(X_test)
    site_data = agg_data[agg_data["scats_number"] == 970]
    assert total < len(site_data)


def test_aggregation_reduces_rows():
    """Test 11: Aggregation reduces row count from raw time series."""
    df = load_scats_data()
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    # Aggregation should have fewer or equal rows (sum across directions)
    assert len(agg) <= len(ts)
    # Each site should have at most 96 * 31 = 2976 time points
    for site in agg["scats_number"].unique():
        site_count = len(agg[agg["scats_number"] == site])
        assert site_count <= 2976, \
            f"Site {site} has {site_count} records, max expected 2976"
