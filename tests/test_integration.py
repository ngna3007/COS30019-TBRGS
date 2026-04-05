"""End-to-end integration tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_loader import load_scats_data, get_site_locations
from src.preprocessing import melt_to_timeseries, aggregate_by_site, add_time_features
from src.graph_builder import load_adjacency
from src.route_finder import find_routes


@pytest.fixture(scope="module")
def full_setup():
    df = load_scats_data()
    locations = get_site_locations(df)
    adjacency = load_adjacency()
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    return locations, adjacency, agg


def _get_predictions(agg, locations, hour, is_weekend=False):
    """Get average traffic predictions for a given hour."""
    agg_t = agg.copy()
    agg_t["hour"] = agg_t["datetime"].dt.hour
    agg_t["is_weekend"] = agg_t["datetime"].dt.dayofweek >= 5
    mask = (agg_t["hour"] == hour) & (agg_t["is_weekend"] == is_weekend)
    predictions = {}
    for site in locations:
        site_data = agg_t[(agg_t["scats_number"] == site) & mask]
        predictions[site] = site_data["volume"].mean() if len(site_data) > 0 else 50
    return predictions


def test_peak_vs_offpeak_travel_time(full_setup):
    """Test 34: Peak hour (8am) travel time > off-peak (2am) travel time."""
    locations, adjacency, agg = full_setup

    peak_preds = _get_predictions(agg, locations, 8)
    offpeak_preds = _get_predictions(agg, locations, 2)

    peak_routes = find_routes(2000, 3002, None, peak_preds, adjacency, locations, k=1)
    offpeak_routes = find_routes(2000, 3002, None, offpeak_preds, adjacency, locations, k=1)

    assert len(peak_routes) > 0 and len(offpeak_routes) > 0

    peak_time = peak_routes[0]["travel_time_seconds"]
    offpeak_time = offpeak_routes[0]["travel_time_seconds"]

    assert peak_time > offpeak_time, \
        f"Peak ({peak_time:.0f}s) should be slower than off-peak ({offpeak_time:.0f}s)"


def test_spec_example_route(full_setup):
    """Test 35: The spec example (O=2000, D=3002) finds valid routes."""
    locations, adjacency, agg = full_setup
    predictions = _get_predictions(agg, locations, 8)

    routes = find_routes(2000, 3002, None, predictions, adjacency, locations, k=5)
    assert len(routes) >= 1, "Should find at least 1 route"

    # First route should have origin as start and destination as end
    assert routes[0]["path"][0] == 2000
    assert routes[0]["path"][-1] == 3002
    assert routes[0]["travel_time_minutes"] > 0
