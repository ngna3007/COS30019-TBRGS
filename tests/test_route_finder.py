"""Tests for route finder (Yen's K-shortest paths)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_loader import load_scats_data, get_site_locations
from src.graph_builder import load_adjacency
from src.route_finder import find_routes


@pytest.fixture(scope="module")
def routing_setup():
    df = load_scats_data()
    locations = get_site_locations(df)
    adjacency = load_adjacency()
    predictions = {site: 50 for site in locations}  # moderate traffic
    return locations, adjacency, predictions


def test_same_origin_destination(routing_setup):
    """Test 30: Same origin and destination returns trivial path."""
    locations, adjacency, predictions = routing_setup
    routes = find_routes(2000, 2000, None, predictions, adjacency, locations, k=5)
    assert len(routes) == 1
    assert routes[0]["path"] == [2000]
    assert routes[0]["travel_time_seconds"] == 0


def test_top_k_returns_multiple_routes(routing_setup):
    """Test 31: Top-5 query returns multiple distinct routes."""
    locations, adjacency, predictions = routing_setup
    routes = find_routes(2000, 3002, None, predictions, adjacency, locations, k=5)
    assert len(routes) >= 2, f"Expected at least 2 routes, got {len(routes)}"
    # All routes should be distinct
    paths = [tuple(r["path"]) for r in routes]
    assert len(set(paths)) == len(paths), "Routes should be distinct"


def test_routes_sorted_by_travel_time(routing_setup):
    """Test 32: Routes are returned in ascending travel time order."""
    locations, adjacency, predictions = routing_setup
    routes = find_routes(2000, 3002, None, predictions, adjacency, locations, k=5)
    times = [r["travel_time_seconds"] for r in routes]
    assert times == sorted(times), "Routes should be sorted by travel time"


def test_route_has_valid_fields(routing_setup):
    """Test 33: Route result contains all required fields."""
    locations, adjacency, predictions = routing_setup
    routes = find_routes(970, 4264, None, predictions, adjacency, locations, k=1)
    assert len(routes) > 0
    route = routes[0]
    assert "path" in route
    assert "travel_time_seconds" in route
    assert "travel_time_minutes" in route
    assert "distance_km" in route
    assert "num_intersections" in route
    assert route["travel_time_seconds"] > 0
    assert route["distance_km"] > 0
    assert route["num_intersections"] >= 2
