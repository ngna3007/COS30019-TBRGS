"""Tests for graph building and adjacency."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_loader import load_scats_data, get_site_locations
from src.graph_builder import (
    build_adjacency, verify_connectivity, haversine_distance,
    build_traffic_graph, ROAD_NETWORK
)


@pytest.fixture(scope="module")
def adjacency():
    df = load_scats_data()
    locations = get_site_locations(df)
    return build_adjacency(locations)


@pytest.fixture(scope="module")
def locations():
    return get_site_locations(load_scats_data())


def test_all_40_sites_connected(adjacency):
    """Test 19: All 40 SCATS sites have at least one edge."""
    for site_id, neighbors in adjacency.items():
        assert len(neighbors) > 0, f"Site {site_id} has no neighbors"


def test_graph_fully_connected(adjacency):
    """Test 20: Graph is a single connected component."""
    connected, components = verify_connectivity(adjacency)
    assert connected, \
        f"Graph is not connected. {len(components)} components: " \
        f"{[sorted(c) for c in components]}"


def test_burke_rd_chain(adjacency):
    """Test 21: Burke Rd has 7 edges connecting 8 consecutive sites."""
    burke_sites = ROAD_NETWORK["BURKE_RD"]
    assert len(burke_sites) == 8
    for i in range(len(burke_sites) - 1):
        s1, s2 = burke_sites[i], burke_sites[i + 1]
        neighbors_of_s1 = [n for n, _ in adjacency[s1]]
        assert s2 in neighbors_of_s1, \
            f"Burke Rd: {s1} should connect to {s2}"


def test_edges_are_bidirectional(adjacency):
    """Test 22: Every edge A->B has a corresponding B->A."""
    for site_id, neighbors in adjacency.items():
        for neighbor_id, dist in neighbors:
            reverse_neighbors = [n for n, _ in adjacency.get(neighbor_id, [])]
            assert site_id in reverse_neighbors, \
                f"Edge {site_id}->{neighbor_id} has no reverse"


def test_haversine_distance():
    """Test 23: Haversine distance is correct for known points."""
    # Melbourne CBD to Hawthorn is roughly 6 km
    dist = haversine_distance(-37.8136, 144.9631, -37.8221, 145.0341)
    assert 5.0 < dist < 8.0, f"Melbourne-Hawthorn should be ~6km, got {dist}"


def test_traffic_graph_builds(adjacency, locations):
    """Test 24: Traffic graph builds correctly from predictions."""
    predictions = {s: 100 for s in locations}
    graph, coords = build_traffic_graph(adjacency, predictions, locations)
    assert len(graph) == 40
    # All edge weights should be positive
    for site_id, neighbors in graph.items():
        for neighbor_id, travel_time in neighbors:
            assert travel_time > 0, \
                f"Edge {site_id}->{neighbor_id} has non-positive travel time"
