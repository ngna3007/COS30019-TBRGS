"""Tests for traffic flow to travel time conversion."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.traffic_conversion import flow_to_speed, compute_travel_time, flow_15min_to_hourly


def test_free_flow_speed():
    """Test 12: Speed is 60 km/hr when flow <= 351 veh/hr."""
    assert flow_to_speed(0) == 60.0
    assert flow_to_speed(100) == 60.0
    assert flow_to_speed(351) == 60.0


def test_congested_speed():
    """Test 13: Speed decreases when flow > 351 veh/hr."""
    speed = flow_to_speed(1000)
    assert 0 < speed < 60, f"Expected 0 < speed < 60, got {speed}"


def test_max_flow_speed():
    """Test 14: At max flow (1500), speed is at minimum (~32 km/hr)."""
    speed = flow_to_speed(1500)
    assert 30 <= speed <= 35, f"Expected ~32 km/hr at max flow, got {speed}"


def test_speed_monotonically_decreasing():
    """Test 15: Speed decreases as flow increases (above threshold)."""
    flows = [400, 600, 800, 1000, 1200, 1500]
    speeds = [flow_to_speed(f) for f in flows]
    for i in range(len(speeds) - 1):
        assert speeds[i] >= speeds[i + 1], \
            f"Speed should decrease: flow {flows[i]}->{flows[i+1]}, speed {speeds[i]}->{speeds[i+1]}"


def test_travel_time_positive():
    """Test 16: Travel time is always positive for valid inputs."""
    for flow in [0, 100, 500, 1000, 1500]:
        for dist in [0.1, 1.0, 5.0]:
            tt = compute_travel_time(dist, flow)
            assert tt > 0, f"Travel time should be > 0, got {tt} for dist={dist}, flow={flow}"


def test_travel_time_includes_intersection_delay():
    """Test 17: Travel time includes 30-second intersection delay."""
    # At free flow (60 km/hr), 1 km takes 60 seconds
    tt = compute_travel_time(1.0, 0)
    assert tt == 90.0, f"Expected 60s travel + 30s delay = 90s, got {tt}"


def test_flow_15min_to_hourly():
    """Test 18: 15-min volume correctly multiplied by 4."""
    assert flow_15min_to_hourly(100) == 400
    assert flow_15min_to_hourly(0) == 0
