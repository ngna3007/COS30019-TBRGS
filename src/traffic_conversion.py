"""Convert traffic flow to speed and travel time using the assignment formula."""

import math
import yaml
import os


def load_traffic_config():
    """Load traffic conversion parameters from config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["traffic"]


def flow_to_speed(flow_per_hour, config=None):
    """
    Convert traffic flow (vehicles/hour) to speed (km/hr).

    Formula: flow = A * speed^2 + B * speed
    where A = -1.4648375, B = 93.75

    Rules:
    - If flow <= 351: speed = 60 km/hr (speed limit, free flow)
    - If flow > 351: solve quadratic for green curve (under capacity)
    - If flow > 1500: clamp to 1500 (minimum speed from formula = 32 km/hr)
    """
    if config is None:
        config = load_traffic_config()

    A = config["quad_a"]       # -1.4648375
    B = config["quad_b"]       # 93.75
    speed_limit = config["speed_limit"]   # 60
    threshold = config["flow_threshold"]  # 351

    if flow_per_hour <= threshold:
        return float(speed_limit)

    # Clamp flow at maximum (discriminant = 0 at flow = 1500)
    flow_per_hour = min(flow_per_hour, 1500.0)

    # Solve: A*v^2 + B*v - flow = 0
    # v = (-B +/- sqrt(B^2 + 4*A*flow)) / (2*A)
    # Note: A is negative, so 4*A*flow is negative
    discriminant = B * B + 4 * A * flow_per_hour

    if discriminant < 0:
        # Beyond max flow, return minimum speed
        return max(B / (2 * abs(A)), 5.0)

    sqrt_disc = math.sqrt(discriminant)

    # Green curve (under capacity) = higher speed root
    # Since A < 0, the higher root is: (-B - sqrt(disc)) / (2*A)
    # Because 2*A is negative, dividing by it flips the sign
    speed_green = (-B - sqrt_disc) / (2 * A)
    speed_red = (-B + sqrt_disc) / (2 * A)

    # Use the green (higher speed) since traffic is assumed under capacity
    speed = max(speed_green, speed_red)
    speed = min(speed, float(speed_limit))
    speed = max(speed, 5.0)  # Floor to prevent division by zero

    return speed


def compute_travel_time(distance_km, flow_per_hour, config=None):
    """
    Compute travel time in seconds for a road segment.

    travel_time = (distance_km / speed) * 3600 + intersection_delay

    Args:
        distance_km: distance between two SCATS sites in km
        flow_per_hour: predicted traffic flow in vehicles/hour at the starting site

    Returns:
        travel_time in seconds
    """
    if config is None:
        config = load_traffic_config()

    speed = flow_to_speed(flow_per_hour, config)
    travel_seconds = (distance_km / speed) * 3600
    travel_seconds += config["intersection_delay"]  # 30 seconds

    return travel_seconds


def flow_15min_to_hourly(volume_15min):
    """Convert a 15-minute volume count to vehicles per hour."""
    return volume_15min * 4
