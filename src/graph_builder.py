"""
Build SCATS adjacency graph from the Boroondara road network.

The adjacency is defined from the VicRoads Boroondara map (IBM data_Bor.pdf).
Each SCATS site sits at an intersection of two roads. Two sites are adjacent
if they are consecutive intersections on the same road with no other SCATS site
between them.

This module uses a hardcoded road network definition derived from the map,
which is more reliable than parsing location description strings.
"""

import json
import math
import os
from collections import deque

from .data_loader import load_config
from .traffic_conversion import compute_travel_time, flow_15min_to_hourly


# ============================================================
# Road network definition from Boroondara map
# Each road lists SCATS site IDs in geographic order (N->S or W->E).
# Only sites present in our 40-site dataset are included.
# ============================================================
ROAD_NETWORK = {
    # N-S roads (listed north to south)
    "BURKE_RD":       [2825, 4030, 4032, 4034, 4035, 3120, 4040, 4043],
    "WARRIGAL_RD":    [3682, 2000, 3685, 970],
    "BALWYN_RD":      [3180, 4057, 4063, 3127],
    "GLENFERRIE_RD":  [4324, 4264, 4270],
    "TOORONGA_RD":    [4272, 4273],
    "TRAFALGAR_RD":   [3804, 3812],
    "PRINCESS_ST":    [2820, 3662],

    # E-W roads (listed west to east)
    "HIGH_ST":        [3662, 4335, 3001, 4321],
    "BARKERS_RD":     [3001, 3002],
    "COTHAM_RD":      [4324, 4034],
    "DONCASTER_RD":   [4030, 4051, 3180],
    "WHITEHORSE_RD":  [4034, 4063, 2200],  # Maroondah Hwy becomes Whitehorse Rd
    "CANTERBURY_RD":  [3120, 3122, 3127, 3126],
    "BURWOOD_RD":     [4262, 4263, 4264, 4266],
    "RIVERSDALE_RD":  [3804, 4040, 4272],
    "TOORAK_RD":      [4043, 4273, 2000],
    "HARP_RD":        [4321, 4032],

    # Additional connections visible on the map
    "BULLEEN_RD":     [2827, 4051],
    "HIGHBURY_RD":    [3685, 2846],  # SE Arterial near High St/Highbury
    "SWAN_ST":        [4812, 4262],  # Swan St connects to Bridge Rd area
    "WALMER_ST":      [4821, 4262],  # Walmer St near Bridge Rd/Burwood
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_adjacency(site_locations, road_network=None):
    """
    Build adjacency from the road network definition.

    For each road, consecutive sites are connected bidirectionally.
    Distance is computed via Haversine from site coordinates.

    Args:
        site_locations: {site_id: (lat, lon)}
        road_network: dict of road_name -> [site_ids in order]
                      defaults to ROAD_NETWORK

    Returns:
        adjacency: {site_id: [(neighbor_id, distance_km), ...]}
    """
    if road_network is None:
        road_network = ROAD_NETWORK

    adjacency = {site: [] for site in site_locations}

    for road_name, sites in road_network.items():
        # Filter to sites that exist in our dataset
        valid_sites = [s for s in sites if s in site_locations]
        if len(valid_sites) < 2:
            continue

        # Connect consecutive pairs
        for i in range(len(valid_sites) - 1):
            s1, s2 = valid_sites[i], valid_sites[i + 1]
            lat1, lon1 = site_locations[s1]
            lat2, lon2 = site_locations[s2]
            dist = round(haversine_distance(lat1, lon1, lat2, lon2), 4)

            # Add bidirectional edge (avoid duplicates)
            if not any(n == s2 for n, _ in adjacency[s1]):
                adjacency[s1].append((s2, dist))
            if not any(n == s1 for n, _ in adjacency[s2]):
                adjacency[s2].append((s1, dist))

    return adjacency


def verify_connectivity(adjacency):
    """
    Verify the graph is fully connected using BFS.

    Returns:
        (is_connected, components) where components is a list of sets.
    """
    all_nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        for n, _ in neighbors:
            all_nodes.add(n)

    if not all_nodes:
        return True, []

    visited = set()
    components = []

    for start in all_nodes:
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            for neighbor, _ in adjacency.get(node, []):
                if neighbor not in component:
                    queue.append(neighbor)
        visited |= component
        components.append(component)

    return len(components) == 1, components


def load_adjacency(filepath=None):
    """
    Load adjacency list from JSON file.

    Returns dict: {site_id(int): [(neighbor_id(int), distance_km(float)), ...]}
    """
    if filepath is None:
        config = load_config()
        base_dir = os.path.dirname(os.path.dirname(__file__))
        filepath = os.path.join(base_dir, config["data"]["adjacency_path"])

    with open(filepath, "r") as f:
        raw = json.load(f)

    adjacency = {}
    for site_str, neighbors in raw.items():
        site_id = int(site_str)
        adjacency[site_id] = [
            (int(nbr["neighbor"]), float(nbr["distance_km"]))
            for nbr in neighbors
        ]
    return adjacency


def save_adjacency(adjacency, filepath):
    """Save adjacency dict to JSON file."""
    json_data = {}
    for site_id, neighbors in sorted(adjacency.items()):
        json_data[str(site_id)] = [
            {"neighbor": n, "distance_km": d}
            for n, d in sorted(neighbors)
        ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2)


def build_traffic_graph(adjacency, predictions, site_locations):
    """
    Build a weighted graph where edge cost = estimated travel time.

    Args:
        adjacency: {site_id: [(neighbor_id, distance_km), ...]}
        predictions: {site_id: predicted_volume_15min}
        site_locations: {site_id: (lat, lon)}

    Returns:
        graph: {site_id: [(neighbor_id, travel_time_seconds), ...]}
        coords: {site_id: (lat, lon)}
    """
    graph = {}
    for site_id in adjacency:
        graph[site_id] = []
        for neighbor_id, distance_km in adjacency[site_id]:
            flow_15min = predictions.get(site_id, 0)
            flow_hourly = flow_15min_to_hourly(flow_15min)
            travel_time = compute_travel_time(distance_km, flow_hourly)
            graph[site_id].append((neighbor_id, travel_time))

    # Ensure all neighbor nodes exist in graph dict
    for neighbors in list(adjacency.values()):
        for neighbor_id, _ in neighbors:
            if neighbor_id not in graph:
                graph[neighbor_id] = []

    return graph, site_locations


def get_all_nodes(adjacency):
    """Get all node IDs from the adjacency graph."""
    nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        for n, _ in neighbors:
            nodes.add(n)
    return sorted(nodes)
