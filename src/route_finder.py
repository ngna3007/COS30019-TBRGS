"""
Top-K route finding using Yen's K-Shortest Paths algorithm.
Integrates ML predictions with graph search for end-to-end TBRGS.
"""

import copy
import math
from .search import a_star, path_cost, Node
from .graph_builder import build_traffic_graph
from .traffic_conversion import flow_15min_to_hourly, compute_travel_time


def yen_k_shortest_paths(graph, coords, origin, destination, k=5):
    """
    Find up to k shortest loopless paths using Yen's algorithm.

    Args:
        graph: {node: [(neighbor, cost), ...]}
        coords: {node: (lat, lon)}
        origin: start node
        destination: goal node
        k: number of paths to find

    Returns:
        list of (path, cost) tuples, sorted by cost
    """
    if origin == destination:
        return [([origin], 0.0)]

    # Find the shortest path
    result = a_star(graph, origin, {destination}, coords)
    path_1, _, _ = result
    if path_1 is None:
        return []

    cost_1 = path_cost(path_1, graph)
    A = [(path_1, cost_1)]  # Confirmed shortest paths
    B = []  # Candidate paths (min-heap by cost)

    for ki in range(1, k):
        prev_path = A[ki - 1][0]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]
            root_cost = path_cost(root_path, graph) if len(root_path) > 1 else 0

            # Create a modified graph: remove edges that would duplicate existing paths
            removed_edges = []
            for (path_a, _) in A:
                if len(path_a) > i and path_a[:i + 1] == root_path:
                    # Remove the edge from spur_node to the next node in this path
                    if i + 1 < len(path_a):
                        next_node = path_a[i + 1]
                        removed_edges.append((spur_node, next_node))

            # Also remove nodes in root_path (except spur_node) to prevent loops
            removed_nodes = set(root_path[:-1])

            # Build modified graph
            mod_graph = {}
            for node, neighbors in graph.items():
                if node in removed_nodes:
                    continue
                mod_graph[node] = [
                    (n, c) for n, c in neighbors
                    if n not in removed_nodes and (node, n) not in removed_edges
                ]

            # Find spur path
            spur_result = a_star(mod_graph, spur_node, {destination}, coords)
            spur_path, _, _ = spur_result
            if spur_path is None:
                continue

            # Combine root + spur
            total_path = root_path[:-1] + spur_path
            total_cost = path_cost(total_path, graph)

            # Add to candidates if not already found
            is_duplicate = any(p == total_path for p, _ in A) or \
                           any(p == total_path for p, _ in B)
            if not is_duplicate:
                B.append((total_path, total_cost))

        if not B:
            break

        # Select the best candidate
        B.sort(key=lambda x: x[1])
        A.append(B.pop(0))

    return A


def find_routes(origin, destination, time_index, predictions,
                adjacency, site_locations, k=5):
    """
    End-to-end route finding with ML-predicted travel times.

    Args:
        origin: SCATS site number (int)
        destination: SCATS site number (int)
        time_index: datetime or index for which to get predictions
        predictions: {site_id: predicted_volume_15min}
        adjacency: {site_id: [(neighbor_id, distance_km), ...]}
        site_locations: {site_id: (lat, lon)}
        k: number of routes to return

    Returns:
        list of dicts: [{
            "path": [site_id, ...],
            "travel_time_seconds": float,
            "travel_time_minutes": float,
            "distance_km": float,
            "num_intersections": int,
        }, ...]
    """
    # Build traffic-weighted graph
    graph, coords = build_traffic_graph(adjacency, predictions, site_locations)

    # Verify origin and destination exist
    if origin not in graph:
        return [{"error": f"Origin site {origin} not found in graph"}]
    if destination not in graph:
        return [{"error": f"Destination site {destination} not found in graph"}]

    # Find top-k paths
    routes = yen_k_shortest_paths(graph, coords, origin, destination, k)

    results = []
    for path, travel_time in routes:
        # Calculate total distance
        total_dist = 0
        for i in range(len(path) - 1):
            for nbr, dist in adjacency.get(path[i], []):
                if nbr == path[i + 1]:
                    total_dist += dist
                    break

        results.append({
            "path": path,
            "travel_time_seconds": travel_time,
            "travel_time_minutes": round(travel_time / 60, 2),
            "distance_km": round(total_dist, 3),
            "num_intersections": len(path),
        })

    return results


def format_route_display(route, site_descriptions=None):
    """Format a route for display."""
    path = route["path"]
    if site_descriptions:
        path_str = " -> ".join(
            f"{s} ({site_descriptions.get(s, '')})" for s in path
        )
    else:
        path_str = " -> ".join(str(s) for s in path)

    return (
        f"Route: {path_str}\n"
        f"Travel Time: {route['travel_time_minutes']:.1f} min "
        f"({route['travel_time_seconds']:.0f} sec)\n"
        f"Distance: {route['distance_km']:.2f} km\n"
        f"Intersections: {route['num_intersections']}"
    )
