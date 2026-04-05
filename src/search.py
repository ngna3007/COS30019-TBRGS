"""
Search algorithms for route finding.
Ported from Assignment 2A with bug fixes:
- IDS: replaced explored set with path-based cycle detection
- visit_all: replaced O(n!) permutations with Held-Karp DP
"""

import math
import heapq
from collections import deque


class Node:
    """Represents a node in the search tree."""
    _counter = 0

    def __init__(self, state, parent=None, g_cost=0, f_cost=0):
        self.state = state
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.order = Node._counter
        Node._counter += 1

    def __lt__(self, other):
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        if self.state != other.state:
            return self.state < other.state
        return self.order < other.order


def get_path(node):
    """Backtrack from goal node to build the path."""
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))


def path_cost(path, graph):
    """Calculate the total edge cost along a path."""
    total = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        for neighbor, cost in graph.get(u, []):
            if neighbor == v:
                total += cost
                break
    return total


def euclidean_distance(coords, node, goals):
    """Euclidean distance from node to nearest goal (heuristic)."""
    if node not in coords:
        return 0
    x1, y1 = coords[node]
    min_dist = float("inf")
    for goal in goals:
        if goal in coords:
            x2, y2 = coords[goal]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            min_dist = min(min_dist, dist)
    return min_dist


def haversine_heuristic(coords, node, goals):
    """
    Haversine-based heuristic for lat/lon coordinates.
    Returns estimated travel time in seconds assuming free-flow speed (60 km/hr).
    """
    if node not in coords:
        return 0
    lat1, lon1 = coords[node]
    min_time = float("inf")
    for goal in goals:
        if goal in coords:
            lat2, lon2 = coords[goal]
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            c = 2 * math.asin(math.sqrt(a))
            dist_km = R * c
            # Estimate time at free flow speed (60 km/hr)
            time_sec = (dist_km / 60.0) * 3600
            min_time = min(min_time, time_sec)
    return min_time


def get_neighbors(graph, state):
    """Get neighbors sorted by ascending node number."""
    return sorted(graph.get(state, []), key=lambda x: x[0])


# ============================================================
# Search Algorithms
# ============================================================

def dfs(graph, start, goals):
    """Depth-First Search."""
    Node._counter = 0
    frontier = [Node(start)]
    explored = set()
    num_nodes = 1

    while frontier:
        node = frontier.pop()
        if node.state in goals:
            return get_path(node), num_nodes, node.state
        if node.state in explored:
            continue
        explored.add(node.state)

        for neighbor, cost in reversed(get_neighbors(graph, node.state)):
            if neighbor not in explored:
                frontier.append(Node(neighbor, node))
                num_nodes += 1

    return None, num_nodes, None


def bfs(graph, start, goals):
    """Breadth-First Search."""
    Node._counter = 0
    frontier = deque([Node(start)])
    explored = set()
    num_nodes = 1

    while frontier:
        node = frontier.popleft()
        if node.state in goals:
            return get_path(node), num_nodes, node.state
        if node.state in explored:
            continue
        explored.add(node.state)

        for neighbor, cost in get_neighbors(graph, node.state):
            if neighbor not in explored:
                frontier.append(Node(neighbor, node))
                num_nodes += 1

    return None, num_nodes, None


def gbfs(graph, start, goals, coords):
    """Greedy Best-First Search."""
    Node._counter = 0
    heuristic = haversine_heuristic
    h_start = heuristic(coords, start, goals)
    frontier = [Node(start, f_cost=h_start)]
    explored = set()
    num_nodes = 1

    while frontier:
        node = heapq.heappop(frontier)
        if node.state in goals:
            return get_path(node), num_nodes, node.state
        if node.state in explored:
            continue
        explored.add(node.state)

        for neighbor, cost in get_neighbors(graph, node.state):
            if neighbor not in explored:
                h = heuristic(coords, neighbor, goals)
                heapq.heappush(frontier, Node(neighbor, node, f_cost=h))
                num_nodes += 1

    return None, num_nodes, None


def a_star(graph, start, goals, coords):
    """A* Search using f(n) = g(n) + h(n)."""
    Node._counter = 0
    heuristic = haversine_heuristic
    h_start = heuristic(coords, start, goals)
    frontier = [Node(start, g_cost=0, f_cost=h_start)]
    explored = set()
    g_best = {start: 0}
    num_nodes = 1

    while frontier:
        node = heapq.heappop(frontier)
        if node.state in goals:
            return get_path(node), num_nodes, node.state
        if node.state in explored:
            continue
        explored.add(node.state)

        for neighbor, cost in get_neighbors(graph, node.state):
            new_g = node.g_cost + cost
            if neighbor not in g_best or new_g < g_best[neighbor]:
                g_best[neighbor] = new_g
                h = heuristic(coords, neighbor, goals)
                f = new_g + h
                heapq.heappush(frontier, Node(neighbor, node, g_cost=new_g, f_cost=f))
                num_nodes += 1

    return None, num_nodes, None


def _is_on_path(node, state):
    """Check if state already appears on the path to this node (cycle detection)."""
    current = node
    while current:
        if current.state == state:
            return True
        current = current.parent
    return False


def _depth_limited_search(graph, start, goals, limit):
    """DFS with depth limit and path-based cycle detection (no explored set)."""
    frontier = [(Node(start), 0)]
    num_nodes = 1

    while frontier:
        node, depth = frontier.pop()
        if node.state in goals:
            return get_path(node), num_nodes, node.state
        if depth >= limit:
            continue
        # Path-based cycle detection instead of explored set
        if _is_on_path(node.parent, node.state):
            continue

        for neighbor, cost in reversed(get_neighbors(graph, node.state)):
            frontier.append((Node(neighbor, node), depth + 1))
            num_nodes += 1

    return None, num_nodes, None


def ids(graph, start, goals):
    """
    Iterative Deepening Search (CUS1).
    Fixed: uses path-based cycle detection instead of explored set.
    """
    Node._counter = 0
    num_nodes = 0
    max_depth = len(graph) + 1

    for depth_limit in range(0, max_depth):
        result, nodes_created, goal = _depth_limited_search(
            graph, start, goals, depth_limit
        )
        num_nodes += nodes_created
        if result is not None:
            return result, num_nodes, goal

    return None, num_nodes, None


def beam_search(graph, start, goals, coords, beam_width=2):
    """Beam Search (CUS2)."""
    Node._counter = 0
    heuristic = haversine_heuristic
    current_level = [Node(start, g_cost=0,
                          f_cost=heuristic(coords, start, goals))]
    explored = set()
    num_nodes = 1

    while current_level:
        for node in current_level:
            if node.state in goals:
                return get_path(node), num_nodes, node.state

        next_level = []
        for node in current_level:
            if node.state in explored:
                continue
            explored.add(node.state)

            for neighbor, cost in get_neighbors(graph, node.state):
                if neighbor not in explored:
                    h = heuristic(coords, neighbor, goals)
                    hops = node.g_cost + 1
                    child = Node(neighbor, node, g_cost=hops, f_cost=h)
                    next_level.append(child)
                    num_nodes += 1

        next_level.sort()
        current_level = next_level[:beam_width]

    return None, num_nodes, None


# ============================================================
# Held-Karp DP for visit_all (replaces O(n!) permutations)
# ============================================================
def visit_all(graph, start, destinations, coords):
    """
    Find the shortest path visiting ALL destinations using Held-Karp DP.
    Complexity: O(n^2 * 2^n) instead of O(n!).
    """
    if not destinations:
        return None, 0, 0

    if len(destinations) == 1:
        path, num_nodes, goal = a_star(graph, start, set(destinations), coords)
        if path:
            cost = path_cost(path, graph)
            return path, cost, num_nodes
        return None, 0, num_nodes

    all_points = [start] + list(destinations)
    n = len(all_points)
    total_nodes = 0

    # Precompute shortest paths between all pairs
    dist = {}
    path_cache = {}
    for i, src in enumerate(all_points):
        for j, dst in enumerate(all_points):
            if i == j:
                continue
            Node._counter = 0
            p, nodes, _ = a_star(graph, src, {dst}, coords)
            total_nodes += nodes
            if p:
                c = path_cost(p, graph)
                dist[(i, j)] = c
                path_cache[(i, j)] = p
            else:
                dist[(i, j)] = float("inf")
                path_cache[(i, j)] = None

    # Held-Karp DP
    # dp[S][i] = min cost to visit all nodes in set S, ending at node i
    # S is a bitmask of visited destination indices (1-indexed in all_points)
    dest_indices = list(range(1, n))
    full_mask = (1 << len(dest_indices)) - 1

    dp = {}
    parent_dp = {}

    # Initialize: start -> each destination
    for idx_pos, di in enumerate(dest_indices):
        mask = 1 << idx_pos
        dp[(mask, di)] = dist.get((0, di), float("inf"))
        parent_dp[(mask, di)] = -1

    # Fill DP
    for mask in range(1, full_mask + 1):
        for idx_pos, di in enumerate(dest_indices):
            if not (mask & (1 << idx_pos)):
                continue
            if (mask, di) not in dp:
                continue
            prev_mask = mask ^ (1 << idx_pos)
            if prev_mask == 0:
                continue
            for idx_pos2, dj in enumerate(dest_indices):
                if not (prev_mask & (1 << idx_pos2)):
                    continue
                if (prev_mask, dj) not in dp:
                    continue
                new_cost = dp[(prev_mask, dj)] + dist.get((dj, di), float("inf"))
                if new_cost < dp.get((mask, di), float("inf")):
                    dp[(mask, di)] = new_cost
                    parent_dp[(mask, di)] = dj

    # Find best ending node
    best_cost = float("inf")
    best_end = -1
    for idx_pos, di in enumerate(dest_indices):
        cost = dp.get((full_mask, di), float("inf"))
        if cost < best_cost:
            best_cost = cost
            best_end = di

    if best_cost == float("inf"):
        return None, 0, total_nodes

    # Reconstruct ordering
    ordering = []
    mask = full_mask
    current = best_end
    while current != -1:
        ordering.append(current)
        prev = parent_dp.get((mask, current), -1)
        idx_pos = dest_indices.index(current)
        mask = mask ^ (1 << idx_pos)
        current = prev

    ordering.reverse()

    # Build full path from cached A* paths
    stops = [0] + ordering
    full_path = None
    for k in range(len(stops) - 1):
        seg_path = path_cache.get((stops[k], stops[k + 1]))
        if seg_path is None:
            return None, 0, total_nodes
        if full_path is None:
            full_path = list(seg_path)
        else:
            full_path.extend(seg_path[1:])

    return full_path, best_cost, total_nodes
