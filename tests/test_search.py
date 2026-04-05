"""Tests for search algorithms (ported from 2A with fixes)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.search import bfs, dfs, a_star, ids, path_cost


def make_test_graph():
    """Create a small test graph for search validation."""
    # Diamond graph: 1->2, 1->3, 2->4, 3->4
    graph = {
        1: [(2, 1), (3, 2)],
        2: [(4, 1)],
        3: [(4, 1)],
        4: [],
    }
    coords = {1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0)}
    return graph, coords


def make_ids_test_graph():
    """
    Graph where explored-set IDS would fail:
    1->2 (cost 1), 1->3 (cost 1), 2->3 (cost 1), 3->5 (cost 1)
    IDS with explored set might miss the shorter path 1->3->5
    because it adds 3 to explored when visiting via 1->2->3
    """
    graph = {
        1: [(2, 1), (3, 1)],
        2: [(3, 1)],
        3: [(5, 1)],
        5: [],
    }
    return graph


def test_bfs_finds_shortest_path():
    """Test 25: BFS finds the shallowest (fewest edges) path."""
    graph, coords = make_test_graph()
    path, num_nodes, goal = bfs(graph, 1, {4})
    assert path is not None
    assert len(path) == 3  # 1->2->4 or 1->3->4 (both 2 edges)
    assert path[0] == 1
    assert path[-1] == 4


def test_astar_finds_optimal_path():
    """Test 26: A* finds the lowest cost path."""
    graph, coords = make_test_graph()
    path, num_nodes, goal = a_star(graph, 1, {4}, coords)
    assert path == [1, 2, 4]  # cost 2, vs [1,3,4] cost 3
    cost = path_cost(path, graph)
    assert cost == 2


def test_ids_fixed_finds_shortest():
    """Test 27: IDS with fixed cycle detection finds the shortest path."""
    graph = make_ids_test_graph()
    path, num_nodes, goal = ids(graph, 1, {5})
    assert path is not None
    assert path == [1, 3, 5], f"IDS should find 1->3->5 (depth 2), got {path}"


def test_ids_no_explored_set_bug():
    """Test 28: IDS does not skip valid shorter paths due to explored set."""
    # Graph: A->B->C->D and A->C->D
    # Old IDS with explored set: visits C via A->B->C at depth 2,
    # adds C to explored, then can't find A->C at depth 1 on next iteration
    graph = {
        1: [(2, 1), (3, 1)],
        2: [(3, 1)],
        3: [(4, 1)],
        4: [],
    }
    path, _, _ = ids(graph, 1, {4})
    assert path is not None
    # Should find path of length 3 (depth 2): 1->3->4
    assert len(path) == 3, f"Expected path length 3, got {len(path)}: {path}"
    assert path == [1, 3, 4] or path == [1, 2, 4], \
        f"Expected [1,3,4] or [1,2,4], got {path}"


def test_search_unreachable_goal():
    """Test 29: All search algorithms return None for unreachable goals."""
    graph = {1: [(2, 1)], 2: [], 3: []}
    coords = {1: (0, 0), 2: (1, 0), 3: (2, 0)}

    path, _, _ = bfs(graph, 1, {3})
    assert path is None

    path, _, _ = dfs(graph, 1, {3})
    assert path is None

    path, _, _ = a_star(graph, 1, {3}, coords)
    assert path is None

    path, _, _ = ids(graph, 1, {3})
    assert path is None
