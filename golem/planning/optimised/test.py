# PyCharm doesn't support Cython apparently
from coop_decomp.planning.optimised.dijkstra_c import test_np_arr, test_get_valid_neighbours_1a, dijkstra_shortest_path

import numpy as np
DTYPE_INT = np.int8


def test_1a_neighbours():
    grid = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=DTYPE_INT)
    goals = np.zeros((4, 2), dtype=DTYPE_INT)
    loc = np.array([1, 1], dtype=DTYPE_INT)
    is_term_agent_enter = 1
    valid_neighbours = test_get_valid_neighbours_1a(grid, goals, loc, is_term_agent_enter)
    print(valid_neighbours)


def test_shortest_path_1a():
    grid = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=DTYPE_INT)
    goals = np.array([[2, 2]], dtype=DTYPE_INT)
    loc = np.array([[0, 0]], dtype=DTYPE_INT)
    goal = np.array([[2, 2]], dtype=DTYPE_INT)
    is_term_agent_enter = 1
    path, cost = dijkstra_shortest_path(1, grid=grid, goals=goals,
                                        joint_start=loc, joint_goal=goal,
                                        is_term_agent_enter=is_term_agent_enter)
    print(path)
    print(cost)


if __name__ == "__main__":
    test_shortest_path_1a()
    # arr = np.arange(16, dtype=DTYPE_INT)
    # tmp = test_np_arr(arr)
    # print(arr, tmp)

