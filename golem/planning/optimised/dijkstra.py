from typing import Set, List, Tuple

import numpy as np

from coop_decomp.planning.optimised import dijkstra_c

DTYPE_INT = np.int8


def dijkstra_shortest_path(
        n_agents: int,
        grid: np.ndarray,
        goals: Set[Tuple[int, int]],
        joint_start: Tuple[Tuple[int, int]],
        joint_goal: Tuple[Tuple[int, int]],
        is_term_agent_enter: bool
) -> Tuple[List[Tuple[Tuple[int, int]]], int]:
    assert type(n_agents) == int and n_agents is not None
    assert type(grid) == np.ndarray and grid is not None

    grid = grid.astype(DTYPE_INT)
    # goals = np.array([list(goal) for goal in goals], dtype=DTYPE_INT)
    # joint_start = np.array([list(el) for el in joint_start], dtype=DTYPE_INT)
    # joint_goal = np.array([list(el) for el in joint_goal], dtype=DTYPE_INT)
    is_term_agent_enter = 1 if is_term_agent_enter else 0

    path, costs = dijkstra_c.dijkstra_shortest_path(n_agents, grid, goals,
                                                    joint_start, joint_goal,
                                                    is_term_agent_enter)

    if n_agents == 1:
        path = [(tup, ) for tup in path]
    else:
        raise NotImplementedError
    # path = [tuple([tuple(loc) for loc in joint_loc]) for joint_loc in path]
    return path, costs


def test_1():
    grid = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    goals = {(2, 2)}
    start = ((0, 0), )
    goal = ((2, 2), )

    is_term_agent_enter = True
    path, cost = dijkstra_shortest_path(1, grid=grid, goals=goals,
                                        joint_start=start, joint_goal=goal,
                                        is_term_agent_enter=is_term_agent_enter)
    print(path)
    print(cost)


def test():
    from coop_decomp.environment.utils import load_config
    from coop_decomp.environment.gridworld import GridWorld, interactive

    config = load_config("5_rooms.json")
    grid = config["grid"]
    named_goals = config["named_goals"]
    named_starts = config["named_starts"]
    goals = {goal: True for goal in named_goals.values()}
    goals_set = set(goals.keys())
    rewards = config["rewards"]

    n_agents = 1

    env = GridWorld(
        grid, n_agents, goals,
        reward_out_mode="joint", terminate_mode="agent_enter",
        is_collisions=True,
        is_virt_term_state=False,
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"]
    )

    joint_start = (tuple(named_starts["L"]), )
    joint_goal = (tuple(named_goals["R"]), )

    env.reset(list(joint_start))

    # interactive(env)

    # planner = Dijkstra(grid, 2, goals_set)
    path, cost = dijkstra_shortest_path(n_agents=n_agents, grid=grid,
                                        goals=goals_set, joint_start=joint_start,
                                        joint_goal=joint_goal, is_term_agent_enter=True)

    print(cost)
    print(path)
    env.animate_path(path)


def benchmark():
    from coop_decomp.environment.utils import load_config
    from coop_decomp.environment.gridworld import GridWorld, interactive
    from coop_decomp.planning.dijkstra import Dijkstra
    from timeit import timeit

    n_repeats = 10000

    config = load_config("5_rooms.json")
    grid = config["grid"]
    named_goals = config["named_goals"]
    named_starts = config["named_starts"]
    goals = {goal: True for goal in named_goals.values()}
    goals_set = set(goals.keys())
    rewards = config["rewards"]

    n_agents = 1

    env = GridWorld(
        grid, n_agents, goals,
        reward_out_mode="joint", terminate_mode="agent_enter",
        is_collisions=True,
        is_virt_term_state=False,
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"]
    )

    joint_start = (tuple(named_starts["L"]),)
    joint_goal = (tuple(named_goals["R"]),)

    env.reset(list(joint_start))

    planner = Dijkstra(grid, n_agents, goals_set)

    # Python Implementation
    print("Python Implementation")
    py_path, py_cost = planner.shortest_path(joint_start, joint_goal)
    print(f"\tpath={py_path}")
    print(f"\tcost={py_cost}")
    py_elapsed = timeit(lambda: planner.shortest_path(joint_start, joint_goal), number=n_repeats)
    print(f"\n\tTime elapsed={py_elapsed}")

    # # py_elapsed = 0
    # for i in range(n_repeats):
    #     py_path, py_cost = planner.shortest_path(joint_start, joint_goal)

    # Cython Implementation
    print("Cython Implementation")
    cy_path, cy_cost = dijkstra_shortest_path(n_agents=n_agents, grid=grid,
                                              goals=goals_set, joint_start=joint_start,
                                              joint_goal=joint_goal, is_term_agent_enter=True)
    print(f"\tpath={cy_path}")
    print(f"\tcost={cy_cost}")
    # for i in range(n_repeats):
    #     cy_path, cy_cost = dijkstra_shortest_path(n_agents=n_agents, grid=grid,
    #                                               goals=goals_set, joint_start=joint_start,
    #                                               joint_goal=joint_goal, is_term_agent_enter=True)

    cy_elapsed = timeit(lambda: dijkstra_shortest_path(n_agents=n_agents, grid=grid,
                                                         goals=goals_set, joint_start=joint_start,
                                                         joint_goal=joint_goal, is_term_agent_enter=True),
                        number=n_repeats)
    print(f"\n\tTime elapsed={cy_elapsed}")

    # cy_elapsed = 0
    # for i in range(n_repeats):
    #     cy_path, cy_cost = dijkstra_shortest_path(n_agents=n_agents, grid=grid,
    #                                               goals=goals_set, joint_start=joint_start,
    #                                               joint_goal=joint_goal, is_term_agent_enter=True)


if __name__ == "__main__":
    benchmark()
    # test()