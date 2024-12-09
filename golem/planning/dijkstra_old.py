import itertools
import os
from typing import Tuple, Optional, Set

import numpy as np
import pygame

from coop_decomp.planning.structures import PriorityQueue

from coop_decomp.planning.planner import Planner
from coop_decomp.environment.utils import load_map_file


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(BASE_PATH)[:-1]
BASE_PATH = os.path.join(*BASE_PATH)


def get_is_valid(grid: np.ndarray, loc: Tuple[int, int]):
    y, x = loc
    h, w = grid.shape
    in_bounds = (0 <= y < h) and (0 <= x < w)
    # grid[y][x] == 0 -> no obstacle
    return in_bounds and grid[y][x] == 0


def get_valid_neighbours_sa(grid: np.ndarray, loc: Tuple[int, int]):
    neighbours = []
    height, width = grid.shape
    y, x = loc
    # UP
    if y > 0 and grid[y - 1][x] == 0:
        neighbours.append((y - 1, x))
    # DOWN
    if y + 1 < height and grid[y + 1][x] == 0:
        neighbours.append((y + 1, x))
    # LEFT
    if x > 0 and grid[y][x - 1] == 0:
        neighbours.append((y, x - 1))
    # RIGHT
    if x + 1 < width and grid[y][x + 1] == 0:
        neighbours.append((y, x + 1))

    return neighbours


def get_valid_neighbours_2a(grid, joint_pos):
    # wait action is allowed
    neighbours1 = get_valid_neighbours_sa(grid, joint_pos[0])
    neighbours2 = get_valid_neighbours_sa(grid, joint_pos[1])
    neighbours1.append(joint_pos[0])
    neighbours2.append(joint_pos[1])

    cand_joint_neighbours = list(itertools.product(neighbours1, neighbours2))
    joint_neighbours = []
    for cand_joint_neighbour in cand_joint_neighbours:
        if cand_joint_neighbour[0] == cand_joint_neighbour[1]:
            # Agents arrive in same spot collision
            continue
        elif cand_joint_neighbour[0] == joint_pos[1] and cand_joint_neighbour[1] == joint_pos[0]:
            # Pass through/head on collision
            continue
        elif cand_joint_neighbour == joint_pos:
            # Both agents can't wait
            continue
        else:
            joint_neighbours.append(cand_joint_neighbour)

    return joint_neighbours


def get_valid_neighbours_2a_sop(grid, joint_pos, joint_goal: Tuple[int, int], goals):
    # wait action is allowed
    if joint_pos[0] != joint_goal[0]:
        cand_neighbours = get_valid_neighbours_sa(grid, joint_pos[0])
        neighbours1 = []
        for neighbour in cand_neighbours:
            if neighbour == joint_goal[0] or neighbour not in goals:
                neighbours1.append(neighbour)

        neighbours1.append(joint_pos[0])
        neighbours1 = [(neighbour, 1) for neighbour in neighbours1]
    else:
        neighbours1 = [(joint_pos[0], 0)]

    if joint_pos[1] != joint_goal[1]:
        cand_neighbours = get_valid_neighbours_sa(grid, joint_pos[1])
        neighbours2 = []
        for neighbour in cand_neighbours:
            if neighbour == joint_goal[1] or neighbour not in goals:
                neighbours2.append(neighbour)

        neighbours2.append(joint_pos[1])
        neighbours2 = [(neighbour, 1) for neighbour in neighbours2]
    else:
        neighbours2 = [(joint_pos[1], 0)]

    # neighbours2 = get_valid_neighbours_sa(grid, joint_pos[1])
    # neighbours2.append(joint_pos[1])

    cand_joint_neighbours = list(itertools.product(neighbours1, neighbours2))
    cand_joint_neighbours = [((n1, n2), c1+c2) for (n1, c1), (n2, c2) in cand_joint_neighbours]

    joint_neighbours = []
    for cand_joint_neighbour, cost in cand_joint_neighbours:
        if cand_joint_neighbour[0] == cand_joint_neighbour[1]:
            # Agents arrive in same spot collision
            continue
        elif cand_joint_neighbour[0] == joint_pos[1] and cand_joint_neighbour[1] == joint_pos[0]:
            # Pass through/head on collision
            continue
        elif cand_joint_neighbour == joint_pos:
            # Both agents can't wait
            continue
        else:
            joint_neighbours.append((cand_joint_neighbour, cost))

    return joint_neighbours


def reconstruct_path(came_from, end_loc):
    path = [end_loc]
    curr_loc = end_loc
    while came_from[curr_loc] is not None:
        curr_loc = came_from[curr_loc]
        path.append(curr_loc)

    path.reverse()
    return path


def dijkstra_sa(grid, start, goal):
    frontier = PriorityQueue()
    frontier.push(start, 0)
    came_from = {start: None}
    costs = {start: 0}

    while not frontier.is_empty():
        curr_loc = frontier.pop()

        if curr_loc == goal:
            break

        valid_neighbours = get_valid_neighbours_sa(grid, curr_loc)

        for next_loc in valid_neighbours:
            new_cost = costs[curr_loc] + 1
            if next_loc not in costs or new_cost < costs[next_loc]:
                costs[next_loc] = new_cost
                priority = new_cost
                frontier.push(next_loc, priority)
                came_from[next_loc] = curr_loc

    path = reconstruct_path(came_from, goal)
    return path, costs[goal]


# Optimising sum of path lengths
def dijkstra_ma_sop(grid, joint_start, joint_goal, goals, n_agents):
    if n_agents == 1:
        raise ValueError("This method is not supported for single agent case")

    frontier = PriorityQueue()
    frontier.push(joint_start, 0)
    came_from = {joint_start: None}
    costs = {joint_start: 0}

    while not frontier.is_empty():
        curr_loc = frontier.pop()

        if curr_loc == joint_goal:
            break

        if n_agents == 2:
            valid_neighbours_costs = get_valid_neighbours_2a_sop(grid, curr_loc, joint_goal, goals)
        else:
            raise NotImplementedError
            # valid_neighbours = get_valid_neighbours_ma(grid, curr_loc)

        for next_loc, curr_cost in valid_neighbours_costs:
            new_cost = costs[curr_loc] + curr_cost
            if next_loc not in costs or new_cost < costs[next_loc]:
                costs[next_loc] = new_cost
                priority = new_cost
                frontier.push(next_loc, priority)
                came_from[next_loc] = curr_loc

    path = reconstruct_path(came_from, joint_goal)
    return path, costs[joint_goal]


# Optimising total path length
def dijkstra_ma_tot(grid, joint_start, joint_goal, n_agents):
    if n_agents == 1:
        raise ValueError("This method is not supported for single agent case")

    frontier = PriorityQueue()
    frontier.push(joint_start, 0)
    came_from = {joint_start: None}
    costs = {joint_start: 0}

    while not frontier.is_empty():
        curr_loc = frontier.pop()

        if curr_loc == joint_goal:
            break

        if n_agents == 2:
            valid_neighbours = get_valid_neighbours_2a(grid, curr_loc)
        else:
            raise NotImplementedError
            # valid_neighbours = get_valid_neighbours_ma(grid, curr_loc)

        for next_loc in valid_neighbours:
            new_cost = costs[curr_loc] + 1
            if next_loc not in costs or new_cost < costs[next_loc]:
                costs[next_loc] = new_cost
                priority = new_cost
                frontier.push(next_loc, priority)
                came_from[next_loc] = curr_loc

    path = reconstruct_path(came_from, joint_goal)
    return path, costs[joint_goal]


def test():
    from coop_decomp.environment.gridworld import GridWorld

    map_file_path = os.path.join(BASE_PATH, "environment", "maps", "3_rooms.txt")
    grid = load_map_file(map_file_path)

    joint_start = ((0, 8), (8, 0))
    # joint_goal = ((7, 1), (1,7))
    joint_goal = ((7, 1), (1,7))

    goals = {
        (1, 1): True,  # BL
        (7, 1): True,  # TL
        (1, 7): True   # BR
    }

    path, cost = dijkstra_ma_sop(grid, joint_start, joint_goal, goals, 2)

    env = GridWorld(map_file_path, 2, goals,
                    reward_out_mode="joint", terminate_mode="agent_enter",
                    is_collisions=True,
                    is_virt_term_state=False)

    # path = [[loc] for loc in path]

    env.animate_path(path)
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
    #             pygame.quit()


if __name__ == "__main__":
    test()
