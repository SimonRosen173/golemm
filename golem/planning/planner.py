import itertools
import os
from typing import List, Tuple, Union, Set

import numpy as np
from abc import abstractmethod


def reconstruct_path(came_from, end_loc):
    path = [end_loc]
    curr_loc = end_loc
    while came_from[curr_loc] is not None:
        curr_loc = came_from[curr_loc]
        path.append(curr_loc)

    path.reverse()
    return path


class Planner:
    def __init__(
            self,
            grid: np.ndarray,
            n_agents: int,
            goals: Union[Set[Tuple[int, int]], List[Set[Tuple[int, int]]]],
            is_collisions=True,
            termination_type="agent_enter",

    ):
        self.grid = grid

        if type(goals) == set:
            goals = [goals for _ in range(n_agents)]
        self.goals: List[Set[Tuple[int, int]]] = goals

        # joint_goals = list(itertools.product(*goals))
        # if is_collisions:
        #     # Filter out joint goals with repeated elements
        #     joint_goals = list(filter(lambda x: len(x) == len(set(x)), joint_goals))
        # self.joint_goals = joint_goals

        self.n_agents = n_agents
        self.is_collisions = is_collisions
        self.termination_type = termination_type
        self.is_agent_enter = termination_type == "agent_enter"
        assert termination_type == "agent_enter" or termination_type == "agent_wait"

    def _get_is_valid(self, loc):
        y, x = loc
        h, w = self.grid.shape
        in_bounds = (0 <= y < h) and (0 <= x < w)
        # grid[y][x] == 0 -> no obstacle
        return in_bounds and self.grid[y][x] == 0

    def _get_valid_neighbours_1a(self, loc, goal=None, agent_index=None):
        if loc in self.goals and self.is_agent_enter:
            # Agent cannot leave goal once it has entered if term mode is agent_enter
            return []
        else:
            y, x = loc
            neighbours = []
            grid = self.grid
            height, width = grid.shape

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

            if self.is_agent_enter:
                assert goal is not None and agent_index is not None
                neighbours = list(filter(lambda el: (el in self.goals[agent_index] and el != goal), neighbours))

            return neighbours

    def _check_collision(self, prev_joint_loc: List[Tuple[int, int]], joint_loc: List[Tuple[int, int]]) -> bool:
        assert len(joint_loc) > 1, "length of joint_loc must be greater than 1"
        n_agents = len(joint_loc)
        if n_agents == 2:
            # same spot collision
            is_same_collision = joint_loc[0] == joint_loc[1]
            # pass through collision
            is_pass_collision = joint_loc[0] == prev_joint_loc[1] and joint_loc[1] == prev_joint_loc[0]
            return is_same_collision or is_pass_collision
        else:
            raise NotImplementedError

    def _remove_collisions(self, curr_joint_loc, next_joint_locs):
        if self.n_agents == 1:
            return next_joint_locs
        out_joint_locs = []

        assert len(curr_joint_loc) == self.n_agents

        for next_joint_loc in next_joint_locs:
            assert len(next_joint_loc) == self.n_agents

            if not self._check_collision(curr_joint_loc, next_joint_loc):
                out_joint_locs.append(next_joint_loc)

        return out_joint_locs

    def _get_valid_neighbours(self, joint_loc, joint_goal):
        n_agents = self.n_agents
        if n_agents == 1:
            return self._get_valid_neighbours_1a(joint_loc)
        else:
            all_neighbours = []
            for i, loc in enumerate(joint_loc):
                curr_neighbours = self._get_valid_neighbours_1a(loc)
                # should probs use filter cause its faster
                # curr_neighbours = list(filter(lambda x: not (x in self.goals[i] and x != joint_goal[i]),
                #                               curr_neighbours))

                # Agent cannot wait when on other goal - TODO: Test
                if loc not in self.goals[i] or loc == joint_goal[i]:
                    curr_neighbours.append(loc)
                all_neighbours.append(curr_neighbours)

            joint_neighbours = list(itertools.product(*all_neighbours))

            if self.is_collisions:
                joint_neighbours = self._remove_collisions(joint_loc, joint_neighbours)

            # joint_neighbour is not valid if all of its constituents are at a goal or
            # it is at desired joint_goal
            valid_joint_neighbours = []
            for joint_neighbour in joint_neighbours:
                is_all_at_goal = True
                for i, neighbour in enumerate(joint_neighbour):
                    if neighbour not in self.goals[i]:
                        is_all_at_goal = False
                        break
                if not (is_all_at_goal and joint_neighbour != joint_goal):
                    valid_joint_neighbours.append(joint_neighbour)

            return joint_neighbours

    @abstractmethod
    def shortest_path(self, *args, **kwargs):
        raise NotImplementedError


def test():
    from coop_decomp.environment.utils import load_map_file
    from coop_decomp.environment.gridworld import GridWorld, interactive

    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.split(base_path)[:-1]
    base_path = os.path.join(*base_path)

    map_file_path = os.path.join(base_path, "environment", "maps", "3_rooms_alt.txt")
    grid = load_map_file(map_file_path)

    goals = {
        (0, 0): True,  # BL
        (10, 0): True,  # TL
        (0, 10): True  # BR
    }
    goals_set = set(goals.keys())

    # env = GridWorld(map_file_path, 2, goals,
    #                 reward_out_mode="joint", terminate_mode="agent_enter",
    #                 is_collisions=True,
    #                 is_virt_term_state=False)
    # interactive(env)

    planner = Planner(grid, 2, goals_set)
    neighbours = planner._get_valid_neighbours([(1, 1), (1, 2)], )
    print(len(neighbours))
    print(neighbours)


if __name__ == "__main__":
    test()
