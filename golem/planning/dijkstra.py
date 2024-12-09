import os
from typing import Tuple, List

import numpy as np

from golem.planning.planner import Planner, reconstruct_path
from golem.planning.structures import PriorityQueue


class Dijkstra(Planner):
    def __init__(
            self,
            grid: np.ndarray,
            n_agents: int,
            goals,
            is_collisions=True,
            opt_path_mode="sop",
            termination_type="agent_enter"
            # **kwargs
    ):

        super().__init__(grid, n_agents, goals, is_collisions, termination_type) #, **kwargs)
        self.opt_path_mode = opt_path_mode

    def _shortest_path_1a(self, start, goal):
        frontier = PriorityQueue()
        frontier.push(start, 0)
        came_from = {start: None}
        costs = {start: 0}

        while not frontier.is_empty():
            curr_loc = frontier.pop()

            if curr_loc == goal:
                break

            valid_neighbours = self._get_valid_neighbours_1a(curr_loc)

            for next_loc in valid_neighbours:
                new_cost = costs[curr_loc] + 1
                if next_loc not in costs or new_cost < costs[next_loc]:
                    costs[next_loc] = new_cost
                    priority = new_cost
                    frontier.push(next_loc, priority)
                    came_from[next_loc] = curr_loc

        path = reconstruct_path(came_from, goal)
        return path, costs[goal]

    def _shortest_path_sop(self, joint_start, joint_goal):
        frontier = PriorityQueue()
        frontier.push(joint_start, 0)
        came_from = {joint_start: None}
        costs = {joint_start: 0}

        while not frontier.is_empty():
            curr_loc = frontier.pop()

            if curr_loc == joint_goal:
                break

            valid_neighbours = self._get_valid_neighbours(curr_loc, joint_goal)
            # calc costs
            # For each agent not already at a goal the cost is increased by one
            curr_cost = 0
            for i, loc in enumerate(curr_loc):
                if self.is_agent_enter:
                    if loc not in self.goals[i]:
                        curr_cost += 1
                else:
                    if loc != joint_goal[i]:
                        curr_cost += 1
            # if curr_cost == 0:
            #     # If curr joint loc is one where all agents are at goals
            #     valid_neighbours_costs = []
            # else:
            #     valid_neighbours_costs = [(neighbour, curr_cost) for neighbour in valid_neighbours]
            valid_neighbours_costs = [(neighbour, curr_cost) for neighbour in valid_neighbours]

            for next_loc, curr_cost in valid_neighbours_costs:
                new_cost = costs[curr_loc] + curr_cost
                if next_loc not in costs or new_cost < costs[next_loc]:
                    costs[next_loc] = new_cost
                    priority = new_cost
                    frontier.push(next_loc, priority)
                    came_from[next_loc] = curr_loc

        path = reconstruct_path(came_from, joint_goal)
        return path, costs[joint_goal]

    def shortest_path(self, joint_start, joint_goal) -> List[Tuple[Tuple[int,int], ...]]:
        if self.n_agents == 1:
            return self._shortest_path_1a(joint_start[0], joint_goal[0])
        else:
            if self.opt_path_mode == "sop":
                return self._shortest_path_sop(joint_start, joint_goal)
            else:
                raise NotImplementedError


def test():
    from golem.environment.gridworld.gridworld import GridWorld, interactive
    from golem.environment.utils import load_config

    # config = load_config("5_rooms_4g.json")
    config = load_config("4x4.json")
    grid = config["grid"]
    named_goals = config["named_goals"]
    named_starts = config["named_starts"]
    goals = {goal: True for goal in named_goals.values()}
    goals_set = set(goals.keys())
    rewards = config["rewards"]

    n_agents = 2

    env = GridWorld(
        config["grid"], n_agents, goals,
        reward_out_mode="sum", terminate_mode="agent_wait",
        is_collisions=True,
        is_virt_term_state=False,
        max_episode_steps=config["max_episode_steps"],
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"], r_min=rewards["r_min"],
    )

    # env = GridWorld(
    #     grid, n_agents, goals,
    #     reward_out_mode="joint", terminate_mode="agent_enter",
    #     is_collisions=True,
    #     is_virt_term_state=False,
    #     desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
    #     step_reward=rewards["step"]
    # )

    # joint_start = ((10, 0), (0, 10))
    joint_start = (named_starts["BL"], named_starts["BR"])
    joint_goal = (named_goals["TL"], named_goals["TR"])
    print(f"Joint start: {joint_start}")
    print(f"Joint goal: {joint_goal}")

    env.reset(list(joint_start))

    # interactive(env)
    # exit()

    planner = Dijkstra(grid, 2, goals_set)
    path, cost = planner.shortest_path(joint_start, joint_goal)

    print(f"Cost: {cost}")
    env.animate_path(path)

    # neighbours = planner._get_valid_neighbours([(1, 1), (1, 2)])
    # print(len(neighbours))
    # print(neighbours)


if __name__ == "__main__":
    test()

