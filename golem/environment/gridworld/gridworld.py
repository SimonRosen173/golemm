import copy
import itertools
from enum import IntEnum
from typing import SupportsFloat, Any, Set, Dict, List, Optional, Union, Tuple
import sys
import random

import gymnasium as gym
import numpy as np
import seaborn as sns
import pygame
from gymnasium.core import ActType, ObsType, RenderFrame

from golem.environment.utils import load_map_file, load_config
from golem.planning.dijkstra import Dijkstra


# from coop_decomp.planning.dijkstra import dijkstra_ma_sop, dijkstra_sa


class GridWorld(gym.Env):
    def __init__(
            self,
            grid: np.ndarray,
            n_agents: int,
            goals: Union[List[Dict[Tuple[int, int], bool]], Dict[Tuple[int, int], bool]],
            is_collisions: bool = True,
            step_reward: float = -0.1,
            desirable_reward: float = 10,
            undesirable_reward: float = -10,
            r_min: Optional[float] = None,
            terminate_mode: str = "agent_enter",  # agent_enter, agent_wait
            max_episode_steps: int = np.inf,
            flatten_state: bool = False,
            is_virt_term_state: bool = False,
            reward_out_mode: str = "sum",  # sum, joint
            # Rendering kwargs
            window_width=512,
            margin_width=10,
            render_mode="human",
            font_size=16,
            agent_colors: Optional[List[str]] = None,
            config: Optional[Dict] = None
    ):
        # Get config stuff
        all_args = locals().copy()
        del all_args["self"]
        del all_args["config"]
        self.config = all_args
        if type(self.config["goals"]) == list:
            self.config["goals"] = [{str(key): val for key, val in agent_goals.items()}
                                    for agent_goals in self.config["goals"]]
        elif type(self.config["goals"]) == dict:
            self.config["goals"] = {str(key): val for key, val in self.config["goals"].items()}
        else:
            raise TypeError
        self.config["n_goals"] = len(self.config["goals"])
        if config is not None:
            assert type(config) == dict
            for key, val in config.items():
                self.config[key] = val

        self.n_agents = n_agents
        self.step_reward = step_reward
        self.desirable_reward = desirable_reward
        self.undesirable_reward = undesirable_reward
        assert self.undesirable_reward < 0
        if r_min is None:
            self.r_min = self.undesirable_reward * 5
        else:
            self.r_min = r_min
        self.r_max = self.desirable_reward

        #########
        # GOALS #
        # ----- #
        if type(goals) == list:
            self.goals: List[Dict[Tuple[int, int], bool]] = goals
        elif type(goals) == dict:
            self.goals: List[Dict[Tuple[int, int], bool]] = [goals for _ in range(n_agents)]
        else:
            raise ValueError

        joint_goals = list(itertools.product(*self.goals))
        if is_collisions:
            # Filter out joint goals with repeated elements
            joint_goals = list(filter(lambda x: len(x) == len(set(x)), joint_goals))
        self.joint_goals = joint_goals
        # ----- #
        # GOALS #
        #########

        self.max_episode_steps = max_episode_steps
        if max_episode_steps != np.inf:
            if self.step_reward * max_episode_steps < self.r_min:
                print("WARNING: Accumulated return from steps may be less than r_min")
        self.curr_episode_steps = 0

        self.is_collisions = is_collisions
        self.flatten_state = flatten_state

        if reward_out_mode == "sum":
            self._reward_out_mode = RewardMode.SUM
        elif reward_out_mode == "joint":
            self._reward_out_mode = RewardMode.JOINT
        else:
            raise ValueError(f"Value of {reward_out_mode} for reward_out_mode is not supported")

        if terminate_mode == "agent_enter":
            self._terminate_mode = TerminateMode.AGENT_ENTER
        elif terminate_mode == "agent_wait":
            self._terminate_mode = TerminateMode.AGENT_WAIT
        else:
            raise ValueError(f"{terminate_mode} is not a valid value for terminate_mode")
        # self.terminate_mode = terminate_mode
        # if self.terminate_mode not in ["agent_enter", "agent_wait"]:
        #     raise ValueError(f"{self.terminate_mode} is not a valid value for terminate_mode")

        if is_virt_term_state and self._terminate_mode == TerminateMode.AGENT_ENTER:
            raise ValueError(f"is_virt_term_state=True is not supported for terminate_mode={terminate_mode}")
        self.is_virt_term_state = is_virt_term_state
        self.vert_term_state = (-1, -1)

        self.n_agent_actions = 5
        self.n_actions = self.n_agent_actions ** self.n_agents

        self.is_dones: np.ndarray = np.zeros(n_agents).astype(bool)

        # self.grid = load_map_file(map_file_path)
        self.grid = grid
        self._grid_height, self._grid_width = self.grid.shape
        # valid states for an agent to occupy
        self._valid_states = list(map(tuple, np.argwhere(self.grid == 0)))
        self.n_valid_states: int = len(self._valid_states)

        # valid start states - excludes goals as valid start states
        self._valid_start_states: List[Set[Tuple[int, int]]] = []
        if terminate_mode == "agent_enter":
            for i in range(n_agents):
                agent_valid_start_states = set(self._valid_states) - set(self.goals[i].keys())
                self._valid_start_states.append(agent_valid_start_states)
        else:
            self._valid_start_states = [set(self._valid_states) for _ in range(self.n_agents)]

        # self._valid_start_states = list(set(self._valid_states) - set(goals.keys()))
        self.n_valid_start_states: List[int] = [len(self._valid_start_states[i]) for i in range(n_agents)]
        # self.reset()

        # (y, x)
        self._curr_joint_pos = [(0, 0) for _ in range(n_agents)]
        self._act_to_move_map: Dict[int, Tuple[int, int]] = {
            Action.NORTH: (1, 0),
            Action.SOUTH: (-1, 0),
            Action.EAST: (0, 1),
            Action.WEST: (0, -1),
            Action.STAY: (0, 0),
        }

        ##################
        # RENDERING VARS #
        ##################
        self.render_mode = render_mode
        self._margin_width = margin_width
        self._window_width = window_width

        # self._max_grid_dim = max(self._grid_width, self._grid_height)
        # self._tile_size = (self.window_size - 2 * self._margin_width) / self._max_grid_dim
        self._tile_size = (window_width - 2 * self._margin_width) / self._grid_width
        self._window_height = self._tile_size * self._grid_height + 2 * self._margin_width
        # self.window_size = (window_width, window_height)  # TODO: Check if correct
        self._window_dims = (self._window_width, self._window_height)

        self._grid_canvas_width = self._grid_width * self._tile_size
        self._grid_canvas_height = self._grid_height * self._tile_size

        self.pygame_font = None
        self.font_size = font_size

        if agent_colors is None:
            agent_colors = sns.color_palette("hls", self.n_agents)
        else:
            assert type(agent_colors) == list
            assert len(agent_colors) == n_agents
            import matplotlib.colors as colors
            agent_colors = [colors.to_rgb(color_name) for color_name in agent_colors]

        agent_colors = [(r * 255, g * 255, b * 255) for r, g, b in agent_colors]

        self.agent_colors = agent_colors

        self.window = None
        ##################

        ###########################
        des_reward = self.desirable_reward
        undes_reward = self.undesirable_reward
        self._goal_rewards: List[Dict[Tuple[int, int]], SupportsFloat] = []
        for i in range(n_agents):
            self._goal_rewards.append({goal: des_reward if is_des else undes_reward for goal, is_des
                                       in self.goals[i].items()})

        ############
        # PLANNING #
        ############
        goal_locations = [set(agent_goals.keys()) for agent_goals in self.goals]
        self.dijkstra = Dijkstra(self.grid, self.n_agents, goals=goal_locations,
                                 is_collisions=self.is_collisions, termination_type=terminate_mode)

    @staticmethod
    def create_env_from_config(config_path, **override_kwargs):
        raise NotImplementedError

    # NOTE: You can only modify desirability of goals
    def modify_goal_values(
            self,
            goals: Union[List[Dict[Tuple[int, int], bool]], Dict[Tuple[int, int], bool]]
    ):
        des_reward = self.desirable_reward
        undes_reward = self.undesirable_reward
        self._goal_rewards: List[Dict[Tuple[int, int]], SupportsFloat] = []
        if type(goals) == dict:
            goals = [goals for _ in range(self.n_agents)]

        # Ensure only values are being changed
        for i, (new_agent_goals, old_agent_goals) in enumerate(zip(goals, self.goals)):
            if set(new_agent_goals.keys()) != set(old_agent_goals.keys()):
                raise ValueError(f"Keys of goals cannot be changed. goals[{i}].keys() != self.goals[{i}].keys()")

        self.goals = goals

        for i in range(self.n_agents):
            self._goal_rewards.append({goal: des_reward if is_des else undes_reward for goal, is_des
                                       in self.goals[i].items()})

    # noinspection PyMethodOverriding
    def reset(
            self,
            joint_start_state: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        info = {}
        self.curr_episode_steps = 0
        if joint_start_state is not None:
            self._curr_joint_pos = joint_start_state
        else:
            n_agents = self.n_agents
            curr_joint_pos = []
            for i in range(n_agents):
                curr_valid_states = self._valid_start_states[i]
                if self.is_collisions:
                    # If there is collisions remove already taken states from potential states to sample
                    curr_valid_states = curr_valid_states - set(curr_joint_pos)
                curr_state = random.sample(list(curr_valid_states), 1)[0]
                curr_joint_pos.append(curr_state)

            self._curr_joint_pos = curr_joint_pos

            # is_replace = not self.is_collisions
            # inds = np.random.choice(list(range(len(self._valid_start_states))), replace=is_replace, size=self.n_agents)
            # self._curr_joint_pos = [self._valid_start_states[i] for i in inds]  # Can optimise

        self.is_dones = np.zeros(self.n_agents, dtype=bool)

        return self._curr_joint_pos, info

    ############
    # DYNAMICS #
    ############
    def step(
            self,
            joint_action: Union[int, List[int]]
    ) -> tuple[Union[np.ndarray, List[Tuple[int, int]]], Union[float, np.ndarray], np.ndarray, bool, dict[str, Any]]:

        out_joint_pos, reward, is_dones, info = self._take_joint_action(joint_action)
        is_truncated = self.curr_episode_steps > self.max_episode_steps
        self.curr_episode_steps += 1
        # TODO: Make is_truncated actually do something to env

        return out_joint_pos, reward, self.is_dones, is_truncated, info

    @staticmethod
    def _check_is_collision_2a(prev_joint_pos, cand_next_joint_pos):
        pos11, pos12 = prev_joint_pos
        pos21, pos22 = cand_next_joint_pos

        is_collide = pos21 == pos22 or (pos11 == pos22 and pos12 == pos21)

        if is_collide:
            next_joint_pos = prev_joint_pos
        else:
            next_joint_pos = cand_next_joint_pos

        return is_collide, next_joint_pos

    @staticmethod
    def _check_is_collision_ma_old(
            prev_joint_pos: np.ndarray,
            cand_next_joint_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        :param prev_joint_pos:
        :param cand_next_joint_pos:
        :return
            - cand_next_joint_pos
            - is_no_ops
            - n_collisions
        """
        assert prev_joint_pos.shape == cand_next_joint_pos.shape
        return np.array([1]), np.array([1]), 1

    @staticmethod
    def _check_is_collision_ma(
            prev_joint_pos: List[Tuple[int, int]],
            cand_next_joint_pos: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[bool], int, str]:
        is_collisions = [False for _ in range(len(prev_joint_pos))]
        info = ""
        n_agents = len(prev_joint_pos)

        all_nodes = set(prev_joint_pos).union(set(cand_next_joint_pos))
        out_edges = {node: (-1, -1) for node in all_nodes}
        in_edges = {node: set() for node in all_nodes}

        for i in range(n_agents):
            out_edges[prev_joint_pos[i]] = cand_next_joint_pos[i]
            if cand_next_joint_pos[i] in in_edges:
                in_edges[cand_next_joint_pos[i]].add(prev_joint_pos[i])
            else:
                in_edges[cand_next_joint_pos[i]] = {prev_joint_pos[i]}

        # pass through collisions
        for node in out_edges:
            if out_edges[node] in in_edges[node] and not out_edges[node] == node:
                # Is pass through collision
                other_node = out_edges[node]
                out_edges[node] = node
                out_edges[other_node] = other_node
                in_edges[node].add(node)
                in_edges[other_node].add(other_node)

                in_edges[node].remove(other_node)
                in_edges[other_node].remove(node)
                # print(f"Pass through collision at {node} & {out_edges[node]}")
                # pass

        # problem_nodes = nodes with multiple in_nodes
        problem_nodes = set()
        for node in in_edges:
            if len(in_edges[node]) > 1:
                problem_nodes.add(node)

        # i = 0
        while len(problem_nodes) > 0:
            curr_node = problem_nodes.pop()
            tmp = copy.copy(in_edges[curr_node])
            for in_node in tmp:
                # i += 1
                if out_edges[in_node] == in_node:
                    pass
                else:
                    problem_nodes.add(in_node)
                    in_edges[curr_node].remove(in_node)
                    out_edges[in_node] = in_node
                    in_edges[in_node].add(in_node)

        n_collisions = 0
        next_joint_pos = []
        for i in range(n_agents):
            next_joint_pos.append(out_edges[prev_joint_pos[i]])
            if cand_next_joint_pos[i] != next_joint_pos[i]:
                n_collisions += 1
                is_collisions[i] = True

        return next_joint_pos, is_collisions, n_collisions, info

    def _is_pos_valid(self, y, x):
        in_bounds = (0 <= x < self._grid_width) and (0 <= y < self._grid_height)
        if not in_bounds:
            return False
        else:
            return self.grid[y, x] == 0

    def _take_joint_action(
            self,
            joint_action: List[int]
    ) -> Tuple[List[Tuple[int, int]], Union[float, np.ndarray], np.ndarray, Dict[str, Any]]:
        n_agents = self.n_agents

        is_just_dones = np.zeros(n_agents, dtype=bool)
        # is_dones = self.is_dones.copy()

        info = {"is_collision": False}
        if self._reward_out_mode == RewardMode.JOINT:
            reward = np.zeros(n_agents, dtype=float)
        elif self._reward_out_mode == RewardMode.SUM:
            reward = 0
        else:
            raise NotImplementedError
        joint_action_taken = joint_action.copy()
        cand_joint_pos = [(-1, -1) for _ in range(n_agents)]

        # Agent Movement Dynamics
        for i in range(n_agents):
            curr_pos = self._curr_joint_pos[i]
            if not self.is_dones[i]:
                action = joint_action[i]
                move_tup = self._act_to_move_map[action]
                cand_pos = curr_pos[0] + move_tup[0], curr_pos[1] + move_tup[1]
                if not self._is_pos_valid(cand_pos[0], cand_pos[1]):
                    cand_pos = curr_pos
                    joint_action_taken[i] = Action.NOOP
            else:
                cand_pos = curr_pos
                joint_action_taken[i] = Action.NOOP

            cand_joint_pos[i] = cand_pos

        # Collision Dynamics
        if self.is_collisions:
            # TODO: Check - this was done at 1 AM
            if n_agents == 1:
                pass
            elif n_agents == 2:
                is_collide, cand_joint_pos = self._check_is_collision_2a(self._curr_joint_pos, cand_joint_pos)
                if is_collide:
                    info["is_collision"] = True
                    for i in range(n_agents):
                        if joint_action_taken[i] != Action.STAY:
                            joint_action_taken[i] = Action.NOOP
            else:
                cand_joint_pos, is_collisions, n_collisions, _ = self._check_is_collision_ma(self._curr_joint_pos,
                                                                                             cand_joint_pos)
                if n_collisions != 0:
                    info["is_collision"] = True
                    info["is_collisions"] = is_collisions
                    for i, is_collision in enumerate(is_collisions):
                        if is_collision:
                            joint_action_taken[i] = Action.NOOP

        # Goal Dynamics
        for i, cand_pos in enumerate(cand_joint_pos):
            # Note: Agent reward of 0 if it has already terminated
            if not self.is_dones[i]:
                if cand_pos in self.goals[i]:
                    if self._terminate_mode == TerminateMode.AGENT_WAIT:
                        is_agent_terminate = joint_action[i] == Action.STAY
                    elif self._terminate_mode == TerminateMode.AGENT_ENTER:
                        is_agent_terminate = True
                    else:
                        raise ValueError
                else:
                    is_agent_terminate = False

                # if (cand_pos in self.goals and self.terminate_mode == TerminateMode.AGENT_WAIT
                #     and joint_action[i] == Action.STAY) or \
                #         (cand_pos in self.goals and self.terminate_mode == TerminateMode.AGENT_ENTER):
                if is_agent_terminate:
                    self.is_dones[i] = True
                    is_just_dones[i] = True
                    term_reward = self.desirable_reward if self.goals[i][cand_pos] else self.undesirable_reward
                    if self._reward_out_mode == RewardMode.JOINT:
                        reward[i] = term_reward
                    elif self._reward_out_mode == RewardMode.SUM:
                        reward += term_reward
                    else:
                        raise NotImplementedError
                else:
                    # reward += self.step_reward
                    if self._reward_out_mode == RewardMode.JOINT:
                        reward[i] = self.step_reward
                    elif self._reward_out_mode == RewardMode.SUM:
                        reward += self.step_reward
                    else:
                        raise NotImplementedError

        self._curr_joint_pos = cand_joint_pos

        if self.is_virt_term_state:
            out_joint_pos = self._curr_joint_pos.copy()
            for i in range(n_agents):
                if self.is_dones[i] and not is_just_dones[i]:
                    out_joint_pos[i] = self.vert_term_state
        else:
            out_joint_pos = self._curr_joint_pos

        return out_joint_pos, reward, self.is_dones, info
    ############

    ###################
    # CALC OPT RETURN #
    ###################
    def calc_opt_joint_return_steps(
            self,
            joint_start,
            joint_goal=None
    ) -> Tuple[float, int, Dict[str, Any]]:
        if self.terminate_mode != "agent_wait":
            raise NotImplementedError
        # if self

        if joint_goal is not None:
            goal_rewards = self._goal_rewards
            path, cost = self.dijkstra.shortest_path(joint_start, joint_goal)
            eps_return = cost * self.step_reward
            info = {"path": path}

            for i, goal in enumerate(joint_goal):
                eps_return += goal_rewards[i][goal]
            tot_steps = cost + self.n_agents  # To account for wait actions
            return eps_return, tot_steps, info
        else:
            best_return = -np.inf
            best_steps = np.inf
            best_path = None
            best_joint_goal = None

            for joint_goal in self.joint_goals:
                curr_return, curr_steps, curr_info = self.calc_opt_joint_return_steps(joint_start, joint_goal)
                if curr_return > best_return:
                    best_return = curr_return
                    best_steps = curr_steps
                    best_joint_goal = joint_goal
                    best_path = curr_info["path"]

            info = {"best_path": best_path, "best_joint_goal": best_joint_goal}

            return best_return, best_steps, info

    # def calc_opt_return_steps_old(self, joint_start, joint_goal=None):
    #     # raise NotImplementedError
    #     if joint_goal is not None:
    #         # TODO: Check if joint_goal is valid
    #         n_agents = self.n_agents
    #         if n_agents == 1:
    #             goal = joint_goal[0]
    #             path, _ = dijkstra_sa(self.grid, joint_start[0], goal)
    #             n_steps = len(path) - 1  # Excluding start
    #             term_reward = self.desirable_reward if self.goals[goal] else self.undesirable_reward
    #             opt_return = self.step_reward * (n_steps - 1) + term_reward
    #             opt_steps = n_steps
    #         elif n_agents == 2:
    #             path, _ = dijkstra_ma_sop(self.grid, joint_start, joint_goal, self.goals, n_agents)
    #             opt_return = 0
    #             opt_steps = 0
    #             for i in range(n_agents):
    #                 curr_goal = joint_goal[i]
    #                 curr_steps = 0
    #                 for j in range(len(path)):
    #                     if path[j][i] == curr_goal:
    #                         break
    #                     else:
    #                         curr_steps += 1
    #                 term_reward = self.desirable_reward if self.goals[curr_goal] else self.undesirable_reward
    #                 curr_return = self.step_reward * (curr_steps - 1) + term_reward
    #                 opt_return += curr_return
    #                 opt_steps += curr_steps
    #         else:
    #             raise NotImplementedError
    #
    #         return opt_return, opt_steps
    #     else:
    #         raise NotImplementedError

    # def calc_opt_return(self, joint_start, joint_goal=None):
    #     if joint_goal is not None:
    #         opt_return, _ = self.calc_opt_return_steps(joint_start, joint_goal)
    #         return opt_return
    #     else:
    #         raise NotImplementedError

    def calc_joint_states(self):
        states = self._valid_states
        if not self.is_collisions:
            states_arr = [states for _ in range(self.n_agents)]
            joint_states = list(itertools.product(*states_arr))
        else:
            joint_states = list(itertools.permutations(states, r=self.n_agents))
        return joint_states

    ##############
    # PROPERTIES #
    ##############
    @property
    # def joint_goals(self) -> List[Tuple[Tuple[int, int], ...]]:
    #     raise NotImplementedError
    #     # goals = list(self.goals.keys())
    #     # if not self.is_collisions:
    #     #     goals_arr = [goals for _ in range(self.n_agents)]
    #     #     joint_goals = list(itertools.product(*goals_arr))
    #     #     return joint_goals
    #     # else:
    #     #     return list(itertools.permutations(goals, r=self.n_agents))
    #     return self._joint_goals

    @property
    def joint_goal_rewards(self) -> Dict[Tuple[Tuple[int, int], ...], float]:
        raise NotImplementedError
        # joint_goals = self.get_joint_goals()
        # des_reward = self.desirable_reward
        # undes_reward = self.undesirable_reward
        # goal_rewards = {goal: des_reward if is_des else undes_reward for goal, is_des in self.goals.items()}
        #
        # joint_goal_rewards = {joint_goal: sum([goal_rewards[goal] for goal in joint_goal]) for joint_goal in joint_goals}
        # return joint_goal_rewards
        return self._joint_goal_rewards

    @property
    def goal_rewards(self):
        return self._goal_rewards

    @property
    def states(self):
        return self._valid_states

    #############
    # RENDERING #
    #############
    # noinspection PyMethodOverriding
    def render(self, render_mode=None):
        if render_mode is None:
            render_mode = self.render_mode

        if render_mode == "human" or render_mode == "rgb_array":
            return self._render_frame()
        else:
            raise ValueError(f"render_mode={render_mode} is not supported")

    def render_path(self, path):
        # window_size = (self._window_width, self._window_height)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                self._window_dims
                # (self.window_size, self.window_size)
            )

        # canvas = pygame.Surface((self.window_size, self.window_size))
        canvas = pygame.Surface(self._window_dims)
        self._render_grid(canvas)  # canvas should be passed by reference

        self._draw_path(canvas, path)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def animate_path(self, path: List[List[Tuple[int, int]]], wait_time=500):
        for joint_loc in path:
            self._curr_joint_pos = joint_loc
            self.render()
            pygame.time.wait(wait_time)

    def _get_canvas_coord(self, x, y):
        cx = x * self._tile_size + self._margin_width
        # cy = self.window_size - y * self._tile_size - self._margin_width
        cy = self._window_height - y * self._tile_size - self._margin_width
        return cx, cy

    # Render grid lines & obstacles
    def _render_grid(self, canvas):
        #############
        # OBSTACLES #
        #############
        obstacles = list(map(tuple, np.argwhere(self.grid == 1)))
        for obstacle in obstacles:
            y, x = obstacle
            color = "black"
            cx, cy = self._get_canvas_coord(x, y)
            rect = pygame.Rect(cx + 1, cy - self._tile_size + 1, self._tile_size, self._tile_size)
            pygame.draw.rect(canvas, color, rect)

        ##############
        # GRID LINES #
        ##############
        line_width = 2
        # Vertical Lines
        for x in range(self._grid_width + 1):
            cx, cy1 = self._get_canvas_coord(x, 0)
            cy2 = cy1 - self._grid_canvas_height
            pygame.draw.line(canvas, (0, 0, 0), (cx, cy1), (cx, cy2), width=line_width)

        # Horizontal Lines
        for y in range(self._grid_height + 1):
            cx1, cy = self._get_canvas_coord(0, y)
            cx2 = cx1 + self._grid_canvas_width
            pygame.draw.line(canvas, (0, 0, 0), (cx1, cy), (cx2, cy), width=line_width)
        ########

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.pygame_font = pygame.font.Font("freesansbold.ttf", self.font_size)
            pygame.display.init()
            self.window = pygame.display.set_mode(
                self._window_dims
                # (self.window_size, self.window_size)
            )

        # canvas = pygame.Surface((self.window_size, self.window_size))
        canvas = pygame.Surface(self._window_dims)
        canvas.fill((255, 255, 255))

        agent_colors = self.agent_colors

        #########
        # GOALS #
        #########
        for i in range(self.n_agents):
            for goal, is_desirable in self.goals[i].items():
                y, x = goal
                color = "green" if is_desirable else "red"
                cx, cy = self._get_canvas_coord(x, y)
                rect = pygame.Rect(cx+1, cy-self._tile_size+1, self._tile_size, self._tile_size)
                pygame.draw.rect(canvas, color, rect)

                line_width = 3
                offset = line_width
                rect_width = self._tile_size - offset + 1
                rect = pygame.Rect(cx + offset - 1, cy - self._tile_size + offset - 1, rect_width, rect_width)
                pygame.draw.rect(canvas, agent_colors[i], rect, width=line_width)

        ##########
        # AGENTS #
        ##########
        text_objs = []
        radius = self._tile_size / 2 - 2
        offset = 2 + radius
        for i, color in enumerate(agent_colors):
            r, g, b = color
            # r, g, b = r * 255, g * 255, b * 255
            y, x = self._curr_joint_pos[i]
            cx, cy = self._get_canvas_coord(x, y)
            cx, cy = cx + offset + 1, cy - offset + 1
            pygame.draw.circle(canvas, (r, g, b), (cx, cy), radius)
            pygame.draw.circle(canvas, (0, 0, 0), (cx, cy), radius, width=1)

            if self.window is not None:
                text_surface = self.pygame_font.render(str(i+1), True, (0, 0, 0)) #, (255, 255, 255))
                rect = text_surface.get_rect()
                rect.center = (cx, cy)
                text_objs.append((text_surface, rect))
            # canvas.blit()

        self._render_grid(canvas)  # canvas should be passed by reference

        #############################
        # RENDER TO WINDOW OR ARRAY #
        #############################
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for surface, rect in text_objs:
                self.window.blit(surface, rect)

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            # self.clock.tick(self.metadata["render_fps"])
        #     return None
        # else:  # rgb_array

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _draw_circle(self, y, x, canvas, color: Tuple[int, int, int], radius, offset):
        r, g, b = color
        # r, g, b = r * 255, g * 255, b * 255
        cx, cy = self._get_canvas_coord(x, y)
        cx, cy = cx + offset + 1, cy - offset + 1
        pygame.draw.circle(canvas, (r, g, b), (cx, cy), radius)

    def _draw_path(self, canvas, path):
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i+1]
            ts = self._tile_size//2

            cx1, cy1 = self._get_canvas_coord(x1, y1)
            cx1, cy1 = cx1 + ts, cy1 - ts
            cx2, cy2 = self._get_canvas_coord(x2, y2)
            cx2, cy2 = cx2 + ts, cy2 - ts

            pygame.draw.line(canvas, "blue", (cx1, cy1), (cx2, cy2), width=2)

        radius = self._tile_size//4
        offset = self._tile_size//2
        red = (255, 0, 0)
        green = (0, 255, 0)
        ys, xs = path[0]
        yg, xg = path[-1]

        self._draw_circle(ys, xs, canvas, green, radius, offset)
        self._draw_circle(yg, xg, canvas, red, radius, offset)

    ##############
    # PROPERTIES #
    ##############
    @property
    def reward_out_mode(self):
        if self._reward_out_mode == RewardMode.JOINT:
            return "joint"
        elif self._reward_out_mode == RewardMode.SUM:
            return "sum"
        else:
            raise ValueError

    @property
    def terminate_mode(self):
        if self._terminate_mode == TerminateMode.AGENT_ENTER:
            return "agent_enter"
        elif self._terminate_mode == TerminateMode.AGENT_WAIT:
            return "agent_wait"
        else:
            raise ValueError


class RewardMode(IntEnum):
    SUM = 0,
    JOINT = 1


class TerminateMode(IntEnum):
    AGENT_WAIT = 0,
    AGENT_ENTER = 1  # Agent terminates upon entering goal


class Action(IntEnum):
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3,
    STAY = 4,
    NOOP = 5


def interactive(env: GridWorld):
    env.render()

    key_action_map = {
        pygame.K_w: Action.NORTH,
        pygame.K_s: Action.SOUTH,
        pygame.K_d: Action.EAST,
        pygame.K_a: Action.WEST,
        pygame.K_x: Action.STAY
    }

    # key_buffer = []
    action_buffer = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                # print(f"Hello: {event.key}")
                # print(pygame.K_KP_ENTER)
                if event.key == 13:
                    # pass
                    if len(action_buffer) < env.n_agents:
                        print(f"Not enough actions chosen. Actions chosen so far = {len(action_buffer)}")
                    elif len(action_buffer) > env.n_agents:
                        print(f"Too many actions chosen. Actions chosen so far = {len(action_buffer)}")
                    else:

                        action = [a.value for a in action_buffer]
                        next_state, reward, is_done, _, info = env.step(action)
                        print(f"next_state={next_state}, reward={reward}, is_done={is_done}, info={info}")
                        env.render()

                    action_buffer = []
                elif event.key in key_action_map.keys():
                    action_buffer.append(key_action_map[event.key])
                elif event.key == pygame.K_r:
                    joint_start_state = env.reset()
                    print(f"Env reset - joint_start_state = {joint_start_state}")
                    env.render()
                elif event.key == pygame.K_p:
                    raise NotImplementedError
                    env_hist = env.env_history
                    logging_config = env_hist.logging_config
                    # history_dict = env.env_history.to_dict()
                    print(f"#####################")
                    print(f"#      HISTORY      #")
                    print(f"#####################")
                    print(f"---------------------")
                    print(f"|       Stats       |")
                    print(f"---------------------")
                    print(f" episode no = {env_hist.episode_no}")
                    print(f" no steps = {env_hist.curr_step}")
                    print(f" cumulative reward = {env_hist.eps_return}")

                    if logging_config["joint_action"]:
                        print(f"---------------------")
                        print(f"|   Joint Actions    |")
                        print(f"---------------------")
                        print(f" Joint actions = {env_hist.joint_action_str_history}")

                    if logging_config["joint_state"]:
                        print(f"---------------------")
                        print(f"|    Joint States    |")
                        print(f"---------------------")
                        print(f" Joint states = {env_hist.joint_state_history}")

                    if logging_config["reward"]:
                        print(f"---------------------")
                        print(f"|      Rewards       |")
                        print(f"---------------------")
                        print(f" Rewards = {env_hist.reward_history}")

                    if logging_config["is_done"]:
                        print(f"---------------------")
                        print(f"|      Is Done       |")
                        print(f"---------------------")
                        print(f" Is done = {env_hist.is_done_history}")

                    if logging_config["info"]:
                        print(f"---------------------")
                        print(f"|       Infos        |")
                        print(f"---------------------")
                        print(f" Infos = {env_hist.info_history}")

                    print(f"#####################")

                    # print(env_hist.frame_history.keys())
                # elif event.key == pygame.K_s:
                #     raise NotImplementedError
                #     env.env_history.save_video("videos", "test", fps=2)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()


def main():
    goals = {
        (0, 0): True,
        (0, 4): True,
        (4, 0): False,
        (4, 4): False
    }
    env = GridWorld("configs/maps/5x5.txt", 2, goals,
                    reward_out_mode="joint", terminate_mode="agent_enter",
                    is_collisions=True,
                    is_virt_term_state=False)
    # env.reset([(1,1), (3, 3)])
    print(f"goal_rewards = {env.goal_rewards}")
    print(f"joint_goals = {env.joint_goals}")
    print(f"joint_goal_rewards = {env.joint_goal_rewards}")
    # print(env.joint_goals)
    # joint_goals = env.get_joint_goals()
    # print(joint_goals)
    # print(env.get_terminal_rewards())

    # exit()
    env.reset()
    interactive(env)

    # env.render()
    # pygame.time.wait(5000)


def test_3rooms():
    config = load_config("4x4.json")
    named_goals = config["named_goals"]
    goals = {goal: True for goal in named_goals.values()}
    rewards = config["rewards"]

    env = GridWorld(
        config["grid"], 3, goals,
        reward_out_mode="joint", terminate_mode="agent_wait",
        is_collisions=True,
        is_virt_term_state=False,
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"]
    )

    # goals = {
    #     (0, 0): True,
    #     (10, 0): True,
    #     (0, 10): True
    # }
    #
    # env = GridWorld("configs/maps/3_rooms_alt.txt", 2, goals,
    #                 reward_out_mode="joint", terminate_mode="agent_enter",
    #                 is_collisions=True,
    #                 is_virt_term_state=False)
    # joint_start = [(10, 1), (1, 10)]
    # #
    # env.reset(joint_start)
    #
    # opt_return, opt_steps = env.calc_opt_return_steps(tuple(joint_start), ((0, 0), (0, 10)))
    # print(f"opt_return = {opt_return}, opt_steps={opt_steps}")
    # env.render()
    # pygame.time.wait(2000)
    interactive(env)


def test_env():
    config = load_config("5_rooms_8g.json")
    named_goals = config["named_goals"]
    goals = {goal: True for goal in named_goals.values()}
    rewards = config["rewards"]

    env = GridWorld(
        config["grid"], 2, goals,
        reward_out_mode="joint", terminate_mode="agent_wait",
        is_collisions=True,
        is_virt_term_state=False,
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"]
    )
    print(f"n_valid_states: {env.n_valid_states}")
    # goals = [{goal: True for goal in named_goals.values()}, {goal: False for goal in named_goals.values()}]
    # env.modify_goal_values(goals)
    env.reset()
    print(env.goal_rewards)

    env.calc_opt_joint_return_steps([])

    interactive(env)


def test_opt_return():
    config = load_config("4x4.json")
    grid = config["grid"]
    named_goals = config["named_goals"]
    named_starts = config["named_starts"]
    # goals = [
    #     {named_goals["TL"]: True, named_goals["TR"]: True,
    #      named_goals["BL"]: False, named_goals["BR"]: False},
    #     {named_goals["TL"]: True, named_goals["TR"]: False,
    #      named_goals["BL"]: True, named_goals["BR"]: False}
    # ]
    goals = [
        {
            named_goals["TL"]: False, named_goals["TR"]: False,
            named_goals["BL"]: False, named_goals["BR"]: True
        },
        {
            named_goals["TL"]: False, named_goals["TR"]: True,
            named_goals["BL"]: False, named_goals["BR"]: False
        }
    ]
    # goals = {goal: True for goal in named_goals.values()}
    # goals_set = set(goals.keys())
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

    eps_return, eps_steps, eps_info = env.calc_opt_joint_return_steps(joint_start)
    print(f"Opt return: {eps_return}")
    print(f"Opt steps: {eps_steps}")

    # print(f"Cost: {cost}")
    env.animate_path(eps_info["best_path"])


def test_opt_return_1():
    config = load_config("5_rooms_4g.json")
    grid = config["grid"]
    named_goals = config["named_goals"]
    named_starts = config["named_starts"]
    goals = [
        {named_goals["U"]: True, named_goals["D"]: True,
         named_goals["L"]: False, named_goals["R"]: False},
        {named_goals["U"]: True, named_goals["D"]: False,
         named_goals["L"]: True, named_goals["R"]: False}
    ]
    # goals = {goal: True for goal in named_goals.values()}
    # goals_set = set(goals.keys())
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
    joint_start = (named_starts["R"], named_starts["L"])
    joint_goal = (named_goals["L"], named_goals["R"])
    print(f"Joint start: {joint_start}")
    print(f"Joint goal: {joint_goal}")

    env.reset(list(joint_start))

    # interactive(env)
    # exit()

    eps_return, eps_steps, eps_info = env.calc_opt_joint_return_steps(joint_start)
    print(f"Opt return: {eps_return}")
    print(f"Opt steps: {eps_steps}")

    # print(f"Cost: {cost}")
    env.animate_path(eps_info["best_path"])


if __name__ == "__main__":
    # main()
    # test_3rooms()
    test_opt_return()
