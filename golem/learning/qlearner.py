import copy
import os
import pickle
from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Set, Union, DefaultDict, Any, SupportsFloat

import numpy as np
import pygame.time

from tqdm import tqdm
from typing_extensions import override

from golem.learning.learner import Learner
from golem.learning.utils import conv_to_wandb
# from coop_decomp.learning.common import pointwise_add, pointwise_add_2d
from golem.utils import copy_to_dict


from golem.planning.dijkstra import Dijkstra


class QLearner(Learner):
    def __init__(
            self,
            env,
            q=None,
            opt_v: Optional[Union[Dict, str]] = None,
            calc_opt_v=False,
            learner_config: Optional[Dict] = None,
            logger_metrics: Optional[List[Tuple[str, str]]] = None,
            **kwargs
    ):
        curr_learner_config = copy.deepcopy(locals())
        del curr_learner_config["learner_config"]
        del curr_learner_config["self"]
        del curr_learner_config["env"]
        if learner_config is None:
            learner_config = {}
        copy_to_dict(learner_config, curr_learner_config)
        # all_args = locals().copy()
        # del all_args["self"]
        # # del all_args["__class__"]
        # self.config = all_args
        # self.config = conv_to_wandb(all_args)
        #
        # if wandb_config is not None and "learner" not in wandb_config:
        #     wandb_config["learner"] = self.config

        # # If learner is in wandb_config then this has already been specified by a subclass
        # if wandb_config is not None and "learner" not in wandb_config:
        #     wandb_config["learner"] = {}

        # copy_to_dict(self.config, wandb_config["learner"])

        # if wandb_metrics is not None:
        #     new_wandb_metrics = [
        #         ("n_joint_goals", "tot_steps")
        #     ]
        #     wandb_metrics.extend(new_wandb_metrics)

        super().__init__(env, **kwargs, learner_config=curr_learner_config, logger_metrics=logger_metrics)

        # Q
        if q is None:
            q = self.create_q_func()

        self._q = q

        # Optimal Value Function
        if opt_v is not None:
            if type(opt_v) == str:
                with open(opt_v, "rb") as f:
                    opt_v = pickle.load(f)
        elif calc_opt_v:
            opt_v = self.calc_opt_v()

        self._opt_v = opt_v

    ######################
    # CREATE/SAVE/LOAD Q #
    ######################
    def create_q_func(self, actions_shape=None) -> DefaultDict[Tuple[Tuple[int, int], ...], np.ndarray]:
        if actions_shape is None:
            actions_shape = tuple([self.n_agent_actions for _ in range(self.n_agents)])
        q = defaultdict(lambda: np.zeros(actions_shape))
        return q

    def load_q_func(
            self,
            file_path: str,
            actions_shape: Optional[Tuple[int, ...]] = None
    ) -> DefaultDict[Tuple[Tuple[int, int], ...], np.ndarray]:
        with open(file_path, "rb") as f:
            raw_q, load_actions_shape = pickle.load(f)
        if actions_shape is not None and load_actions_shape != actions_shape:
            raise ValueError("Loaded q func does not have same actions_shape as that specified")
        actions_shape = load_actions_shape
        q = self.create_q_func(actions_shape)

        for joint_state in raw_q.keys():
            q[joint_state] = np.asarray(raw_q[joint_state])

        return q

    @staticmethod
    def save_q_func(q: DefaultDict[Tuple[Tuple[int, int], ...], np.ndarray], file_path: str):
        q_dict = {}
        for joint_state in q.keys():
            q_dict[joint_state] = q[joint_state].tolist()

        actions_shape = next(iter(q.values())).shape
        save_obj = (q_dict, actions_shape)

        with open(file_path, "wb") as f:
            # pickle.dump(q_dict, f)
            pickle.dump(save_obj, f)

    def save_q(self, file_path):
        self.save_q_func(self._q, file_path)

    def load_q(self, file_path):
        self._q = self.load_q_func(file_path)

    ######################
    #        END         #
    # CREATE/SAVE/LOAD Q #
    ######################

    #####################
    # CHECK CONVERGENCE #
    #####################
    def calc_opt_v(self, use_prog_bar=True):
        termination_mode = self.env.terminate_mode
        assert termination_mode == "agent_enter" or termination_mode == "agent_wait"
        is_agent_enter = termination_mode == "agent_enter"

        # S x G version of WVF
        # vq -> VQ(s, g) = max_a{Q(s, g, a)}
        n_agents = self.n_agents
        goals = self.env.goals
        joint_states = self.env.calc_joint_states()
        joint_goals = self.env.joint_goals

        n_vals = len(joint_states) * len(joint_goals)

        dijkstra = Dijkstra(grid=self.env.grid, n_agents=self.n_agents,
                            goals=self.env.goals, is_collisions=self.env.is_collisions,
                            termination_type=self.env.terminate_mode)

        prog_bar = tqdm(total=n_vals, desc="# Costs", disable=not use_prog_bar)

        # all_opt_costs signifies the number of step_rewards accumulated for each joint_state, joint_goal pair
        all_opt_costs = {}
        for joint_state in joint_states:
            all_opt_costs[joint_state] = {}
            for joint_goal in joint_goals:
                is_pair_valid = True
                cost_incr = 0

                if is_agent_enter:
                    for state, goal in zip(joint_state, joint_goal):
                        if state in goals:
                            if state != goal:
                                is_pair_valid = False
                                break
                            else:
                                # Cost for starting at goal is zero so this accounts for later step where we minus cost
                                cost_incr += 1

                if is_pair_valid:
                    _, cost = dijkstra.shortest_path(joint_state, joint_goal)
                    # No cost per agent for moving to goal, but we must account for case where it started at a goal
                    if is_agent_enter:
                        cost = cost - n_agents + cost_incr
                    # else:
                    #     # There is a cost to taking wait action - Need to account for when agent starts at goal
                    #     cost = cost + cost_incr
                    all_opt_costs[joint_state][joint_goal] = cost
                    if use_prog_bar:
                        prog_bar.update()

        prog_bar.close()

        goal_rewards = self.env.goal_rewards
        r_step = self.env.step_reward
        opt_vq = {}
        for joint_state in joint_states:
            opt_vq[joint_state] = {}

        for joint_goal in joint_goals:
            for joint_state in joint_states:
                if joint_state in all_opt_costs.keys() and joint_goal in all_opt_costs[joint_state].keys():
                    # Return accumulated return from step reward
                    curr_return = all_opt_costs[joint_state][joint_goal] * r_step
                    for state, goal in zip(joint_state, joint_goal):
                        # This condition can probably be simplified
                        if not is_agent_enter or (is_agent_enter and state not in goals):
                            curr_return += goal_rewards[goal]

                    opt_vq[joint_state][joint_goal] = curr_return

        opt_v = {}
        # Compute V from VQ (should do more efficiently)
        for joint_state in opt_vq.keys():
            curr_dict = opt_vq[joint_state]
            # opt_v[joint_state] = np.amax([curr_dict[joint_goal] for joint_goal in curr_dict.keys()], axis=0)
            opt_v[joint_state] = np.max([curr_dict[joint_goal] for joint_goal in curr_dict.keys()])

        return opt_v

    @staticmethod
    def check_v_equal(v1, v2, tolerance=0.0, ignore_missing=True):
        for joint_state in v1.keys():
            if joint_state in v2.keys():
                if abs(v1[joint_state] - v2[joint_state]) > tolerance:
                    return False
            elif not ignore_missing:
                raise ValueError(f"joint_state = {joint_state} not in v2")
        return True

    @staticmethod
    def calc_v(q):
        v = {}
        for joint_state in q:
            v[joint_state] = q[joint_state].max()

        return v

    def check_convergence(self, tolerance=None) -> bool:
        if tolerance is None:
            tolerance = self.eval_conv_tol
        if self._opt_v is None:
            self._opt_v = self.calc_opt_v(use_prog_bar=False)

        v = self.calc_v(self._q)
        return self.check_v_equal(self._opt_v, v, tolerance)

    #####################
    #       END         #
    # CHECK CONVERGENCE #
    #####################

    # noinspection PyMethodOverriding
    def greedy_policy(self, joint_state):
        action_arr = self._q[joint_state]
        flat_ind = np.random.choice(np.flatnonzero(action_arr == action_arr.max()))
        joint_action = list(np.unravel_index(flat_ind, action_arr.shape))
        return joint_action

    def learn(
            self,
            no_steps=np.inf,
            no_episodes=np.inf,
            eval_conv_freq_eps=None,
            eval_conv_freq_steps=None,
            eval_freq_eps=None,
            eval_freq_steps=None
    ):
        assert self.env.reward_out_mode == "sum"
        max_steps = min(self.max_steps, self.curr_step + no_steps)
        max_episodes = min(self.max_episodes, self.curr_episode + no_episodes)
        if max_steps == np.inf and max_episodes == np.inf and \
                eval_conv_freq_eps is None and eval_conv_freq_steps is None:
            raise ValueError("Arguments and/or init args are invalid. Learning would never stop")

        if self._use_prog_bar:
            if max_steps != np.inf:
                total = max_steps
                desc = "Steps"
                # prog_bar.reset(total=max_steps)
                # prog_bar.set_description("Steps")
            elif max_episodes != np.inf:
                total = max_episodes
                desc = "Episodes"
                # prog_bar.reset(total=max_episodes)
                # prog_bar.set_description("Episodes")
            else:
                total = None
                desc = ""
                # prog_bar.reset(total=np.inf)  # No idea if this will work

            prog_bar = tqdm(total=total, desc=desc)
        else:
            prog_bar = tqdm(disable=True)

        def stop_cond():
            return self.curr_step > max_steps or self.curr_episode > max_episodes or self.is_converged

        step_history = []
        return_history = []

        env = self.env
        # n_agents = self.env.n_agents
        alpha = self.alpha
        gamma = self.gamma
        q = self._q
        eps = self.epsilon

        while not stop_cond():
            prog_dict = {
                "avg_steps": np.mean(step_history[-10:]),
                "avg_return": np.mean(return_history[-10:]),
                "epsilon": eps
            }
            is_all_done = False
            eps_return = 0
            eps_steps = 0

            # Reset Env
            curr_joint_state, _ = tuple(env.reset())
            # Augment joint_state to indicate agent level 'is_dones'
            curr_joint_state = tuple([(state, False) for state in curr_joint_state])

            is_truncated = False
            # is_truncated is true if max steps for episode has been reached

            while not is_all_done and not stop_cond() and not is_truncated:
                joint_action = self.act(joint_state=curr_joint_state, epsilon=eps)

                next_joint_state, reward, joint_is_done, is_truncated, info = env.step(joint_action)
                # Augment state space with agent level is_dones
                next_joint_state = tuple(zip(next_joint_state, joint_is_done))
                # next_joint_state = tuple(next_joint_state)
                is_all_done = np.all(joint_is_done)
                # is_any_done = np.any(joint_is_done)

                # reward = np.sum(joint_reward)

                # VALUE UPDATE
                s = curr_joint_state
                s_next = next_joint_state
                a = tuple(joint_action)

                # Do not need '* (not is_all_done)' since this is handled by state space augmented with termination status
                td_target = reward + gamma * np.max(q[s_next])  #* (not is_all_done)
                td_error = td_target - q[s][a]

                q[s][a] = q[s][a] + alpha * td_error

                curr_joint_state = next_joint_state

                if self.is_decaying_eps and self.decay_type == "step":
                    eps = self.decay_epsilon()

                if eval_conv_freq_steps is not None and self.curr_step != 0 \
                        and self.curr_step % eval_conv_freq_steps == 0:
                    self.is_converged = self.check_convergence()
                    if self.is_converged:
                        print("Converged!")

                # Eval policy
                if eval_freq_steps is not None and self.curr_step % eval_freq_steps == 0:
                    eval_steps, eval_return = self.evaluate(self.eval_joint_start)
                    eval_log_dict = {
                        "tot_steps": self.curr_step,
                        "tot_episodes": self.curr_episode,
                        "eval_steps": eval_steps,
                        "eval_return": eval_return
                    }
                    self.logger.log(eval_log_dict)

                eps_steps += 1
                self.curr_step += 1
                eps_return += reward

            # Update progress bar each episode
            if self._use_prog_bar:
                prog_bar.update(eps_steps)
                prog_bar.set_postfix(prog_dict)

            # End of episode
            # Eval Convergence
            if eval_conv_freq_eps is not None and self.curr_episode != 0 \
                    and self.curr_episode % eval_conv_freq_eps == 0:
                self.is_converged = self.check_convergence()
                if self.is_converged:
                    print("Converged!")

            # Eval policy
            if eval_freq_eps is not None and self.curr_episode % eval_freq_eps == 0:
                eval_steps, eval_return = self.evaluate(self.eval_joint_start)
                eval_log_dict = {
                    "tot_steps": self.curr_step,
                    "tot_episodes": self.curr_episode,
                    "eval_steps": eval_steps,
                    "eval_return": eval_return
                }
                self.logger.log(eval_log_dict)

            step_history.append(eps_steps)
            return_history.append(eps_return)
            self.curr_episode += 1

            if self.is_decaying_eps and self.decay_type == "episode":
                eps = self.decay_epsilon()

            log_dict = {
                "tot_steps": self.curr_step,
                "epsilon": eps,
                "eps_return": return_history[-1],
                "eps_steps": step_history[-1],
                "tot_episodes": self.curr_episode
            }
            self.logger.log(log_dict)

        prog_bar.close()

    @override
    def evaluate(self, joint_start_state: List[Tuple[int, int]], max_steps=None):
        if max_steps is None:
            max_steps = self.max_eval_steps

        env = self.env
        joint_state, _ = env.reset(joint_start_state)
        joint_state = tuple([(state, False) for state in joint_state])
        is_done = False
        is_truncated = False
        curr_step = 0
        curr_return = 0

        while not is_done and not is_truncated and curr_step < max_steps:
            joint_action = self.greedy_policy(joint_state)
            joint_state, reward, joint_is_done, is_truncated, info = env.step(joint_action)
            joint_state = tuple(zip(joint_state, joint_is_done))

            curr_return += reward
            curr_step += 1
            is_done = np.all(joint_is_done)

        return curr_step, curr_return

    def vis_policy(self, joint_start, max_steps=100, tick_wait_ms=1000):
        env = self.env
        curr_joint_state, _ = env.reset(joint_start)
        env.render()
        pygame.time.wait(tick_wait_ms)
        is_all_done = False
        curr_step = 0
        while not is_all_done and curr_step < max_steps:
            joint_action = self.act(curr_joint_state, 0)
            curr_joint_state, reward, is_dones, is_truncated, info = env.step(joint_action)
            is_all_done = np.all(is_dones)
            env.render()
            pygame.time.wait(tick_wait_ms)
            curr_step += 1


def test():
    from golem.environment.utils import load_config
    from golem.environment.gridworld.gridworld import GridWorld, interactive

    wandb_kwargs = {
        "entity": "simonrosen42",
        "project": "golem-gs-tab",
        "group": "debug",
        "job_type": "debug",
        "mode": "online",
        "notes": "debug"
    }

    n_agents = 2
    map_name = "4x4"
    is_collisions = True
    # alg = "decomp"

    config = load_config(f"{map_name}.json")
    named_goals = config["named_goals"]
    goals = {goal: True for goal in named_goals.values()}
    rewards = config["rewards"]

    env = GridWorld(
        config["grid"], n_agents, goals,
        reward_out_mode="joint", terminate_mode="agent_wait",
        is_collisions=is_collisions,
        is_virt_term_state=False,
        max_episode_steps=config["max_episode_steps"],
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"], r_min=rewards["r_min"]
    )

    print(f"# states = {len(env.states)}")
    print(f"# joint states = {len(env.calc_joint_states())}")
    interactive(env)


if __name__ == "__main__":
    test()
    # test_decomp()
