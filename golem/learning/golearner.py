import copy
import itertools
import os
import pickle
from collections import defaultdict

import numpy as np
import pygame.time

from tqdm import tqdm

from golem.environment.gridworld.gridworld import GridWorld
from golem.learning.learner import Learner
from golem.learning.utils import conv_to_wandb
from golem.learning.common import pointwise_add
from golem.utils import copy_to_dict, save_file, load_file
from golem.planning.dijkstra import Dijkstra

from typing import Optional, Dict, Tuple, List, Set, Any, Union, SupportsFloat, SupportsInt
from typing_extensions import override

JointPos = Tuple[Tuple[int, int], ...]


# Learns from multiple rewards
class JointGOLearner(Learner):
    DEFAULT_LOG_VERBOSITY = {
        "eval": "",
        "eval_conv": "agr"
    }

    def __init__(
            self,
            env,
            q=None,
            q_init_mode="zero",  # zero, optimistic
            # joint_goal_library: Optional[Set] = None,
            goal_library: Optional[List[Set]] = None,
            # Optimality checks
            comp_opt_vq: bool = True,
            opt_vq: Optional[str] = None,
            opt_vq_costs: Optional[str] = None,
            save_opt_vq_path: Optional[str]= None,
            save_opt_vq_costs_path: Optional[str] = None,
            eval_convergence_type: Optional[str] = None,  # None, "starts", "full"
            # eval_conv_joint_goals: Optional[List[Tuple[Tuple[int, int], ...]]] = None,
            eval_conv_joint_starts: Optional[List[Tuple[Tuple[int, int], ...]]] = None,
            # Evaluation
            eval_joint_goal: Optional[JointPos] = None,
            eval_joint_start: Optional[JointPos] = None,
            max_eval_steps: SupportsInt = np.inf,
            # eval_opt_joint_starts: bool = False,
            # wandb_kwargs
            learner_config: Optional[Dict] = None,
            wandb_kwargs: Optional[Dict] = None,
            logger_metrics: Optional[List[Tuple[str, str]]] = None,
            log_verbosity: Optional[Dict[str, str]] = None,
            **kwargs
    ):
        all_args = copy.copy(locals())
        del all_args["self"]
        if all_args["q"] is not None and type(all_args["q"]) != list:
            all_args["q"] = f"{type(all_args['q'])}"

        del all_args["env"]
        del all_args["learner_config"]
        del all_args["__class__"]
        del all_args["wandb_kwargs"]

        curr_config = conv_to_wandb(all_args)
        curr_config["name"] = "JointGOLearner"

        copy_to_dict(kwargs, curr_config)
        del curr_config["kwargs"]

        if learner_config is None:
            learner_config = {}

        copy_to_dict(learner_config, curr_config)
        # for key, val in curr_config.items():
        #     print(f"{key}, {val} - {type(key)}, {type(val)}")
        # exit()
        # print(curr_config)
        # exit()

        # curr_config = {} # TEMP

        # # If learner is in wandb_config then this has already been specified by a subclass
        # if wandb_config is not None and "learner" not in wandb_config:
        #     wandb_config["learner"] = self.config

        if logger_metrics is None:
            logger_metrics = []

        new_logger_metrics = [
            ("n_joint_goals", "tot_steps")
        ]
        logger_metrics.extend(new_logger_metrics)

        super().__init__(env, **kwargs, wandb_kwargs=wandb_kwargs,
                         learner_config=curr_config, logger_metrics=logger_metrics)

        self._q_init_mode = q_init_mode

        if q is None:
            self._q = self._create_wvf()
        elif type(q) == str:
            # Assume q is path
            self._q, goal_library = self.load_q(q)
        elif type(q) == defaultdict:
            self._q = q
        else:
            raise ValueError(f"q is of wrong type (type(q) = {type(q)})")

        if goal_library is None:
            self._goal_library = [set() for _ in range(self.n_agents)]
            self._joint_goal_library = set()  # Used for efficiency
        else:
            # TODO: Test
            assert type(goal_library) == list, "goal_library must be a list"
            assert len(goal_library) == self.n_agents, f"goal_library must be of size n_agents={self.n_agents}"
            self._goal_library = goal_library

            joint_goals = list(itertools.product(*[list(el) for el in goal_library]))
            joint_goals = list(filter(lambda x: len(set(x)) == len(x), joint_goals))  # Remove joint goals with repeats
            self._joint_goal_library = set(joint_goals)
        #     self._joint_goal_library = set()
        # else:
        #     self._joint_goal_library = joint_goal_library

        self._log_verbosity = copy.copy(self.DEFAULT_LOG_VERBOSITY)
        copy_to_dict(log_verbosity, self._log_verbosity)
        self._log_verbosity: Dict[str, int] = self.conv_verbosity(self._log_verbosity)

        # eval convergence
        self.eval_joint_start = eval_joint_start
        self.eval_joint_goal = eval_joint_goal
        self.max_eval_steps = max_eval_steps

        self.eval_convergence_type = eval_convergence_type
        self.is_eval_convergence = eval_convergence_type is not None

        self.opt_vq: Optional[Dict[JointPos, Dict[JointPos, SupportsFloat]]] = None
        self.opt_vq_costs: Optional[Dict[JointPos, Dict[JointPos, int]]] = None
        self.opt_eval_returns: Optional[Dict[JointPos, Dict[JointPos, SupportsFloat]]] = None
        self.opt_eval_steps: Optional[Dict[JointPos, Dict[JointPos, int]]] = None

        self.save_opt_vq_path = save_opt_vq_path
        self.save_opt_vq_costs_path = save_opt_vq_costs_path
        self.eval_conv_joint_starts = eval_conv_joint_starts
        # self.eval_conv_joint_goals = eval_conv_joint_goals

        if eval_convergence_type == "full":
            if opt_vq is not None:
                if type(opt_vq) == str:
                    self.opt_vq = load_file(opt_vq)
                    # with open(opt_vq, "rb") as f:
                    #      = pickle.load(f)
                else:
                    self.opt_vq = opt_vq
            elif comp_opt_vq:
                if opt_vq_costs is not None:
                    if type(opt_vq_costs) == str:
                        opt_vq_costs = load_file(opt_vq_costs)
                        # with open(opt_vq_costs, "rb") as f:
                        #     opt_vq_costs = pickle.load(f)
                        self.opt_vq_costs = opt_vq_costs
                    elif type(opt_vq_costs) == dict:
                        # No change required
                        self.opt_vq_costs = opt_vq_costs
                    else:
                        raise ValueError(f"opt_vq_costs is of wrong type (type(opt_vq_costs) = {type(opt_vq_costs)})")
                self.opt_vq = self.comp_optimal_vq()
            else:
                self.opt_vq = None
        elif eval_convergence_type == "starts":
            assert self.eval_conv_joint_starts is not None
            self.comp_opt_eval_returns()

            # comp eval metrics
            eval_metrics = []
            step_metric = "tot_steps"
            eval_conv_verbosity = self._log_verbosity["eval_conv"]
            # Top level agr
            if eval_conv_verbosity > 0:
                for agr_type in ["avg", "min", "max"]:
                    for val1 in ["opt", "act", "diff"]:
                        for val2 in ["return", "steps"]:
                            # eval_avg_opt_return
                            eval_metrics.append((f"eval_{agr_type}_{val1}_{val2}", step_metric))
                    eval_metrics.append((f"eval_{agr_type}_n_converged", step_metric))
                # eval_metrics.append(("eval_min_n_goals_converged", step_metric))
                # eval_metrics.append(("eval_max_n_goals_converged", step_metric))

            # Joint State Agr
            if eval_conv_verbosity > 1:
                for js in self.eval_conv_joint_starts:
                    for agr_type in ["avg", "min", "max"]:
                        for val1 in ["opt", "act", "diff"]:
                            for val2 in ["return", "steps"]:
                                eval_metrics.append((f"eval_{js}_{agr_type}_{val1}_{val2}", step_metric))
                        eval_metrics.append((f"eval_{js}_{agr_type}_n_converged", step_metric))
                    eval_metrics.append((f"eval_{js}_n_converged", step_metric))
                    eval_metrics.append((f"eval_{js}_perc_converged", step_metric))
                    # f"eval_{js}_n_converged"

            if eval_conv_verbosity > 2:
                for js in self.eval_conv_joint_starts:
                    for jg in self.env.joint_goals:
                        for val1 in ["opt", "act", "diff"]:
                            for val2 in ["return", "steps"]:
                                eval_metrics.append((f"eval_{js}_{val1}_{val2}", step_metric))

            self.logger.add_metrics(eval_metrics)
            # if eval_conv_verbosity == "all" or eval_conv_verbosity == "agr":
            #     eval_metrics.append(("eval_avg_act_return", step_metric))
            #     eval_metrics.append(("eval_avg_opt_return", step_metric))
            #     eval_metrics.append(("eval_avg_diff_return", step_metric))
            #     eval_metrics.append(("eval_avg_act_steps", step_metric))
            #     eval_metrics.append(("eval_avg_opt_steps", step_metric))
            #     eval_metrics.append(("eval_avg_diff_steps", step_metric))
            #
            #     eval_metrics.append(("eval_min_act_return", step_metric))
            #     eval_metrics.append(("eval_min_opt_return", step_metric))
            #     eval_metrics.append(("eval_min_diff_return", step_metric))
            #     eval_metrics.append(("eval_min_act_steps", step_metric))
            #     eval_metrics.append(("eval_min_opt_steps", step_metric))
            #     eval_metrics.append(("eval_min_diff_steps", step_metric))
            #
            #     eval_metrics.append(("eval_max_act_return", step_metric))
            #     eval_metrics.append(("eval_max_opt_return", step_metric))
            #     eval_metrics.append(("eval_max_diff_return", step_metric))
            #     eval_metrics.append(("eval_max_act_steps", step_metric))
            #     eval_metrics.append(("eval_max_opt_steps", step_metric))
            #     eval_metrics.append(("eval_max_diff_steps", step_metric))
            #
            #     eval_metrics.append(("eval_avg_n_goals_converged", step_metric))
            #     eval_metrics.append(("eval_min_n_goals_converged", step_metric))
            #     eval_metrics.append(("eval_max_n_goals_converged", step_metric))

            # if eval_conv_verbosity == "all":
            #     for js in self.eval_conv_joint_starts:
            #         for jg in self.env.joint_goals:
            #             prefix = f"eval_{str(js)}_{str(jg)}"
            #             eval_metrics.append((f"{prefix}_act_return", step_metric))
            #             eval_metrics.append((f"{prefix}_opt_return", step_metric))
            #             eval_metrics.append((f"{prefix}_diff_return", step_metric))
            #             eval_metrics.append((f"{prefix}_act_steps", step_metric))
            #             eval_metrics.append((f"{prefix}_opt_steps", step_metric))
            #             eval_metrics.append((f"{prefix}_diff_steps", step_metric))
        else:
            raise ValueError(f"eval_convergence_type = {eval_convergence_type} is not supported")

    @staticmethod
    def conv_verbosity(verbosity):
        eval_conv_verb = verbosity["eval_conv"]
        if eval_conv_verb is None:
            log_verbosity = 0
        elif eval_conv_verb == "agr":
            log_verbosity = 1
        elif eval_conv_verb == "js_agr":
            log_verbosity = 2
        elif eval_conv_verb == "all":
            log_verbosity = 3
            raise NotImplementedError
        else:
            raise ValueError(f"eval_conv_verb = {eval_conv_verb} is invalid")
        verbosity["eval_conv"] = log_verbosity

        return verbosity

    def _create_wvf(self, actions_shape=None):
        q_init_mode = self._q_init_mode
        if actions_shape is None:
            actions_shape = tuple([self.n_agent_actions for _ in range(self.n_agents)])
        if q_init_mode == "zero":
            q = defaultdict(lambda: defaultdict(lambda: np.zeros(actions_shape)))
        elif q_init_mode == "optimistic":
            fill_val = self.env.r_max * self.n_agents
            q = defaultdict(lambda: defaultdict(lambda: np.full(actions_shape, fill_val)))
        else:
            raise ValueError(f"q_init_mode={q_init_mode} is not supported for _create_wvf")
        return q

    ####################
    # SAVING & LOADING #
    ####################

    @staticmethod
    def save_wvf(wvf, goal_library, file_path):
        tmp_wvf = {}
        for joint_state in wvf.keys():
            tmp_wvf[joint_state] = dict()
            for joint_goal in wvf[joint_state].keys():
                tmp_wvf[joint_state][joint_goal] = wvf[joint_state][joint_goal].tolist()

        # Assuming goal_library is serializable
        obj = (tmp_wvf, goal_library)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    def load_wvf(self, file_path, actions_shape=None):
        wvf = self._create_wvf(actions_shape)
        with open(file_path, "rb") as f:
            tup = pickle.load(f)
            goal_library: List[Set] = tup[1]
            raw_wvf: Dict[Dict[Tuple, List]] = tup[0]

        for joint_state in raw_wvf.keys():
            for joint_goal in raw_wvf[joint_state].keys():
                wvf[joint_state][joint_goal] = np.asarray(raw_wvf[joint_state][joint_goal])

        return wvf, goal_library

    def save_q(self, file_path=None):
        if file_path is None:
            assert self.save_model_path is not None
            file_path = self.save_model_path

        self.save_wvf(self._q, self._goal_library, file_path)
        self.logger.log_artifact(file_path, name="wvf_gl", artifact_type="model")

    def load_q(self, file_path, actions_shape=None):
        self._q, self._goal_library = self.load_wvf(file_path, actions_shape)
        return self._q, self._goal_library

    ########################
    # END SAVING & LOADING #
    ########################

    ##############
    # PROPERTIES #
    ##############
    @property
    def q(self):
        return self._q

    ##################
    # END PROPERTIES #
    ##################

    # noinspection PyMethodOverriding
    def greedy_policy(self, joint_state, joint_goal=None):
        q = self._q
        if joint_goal is not None:
            curr_q = q[joint_state][joint_goal]
            flat_ind = np.random.choice(np.flatnonzero(q[joint_state][joint_goal] == q[joint_state][joint_goal].max()))
            tmp = list(np.unravel_index(flat_ind, q[joint_state][joint_goal].shape))
            return tmp
        else:
            if len(q[joint_state].keys()) == 0:
                # We must use random policy if there are no values to use
                return self.random_policy()
            else:
                q_arrs = [q[joint_state][g] for g in q[joint_state].keys()]
                qvf = np.max(q_arrs, axis=0)
                # take argmax on QVF
                flat_ind = np.random.choice(np.flatnonzero(qvf == qvf.max()))
                joint_action = list(np.unravel_index(flat_ind, qvf.shape))
                return joint_action

    def sample_joint_goal(self) -> Optional[Tuple[Any, ...]]:
        joint_goal_library = list(self._joint_goal_library)
        if len(joint_goal_library) == 0:
            # No goals exist in goal library
            return None
        else:
            ind = np.random.randint(0, len(joint_goal_library))
            return joint_goal_library[ind]
            # jg = []
            # jg_set = set()
            # for i, agent_goals in enumerate(goal_library):
            #     curr_goals = agent_goals - jg_set
            #     ind = np.random.randint(0, len(curr_goals))
            #     g = list(curr_goals)[ind]
            #     jg.append(g)
            #     jg_set.add(g)

        # return tuple(jg)

    def add_goal(self, goal, agent_id):
        # TODO: TEST THIS
        goal_libray = self._goal_library
        if goal not in goal_libray[agent_id]:  # Do nothing if goal has already been found
            goal_libray[agent_id].add(goal)
            goal_arrs = []
            for i, s in enumerate(goal_libray):
                curr_arr = list(s - {goal})
                goal_arrs.append(curr_arr)
            new_joint_goals = list(itertools.product(*goal_arrs))
            self._joint_goal_library = self._joint_goal_library.union(new_joint_goals)

    def learn(
            self,
            no_steps=np.inf,
            no_episodes=np.inf,
            # eval_conv_freq_eps=None,
            # eval_conv_freq_steps=None,
            # eval_freq_eps=None,
            # eval_freq_steps=None
    ):
        # I don't think this code requires modification for termination mode
        terminate_mode = self.env.terminate_mode
        assert terminate_mode == "agent_enter" or terminate_mode == "agent_wait"

        max_steps = min(self.max_steps, self.curr_step + no_steps)
        max_episodes = min(self.max_episodes, self.curr_episode + no_episodes)

        eval_conv_freq_eps = self.eval_conv_freq_eps
        eval_conv_freq_steps = self.eval_conv_freq_steps
        eval_freq_eps = self.eval_freq_eps
        eval_freq_steps = self.eval_freq_steps

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
        n_agents = self.env.n_agents
        alpha = self.alpha
        gamma = self.gamma
        q = self._q
        goal_library = self._goal_library
        # joint_goal_library = self._joint_goal_library
        eps = self.epsilon
        r_min = self.env.r_min

        while not stop_cond():
            # tot_n_goals = sum([len(agent_goals) for agent_goals in goal_library])

            prog_dict = {
                "avg_steps": np.mean(step_history[-10:]),
                "avg_return": np.mean(return_history[-10:]),
                "n_joint_goals": len(self._joint_goal_library),
                "epsilon": eps
            }
            is_all_done = False
            eps_return = 0
            eps_steps = 0

            # Reset Env
            curr_joint_state, info = tuple(env.reset())
            # augment joint state with done status
            curr_joint_state = tuple(zip(curr_joint_state, [False for _ in range(n_agents)]))

            joint_goal = self.sample_joint_goal()
            # # Sample Joint Goal
            # if len(joint_goal_library) == 0:
            #     joint_goal = None
            # else:
            #     goals_arr = list(joint_goal_library)
            #     goal_ind = np.random.randint(len(goals_arr))
            #     joint_goal = goals_arr[goal_ind]

            # already_dones = np.zeros(n_agents, dtype=bool)

            is_truncated = False
            # is_truncated is true if max steps for episode has been reached

            while not is_all_done and not stop_cond() and not is_truncated:
                if joint_goal is None:
                    joint_action = self.random_policy()
                else:
                    joint_action = self.act(joint_state=curr_joint_state, epsilon=eps, joint_goal=joint_goal)

                next_joint_state, joint_reward, joint_is_done, is_truncated, info = env.step(joint_action)
                next_joint_state = tuple(zip(tuple(next_joint_state), joint_is_done))
                is_all_done = np.all(joint_is_done)
                is_any_done = np.any(joint_is_done)

                if is_any_done:
                    curr_reward = 0

                    for i in range(n_agents):
                        # Augment agent level rewards - if agent has reached wrong goal then it gets
                        # reward of r_min, otherwise it gets env specified reward

                        was_agent_done = curr_joint_state[i][1]  # If term status of agent is True for curr state
                        if joint_is_done[i] and not was_agent_done:
                            reached_agent_goal = next_joint_state[i][0]
                            if joint_goal is not None and reached_agent_goal != joint_goal[i]:
                                curr_reward += r_min
                            else:
                                curr_reward += joint_reward[i]

                            # Required for alg - rest is for logging
                            self.add_goal(reached_agent_goal, i)
                        else:
                            curr_reward += joint_reward[i]

                        # if joint_is_done[i] and joint_reward[i] != 0:  # Reward is 0 for the steps after terminating
                        #     # If currently considered agent reached wrong goal and just terminated
                        #     # Note: In this alg next_joint_state is goal
                        #     # Will be same for agent_wait mode since next_state == curr_state
                        #     if next_joint_state[i] != joint_goal[i]:
                        #         curr_reward += r_min
                        #     else:
                        #         curr_reward += joint_reward[i]
                        # else:
                        #     curr_reward += joint_reward[i]

                    eps_return += curr_reward
                    # # For algorithm
                    # if is_all_done:
                    #     joint_goal_library.add(next_joint_state)
                else:
                    eps_return += np.sum(joint_reward)

                for learn_joint_goal in self._joint_goal_library:
                    ###############
                    # DESCRIPTION #
                    # This has been edited to modify the reward at an agent level based off which agent arrived at
                    # the correct goal assigned to it
                    ###############

                    if is_any_done:
                        r_learn = 0
                        # TODO: Optimise once this is working
                        for i in range(n_agents):
                            # # Only do this if agent just terminated
                            # if joint_is_done[i] and joint_reward[i] != 0:  # Reward is 0 for the steps after terminating
                            #     # If currently considered agent reached wrong goal and just terminated
                            #     if next_joint_state[i][0] != learn_joint_goal[i]:
                            #         r_learn += r_min
                            #     else:
                            #         r_learn += joint_reward[i]
                            # else:
                            #     r_learn += joint_reward[i]
                            was_agent_done = curr_joint_state[i][1]  # If term status of agent is True for curr state
                            if joint_is_done[i] and not was_agent_done:
                                reached_agent_goal = next_joint_state[i][0]
                                if reached_agent_goal != learn_joint_goal[i]:
                                    r_learn += r_min
                                else:
                                    r_learn += joint_reward[i]
                            else:
                                r_learn += joint_reward[i]
                    else:
                        r_learn = np.sum(joint_reward)

                    # VALUE UPDATE
                    s = curr_joint_state
                    s_next = next_joint_state
                    g_prime = learn_joint_goal
                    a = tuple(joint_action)

                    td_target = r_learn + gamma * np.max(q[s_next][g_prime]) * (not is_all_done)
                    td_error = td_target - q[s][g_prime][a]

                    q[s][g_prime][a] = q[s][g_prime][a] + alpha * td_error

                # # Set agent as being done
                # if is_any_done:
                #     for i in range(n_agents):
                #         if joint_is_done[i] and not already_dones[i]:
                #             already_dones[i] = True

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
                    eval_steps, eval_return = self.evaluate(self.eval_joint_start, self.eval_joint_goal)
                    eval_log_dict = {
                        "tot_steps": self.curr_step,
                        "tot_episodes": self.curr_episode,
                        "eval_steps": eval_steps,
                        "eval_return": eval_return
                    }
                    self.logger.log(eval_log_dict)

                if self.save_model_step_freq is not None and self.curr_step % self.save_model_step_freq == 0:
                    self.save_q()

                eps_steps += 1
                self.curr_step += 1

            # Update progress bar each episode
            if self._use_prog_bar:
                prog_bar.update(eps_steps)
                prog_bar.set_postfix(prog_dict)

            # End of episode
            if eval_conv_freq_eps is not None and self.curr_episode != 0 \
                    and self.curr_episode % eval_conv_freq_eps == 0:
                self.is_converged = self.check_convergence()
                print("Converged!")

            # Eval policy
            if eval_freq_eps is not None and self.curr_episode % eval_freq_eps == 0:
                eval_steps, eval_return = self.evaluate(self.eval_joint_start, self.eval_joint_goal)
                eval_log_dict = {
                    "tot_steps": self.curr_step,
                    "tot_episodes": self.curr_episode,
                    "eval_steps": eval_steps,
                    "eval_return": eval_return
                }
                self.logger.log(eval_log_dict)

            if self.save_model_eps_freq is not None and self.curr_episode % self.save_model_eps_freq == 0:
                self.save_q()

            step_history.append(eps_steps)
            return_history.append(eps_return)
            self.curr_episode += 1

            if self.is_decaying_eps and self.decay_type == "episode":
                eps = self.decay_epsilon()

            log_dict = {
                "tot_steps": self.curr_step,
                "epsilon": eps,
                "n_joint_goals": len(self._joint_goal_library),
                "eps_return": return_history[-1],
                "eps_steps": step_history[-1],
                "tot_episodes": self.curr_episode
            }
            self.logger.log(log_dict)

        if self.save_model_path is not None:
            self.save_q()

        prog_bar.close()

    @override
    def evaluate(self, joint_start_state, joint_goal=None, max_steps=None) -> Tuple[int, float]:
        if max_steps is None:
            max_steps = self.max_eval_steps

        env = self.env
        joint_state, info = env.reset(joint_start_state)
        # is_done = False
        joint_is_done = np.zeros(self.n_agents, dtype=bool)
        joint_state = tuple(zip(joint_state, joint_is_done))

        is_truncated = False
        curr_step = 0
        curr_return = 0

        # print("EVALUATE")
        while not np.all(joint_is_done) and not is_truncated and curr_step < max_steps:
            for i in range(self.n_agents):
                if not joint_is_done[i]:
                    curr_step += 1

            joint_action = self.greedy_policy(tuple(joint_state), joint_goal)
            joint_state, joint_reward, joint_is_done, is_truncated, info = env.step(joint_action)
            # print(f"joint_state, joint_reward, joint_is_done, is_truncated, info = {joint_state, joint_reward, joint_is_done, is_truncated, info}")
            joint_state = tuple(zip(tuple(joint_state), joint_is_done))

            curr_return += sum(joint_reward)

            # is_done = np.all(joint_is_done)
        # print("END EVALUATE")

        return curr_step, curr_return

    def visualise_episode(
            self,
            joint_start_state,
            joint_goal,
            max_steps=np.inf,
            video_file_path=None,
            wait_ms=500,
    ):
        if video_file_path is not None:
            raise NotImplementedError

        env = self.env

        joint_state, info = tuple(env.reset(joint_start_state))
        env.render()
        # pygame.time.wait(2000)

        is_done = False
        is_dones = np.zeros(self.n_agents, dtype=bool)
        is_truncated = False
        curr_step = 0
        curr_return = 0

        while not is_done and not is_truncated and curr_step < max_steps:
            joint_state = tuple(zip(tuple(joint_state), is_dones))
            joint_action = self.greedy_policy(joint_state, joint_goal)
            pygame.time.wait(wait_ms)
            joint_state, joint_reward, joint_is_done, is_truncated, info = env.step(joint_action)
            env.render()

            curr_return += sum(joint_reward)
            curr_step += 1
            is_done = np.all(joint_is_done)

    def check_convergence(self, tolerance=None) -> bool:
        if self.eval_convergence_type == "full":
            if tolerance is None:
                tolerance = self.eval_conv_tol
            if self.opt_vq is None:
                # self.opt_vq = self.comp_optimal_vq(use_prog_bar=False)
                self.comp_optimal_vq(use_prog_bar=False)

            vq = self.comp_vq(self._q)
            return self.check_vq_equal(self.opt_vq, vq, tolerance)
        elif self.eval_convergence_type == "starts":
            return self.comp_eval_starts_conv()
        else:
            raise ValueError

    @staticmethod
    def comp_vq(wvf):
        vq = {}
        for joint_state in wvf.keys():
            vq[joint_state] = {}
            for joint_goal in wvf[joint_state]:
                vq[joint_state][joint_goal] = wvf[joint_state][joint_goal].max()
        return vq

    @staticmethod
    def check_vq_equal(vq1, vq2, tolerance=0.0, ignore_missing=True):
        for joint_state in vq1.keys():
            if joint_state in vq2.keys():
                for joint_goal in vq1[joint_state].keys():
                    if joint_goal in vq2[joint_state].keys():
                        if abs(vq1[joint_state][joint_goal] - vq2[joint_state][joint_goal]) > tolerance:
                            return False
                    elif not ignore_missing:
                        raise ValueError(f"joint_goal = {joint_goal} not in vq2[{joint_state}]")
            elif not ignore_missing:
                raise ValueError(f"joint_state = {joint_state} not in vq2")

        return True

    def comp_optimal_vq(
            self,
            # opt_vq_costs: Optional[Dict[Any, Dict[Any, int]]] = None,
            use_prog_bar=True
    ) -> Dict[Any, Dict[Any, SupportsFloat]]:
        # opt_vq_costs is independent of task rewards
        all_opt_costs = self.opt_vq_costs

        save_opt_vq_costs_path = self.save_opt_vq_costs_path  # Will save opt_vq_costs to path  if not none
        save_opt_vq_path = self.save_opt_vq_path  # Will save opt_vq to path if not none

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
        goals = [set(agent_goals.keys()) for agent_goals in self.env.goals]

        dijkstra = Dijkstra(grid=self.env.grid, n_agents=self.n_agents,
                            goals=goals, is_collisions=self.env.is_collisions,
                            termination_type=self.env.terminate_mode)

        # all_opt_costs signifies the number of step_rewards accumulated for each joint_state, joint_goal pair
        if all_opt_costs is not None:
            prog_bar = tqdm(total=n_vals, desc="# Costs", disable=not use_prog_bar)
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

            self.opt_vq_costs = all_opt_costs
            if save_opt_vq_costs_path is not None:
                save_file(self.opt_vq_costs, save_opt_vq_costs_path)
        else:
            # TODO: Do type checking & validation
            pass

        goal_rewards = self.env.goal_rewards
        r_step = self.env.step_reward
        opt_vq = {}
        for joint_state in joint_states:
            opt_vq[joint_state] = {}

        for joint_goal in joint_goals:
            for joint_state in joint_states:
                if joint_state in all_opt_costs.keys() and joint_goal in all_opt_costs[joint_state].keys():
                    # Return accumulated from step reward
                    curr_return = all_opt_costs[joint_state][joint_goal] * r_step
                    for i, (state, goal) in enumerate(zip(joint_state, joint_goal)):
                        # This condition can probably be simplified
                        if not is_agent_enter or (is_agent_enter and state not in goals):
                            curr_return += goal_rewards[i][goal]

                    opt_vq[joint_state][joint_goal] = curr_return

        if save_opt_vq_path is not None:
            save_file(opt_vq, save_opt_vq_path)

        self.opt_vq = opt_vq
        return opt_vq

    def save_opt_vq(self, file_path):
        save_file(self.opt_vq, file_path)

    def save_opt_vq_costs(self, file_path):
        save_file(self.opt_vq_costs, file_path)

    def comp_opt_eval_returns(self):
        opt_eval_returns = {}
        opt_eval_steps = {}
        joint_starts = self.eval_conv_joint_starts
        joint_goals = self.env.joint_goals

        for js in joint_starts:
            opt_eval_returns[js] = {}
            opt_eval_steps[js] = {}
            for jg in joint_goals:
                opt_return, opt_steps, _ = self.env.calc_opt_joint_return_steps(js, jg)
                opt_eval_returns[js][jg] = opt_return
                opt_eval_steps[js][jg] = opt_steps
        self.opt_eval_returns = opt_eval_returns
        self.opt_eval_steps = opt_eval_steps

    def comp_eval_starts_conv(self):
        assert self.opt_eval_returns is not None
        logger = self.logger
        max_steps = self.max_eval_steps
        tol = self.eval_conv_tol

        log_verbosity = self._log_verbosity["eval_conv"]
        #
        # if log_verbosity_str is None:
        #     log_verbosity = 0
        # elif log_verbosity_str == "agr":
        #     log_verbosity = 1
        # elif log_verbosity_str == "js_agr":
        #     log_verbosity = 2
        # elif log_verbosity_str == "all":
        #     log_verbosity = 3
        #     raise NotImplementedError
        # else:
        #     raise ValueError(f"log_verbosity_str = {log_verbosity_str} is invalid")
        # is_log_agr = log_verbosity == "agr" or log_verbosity == "all"
        # is_log_all = log_verbosity == "all"

        is_converged = True
        joint_goals = self.env.joint_goals

        n_els = 0

        min_n_converged = np.inf
        max_n_converged = -np.inf
        sum_n_converged = 0
        avg_n_converged = 0

        min_diff_return = np.inf
        max_diff_return = -np.inf
        sum_diff_return = 0
        avg_diff_return = 0

        min_diff_steps = np.inf
        max_diff_steps = -np.inf
        sum_diff_steps = 0
        avg_diff_steps = 0

        js_metrics = {}
        for key in ["diff_return", "diff_steps"]:
            js_metrics[key] = {
                "min": {},
                "max": {},
                "avg": {},
                "sum": {}
            }
        js_metrics["n_converged"] = {}
        js_metrics["perc_converged"] = {}

        for js in self.eval_conv_joint_starts:
            curr_min_diff_return = np.inf
            curr_max_diff_return = -np.inf
            curr_sum_diff_return = 0

            curr_min_diff_steps = np.inf
            curr_max_diff_steps = -np.inf
            curr_sum_diff_steps = 0

            curr_n_converged = 0

            for jg in joint_goals:
                eval_steps, eval_return = self.evaluate(js, jg, max_steps=max_steps)
                opt_steps = self.opt_eval_steps[js][jg]
                opt_return = self.opt_eval_returns[js][jg]
                diff_steps = opt_steps - eval_steps
                diff_return = opt_return - eval_return
                if abs(diff_return) <= tol:
                    curr_n_converged += 1
                else:
                    is_converged = False

                curr_min_diff_return = min(curr_min_diff_return, diff_return)
                curr_max_diff_return = max(curr_max_diff_return, diff_return)
                curr_sum_diff_return += diff_return

                curr_min_diff_steps = min(curr_min_diff_steps, diff_steps)
                curr_max_diff_steps = max(curr_max_diff_steps, diff_steps)
                curr_sum_diff_steps += diff_steps

            js_metrics["diff_return"]["min"][js] = curr_min_diff_return
            js_metrics["diff_return"]["max"][js] = curr_max_diff_return
            js_metrics["diff_return"]["avg"][js] = curr_sum_diff_return/len(joint_goals)

            js_metrics["diff_steps"]["min"][js] = curr_min_diff_steps
            js_metrics["diff_steps"]["max"][js] = curr_max_diff_steps
            js_metrics["diff_steps"]["avg"][js] = curr_sum_diff_steps / len(joint_goals)

            js_metrics["n_converged"][js] = curr_n_converged
            js_metrics["perc_converged"][js] = curr_n_converged/len(joint_goals)

            min_diff_return = min(curr_min_diff_return, min_diff_return)
            max_diff_return = max(curr_max_diff_return, max_diff_return)
            sum_diff_return += curr_sum_diff_return

            min_diff_steps = min(curr_min_diff_steps, min_diff_steps)
            max_diff_steps = max(curr_max_diff_steps, max_diff_steps)
            sum_diff_steps += curr_sum_diff_steps

            min_n_converged = min(min_n_converged, curr_n_converged)
            max_n_converged = max(max_n_converged, curr_n_converged)
            sum_n_converged += curr_n_converged

        n_joint_states = len(self.eval_conv_joint_starts)
        n_joint_goals = len(joint_goals)
        tot_els = n_joint_states * n_joint_goals

        avg_n_converged = sum_n_converged/tot_els
        avg_diff_steps = sum_diff_steps/tot_els
        avg_diff_return = sum_diff_return/tot_els

        log_dict = {
            "tot_steps": self.curr_step,
            "tot_episodes": self.curr_episode
        }
        # Agr
        if log_verbosity > 0:
            log_dict["eval_avg_diff_return"] = avg_diff_return
            log_dict["eval_avg_diff_steps"] = avg_diff_steps
            log_dict["eval_avg_n_converged"] = avg_n_converged

            log_dict["eval_max_diff_return"] = max_diff_return
            log_dict["eval_max_diff_steps"] = max_diff_steps
            log_dict["eval_max_n_converged"] = max_n_converged

            log_dict["eval_min_diff_return"] = min_diff_return
            log_dict["eval_min_diff_steps"] = min_diff_steps
            log_dict["eval_min_n_converged"] = min_n_converged

        if log_verbosity > 1:
            for js in self.eval_conv_joint_starts:
                for agr_type in ["avg", "min", "max"]:
                    for val1 in ["diff"]:  # "opt", "act",
                        for val2 in ["return", "steps"]:
                            # js_metrics["diff_return"]["min"][js]
                            log_key = f"eval_{js}_{agr_type}_{val1}_{val2}"
                            log_dict[log_key] = js_metrics[f"{val1}_{val2}"][agr_type][js]

                log_dict[f"eval_{js}_n_converged"] = js_metrics["n_converged"][js]
                log_dict[f"eval_{js}_perc_converged"] = js_metrics["perc_converged"][js]

        logger.log(log_dict)

        return is_converged


class InferJointGOLearner(JointGOLearner):
    def __init__(
            self,
            env,
            learn_goal_rewards: Union[List[Dict[Any, bool]], Dict[Any, bool]],
            # infer_tas
            *args,
            infer_tasks: Optional[List[Union[List[Dict[Any, bool]], Dict[Any, bool]]]] = None,
            log_all_tasks=False,
            learner_config: Optional[Dict] = None,
            wandb_kwargs: Optional[Dict] = None,
            logger_metrics: Optional[List[Tuple[str, str]]] = None,
            **kwargs
    ):
        all_args = copy.copy(locals())
        del all_args["self"]
        if "q" in all_args and all_args["q"] is not None and type(all_args["q"]) != list:
            all_args["q"] = f"{type(all_args['q'])}"

        del all_args["env"]
        del all_args["learner_config"]
        del all_args["__class__"]
        del all_args["wandb_kwargs"]

        curr_config = conv_to_wandb(all_args)
        curr_config["name"] = "InferJointGOLearner"

        copy_to_dict(kwargs, curr_config)
        del curr_config["kwargs"]

        if learner_config is None:
            learner_config = {}

        copy_to_dict(curr_config, learner_config)

        self.infer_tasks = infer_tasks
        self.log_all_tasks = log_all_tasks

        new_logger_metrics = ["avg_task_opt_steps", "avg_task_opt_return",
                              "avg_task_act_steps", "avg_task_act_return",
                              "avg_task_diff_steps", "avg_task_diff_return",
                              "n_tasks_completed"]

        if log_all_tasks:
            for i in range(len(infer_tasks)):
                # task_logger_metrics = [""]
                task_logger_metrics = [f"{i}_task_opt_steps", f"{i}_task_opt_return",
                                      f"{i}_task_act_steps", f"{i}_task_act_return",
                                      f"{i}_task_diff_steps", f"{i}_task_diff_return"]
                new_logger_metrics.extend(task_logger_metrics)

        new_logger_metrics = [(metric, "tot_steps") for metric in new_logger_metrics]

        if logger_metrics is None:
            logger_metrics = new_logger_metrics
        else:
            logger_metrics.extend(new_logger_metrics)

        super().__init__(env, *args,
                         learner_config=learner_config,
                         wandb_kwargs=wandb_kwargs, logger_metrics=logger_metrics,
                         **kwargs)

        if type(learn_goal_rewards) == dict:
            learn_goal_rewards = [learn_goal_rewards for _ in range(self.n_agents)]
        elif type(learn_goal_rewards) == list:
            assert len(learn_goal_rewards) == self.n_agents, f"learn_goal_rewards must be of length n_agents={self.n_agents}"
        else:
            raise TypeError(f"learn_goal_rewards is of wrong type. type(learn_goal_rewards) = {type(learn_goal_rewards)}")
        self._learn_goal_rewards = learn_goal_rewards

        self._infer_goal_rewards = None


    def set_infer_rewards(
            self,
            infer_goal_rewards: Union[List[Dict[Any, bool]], Dict[Any, bool]]
    ):
        if type(infer_goal_rewards) == dict:
            infer_goal_rewards = [infer_goal_rewards for _ in range(self.n_agents)]
        elif type(infer_goal_rewards) == list:
            assert len(infer_goal_rewards) == self.n_agents, f"infer_goal_rewards must be of length n_agents={self.n_agents}"
        else:
            raise TypeError(f"infer_goal_rewards is of wrong type. type(infer_goal_rewards) = {type(infer_goal_rewards)}")

        self._infer_goal_rewards = infer_goal_rewards

    @staticmethod
    def _calc_goal_reward_sum(joint_state, joint_goal, goal_rewards):
        reward = 0
        # Termination status is concatenated to state
        for i, ((state, is_term), goal) in enumerate(zip(joint_state, joint_goal)):
            if not is_term:
                agent_goal_rewards = goal_rewards[i]
                reward += agent_goal_rewards[goal]

        return reward

    def _calc_q(self, joint_state, joint_goal):
        assert self._infer_goal_rewards is not None
        infer_goal_rewards = self._infer_goal_rewards
        learn_goal_rewards = self._learn_goal_rewards

        learn_term_val = self._calc_goal_reward_sum(joint_state, joint_goal, learn_goal_rewards)
        infer_term_val = self._calc_goal_reward_sum(joint_state, joint_goal, infer_goal_rewards)

        q_arr = self._q[joint_state][joint_goal] - learn_term_val + infer_term_val
        return q_arr

    def greedy_policy(self, joint_state, joint_goal=None):
        q = self._q
        if joint_goal is not None:
            q_arr = self._calc_q(joint_state, joint_goal)  # q[joint_state][joint_goal]
            flat_ind = np.random.choice(np.flatnonzero(q_arr == q_arr.max()))
            tmp = list(np.unravel_index(flat_ind, q_arr.shape))
            # flat_ind = np.random.choice(np.flatnonzero(q[joint_state][joint_goal] == q[joint_state][joint_goal].max()))
            # tmp = list(np.unravel_index(flat_ind, q[joint_state][joint_goal].shape))
            return tmp
        else:
            if len(q[joint_state].keys()) == 0:
                # We must use random policy if there are no values to use
                return self.random_policy()
            else:
                q_arrs = [self._calc_q(joint_state, g) for g in q[joint_state].keys()]
                # q_arrs = [q[joint_state][g] for g in q[joint_state].keys()]
                qvf = np.max(q_arrs, axis=0)
                # take argmax on QVF
                flat_ind = np.random.choice(np.flatnonzero(qvf == qvf.max()))
                joint_action = list(np.unravel_index(flat_ind, qvf.shape))
                return joint_action

    def evaluate_infer_tasks(self):
        pass

    # def eval_infer_multi(self, tasks: List[Union[List[Dict[Any, bool]], Dict[Any, bool]]]):
    #     for task in tasks:
    #         pass
        # env: GridWorld = self.env
        # n_tasks_completed = 0
        # for curr_task in tasks:
        #     env.modify_goal_values(curr_task)
        #     infer_goal_rewards = copy.copy(env.goal_rewards)
        #     self.set_infer_rewards(infer_goal_rewards)

    def evaluate_task(
            self,
            task,
            joint_start: Tuple[Tuple[int, int], ...],
            tol=0,
            max_steps=1000
    ):
        env = self.env
        orig_task = copy.deepcopy(env.goals)
        env.modify_goal_values(task)
        self.set_infer_rewards(env.goal_rewards)

        # Do stuff
        opt_return, opt_steps, opt_info = env.calc_opt_joint_return_steps(joint_start)
        act_steps, act_return = self.evaluate(joint_start, max_steps=max_steps)
        task_completed = abs(act_return - opt_return) <= tol

        # print("Opt path: ", info["best_path"])
        # self.env.animate_path(info["best_path"])

        # Modify back
        env.modify_goal_values(orig_task)

        return (opt_return, act_return), (opt_steps, act_steps), opt_info, task_completed


# def pointwise_add(arrs: List[np.ndarray]):
#     if len(arrs) == 2:
#         a, b = arrs[0], arrs[1]
#         assert a.shape == b.shape
#         n_els = a.shape[0]
#         a = a.reshape((n_els, 1))
#         b = b.reshape((1, n_els))
#
#         ar = np.repeat(a, n_els, axis=1)
#         br = np.repeat(b, n_els, axis=0)
#
#         return ar + br
#     else:
#         raise NotImplementedError

def main():
    pass


if __name__ == "__main__":
    main()
    # test()
