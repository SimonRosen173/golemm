import copy
import time
from abc import abstractmethod
from typing import Optional, Dict, Tuple, List

import numpy as np
import wandb
from tqdm import tqdm

from golem.environment.gridworld.gridworld import GridWorld
from golem.utils import copy_to_dict
from golem.logger import Logger

from golem.learning.utils import conv_to_wandb

# import faulthandler
# faulthandler.enable()


class EpsilonDecayer:
    # Decay epsilon by decay_rate each step
    def __init__(self, start_eps, end_eps, eps_steps=None, decay_rate=None):
        if decay_rate is None:
            decay_rate = (start_eps - end_eps)/eps_steps

        if eps_steps is None and decay_rate is None:
            raise ValueError("eps_steps and decay_rate cannot both be None, one of them must be specified")

        self.decay_rate = decay_rate
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.eps_steps = eps_steps

        self.curr_eps = self.start_eps

    def decay(self):
        if self.curr_eps - self.decay_rate >= self.end_eps:
            self.curr_eps -= self.decay_rate

        return self.curr_eps


class Learner:
    def __init__(
            self,
            env: GridWorld,
            max_steps=np.inf, max_episodes=np.inf,
            alpha=1, gamma=1,
            # Evaluation
            eval_conv_freq_eps=None,
            eval_conv_freq_steps=None,
            eval_freq_eps=None,
            eval_freq_steps=None,
            max_eval_steps=1000,
            eval_joint_start=None,
            eval_conv_tol=1e-4,
            # ...
            epsilon=0.1,
            is_decaying_eps=False,
            decay_type="step",  # step or episode
            start_epsilon=0.99, end_epsilon=0.01, decay_rate=None, epsilon_steps=None,
            # Logging stuff
            log_mode: str = "wandb",
            learner_config: Optional[Dict] = None,
            wandb_kwargs: Optional[Dict] = None,
            logger_metrics: Optional[List[Tuple[str, str]]] = None,
            use_prog_bar=True,
            run_config_path: Optional[str] = None,
            exp_config_path: Optional[str] = None,
            save_model_path: Optional[str] = None,
            save_model_eps_freq: Optional[int] = None,
            save_model_step_freq: Optional[int] = None
            # prog_bar_kwargs: Optional[Dict] = None
    ):
        curr_learner_config = copy.copy(locals())
        env_config = curr_learner_config["env"].config

        del curr_learner_config["env"]
        del curr_learner_config["learner_config"]
        del curr_learner_config["self"]
        del curr_learner_config["wandb_kwargs"]

        curr_learner_config["name"] = self.__class__.__name__

        copy_to_dict(learner_config, curr_learner_config)

        # _locals = locals().copy()
        self.config = {
            "learner": conv_to_wandb(curr_learner_config),
            "env": conv_to_wandb(env_config)
        }

        # config["env"] = config["env"].config
        # print(_locals)
        # exit()

        self.env = env
        self.n_agents = env.n_agents
        self.n_agent_actions = env.n_agent_actions
        self.actions_shape = tuple([self.n_agent_actions for _ in range(self.n_agents)])

        ###################
        # GEN HYPERPARAMS #
        ###################
        self.max_eval_steps = max_eval_steps
        self.eval_joint_start = eval_joint_start

        self.eval_conv_freq_eps = eval_conv_freq_eps
        self.eval_conv_freq_steps = eval_conv_freq_steps
        self.eval_freq_steps = eval_freq_steps
        self.eval_freq_eps = eval_freq_eps

        self.curr_step = 0
        self.curr_episode = 0
        self.is_converged = False

        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma

        # self.eval_conv_freq_eps = eval_conv_freq_eps
        # self.eval_conv_freq_steps = eval_conv_freq_steps
        self.eval_conv_tol = eval_conv_tol

        ###########
        # EPSILON #
        ###########
        self.is_decaying_eps = is_decaying_eps
        self._epsilon = epsilon
        self.decay_type = decay_type
        if is_decaying_eps:
            self._eps_decayer = EpsilonDecayer(start_eps=start_epsilon, end_eps=end_epsilon,
                                               eps_steps=epsilon_steps, decay_rate=decay_rate)
        else:
            self._eps_decayer = None

        #########
        # OTHER #
        #########
        self.save_model_eps_freq = save_model_eps_freq
        self.save_model_step_freq = save_model_step_freq
        self.save_model_path = save_model_path

        ##########
        # LOGGER #
        ##########
        if wandb_kwargs is None:
            wandb_kwargs = {
                "mode": "disabled"
            }

        if "config" in wandb_kwargs:
            raise ValueError("wandb config param should be specified by wandb_config not in wandb_hyperparams")

        if logger_metrics is None:
            logger_metrics = []

        default_metrics = [
            ("tot_steps", None),
            ("tot_episodes", "tot_steps"),
            ("eps_steps", "tot_episodes"),
            ("eps_return", "tot_episodes"),
            ("epsilon", "tot_steps"),
            ("eval_steps", "tot_steps"),
            ("eval_return", "tot_steps")
        ]
        logger_metrics.extend(default_metrics)

        self.logger = Logger(metrics=logger_metrics, log_mode=log_mode, config=self.config,
                             wandb_kwargs=wandb_kwargs)

        if run_config_path is not None:
            self.logger.log_file(run_config_path)

        if exp_config_path is not None:
            self.logger.log_file(exp_config_path)

        # self.wandb_run: wandb.run = wandb.init(reinit=True, **wandb_kwargs)

        # if wandb_metrics is not None:
        #     new_wandb_metrics = [
        #         ("tot_steps", None),
        #         ("tot_episodes", "tot_steps"),
        #         ("eps_steps", "tot_episodes"),
        #         ("eps_return", "tot_episodes"),
        #         ("epsilon", "tot_steps"),
        #         ("eval_steps", "tot_steps"),
        #         ("eval_return", "tot_steps")
        #     ]
        #     wandb_metrics.extend(new_wandb_metrics)
        #
        #     for name, step_metric in wandb_metrics:
        #         self.wandb_run.define_metric(name, step_metric=step_metric)

        ########
        # TQDM #
        ########
        self._use_prog_bar = use_prog_bar
        # if prog_bar_kwargs is None:
        #     prog_bar_kwargs = {
        #         "disable": True
        #     }
        #     self._use_prog_bar = False
        # else:
        #     self._use_prog_bar = True

        # self._prog_bar = tqdm(**prog_bar_kwargs)

    # def stop_condition(self) -> bool:
    #     return self.curr_step > self.max_steps or self.curr_episode > self.max_episodes or self.is_converged

    def random_policy(self) -> List[int]:
        # noinspection PyTypeChecker
        return np.random.randint(0, self.n_agent_actions, self.n_agents).tolist()

    def act(self, joint_state, epsilon: float, *args, **kwargs) -> List[int]:
        if np.random.rand() < epsilon:
            return self.random_policy()
        else:
            return self.greedy_policy(joint_state, *args, **kwargs)

    @abstractmethod
    def check_convergence(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, joint_start_state, *args, max_steps=np.inf, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def greedy_policy(self, joint_state, *args, **kwargs):
        raise NotImplementedError

    def decay_epsilon(self):
        if self.is_decaying_eps:
            return self._eps_decayer.decay()
        else:
            raise ValueError

    @property
    def epsilon(self):
        if self.is_decaying_eps:
            return self._eps_decayer.curr_eps
        else:
            return self._epsilon

    def vis_episode(self, start_state, step_time=0.2):
        env = self.env
        s, _ = env.reset(start_state)
        is_done = False
        eps_return = 0
        eps_steps = 0
        is_dones = np.zeros((self.n_agents, ), dtype=bool)

        while not is_done:
            env.render(render_mode="human")
            time.sleep(step_time)

            act_s = tuple(zip(s, is_dones))
            a = self.act(act_s, epsilon=0)
            s, r, is_dones, is_truncated, info = env.step(a)
            print(a)
            eps_return += r
            eps_steps += 1
            is_done = np.all(is_dones) or is_truncated

        print(f"Episode return = {eps_return}")
        print(f"Episode steps = {eps_steps}")

    def finish(self):
        self.logger.finish()

    def close(self):
        self.finish()
