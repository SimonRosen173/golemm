import copy
import itertools
import os
from typing import Tuple

from golem.environment.gridworld.gridworld import GridWorld, interactive
from golem.environment.utils import load_config
from golem.learning.qlearner import QLearner
from golem.learning.golearner import JointGOLearner, InferJointGOLearner


ENTITY = "simonrosen42"
GROUP = "debug"
PROJECT = "golem-gs-tab"


def create_env(map_name, n_agents,
               eval_goals, eval_starts,
               is_collisions=True, reward_out_mode="joint"):
    # n_agents = 2
    # # map_name = "4x4"
    # is_collisions = True

    config = load_config(f"{map_name}.json")
    named_goals = config["named_goals"]
    goals = {goal: True for goal in named_goals.values()}
    rewards = config["rewards"]
    named_starts = config["named_starts"]
    eval_joint_start = tuple([named_starts[start] for start in eval_starts])
    # eval_starts = config["eval_joint_starts"]

    # eval_joint_start = (named_starts["ML"], named_starts["MR"])
    # eval_joint_start = ((1, 1), (1, 2))
    # eval_joint_goal = ((0, 0), (0, 3))
    eval_joint_goal = tuple([named_goals[goal] for goal in eval_goals])
    # eval_joint_goal = (named_goals["L"], named_goals["R"])

    env = GridWorld(
        config["grid"], n_agents, goals,
        reward_out_mode=reward_out_mode, terminate_mode="agent_wait",
        is_collisions=is_collisions,
        is_virt_term_state=False,
        max_episode_steps=config["max_episode_steps"],
        desirable_reward=rewards["desirable"], undesirable_reward=rewards["undesirable"],
        step_reward=rewards["step"], r_min=rewards["r_min"],
    )

    return env, eval_joint_start, eval_joint_goal, config


def test_learn():
    wandb_kwargs = {
        "entity": ENTITY,
        "project": PROJECT,
        "group": "debug",
        "job_type": "debug",
        "mode": "online",
        "notes": "WVF"
    }

    map_name = "4x4"
    n_agents = 2
    env, eval_joint_start, eval_joint_goal, config = create_env(map_name, n_agents,
                                                                eval_starts=["TL", "TR"],
                                                                eval_goals=["BR", "BL"],
                                                                reward_out_mode="joint")
    eval_starts = [(2, 1), (2, 2), (1, 1), (1, 2)]
    eval_joint_starts = list(itertools.permutations(eval_starts, r=2))
    # eval_joint_starts = itertools.product(eval_starts, eval_starts)

    # env, eval_joint_start, eval_joint_goal, config = create_env(map_name, n_agents)
    eval_joint_goal = None  # I.e. evaluate based using argmax_a{max_g{Q(s,g,a)}}
    q_init_mode = "zero"
    max_steps = 1000000

    # interactive(env)

    # interactive(env)
    model_path = f"models/wvf_{map_name}_{q_init_mode}_{max_steps}.pkl"

    is_train = True
    if is_train:
        # wandb_kwargs["mode"] = "online"
        goal_library = [set(agent_goals.keys()) for agent_goals in env.goals]
        q = None
    else:
        wandb_kwargs["mode"] = "disabled"
        goal_library = None
        q = model_path

    learner = JointGOLearner(env, wandb_kwargs=wandb_kwargs,
                             q_init_mode=q_init_mode,
                             eval_convergence_type="starts",
                             comp_opt_vq=False, eval_conv_joint_starts=eval_joint_starts,
                             goal_library=goal_library, q=q,
                             log_verbosity={"eval_conv": "js_agr"},
                             # eval_joint_start=eval_joint_start, eval_joint_goal=eval_joint_goal,
                             epsilon=0.25)
    # learner.load_q(model_path)
    # learner.save_q(model_path)

    # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=None)
    # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=eval_joint_goal)

    if is_train:
        # learner.learn(no_steps=max_steps, eval_freq_eps=250)
        learner.learn(no_steps=max_steps, eval_conv_freq_steps=1000)
        # learner.save_q(model_path)
    else:
        learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=eval_joint_goal)
    learner.close()


def test_qlearn():
    wandb_kwargs = {
        "entity": ENTITY,
        "project": PROJECT,
        "group": "debug",
        "job_type": "debug",
        "mode": "online",
        "notes": "Q"
    }

    map_name = "5_rooms_4g"
    n_agents = 2
    env, eval_joint_start, eval_joint_goal, config = create_env(map_name, n_agents,
                                                                ["TL", "TR"], ["TL", "TR"],
                                                                reward_out_mode="sum")
    # eval_joint_goal = None  # I.e. evaluate based using argmax_a{max_g{Q(s,g,a)}}
    q_init_mode = "zero"
    max_steps = 250000

    # interactive(env)

    # interactive(env)
    model_path = f"models/q_{map_name}_{q_init_mode}_{max_steps}.pkl"

    is_train = True
    if is_train:
        # wandb_kwargs["mode"] = "online"
        goal_library = [set(agent_goals.keys()) for agent_goals in env.goals]
        q = None
    else:
        wandb_kwargs["mode"] = "disabled"
        goal_library = None
        q = model_path

    learner = QLearner(env, wandb_kwargs=wandb_kwargs,
                       # q_init_mode=q_init_mode,
                       # comp_opt_vq=False, q=q,
                       eval_joint_start=eval_joint_start, epsilon=0.25)
    # learner.load_q(model_path)
    # learner.save_q(model_path)

    # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=None)
    # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=eval_joint_goal)

    if is_train:
        learner.learn(no_steps=max_steps, eval_freq_eps=250)
        learner.save_q(model_path)
    else:
        raise NotImplementedError
        # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=eval_joint_goal)
    learner.close()


def test_infer():
    wandb_kwargs = {
        "entity": ENTITY,
        "project": PROJECT,
        "group": "task-gen",
        "job_type": "debug",
        "mode": "disabled",
        "notes": ""
    }

    map_name = "4x4"
    n_agents = 2
    env, eval_joint_start, eval_joint_goal, config = create_env(map_name, n_agents, ["TL", "TR"], ["TL", "TR"])
    q_init_mode = "zero"
    max_steps = 1000000

    eval_joint_goal = None  # I.e. evaluate based using argmax_a{max_g{Q(s,g,a)}}
    named_goals = config["named_goals"]
    # named_starts = config["named_starts"]
    # eval_joint_start[]

    # interactive(env)
    #
    # interactive(env)
    model_path = f"models/q_{map_name}_{q_init_mode}_{max_steps}.pkl"

    wandb_kwargs["mode"] = "online"
    goal_library = None
    q = model_path
    learn_goal_rewards = copy.copy(env.goal_rewards)

    infer_tasks = [
        [
            {
                named_goals["TL"]: False, named_goals["TR"]: False,
                named_goals["BL"]: False, named_goals["BR"]: True
            },
            {
                named_goals["TL"]: False, named_goals["TR"]: True,
                named_goals["BL"]: False, named_goals["BR"]: False
            }
        ],
        [
            {
                named_goals["TL"]: False, named_goals["TR"]: True,
                named_goals["BL"]: False, named_goals["BR"]: True
            },
            {
                named_goals["TL"]: False, named_goals["TR"]: True,
                named_goals["BL"]: False, named_goals["BR"]: True
            }
        ]
    ]

    learner = InferJointGOLearner(env=env, learn_goal_rewards=learn_goal_rewards, infer_tasks=infer_tasks,
                                  wandb_kwargs=wandb_kwargs,
                                  goal_library=goal_library, q=q, comp_opt_vq=False,
                                  eval_joint_start=eval_joint_start, eval_joint_goal=eval_joint_goal)

    infer_goals = [
        {
            named_goals["TL"]: False, named_goals["TR"]: False,
            named_goals["BL"]: False, named_goals["BR"]: True
        },
        {
            named_goals["TL"]: False, named_goals["TR"]: True,
            named_goals["BL"]: False, named_goals["BR"]: False
        }
    ]
    eval_joint_start = ((1, 0), (1, 3))

    (opt_return, act_return), (opt_steps, act_steps), opt_info, task_completed = learner.evaluate_task(infer_goals, eval_joint_start)
    print(f"Return: \n")
    print(f"\t Optimal: {opt_return}")
    print(f"\t Actual: {act_return}\n")

    print(f"Steps: \n")
    print(f"\t Optimal: {opt_steps}")
    print(f"\t Actual: {act_steps}\n")
    #
    print(f"Task completed: {task_completed}")

    env.modify_goal_values(infer_goals)
    infer_goal_rewards = copy.copy(env.goal_rewards)
    print(infer_goal_rewards)
    learner.set_infer_rewards(infer_goal_rewards)
    # interactive(env)

    # learner.visualise_episode(joint_start_state=eval_joint_start, joint_goal=None)
    learner.close()


if __name__ == "__main__":
    test_learn()
    # test_qlearn()
    # test_infer()
