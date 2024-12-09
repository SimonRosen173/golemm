def run(config, run_type):
    if run_type == "train_to_convergence":
        train_to_convergence(config)
    elif run_type == "train_until_quota":
        train_until_quota(config)
    else:
        raise ValueError(f"run_type = {run_type} is not supported")


def train_to_convergence(config):
    from golem.environment.gridworld.gridworld import GridWorld

    env_config = config["env"]
    env_params = env_config["params"]
    if env_config["name"] == "GridWorld":
        env = GridWorld(**env_params)
    else:
        raise ValueError(f"env = {env_config['name']} is not supported")

    learner_config = config["learner"]
    learner_params = config["learner"]["params"]
    if learner_config["name"] == "JointGOLearner":
        from golem.learning.golearner import JointGOLearner
        learner = JointGOLearner(env=env, **learner_params)
    else:
        raise ValueError(f"learner = {learner_config['name']} is not supported")

    learner.learn()
    learner.finish()


def train_until_quota(config):
    raise NotImplementedError
