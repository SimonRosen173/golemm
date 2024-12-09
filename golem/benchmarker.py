import os
import subprocess
import sys
sys.path.append('..')


# BENCHMARKER:
# 1. Init env
# 2. Init learner
# 3. Do learning
# 4. Do evaluation
# 5. Handle logging

PATH_SEP = os.path.sep
BASE_PATH_ARR = os.path.abspath(__file__).split(PATH_SEP)[:-1]

# RUNS_PATH = BASE_PATH_ARR.copy()
# RUNS_PATH.append("runs")
# RUNS_PATH = PATH_SEP.join(RUNS_PATH)

CONFIGS_PATH = BASE_PATH_ARR.copy()
CONFIGS_PATH.append("exp_configs")
CONFIGS_PATH = PATH_SEP.join(CONFIGS_PATH)

BASE_PATH = os.path.normpath(PATH_SEP.join(BASE_PATH_ARR))
# CONFIGS_PATH = f"{os.path.abspath(__file__)}\\configs"

RUNS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "runs"))
SLURMS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "slurms"))
QSUBS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "qsubs"))

PROJECT_NAME = "AAMAS24-GOLEMM-RL"
ENTITY = "simonrosen42"

# TODO: Fix
CLUSTER_CODE_PATH = "/home-mscluster/srosen/code/zero-shot-comp-marl/mzeroshot/"
CLUSTER_LOGS_PATH = "/home-mscluster/srosen/exp_logs"
CLUSTER_CODE_LOGS_PATH = "/home-mscluster/srosen/logs/zero-shot-comp-marl/"
CLUSTER_JOB_LOGS_PATH = "/home-mscluster/srosen/cluster_logs/"

CHPC_WANDB_PATH = "/mnt/lustre/users/srosen/wandb_data"
CHPC_CODE_PATH = "/mnt/lustre/users/srosen/code/zero-shot-comp-marl/mzeroshot/"
CHPC_LOGS_PATH = "/mnt/lustre/users/srosen/exp_logs"
CHPC_WALL_TIME = "36:00:00"
# CHPC_CODE_LOGS_PATH = "/mnt/lustre/users/srosen/exp_logs/"
# CHPC_JOB_LOGS_PATH = "/home-mscluster/srosen/cluster_logs/"


def print_system_specs():
    import platform, multiprocessing, psutil
    print("################")
    print("# SYSTEM SPECS #")
    print("################")
    print("|    GENERAL   |")
    print("|--------------|")

    print(f"Machine: {platform.machine()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Platform: {platform.platform()}")
    print(f"Platform System: {platform.system()}")
    print(f"Platform Release: {platform.release()}")
    print(f"Platform Version: {platform.version()}")

    print("|--------------|")
    print("|     CPU      |")
    print("|--------------|")
    print(f"Processor Name: {platform.processor()}")
    print(f"No cores: {multiprocessing.cpu_count()}")

    print("|--------------|")
    print("|     RAM      |")
    print("|--------------|")
    print(f"Ram: {str(round(psutil.virtual_memory().total / (1024.0 ** 3), 2)) + ' GB'}")
    print("################")


def create_or_clear_folder(folder_path):
    import shutil

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.mkdir(folder_path)


class Benchmarker:
    def __init__(self,
                 exp_config_path="",
                 is_local=False
                 ):

        exp_config_path = os.path.normpath(os.path.join(BASE_PATH, exp_config_path))
        self.exp_config = load_exp_config(file_path=exp_config_path) #, file_name=exp_config_name)

        cluster_config = self.exp_config["cluster"]
        self.cluster_name = cluster_config["cluster_name"]
        self.partition = cluster_config["partition"]
        self.max_nodes = cluster_config["max_nodes"]
        self.job_name_prefix = cluster_config["job_name_prefix"]
        self.run_logs_folder = cluster_config["run_logs_folder"]

        if self.cluster_name == "msl":
            self.slurm_logs_folder = cluster_config["slurm_logs_folder"]
        elif self.cluster_name == "chpc":
            self.qsubs_logs_folder = cluster_config["qsub_logs_folder"]

        # self.qsub_logs_folder = cluster_config["qsub_logs_folder"]
        self.runs_per_node = cluster_config["runs_per_node"]

        # self.runs_per_node = runs_per_node
        self.no_runs = 0
        self.n_nodes = 0
        # self.log_folder = log_folder
        self.slurms_folder = None
        self.qsubs_folder = None

        if self.cluster_name == "msl":
            self.slurms_folder = os.path.join(SLURMS_FOLDER, cluster_config["slurm_folder"])
            self.slurm_logs_folder = f"{CLUSTER_LOGS_PATH}/slurms/{self.slurm_logs_folder}"
            self.qsubs_logs_folder = None
            self.run_logs_folder = f"{CLUSTER_LOGS_PATH}/runs/{self.run_logs_folder}"
            self.wandb_base_folder = None
        elif self.cluster_name == "chpc":
            self.qsubs_folder = os.path.join(QSUBS_FOLDER, cluster_config["qsubs_folder"])
            self.qsubs_logs_folder = f"{CHPC_LOGS_PATH}/qsubs/{self.qsubs_logs_folder}"
            self.slurm_logs_folder = None
            self.run_logs_folder = f"{CHPC_LOGS_PATH}/runs/{self.run_logs_folder}"
            self.wandb_base_folder = f"{CHPC_WANDB_PATH}/{cluster_config['wandb_folder']}"

        self.runs_folder = os.path.join(RUNS_FOLDER, cluster_config["runs_folder"])


        # self.max_nodes = max_nodes
        # self.runs_per_node = runs_per_node
        # self.partition = partition
        # self.job_name_prefix = job_name_prefix
        # self.no_runs = 0
        # self.n_nodes = 0
        # self.log_folder = log_folder
        self.is_local = is_local

        if not is_local:
            self.make_log_folders()
            if self.cluster_name == "chpc":
                create_or_clear_folder(self.wandb_base_folder)

    def make_log_folders(self):
        # CLUSTER_LOGS_PATH =
        # code_logs_path = f"{CLUSTER_CODE_LOGS_PATH}{self.log_folder}"
        # job_logs_path = f"{CLUSTER_JOB_LOGS_PATH}{self.log_folder}"
        create_or_clear_folder(self.run_logs_folder)
        if self.cluster_name == "msl":
            create_or_clear_folder(self.slurm_logs_folder)
        elif self.cluster_name == "chpc":
            create_or_clear_folder(self.qsubs_logs_folder)
        else:
            raise ValueError

    def create_run_configs(self):
        import copy
        import itertools
        import pickle

        import numpy as np

        from golem.utils import copy_to_dict

        create_or_clear_folder(self.runs_folder)

        exp_config = self.exp_config
        all_named_goals = exp_config["env_config"]["named_goals"]
        if "step_quota" not in exp_config:
            exp_config["step_quota"] = [None for _ in exp_config["named_goals"]]
        self.no_runs = 0
        # clear_folder(RUNS_PATH)
        # learners, repeats_per_run
        for step_quota, named_goals_keys, desirable_joint_goals, no_goals in zip(exp_config["step_quota"],
                                                                                exp_config["named_goals"],
                                                                                exp_config["desirable_joint_goals"],
                                                                                exp_config["no_goals"]):
            env_config = copy.deepcopy(exp_config["env_config"])
            n_agents = exp_config["n_agents"]
            env_config["n_agents"] = exp_config["n_agents"]
            env_config["named_goals"] = {key: all_named_goals[key] for key in named_goals_keys}
            env_config["goals"] = set(env_config["named_goals"].values())
            env_config["n_goals"] = len(env_config["goals"])
            env_config["joint_start"] = exp_config["eval_starts"][0]

            for run_no in range(exp_config["par_repeats_per_run"]):
                curr_env_config = copy.deepcopy(env_config)
                if no_goals is not None:
                    named_goals = np.random.choice(list(all_named_goals.keys()), no_goals, replace=False).tolist()
                    curr_desirables = list(itertools.permutations(named_goals, n_agents))
                    curr_desirables = [list(el) for el in curr_desirables]
                    desirable_joint_goals = set(tuple([all_named_goals[goal] for goal in joint_goal]) for joint_goal in curr_desirables)

                    # env_config["named_goals"] = named_goals

                    curr_env_config["named_goals"] = {key: all_named_goals[key] for key in named_goals}
                    curr_env_config["goals"] = set(curr_env_config["named_goals"].values())
                     # = tmp_named_goals

                for learner_name in exp_config["learners"]:
                    run_config = {
                        "env_config": curr_env_config,
                        "learner": learner_name,
                        "hyperparams": exp_config["hyperparams_static"]["common"],
                        "max_episodes": exp_config["max_episodes"],
                        "eval_freq": exp_config["eval_freq"],
                        "desirable_joint_goals": desirable_joint_goals,
                        "eval_starts": exp_config["eval_starts"],
                        "run_no": run_no,
                        "logger_kwargs": exp_config["logger_kwargs"],
                        "step_quota": step_quota
                    }
                    run_config["logger_kwargs"]["exp_config_path"] = exp_config["exp_config_path"]
                    copy_to_dict(exp_config["hyperparams_static"][learner_name], run_config["hyperparams"])
                    copy_to_dict(exp_config["hyperparams_multi"][learner_name], run_config["hyperparams"])

                    tags = [learner_name, exp_config["env_config_name"]]
                    run_config["logger_kwargs"]["tags"] = tags
                    run_config["wandb_config"] = {
                        "learner": learner_name
                    }

                    run_path = os.path.join(self.runs_folder, f"run{self.no_runs}.pkl")
                    run_config["logger_kwargs"]["run_config_path"] = run_path
                    with open(run_path, "wb") as f:
                        pickle.dump(run_config, f)

                    self.no_runs += 1

            # for learner_name, run_no in itertools.product(exp_config["learners"], range(exp_config["repeats_per_run"])):
            #     run_config = {
            #         "env_config": env_config,
            #         "learner": learner_name,
            #         "hyperparams": exp_config["hyperparams_static"]["common"],
            #         "max_episodes": exp_config["max_episodes"],
            #         "eval_freq": exp_config["eval_freq"],
            #         "desirable_joint_goals": desirable_joint_goals,
            #         "eval_starts": exp_config["eval_starts"],
            #         "run_no": run_no,
            #         "logger_kwargs": exp_config["logger_kwargs"],
            #         "step_quota": step_quota
            #     }
            #     run_config["logger_kwargs"]["exp_config_path"] = exp_config["exp_config_path"]
            #     copy_to_dict(exp_config["hyperparams_static"][learner_name], run_config["hyperparams"])
            #     copy_to_dict(exp_config["hyperparams_multi"][learner_name], run_config["hyperparams"])
            #
            #     tags = [learner_name, exp_config["env_config_name"]]
            #     run_config["logger_kwargs"]["tags"] = tags
            #     run_config["wandb_config"] = {
            #         "learner": learner_name
            #     }
            #
            #     run_path = os.path.join(RUNS_PATH, f"run{self.no_runs}.pkl")
            #     run_config["logger_kwargs"]["run_config_path"] = run_path
            #     with open(run_path, "wb") as f:
            #         pickle.dump(run_config, f)
            #
            #     self.no_runs += 1

        # for learner_name in exp_config["learners"]:
        #     for desirable_joint_goals in exp_config["desirable_joint_goals"]:
        #         for run_no in range(exp_config["repeats_per_run"]):
        #             run_config = {
        #                 "env_config": env_config,
        #                 "learner": learner_name,
        #                 "hyperparams": exp_config["hyperparams_static"]["common"],
        #                 "max_episodes": exp_config["max_episodes"],
        #                 "eval_freq": exp_config["eval_freq"],
        #                 "desirable_joint_goals": desirable_joint_goals,
        #                 "eval_starts": exp_config["eval_starts"],
        #                 "run_no": run_no,
        #                 "logger_kwargs": exp_config["logger_kwargs"]
        #             }
        #             copy_to_dict(exp_config["hyperparams_multi"][learner_name], run_config["hyperparams"])
        #
        #             tags = [learner_name, exp_config["env_config_name"]]
        #             run_config["logger_kwargs"]["tags"] = tags
        #             run_config["wandb_config"] = {
        #                 "learner": learner_name
        #             }
        #
        #             run_path = os.path.join(RUNS_PATH, f"run{i}.pkl")
        #             with open(run_path, "wb") as f:
        #                 pickle.dump(run_config, f)
        #
        #             i += 1

        self.n_nodes = self.no_runs // self.runs_per_node
        if self.no_runs % self.runs_per_node != 0:
            self.n_nodes += 1

        if self.n_nodes > self.max_nodes:
            raise ValueError("No of nodes exceeds maximum")

        # if (self.no_runs // self.runs_per_node + self.no_runs % self.runs_per_node != 0) \
        #         > self.max_nodes:
        #     raise ValueError("No of nodes exceeds maximum")

    def load_slurm_template(self):
        template_path = os.path.join(BASE_PATH, "template.slurm")
        with open(template_path, "r") as f:
            file_contents = "".join(f.readlines())

        file_contents = file_contents.replace("{partition}", self.partition)
        file_contents = file_contents.replace("{base_path}", BASE_PATH)
        file_contents = file_contents.replace("{slurm_logs_folder}", self.slurm_logs_folder)
        # file_contents = file_contents.replace("{cluster_job_logs_path}", self.slurm_logs_folder)
        return file_contents

    def load_qsub_template(self):
        template_path = os.path.join(BASE_PATH, "template.qsub")
        with open(template_path, "r") as f:
            file_contents = "".join(f.readlines())

        file_contents = file_contents.replace("{partition}", self.partition)
        file_contents = file_contents.replace("{base_path}", BASE_PATH)
        file_contents = file_contents.replace("{qsubs_logs_folder}", self.qsubs_logs_folder)
        file_contents = file_contents.replace("{wall_time}", CHPC_WALL_TIME)
        file_contents = file_contents.replace("{n_cpus}", str(self.runs_per_node))

        # file_contents = file_contents.replace("{cluster_job_logs_path}", self.slurm_logs_folder)
        return file_contents

    # def create_bash_scripts(self):
    #     pass
    #
    # def exec_bash_scripts(self):
    #     pass

    def create_slurms(self):
        runs_sub_folder = self.exp_config["cluster"]["slurm_folder"]
        create_or_clear_folder(self.slurms_folder)

        n_seq_repeats = self.exp_config["cluster"]["seq_repeats_per_run"]

        slurm_template = self.load_slurm_template()
        no_slurms = self.no_runs//self.runs_per_node
        if self.no_runs % self.runs_per_node != 0:
            no_slurms += 1

        # slurm_folder = f"{CLUSTER_CODE_PATH}slurms/"
        # slurm_folder = f"{BASE_PATH}{PATH_SEP}slurms{PATH_SEP}"
        logs_path = self.run_logs_folder
        # logs_path = f"{CLUSTER_CODE_LOGS_PATH}{self.log_folder}/"
        # logs_folder = f"{BASE_PATH}{PATH_SEP}logs{PATH_SEP}"

        # clear_folder(slurm_folder)

        for slurm_id in range(no_slurms):
            start_id = slurm_id * self.runs_per_node
            end_id = min(start_id + self.runs_per_node, self.no_runs)
            job_name = f"{self.job_name_prefix}_{slurm_id}"

            slurm_str = slurm_template.replace("{job_name}", job_name)
            run_commands = ""
            for i in range(start_id, end_id):
                run_commands += f'echo "Attempting to run run{i}.pkl"\n'
                run_commands += f"python3 {CLUSTER_CODE_PATH}benchmarker.py --run {runs_sub_folder}/run{i}.pkl " \
                                f"--node $SLURM_JOB_NODELIST " \
                                f"-rr {n_seq_repeats}" \
                                f" > {logs_path}/run{i}.out 2>&1 & \n"
            slurm_str = slurm_str.replace("{run_commands}", run_commands)
            # slurm_str = slurm_str.replace("{start_id}", str(start_id))
            # slurm_str = slurm_str.replace("{end_id}", str(end_id))
            slurm_path = os.path.join(self.slurms_folder, f"job_{slurm_id}.slurm")
            # f"{slurm_folder}job_{slurm_id}.slurm"
            with open(slurm_path, "w") as f:
                f.write(slurm_str)
    # create

    def create_qsubs(self):
        runs_sub_folder = self.exp_config["cluster"]["qsubs_folder"]
        create_or_clear_folder(self.qsubs_folder)

        n_seq_repeats = self.exp_config["cluster"]["seq_repeats_per_run"]

        qsub_template = self.load_qsub_template()
        no_qsubs = self.no_runs//self.runs_per_node
        if self.no_runs % self.runs_per_node != 0:
            no_qsubs += 1

        # qsub_folder = f"{CLUSTER_CODE_PATH}qsubs/"
        # qsub_folder = f"{BASE_PATH}{PATH_SEP}qsubs{PATH_SEP}"
        logs_path = self.run_logs_folder
        # logs_path = f"{CLUSTER_CODE_LOGS_PATH}{self.log_folder}/"
        # logs_folder = f"{BASE_PATH}{PATH_SEP}logs{PATH_SEP}"

        # clear_folder(qsub_folder)

        for qsub_id in range(no_qsubs):
            start_id = qsub_id * self.runs_per_node
            end_id = min(start_id + self.runs_per_node, self.no_runs)
            job_name = f"{self.job_name_prefix}_{qsub_id}"

            qsub_str = qsub_template.replace("{job_name}", job_name)
            run_commands = ""
            for i in range(start_id, end_id):
                run_commands += f'echo "Attempting to run run{i}.pkl"\n'
                run_commands += f"python3 {CHPC_CODE_PATH}benchmarker.py --run {runs_sub_folder}/run{i}.pkl " \
                                f"--node $PBS_NODEFILE " \
                                f"-rr {n_seq_repeats}" \
                                f" > {logs_path}/run{i}.out 2>&1 & \n"
            qsub_str = qsub_str.replace("{run_commands}", run_commands)

            curr_wandb_folder = f"{self.wandb_base_folder}/{qsub_id}"

            if not self.is_local:
                create_or_clear_folder(curr_wandb_folder)
            qsub_str = qsub_str.replace("{wandb_folder}", curr_wandb_folder)

            # qsub_str = qsub_str.replace("{start_id}", str(start_id))
            # qsub_str = qsub_str.replace("{end_id}", str(end_id))
            qsub_path = os.path.join(self.qsubs_folder, f"job_{qsub_id}.qsub")
            # f"{qsub_folder}job_{qsub_id}.qsub"
            with open(qsub_path, "w") as f:
                f.write(qsub_str)


    def exec_slurms(self):
        slurm_files = os.listdir(self.slurms_folder)
        assert len(slurm_files) <= self.max_nodes, "Too many slurm files requested"

        for file in slurm_files:
            curr_file_path = os.path.join(self.slurms_folder, file)
            subprocess.run(["sbatch", curr_file_path])


def exec_slurms(slurm_folder):
    slurm_folder = os.path.join(SLURMS_FOLDER, slurm_folder)

    # self.slurm_logs_folder = f"{CLUSTER_LOGS_PATH}/slurms/{self.slurm_logs_folder}"
    # slurm_folder = BASE_PATH + PATH_SEP + "slurms" + PATH_SEP

    slurm_files = os.listdir(slurm_folder)
    assert len(slurm_files) <= 20, "Too many slurm files requested"

    for file in slurm_files:
        subprocess.run(["sbatch", f"{slurm_folder}{file}"])


def exec_bash_scripts(start_id, end_id, node=None):
    benchmarker_path = BASE_PATH_ARR.copy()
    benchmarker_path.extend(["benchmarker.py"])
    benchmarker_path = PATH_SEP.join(benchmarker_path)

    bash_path = BASE_PATH_ARR.copy()
    bash_path.extend(["bash_scripts", "run.bash"])
    bash_path = PATH_SEP.join(bash_path)
    # print(benchmarker_path)
    # exit()

    for i in range(start_id, end_id + 1):
        subprocess.Popen(('gnome-terminal', '--', bash_path, benchmarker_path,
                          f"run{i}.pkl", node))


def run(run_config_path, node=None):
    import pickle

    from learning.qlearning import QLearning
    from learning.go_qlearning import GOLearning, ImprovedGOLearning, BaseGOLearning

    from magw.gridworld import GridWorld
    from utils.common import copy_to_dict

    # Print System Info
    print_system_specs()

    with open(run_config_path, "rb") as f:
        run_config = pickle.load(f)

    if run_config["step_quota"] is not None:
        run_sq(run_config, node)
    else:
        env_config = run_config["env_config"]

        env = GridWorld(n_agents=env_config["n_agents"],
                        grid=env_config["grid"],
                        goals=env_config["goals"],
                        desirable_joint_goals=run_config["desirable_joint_goals"],
                        joint_start_state=env_config["joint_start"],
                        grid_input_type=env_config["grid_input_type"],
                        rewards_config=env_config["rewards_config"],
                        flatten_state=env_config["flatten_state"],
                        is_rendering=env_config["is_rendering"],
                        dynamics_config=env_config["dynamics_config"],
                        logging_config=env_config["env_logging_config"])

        run_config["wandb_config"]["node"] = node

        kwargs = {
            "env": env,
            "max_episodes": run_config["max_episodes"],
            "eval_starts": run_config["eval_starts"],
            "logger_kwargs": run_config["logger_kwargs"],
            "wandb_config": run_config["wandb_config"]
        }
        copy_to_dict(run_config["hyperparams"], kwargs)

        if run_config["learner"] == "qlearning":
            kwargs["desirable_joint_goals"] = run_config["desirable_joint_goals"]
            learner = QLearning(**kwargs)
        elif run_config["learner"] == "golearning":
            learner = GOLearning(**kwargs)
        elif run_config["learner"] == "improved_golearning":
            learner = ImprovedGOLearning(**kwargs)
        elif run_config["learner"] == "base_golearning":
            learner = BaseGOLearning(**kwargs)
        else:
            raise NotImplementedError(f"learner={run_config['learner']} is not supported")

        learner.learn_and_eval(run_config["eval_freq"])

        learner.close()
    # learner = QLearning(env=env,
    #                     eval_starts=run_config["eval_starts"],
    #                     logger_kwargs=run_config["logger_kwargs"],
    #                     wandb_config=run_config["wandb_config"],
    #                     **run_config["hyperparams"])
    # print(run_config)


def run_debug(run_config_path):
    import pickle
    from magw.gridworld import GridWorld, interactive

    # Print System Info
    print_system_specs()

    with open(run_config_path, "rb") as f:
        run_config = pickle.load(f)

    env_config = run_config["env_config"]

    if '/' in env_config["map_config_file"]:
        tmp_arr = env_config["map_config_file"].split("/")[6:]
        tmp_path = "\\".join(tmp_arr)
        env_config["map_config_file"] = os.path.join(BASE_PATH, tmp_path)

    if '/' in env_config["grid"]:
        tmp_arr = env_config["grid"].split("/")[6:]
        tmp_path = "\\".join(tmp_arr)
        env_config["grid"] = os.path.join(BASE_PATH, tmp_path)

    env = GridWorld(n_agents=env_config["n_agents"],
                    grid=env_config["grid"],
                    goals=env_config["goals"],
                    desirable_joint_goals=run_config["desirable_joint_goals"],
                    joint_start_state=env_config["joint_start"],
                    grid_input_type=env_config["grid_input_type"],
                    rewards_config=env_config["rewards_config"],
                    flatten_state=env_config["flatten_state"],
                    is_rendering=env_config["is_rendering"],
                    dynamics_config=env_config["dynamics_config"],
                    logging_config=env_config["env_logging_config"])

    interactive(env)

    # if run_config["step_quota"] is not None:
    #     run_sq(run_config, node)
    # else:
    #     env_config = run_config["env_config"]
    #
    #
    #
    #     run_config["wandb_config"]["node"] = node
    #
    #     kwargs = {
    #         "env": env,
    #         "max_episodes": run_config["max_episodes"],
    #         "eval_starts": run_config["eval_starts"],
    #         "logger_kwargs": run_config["logger_kwargs"],
    #         "wandb_config": run_config["wandb_config"]
    #     }
    #     copy_to_dict(run_config["hyperparams"], kwargs)
    #
    #     if run_config["learner"] == "qlearning":
    #         kwargs["desirable_joint_goals"] = run_config["desirable_joint_goals"]
    #         learner = QLearning(**kwargs)
    #     elif run_config["learner"] == "golearning":
    #         learner = GOLearning(**kwargs)
    #     elif run_config["learner"] == "improved_golearning":
    #         learner = ImprovedGOLearning(**kwargs)
    #     elif run_config["learner"] == "base_golearning":
    #         learner = BaseGOLearning(**kwargs)
    #     else:
    #         raise NotImplementedError(f"learner={run_config['learner']} is not supported")
    #
    #     learner.learn_and_eval(run_config["eval_freq"])
    #
    #     learner.close()


def run_sq(run_config, node):
    import copy
    import itertools
    import more_itertools
    import random

    import wandb

    from magw.gridworld import GridWorld

    from utils.common import copy_to_dict
    from learning.qlearning import QLearning
    from learning.go_qlearning import GOLearning

    step_quota = run_config["step_quota"]
    tot_steps = 0
    no_tasks = 0

    env_config = run_config["env_config"]

    using_wandb = run_config["logger_kwargs"]["using_wandb"]
    # using_wandb = False  # TODO: TEMP

    if using_wandb:
        mode = "online"
    else:
        mode = "disabled"

    # run_config["n_goals"] = len(env_config["goals"])
    job_type = run_config["logger_kwargs"]["job_type"]
    tags = run_config["logger_kwargs"]["tags"]
    group = run_config["logger_kwargs"]["group"]
    notes = run_config["logger_kwargs"]["notes"]
    wandb_config = copy.deepcopy(run_config)
    wandb_config["desirable_joint_goals"] = list(wandb_config["desirable_joint_goals"])
    wandb_config["env_config"]["goals"] = list(wandb_config["env_config"]["goals"])

    wandb_run = wandb.init(project=PROJECT_NAME, entity=ENTITY,
                           mode=mode, config=wandb_config, job_type=job_type,
                           tags=tags, group=group, notes=notes)
    # wandb_run = wandb.init(project=PROJECT_NAME, entity=ENTITY,
    #                        mode="disabled", config=wandb_config, job_type=job_type,
    #                        tags=tags, group=group, notes=notes)

    wandb_run.define_metric("episode")
    wandb_run.define_metric("return", step_metric="episode")

    run_config["logger_kwargs"]["using_wandb"] = False
    # all_desirable_jg = list(itertools.permutations(env_config["goals"], env_config["n_agents"]))

    wandb_run.define_metric("tot_steps")
    wandb_run.define_metric("no_tasks", step_metric="tot_steps")

    goals = env_config["goals"]
    # Gives all possible tasks represented as lists of only desirable goals
    tasks_powerset = list(more_itertools.powerset(goals))
    # Probs correct?
    all_tasks = [{goal: (True if goal in task else False) for goal in goals} for task in tasks_powerset]
    random.shuffle(all_tasks)
    # all_tasks = list(reversed(all_tasks))  # TODO: Temp
    n_agents = env_config["n_agents"]
    assert n_agents == 2

    all_joint_goals = list(itertools.permutations(env_config["goals"], env_config["n_agents"]))

    while tot_steps <= step_quota and no_tasks < len(all_tasks):
        curr_task = all_tasks[no_tasks]
        print(f"Curr task: {curr_task}")

        curr_task_jgs = {jg: curr_task[jg[0]] and curr_task[jg[1]] for jg in all_joint_goals}
        desirable_joint_goals = set([jg for jg, is_desirable in curr_task_jgs.items() if is_desirable])

        curr_max_steps = step_quota - tot_steps
        # desirable_joint_goals =
        # arr = all_desirable_jg
        # n_els = np.random.randint(1, len(arr)+1)
        # inds = list(range(len(arr)))
        # samp_inds = list(np.random.choice(inds, n_els, replace=False))
        # samp_els = np.array(arr)[samp_inds].tolist()
        # samp_els = [tuple([tuple(goal) for goal in joint_goal]) for joint_goal in samp_els]
        # desirable_joint_goals = set(samp_els)

        # n_els = np.random.randint(1, len(all_desirable_jg)+1)
        # desirable_joint_goals = set(np.random.choice(all_desirable_jg, n_els, replace=False))

        env = GridWorld(n_agents=env_config["n_agents"],
                        grid=env_config["grid"],
                        goals=env_config["goals"],
                        desirable_joint_goals=desirable_joint_goals,
                        joint_start_state=env_config["joint_start"],
                        grid_input_type=env_config["grid_input_type"],
                        rewards_config=env_config["rewards_config"],
                        flatten_state=env_config["flatten_state"],
                        is_rendering=env_config["is_rendering"],
                        dynamics_config=env_config["dynamics_config"],
                        logging_config=env_config["env_logging_config"])

        # interactive(env)

        run_config["wandb_config"]["node"] = node

        kwargs = {
            "env": env,
            "max_steps": curr_max_steps,
            "eval_starts": run_config["eval_starts"],
            "logger_kwargs": run_config["logger_kwargs"],
            "wandb_config": run_config["wandb_config"]
        }

        copy_to_dict(run_config["hyperparams"], kwargs)

        if run_config["learner"] == "qlearning":
            kwargs["desirable_joint_goals"] = run_config["desirable_joint_goals"]
            learner = QLearning(**kwargs)
        elif run_config["learner"] == "golearning":
            learner = GOLearning(**kwargs)
        else:
            raise NotImplementedError(f"learner={run_config['learner']} is not supported")

        is_converged = learner.learn()

        tot_steps += learner.step_no
        no_tasks += 1

        if is_converged:
            print(f"Task learnt! No of tasks learnt = {no_tasks}")
            log_dict = {
                "no_tasks": no_tasks,
                "total_steps": tot_steps
            }
            wandb_run.log(log_dict)

        learner.close()

    wandb_run.finish()


# Do multiple runs in parallel using multiprocessing
def run_multi(run_config_paths, node=None):
    import multiprocessing
    from functools import partial

    run_partial = partial(run, node=node)

    with multiprocessing.Pool(processes=len(run_config_paths)) as pool:
        pool.map(run_partial, run_config_paths)


# Note: Desirable joint goals are specified in run configs
def load_env_config(file_path=None, file_name=None):
    import json
    from magw.utils.loadconfig import load_config as load_map_config

    if file_name is not None:
        file_path = os.path.join(CONFIGS_PATH, file_name)
        # file_path = f"{CONFIGS_PATH}\\envs\\{file_name}"

    with open(file_path) as f:
        env_config = json.load(f)

    if env_config["is_map_local"]:
        grid = env_config["grid"]
        map_folder_path = f"{BASE_PATH}{PATH_SEP}maps{PATH_SEP}"

        if env_config["grid_input_type"] == "map_name":
            map_path = f"{map_folder_path}{grid}.map"
            env_config["grid"] = map_path
            env_config["grid_input_type"] = "file_path"
        elif env_config["grid_input_type"] == "file_path":
            pass
        else:
            raise ValueError(f"grid_input_type=={env_config['grid_input_type']} is not supported")

        if env_config["map_config_type"] == "name":
            config_path = f"{map_folder_path}{env_config['map_config_file']}"
            env_config['map_config_file'] = config_path
            env_config["map_config_type"] = "file_path"

    if env_config["map_config_type"] == "name":
        map_config = load_map_config(file_name=env_config["map_config_file"])
    elif env_config["map_config_type"] == "file_path":
        map_config = load_map_config(file_path=env_config["map_config_file"])
    else:
        raise ValueError(f"map_config_type=={env_config['map_config_type']} is not supported")

    named_map_goals = map_config["named_goals"]
    named_goals = {name: named_map_goals[name] for name in map_config["named_goals"]}
    goals = set([named_map_goals[name] for name in map_config["named_goals"]])

    env_config["named_goals"] = named_goals
    env_config["goals"] = goals
    env_config["n_goals"] = len(goals)
    env_config["named_starts"] = map_config["named_start_locs"]

    return env_config


def load_exp_config(file_path=None):  # , file_name=None):
    import json, itertools
    import numpy as np

    # if file_name is not None:
    #     file_path = os.path.join(CONFIGS_PATH, "exps", file_name)
        # file_path = f"{CONFIGS_PATH}\\exps\\{file_name}"

    with open(file_path) as f:
        exp_config = json.load(f)

    if exp_config["logger_kwargs"]["wandb_mode"] not in ["offline", "online", "disabled"]:
        raise ValueError(f'{exp_config["logger_kwargs"]["wandb_mode"]} is not a valid value for logger_kwargs.wandb_mode')

    exp_config["exp_config_path"] = file_path

    if exp_config["is_env_config_magw"]:
        env_config = load_env_config(file_name=exp_config["env_config_name"])
    else:
        env_config_name = exp_config["env_config_name"]
        env_config_path = os.path.join(CONFIGS_PATH, "envs", env_config_name)
        # env_config_path = f"{CONFIGS_PATH}\\envs\\{env_config_name}"
        env_config = load_env_config(file_path=env_config_path)

    n_agents = exp_config["n_agents"]
    # env_config["n_agents"] = exp_config["n_agents"]
    exp_config["env_config"] = env_config

    all_named_goals = env_config["named_goals"]

    if "named_goals" not in exp_config:
        if "no_goals" in exp_config:
            exp_config["named_goals"] = []
            for curr_no_goals in exp_config["no_goals"]:
                tmp_named_goals = np.random.choice(list(all_named_goals.keys()), curr_no_goals, replace=False).tolist()
                exp_config["named_goals"].append(tmp_named_goals)
        else:
            raise ValueError("Either named_goals or no_goals must be set")

    if "no_goals" not in exp_config:
        exp_config["no_goals"] = [None for _ in exp_config["named_goals"]]

    desirable_arr = exp_config["named_desirable_joint_goals"]
    for i, arr in enumerate(desirable_arr):
        if arr == "all":
            curr_named_goals = exp_config["named_goals"][i]
            curr_desirables = list(itertools.permutations(curr_named_goals, n_agents))
            curr_desirables = [list(el) for el in curr_desirables]
            desirable_arr[i] = curr_desirables
    desirable_arr = [set(tuple([all_named_goals[goal] for goal in joint_goal])
                         for joint_goal in joint_goals) for joint_goals in desirable_arr]
    exp_config["desirable_joint_goals"] = desirable_arr

    # named_goals_arr = exp_config["named_goals"]
    # # goals_arr

    named_starts = env_config["named_starts"]
    exp_config["eval_starts"] = [[named_starts[start] for start in joint_start] for joint_start
                                 in exp_config["named_eval_starts"]]

    return exp_config


# def main():
#     # load_exp_config(file_path=f"{CONFIGS_PATH}\\exps\\exp1.config")
#     benchmarker = Benchmarker(exp_config_name="exp1.config")
#     run(f"{RUNS_PATH}\\run0.pkl")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--create")
    parser.add_argument("--run")
    parser.add_argument("--run-debug")
    parser.add_argument("--run-multi")
    parser.add_argument("--run-bash")
    parser.add_argument("--run-slurms")
    parser.add_argument("--node")

    # parser.add_argument("-rn", "--runs_per_node")
    parser.add_argument("-rr", "--repeats_per_run", type=int, default=1)
    # parser.add_argument("-j", "--job_name")
    # parser.add_argument("-p", "--partition")
    # parser.add_argument("-lf", "--log_folder")

    parser.add_argument("-il", "--is_local", action="store_true")

    args = vars(parser.parse_args())
    # if (args["create"] is not None) and (args["run"] is not None):
    #     raise ValueError("Either --create or --run must be specified not both")
    # if (args["create"] is None) and (args["run"] is None):
    #     raise ValueError("Either --create or --run must be specified")

    if args["create"] is not None:
        # assert args["runs_per_node"] is not None, "--runs_per_node arg must be set"
        # assert args["partition"] is not None, "--partition arg must be set"
        print(f"Creating for {args['create']}")

        benchmarker = Benchmarker(exp_config_path=args["create"],
                                  # runs_per_node=int(args["runs_per_node"]),
                                  # partition=args["partition"], log_folder=args["log_folder"],
                                  is_local=args["is_local"]
                                  )
        print("Making run configs...")
        benchmarker.create_run_configs()
        if benchmarker.cluster_name == "msl":
            print("Making slurms...")
            benchmarker.create_slurms()
        elif benchmarker.cluster_name == "chpc":
            print("Making qsubs...")
            benchmarker.create_qsubs()
        print("Done :P")
        # print(f"Create: {args['create']}")
    elif args["run"] is not None:
        run_path = os.path.join(RUNS_FOLDER, args["run"])

        for i in range(args["repeats_per_run"]):
            print(f"Run: {args['run']}.\n\trepeat:{i+1}")
            run(run_path, args["node"])
    elif args["run_debug"] is not None:
        run_path = os.path.join(RUNS_FOLDER, args["run_debug"])
        run_debug(run_path)
    elif args["run_multi"] is not None:
        raise NotImplementedError
        # # print(f"Run_multi: {args['run']}")
        # run_range = args["run_multi"].split(",")
        # run_range = tuple([int(el) for el in run_range])
        # run_paths = [f"run{run_no}.pkl" for run_no in range(run_range[0], run_range[1] + 1)]
        # run_paths = [os.path.join(RUNS_PATH, run_path) for run_path in run_paths]
        # run_multi(run_paths, args["node"])
        # run_path = os.path.join(RUNS_PATH, args["run"])
        # run(run_path, args["node"])
    elif args["run_bash"] is not None:
        print("Running bash")
        start_id, end_id = args["run_bash"].split(",")
        start_id, end_id = int(start_id), int(end_id)
        exec_bash_scripts(start_id, end_id, args["node"])
    elif args["run_slurms"] is not None:
        # benchmarker = Benchmarker(exp_config_path=args["run_slurms"], is_local=False)
        # benchmarker.exec_slurms()
        print(f"Running slurms for {args['run_slurms']}")
        exec_slurms(args["run_slurms"])
        # print(slurm_files)
    else:
        raise ValueError


def test():
    benchmarker = Benchmarker(exp_config_name="exp1.config",
                              partition="stampede", job_name_prefix="job")
    benchmarker.load_slurm_template()


def test_method():
    create_or_clear_folder(r"C:\Users\simon\Documents\Varsity\2022\Code\zero-shot-comp-marl\mzeroshot\test_dir")


if __name__ == "__main__":
    # test_method()
    main()
    # test()

