import json
import os

import numpy as np


BASE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

# BASE_PATH = os.path.split(BASE_PATH)[0]
# BASE_PATH = BASE_PATH.split(os.path.sep)[:-1]
# BASE_PATH = posixpath.split(BASE_PATH)[:-1]
CONFIGS_PATH = os.path.normpath(os.path.join(BASE_PATH, "configs"))


def load_map_file(file_path: str) -> np.ndarray:
    with open(file_path) as f:
        grid_arr = []
        for line in f:
            curr_arr = line.split(" ")
            curr_arr = [int(el) for el in curr_arr]
            grid_arr.append(curr_arr)

        grid = np.asarray(grid_arr)

    grid = grid[1:-1, 1:-1]
    grid = np.flip(grid, axis=0)

    return grid


def load_config(config_name):
    config_path = os.path.join(CONFIGS_PATH, config_name)
    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict["named_goals"] = {key: tuple(val) for key, val in config_dict["named_goals"].items()}
    config_dict["named_starts"] = {key: tuple(val) for key, val in config_dict["named_starts"].items()}
    map_path = os.path.join(CONFIGS_PATH, "maps", config_dict["grid_name"])
    config_dict["grid"] = load_map_file(map_path)

    return config_dict


def test():
    config = load_config("4x4.json")
    print(config)


if __name__ == "__main__":
    test()