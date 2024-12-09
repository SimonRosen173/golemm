from typing import Dict
from copy import deepcopy


# Do in place
def conv_to_wandb(raw_dict: Dict) -> Dict:
    conv_dict = rec_conv_dict(raw_dict)
    return conv_dict


# Recursively convert dict
def rec_conv_dict(x):
    if type(x) == dict:
        new_x = {}
        for key, val in x.items():
            if type(key) == tuple:
                key = str(key)
            new_x[key] = rec_conv_dict(val)
        return new_x
    elif type(x) == set or type(x) == tuple or type(x) == list:
        new_x = []
        for val in x:
            new_x.append(rec_conv_dict(val))
        return new_x
    else:
        return x
