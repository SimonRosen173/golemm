from typing import List

import numpy as np


def pointwise_add(arrs: List[np.ndarray]):
    new_arrs = []
    n_dims = len(arrs)
    n_els = arrs[0].shape[0]

    for i, arr in enumerate(arrs):
        curr_shape = [1 for _ in range(n_dims)]
        curr_shape[i] = n_els
        arr = arr.reshape(tuple(curr_shape))

        for j in range(n_dims):
            if i != j:
                arr = np.repeat(arr, n_els, axis=j)
 
        new_arrs.append(arr)

    return np.sum(new_arrs, axis=0)


# TODO: Test
def pointwise_add_2d(arrs: List[np.ndarray], inds, n_els_arr, n_dims):
    new_arrs = []
    # n_els = arrs[0].shape[0]
    for (i, j), arr in zip(inds, arrs):
        curr_shape = [1 for _ in range(n_dims)]
        curr_shape[i] = n_els_arr[i]
        curr_shape[j] = n_els_arr[j]

        arr = arr.reshape(tuple(curr_shape))

        for k in range(n_dims):
            if k != i and k != j:
                arr = np.repeat(arr, n_els_arr[k], axis=k)

        new_arrs.append(arr)

    return np.sum(new_arrs, axis=0)
