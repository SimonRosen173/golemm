# _heapq is cython version
from _heapq import heapify, heappop, heappush

# numpy import stuff
import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()

DTYPE_INT = np.int8

ctypedef cnp.int8_t DTYPE_INT_t

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def test_np_arr(cnp.ndarray[DTYPE_INT_t, ndim=1] a):
#     return np.max(a)


# # ...
# cdef hash_np_arr(cnp.ndarray[DTYPE_INT_t, ndim=1] arr) -> str:
#     pass


# NOTE: Ensure that none of these variables are none -> Can lead to data corruption
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef list _get_valid_neighbours_1a(
        cnp.ndarray[DTYPE_INT_t, ndim=2] grid: np.ndarray,
        # cnp.ndarray[DTYPE_INT_t, ndim=2] goals: np.ndarray,
        set goals,
        # cnp.ndarray[DTYPE_INT_t, ndim=1] desired_goal: np.ndarray,
        tuple loc,
        int is_term_agent_enter
):
    # assert valid_neighbours.shape[0] == 4

    cdef int height = grid.shape[0]
    cdef int width = grid.shape[1]

    cdef int n_valid = 0
    # cdef np.ndarray[DTYPE_INT_t, ndim=2] valid_neighbours = np.zeros((4, 2), dtype=DTYPE_INT)

    cdef int y = loc[0]
    cdef int x = loc[1]

    cdef list valid_neighbours = []

    # Dunno if 'loc in goals' is efficient
    if is_term_agent_enter == 1 and loc in goals:
        # print("?")
        pass
    else:
        # print(y, x, height, width)

        # UP
        if y + 1 < height and grid[y+1][x] == 0:
            # print("UP")
            valid_neighbours.append((y+1, x))
            # valid_neighbours[n_valid][0] = y+1
            # valid_neighbours[n_valid][1] = x

        # DOWN
        if y - 1 >= 0 and grid[y - 1][x] == 0:
            valid_neighbours.append((y - 1, x))
            # print("DOWN")
            # valid_neighbours[n_valid][0] = y - 1
            # valid_neighbours[n_valid][1] = x

        # RIGHT
        if x + 1 < width and grid[y][x + 1] == 0:
            # print("RIGHT")
            valid_neighbours.append((y, x+1))
            # valid_neighbours[n_valid][0] = y
            # valid_neighbours[n_valid][1] = x + 1

        # LEFT
        if x - 1 >= 0 and grid[y][x - 1] == 0:
            # print("LEFT")
            valid_neighbours.append((y, x-1))
            # valid_neighbours[n_valid][0] = y
            # valid_neighbours[n_valid][1] = x - 1
            # n_valid += 1

    return valid_neighbours


# 1a for definition of curr_loc
cdef list reconstruct_path_1a(dict came_from, tuple end_loc):
    cdef list path = []
    # cdef bytes curr_loc_hash = end_loc_hash
    # cdef cnp.ndarray[DTYPE_INT_t, ndim=1] curr_loc =
    cdef tuple curr_loc = end_loc
    path.append(curr_loc)

    while came_from[curr_loc] is not None:
        curr_loc = came_from[curr_loc]
        path.append(curr_loc)

    path.reverse()
    return path


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef tuple _dijkstra_shortest_path_1a(
        cnp.ndarray[DTYPE_INT_t, ndim=2] grid: np.ndarray,
        set goals,
        tuple start,
        tuple desired_goal,
        int is_term_agent_enter
):
    cdef int height = grid.shape[0]
    cdef int width = grid.shape[1]
    # cdef bytes des_goal_hash = desired_goal.tobytes()
    # cdef cnp.ndarray[DTYPE_INT_t, ndim=2] valid_neighbours = np.zeros((4, 2), dtype=DTYPE_INT)

    # print(frontier)

    # Store hash and original array
    # cdef dict hash_to_arr = {}

    # hash np array start by converting to bytes
    # cdef bytes start_hash = start.tobytes()
    # hash_to_arr[start_hash] = start

    # Inefficient?
    cdef list frontier = []
    heappush(frontier, (0, start))

    cdef dict came_from = {start: None}
    cdef dict costs = {start: 0}

    # cdef cnp.ndarray[DTYPE_INT_t, ndim = 1] curr_loc  # = np.zeros(2, dtype=DTYPE_INT)
    # cdef bytes curr_loc_hash
    # cdef int n_valid
    cdef tuple curr_loc

    # cdef cnp.ndarray[DTYPE_INT_t, ndim = 1] next_loc
    cdef tuple next_loc
    # cdef bytes next_loc_hash
    cdef int new_cost
    cdef int tmp_cost

    cdef list valid_neighbours
    cdef tuple neighbour
    # cdef int is_found = 0

    # cdef set goals_hashes = set()
    # for i in range(goals.shape[0]):
    #     goals_hashes.add(goals[i].tobytes())

    while not len(frontier) == 0:
        tmp_cost, curr_loc = heappop(frontier)

        if curr_loc == desired_goal:
            # print(f"GOAL FOUND\ndes_goal_hash in hash_to_arr: {curr_loc_hash in hash_to_arr}")
            break

        # valid_neighbours = np.zeros((4, 2), dtype=DTYPE_INT)

        valid_neighbours = _get_valid_neighbours_1a(grid, goals, curr_loc, is_term_agent_enter)

        new_cost = costs[curr_loc] + 1
        for i in range(len(valid_neighbours)):
            neighbour = valid_neighbours[i]
            # next_loc = valid_neighbours[i]
            # next_loc_hash = valid_neighbours[i].tobytes()
            if neighbour not in costs or new_cost < costs[neighbour]:
                # hash_to_arr[next_loc_hash] = valid_neighbours[i].copy()
                costs[neighbour] = new_cost
                heappush(frontier, (new_cost, neighbour))  # Probs optimise
                came_from[neighbour] = curr_loc

    # print(came_from)
    cdef list path = reconstruct_path_1a(came_from, desired_goal)
    return path, costs[desired_goal]
    # print(came_from)
    # print(np.frombuffer(start_bytes))


cdef _dijkstra_shortest_path_ma(
        int n_agents,
        cnp.ndarray[DTYPE_INT_t, ndim=2] grid: np.ndarray,
        cnp.ndarray[DTYPE_INT_t, ndim=2] goals: np.ndarray,
        cnp.ndarray[DTYPE_INT_t, ndim=1] start: np.ndarray,
        cnp.ndarray[DTYPE_INT_t, ndim=1] desired_goal: np.ndarray,
        int is_term_agent_enter
):
    pass



def dijkstra_shortest_path(
        int n_agents,
        cnp.ndarray[DTYPE_INT_t, ndim=2] grid: np.ndarray,
        set goals,
        tuple joint_start,
        tuple joint_goal,
        int is_term_agent_enter
):
    assert grid.dtype == DTYPE_INT

    cdef int height = grid.shape[0]
    cdef int width = grid.shape[1]

    if n_agents == 1:
        return _dijkstra_shortest_path_1a(
            grid=grid, goals=goals, start=joint_start[0],
            desired_goal=joint_goal[0], is_term_agent_enter=is_term_agent_enter
        )
    else:
        pass



# def test_get_valid_neighbours_1a(
#         cnp.ndarray[DTYPE_INT_t, ndim=2] grid: np.ndarray,
#         cnp.ndarray[DTYPE_INT_t, ndim=2] goals: np.ndarray,
#         cnp.ndarray[DTYPE_INT_t, ndim=1] loc: np.ndarray,
#         # cnp.ndarray[DTYPE_INT_t, ndim=2] valid_neighbours: np.ndarray,  # Pass by reference
#         int is_term_agent_enter
# ):
#     cdef cnp.ndarray[DTYPE_INT_t, ndim=2] valid_neighbours = np.zeros((4, 2), dtype=DTYPE_INT)
#     cdef n_valid = _get_valid_neighbours_1a(grid, goals, loc, valid_neighbours, is_term_agent_enter)
#     return valid_neighbours[:n_valid]
