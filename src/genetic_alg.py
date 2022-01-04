"""
Contains high-level algorithm code.
"""

import math
import numpy as np
from tqdm import tqdm

from .genetic_op import *
from .utils import round_nearest


def snrpga2(D_m, T_m, D_t, D, t2i, window_size, g_start, g_end,
            C, mng=1000, init_size=100, obj_func='distance'):
    """
    High-level code for SNRPGA2 genetic algorithm.

    Parameters:
    - D_m: 3D np.ndarray (time-dependent distance matrix)
    - T_m: 3D np.ndarray (time-dependent time travel matrix)
    - D_t: 1D np.ndarray (delivery times for each location)
    - D: 1D np.ndarray (demand for each location)
    - t2i: dict (map time to index of T on axis 0)
    - window_size: int (time window size)
    - g_start: int (global start time, aka opening of depot)
    - g_end: int (global end time, aka closing of depot)
    - C: float (capacity of each vehicle)
    - mng: int (max number of generations)
    - init_size: int (size of random initial population)
    - obj_func: str ('num_vehicles' or 'distance')

    Returns:
    - sol_res: tuple (detailed info of best chromosome)
    - sol_score: float (best chromosome's fitness score)
    """

    # Perform assertion checks to ensure feasibility of solution
    assert D_m.shape[1] >= 3, "At least three customers required"
    assert D.max() <= C, "Largest demand must not exceed vehicle capacity"
    for k in range(1, D.shape[0]):
        time_to_k = T_m[0, 0, k]
        t_k = min(round_nearest(g_start+time_to_k+D_t[k], window_size), g_end)
        i_k = t2i[int(t_k)]
        time_from_k = T_m[i_k, k, 0]
        assert time_to_k + D_t[k] + time_from_k <= g_end - g_start, \
            "Every customer must be roundtrip reachable from depot within " + \
                "global time frame (accounting for both travel time and " + \
                "waiting time according to customer time windows)"


    # Generate random initial population
    n = D_m.shape[1] - 1  # number of customers
    L = [np.random.choice(np.arange(1, n + 1), n, replace=False)
            for _ in range(init_size)]

    # Evaluate fitness function of init pop
    L_res = [assign_routes(ch, C, D_t, D, T_m, t2i, window_size,
                    g_start, g_end) for ch in L]
    L_scores = [eval_fitness(res[0], res[2], obj_func, D_m, res[4],
                    t2i, window_size) for res in L_res]


    pbar = tqdm(range(1, mng + 1), desc='Generation count')
    for _ in pbar:
        # Select subpopulation
        S_ind = subpop_selection(L_scores)

        # Run tournament selection to generate pairs
        pairs = tournament_selection(S_ind,
                    math.ceil(len(S_ind) / 2),
                    math.ceil(len(S_ind) / 10))

        # Create next generation through each candidate pair
        for pair in pairs:
            i1, i2 = pair[0], pair[1]

            # Produce offspring through crossover and mutation
            c1, c2 = L[i1], L[i2]
            d1, d2, modified_c = crossover(c1, c2)
            d1, d2, modified_m = mutation(d1, d2)

            if not modified_c and not modified_m:
                continue  # skip if offspring are identical to parents

            # Evaluate fitness function of offspring
            d1_res = assign_routes(d1, C, D_t, D, T_m, t2i,
                            window_size, g_start, g_end)
            d1_score = eval_fitness(d1_res[0], d1_res[2], obj_func, D_m,
                            d1_res[4], t2i, window_size)
            d2_res = assign_routes(d2, C, D_t, D, T_m, t2i,
                            window_size, g_start, g_end)
            d2_score = eval_fitness(d2_res[0], d2_res[2], obj_func, D_m,
                            d2_res[4], t2i, window_size)

            # Replace in population if fitness score is lower
            replace_in_pop(L, L_scores, L_res, i1, d1, d1_score, d1_res)
            replace_in_pop(L, L_scores, L_res, i2, d2, d2_score, d2_res)

        pbar.set_description("Best score = %.3f" % min(L_scores))


    # Retrieve information for top chromosome in last population
    sol_ind = subpop_selection(L_scores, n=1, mode='top')[0]
    sol_res, sol_score = L_res[sol_ind], L_scores[sol_ind]

    return sol_res, sol_score