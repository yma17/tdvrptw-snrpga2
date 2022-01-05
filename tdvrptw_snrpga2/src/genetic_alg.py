"""
Contains high-level algorithm code.
"""

import copy
import math
import numpy as np
from random import randint
from tqdm import tqdm

from tdvrptw_snrpga2.src.genetic_op import *
from tdvrptw_snrpga2.src.utils import round_nearest


def snrpga2(D_m, T_m, D_t, D, E, t2i, window_size, g_start, g_end, C,
            mng=1000, init_size=100, obj_func='dt',
            init='random_sample', w_t=1.0):
    """
    High-level code for SNRPGA2 genetic algorithm.

    Parameters:
    - D_m: 2D np.ndarray (distance matrix)
    - T_m: 3D np.ndarray (time-dependent time travel matrix)
    - D_t: 1D np.ndarray (delivery times for each location)
    - D: 1D np.ndarray (demand for each location)
    - E: 1D np.ndarray (end time for each location)
    - t2i: dict (map time to index of T on axis 0)
    - window_size: int (time window size)
    - g_start: int (global start time, aka opening of depot)
    - g_end: int (global end time, aka closing of depot)
    - C: float (capacity of each vehicle)
    - mng: int (max number of generations)
    - init_size: int (size of random initial population)
    - obj_func: str
    - init: str (initialization strategy)
        - 'random_sample', 'random_seq', 'tsp'
    - w_t: float

    Returns:
    - sol_res: tuple (detailed info of best chromosome)
    - sol_score: float (best chromosome's fitness score)
    """

    # Perform assertion checks to ensure feasibility of solution
    assert init in ['random_sample', 'random_seq', 'tsp']
    assert D_m.shape[0] >= 3, "At least three customers required"
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
    n = D_m.shape[0] - 1  # number of customers
    if init == 'random_sample':
        L = [np.random.choice(np.arange(1, n + 1), n, replace=False)
                for _ in range(init_size)]
    elif init == 'random_seq':
        i = randint(1, n)
        L = [np.hstack([np.arange(i, n + 1), np.arange(1, i)])
                for _ in range(init_size)]
    else:  # init == 'tsp'
        # TODO - implement
        pass

    # Evaluate fitness function of init pop
    L_res = [assign_routes(ch, C, D_t, D, E, T_m, t2i, window_size,
                    g_start, g_end) for ch in L]
    L_scores = [eval_fitness(obj_func, routes=res[0], dist_matrix=D_m,
                        depot_arrivals=res[4], g_start=g_start, w_t=w_t)
                        for res in L_res]

    pbar = tqdm(range(1, mng + 1), desc='Generation count')
    for _ in pbar:
        # Select subpopulation
        S_ind = subpop_selection(L_scores)

        # Run tournament selection to generate pairs
        pairs = tournament_selection(S_ind,
                    math.ceil(len(S_ind)), 5)

        # Create next generation through each candidate pair
        L_copy = copy.deepcopy(L)
        for pair in pairs:
            i1, i2 = pair[0], pair[1]

            # Produce offspring through crossover and mutation
            c1, c2 = L_copy[i1], L_copy[i2]
            d1, d2, modified_c = crossover(c1, c2)
            d1, d2, modified_m = mutation(d1, d2)

            if not modified_c and not modified_m:
                continue  # skip if offspring are identical to parents

            d1_res = assign_routes(d1, C, D_t, D, E, T_m, t2i,
                            window_size, g_start, g_end)
            d2_res = assign_routes(d2, C, D_t, D, E, T_m, t2i,
                            window_size, g_start, g_end)

            # Evaluate fitness function of offspring
            d1_score = eval_fitness(obj_func, routes=d1_res[0],
                                    dist_matrix=D_m, depot_arrivals=d1_res[4],
                                    g_start=g_start, w_t=w_t)
            d2_score = eval_fitness(obj_func, routes=d2_res[0],
                                    dist_matrix=D_m, depot_arrivals=d2_res[4],
                                    g_start=g_start, w_t=w_t)

            # Replace in population if fitness score is lower
            replace_in_pop(L, L_scores, L_res, i1, d1, d1_score, d1_res)
            replace_in_pop(L, L_scores, L_res, i2, d2, d2_score, d2_res)

        pbar.set_description("Best score = %.3f" % min(L_scores))


    # Retrieve information for top chromosome in last population
    sol_ind = subpop_selection(L_scores, n=1, mode='top')[0]
    sol_res, sol_score = L_res[sol_ind], L_scores[sol_ind]

    return sol_res, sol_score
