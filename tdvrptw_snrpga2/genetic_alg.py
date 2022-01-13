"""
Contains high-level algorithm code.
"""

import copy
import math
import numpy as np
from random import randint
from tqdm import tqdm

from .genetic_op import *
from .utils import take_closest


def snrpga2(D_m, T_m, D_t, D, E, g_start, t2i, times, C, params):
    """
    High-level code for SNRPGA2 genetic algorithm.

    Parameters:
    - D_m: 3D np.ndarray (time-dependent distance matrix)
    - T_m: 3D np.ndarray (time-dependent time travel matrix)
    - D_t: 1D np.ndarray (delivery times for each location)
    - D: 1D np.ndarray (demand for each location)
    - E: 1D np.ndarray (end time for each location)
    - g_start: float (global start time aka departure from depot)
    - t2i: dict (map time to index of D_m and T_m on axis 0)
    - times: list (times of indices of axis 0 of D_m and T_m)
    - C: float (capacity of each vehicle)
    - params: dict (algorithm hyperparameters)

    Returns:
    - sol_res: tuple (detailed info of best chromosome)
    - sol_score: float (best chromosome's fitness score)
    """

    cache = {}

    # Perform assertion checks to ensure feasibility of solution
    assert params["init_method"] in ['random_sample', 'random_seq', 'tsp']
    assert D_m.shape[1] >= 3, "At least three customers required"
    assert D.max() <= C, "Largest demand must not exceed vehicle capacity"
    for k in range(1, D.shape[0]):
        time_to_k = T_m[0, 0, k]
        t_k = take_closest(times,
                           g_start + time_to_k + D_t[k],
                           cache=cache,
                           granularity=params["cache_gran"])
        i_k = t2i[t_k]
        time_from_k = T_m[i_k, k, 0]
        assert time_to_k + D_t[k] + time_from_k <= E[0] - g_start, \
            "Every customer must be roundtrip reachable from depot within " + \
                "global time frame (accounting for both travel time and " + \
                "waiting time according to customer time windows)"


    # Generate random initial population
    n = D_m.shape[1] - 1  # number of customers
    if params["init_method"] == 'random_sample':
        L = [np.random.choice(np.arange(1, n + 1), n, replace=False)
                for _ in range(params["pop_size"])]
    elif params["init_method"] == 'random_seq':
        i = randint(1, n)
        L = [np.hstack([np.arange(i, n + 1), np.arange(1, i)])
                for _ in range(params["pop_size"])]
    else:  # init == 'tsp'
        # TODO - implement
        pass

    # Evaluate fitness function of init pop
    L_res = [assign_routes(ch, C, D_t, D, E, T_m, g_start, times, t2i,
                             cache=cache, granularity=params["cache_gran"])
                             for ch in L]
    L_scores = [eval_fitness(params["obj_func"], routes=res[0],
                                departure_times=res[3], dist_matrix=D_m,
                                depot_arrivals=res[4], times=times, t2i=t2i,
                                cache=cache, granularity=params["cache_gran"],
                                g_start=g_start, w_t=params["w_t"])
                                for res in L_res]

    pbar = tqdm(range(1, params["mng"] + 1), desc='Generation count')
    for _ in pbar:
        # Select subpopulation
        S_ind = subpop_selection(L_scores)

        # Run tournament selection to generate pairs
        pairs = tournament_selection(S_ind,
                    math.ceil(len(S_ind)), 5, p=params["ts_prob"])

        # Create next generation through each candidate pair
        L_copy = copy.deepcopy(L)
        for pair in pairs:
            i1, i2 = pair[0], pair[1]

            # Produce offspring through crossover and mutation
            c1, c2 = L_copy[i1], L_copy[i2]
            d1, d2, modified_c = crossover(c1, c2, x_prob=params["x_prob"])
            d1, d2, modified_m = mutation(d1, d2, m_prob=params["m_prob"])

            if not modified_c and not modified_m:
                continue  # skip if offspring are identical to parents

            d1_res = assign_routes(d1, C, D_t, D, E, T_m, g_start, times,
                            t2i, cache=cache, granularity=params["cache_gran"])
            d2_res = assign_routes(d2, C, D_t, D, E, T_m, g_start, times,
                            t2i, cache=cache, granularity=params["cache_gran"])

            # Evaluate fitness function of offspring
            d1_score, d2_score = [eval_fitness(params["obj_func"],
                                    g_start=times[0], routes=res[0],
                                    departure_times=res[3], dist_matrix=D_m,
                                    depot_arrivals=res[4], times=times,
                                    t2i=t2i, cache=cache, w_t=params["w_t"],
                                    granularity=params["cache_gran"])
                                    for res in (d1_res, d2_res)]

            # Replace in population if fitness score is lower
            replace_in_pop(L, L_scores, L_res, i1, d1, d1_score, d1_res)
            replace_in_pop(L, L_scores, L_res, i2, d2, d2_score, d2_res)

        pbar.set_description("Best score = %.3f" % min(L_scores))


    # Retrieve information for top chromosome in last population
    sol_ind = subpop_selection(L_scores, n=1, mode='top')[0]
    sol_res, sol_score = L_res[sol_ind], L_scores[sol_ind]

    return sol_res, sol_score
