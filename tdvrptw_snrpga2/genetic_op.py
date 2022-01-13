"""
Contains helper functions for SNRPGA2 genetic algorithm operations.
"""

import math
import numpy as np
from random import random, randint, sample

from .utils import take_closest


def subpop_selection(scores, n=0.3, mode='prop'):
    """
    "SUBPOPULATION SELECTION" MODULE OF GENETIC ALGORITHM.
    Sorts population by fitness score in ascending order.
    Then, takes a subset of the sorted scores to form a subpopulation.

    Parameters:
    - scores: list of float (already-computed fitness scores)
    - n: float or int (proportion or number to keep)
    - mode: str ('prop' or 'top')

    Returns:
    - subpop_ind: list of int (indices of subpopulation chromosomes)
    """
    assert mode == 'prop' or mode == 'top'
    if mode == 'prop':
        assert n > 0 and n <= 1
    else:
        assert isinstance(n, int) and n >= 1

    # Sort by fitness score
    ind = list(range(len(scores)))
    _, sorted_ind = zip(*sorted(zip(scores, ind)))

    # Retrieve top proportion or top number
    if mode == 'prop':
        subpop_ind = list(sorted_ind)[:math.ceil(len(scores) * n)]
    else:
        subpop_ind = list(sorted_ind)[:n]

    return subpop_ind


def tournament_selection(S_ind, n, k, p=0.9):
    """
    "TOURNAMENT SELECTION" MODULE OF GENETIC ALGORITHM.
    Select pairs of two for crossover.

    Parameters:
    - S_ind: list of int (indices of subpopulation)
        - output of subpop_selection()
    - n: int (number of tournaments to run)
    - k: int (number of individuals in each tournament)
    - p: float (probability of selecting indiv with best fitness)

    Returns:
    - sel_pairs: list of np.ndarray (index pairs)
    """

    prob = [p * ((1 - p) ** i) for i in range(k)]
    prob[-1] = 1.0 - sum(prob[:-1])

    # sel_pairs = []
    # x_list = np.random.choice(range(len(S_ind)), len(S_ind), replace=False)
    # for i in range(0, len(x_list) - 1, 2):
    #     sel_pairs += [np.array([x_list[i], x_list[i + 1]])]

    sel_pairs = []
    sel_set = set()
    for _ in range(n):
        # Get indices of contestants
        x_list = sample(range(len(S_ind)), k)
        x_list.sort()
        C_ind = [S_ind[x] for x in x_list]

        # Select pair of two contestants
        if p < 1.0:
            pair = np.random.choice(C_ind, 2, replace=False, p=prob)
        else:
            pair = C_ind[:2]
        sel_pairs += [pair]
        sel_set.add(pair[0])
        sel_set.add(pair[1])
    
    # Ensure every chromosome in subpopulation is included
    #   in at least one pair
    ind_left = list(set(S_ind) - sel_set)
    for i in range(0, len(ind_left) - 1, 2):
        sel_pairs += [np.array([ind_left[i], ind_left[i + 1]])]
    if len(ind_left) % 2 == 0:
        sel_pairs += [np.array([S_ind[0], ind_left[-1]])]

    # # Ensure every chromosome in subpopulation is included
    # #   in at least one pair
    # ind_left = list(set(S_ind) - sel_set)
    # S_ind_k = S_ind[:k]
    # for i in range(len(ind_left)):
    #     x = np.random.choice(S_ind_k, 1, p=prob)
    #     sel_pairs += [np.array([x[0], ind_left[i]])]

    return sel_pairs


def crossover(c1, c2, x_prob=0.7):
    """
    "CROSSOVER" MODULE OF GENETIC ALGORITHM.
    Performs random sequence iteration-based crossover (RSIX).
    Parameters:
    - c1, c2: np.ndarray (representing parent chromosomes)
    - prob: float (crossover probability)
    Returns:
    - d1, d2: np.ndarray (representing offspring chromosomes)
    - bool (True if offspring are different than parents)
    """

    p = random()
    if p >= x_prob:
        return c1, c2, False

    # Generate two crossover points of length three
    c1_i, c2_i = randint(0, c1.shape[0] - 3), randint(0, c2.shape[0] - 3)
    c1_pt, c2_pt = c1[c1_i:c1_i + 3].tolist(), c2[c2_i:c2_i + 3].tolist()
    c, c_i, c_pt = [c1, c2], [c1_i, c2_i], [c1_pt, c2_pt]

    # Swapping operation, wrt problem constraints
    d = []
    for x in range(2):
        d_next = []
        for i in range(len(c[x])):
            if i == c_i[x]:
                d_next += c_pt[(x + 1) % 2]
            if c[x][i] not in c_pt[(x + 1) % 2]:
                d_next += [c[x][i]]
        d += [d_next]
    
    d1, d2 = np.array(d[0]), np.array(d[1])
    assert c1.shape[0] == d1.shape[0] and c2.shape[0] == d2.shape[0]
    return d1, d2, True


def mutation(d1, d2, m_prob=0.3):
    """
    "MUTATION" MODULE OF GENETIC ALGORITHM.
    Performs mutation on a chromosome/offspring.
    Parameters:
    - d1, d2: np.ndarray (offspring)
    - m_prob: float (mutation probability)
    Returns:
    - d1, d2: np.ndarray (mutated offspring)
    - bool (True if mutation occurred)
    """
    
    p = random()
    if p >= m_prob:
        return d1, d2, False

    # Select two genes from each chromosome to swap
    d1_i = np.random.choice(d1.shape[0], 2, replace=False)
    d2_i = np.random.choice(d2.shape[0], 2, replace=False)
    d1[d1_i[0]], d1[d1_i[1]] = d1[d1_i[1]], d1[d1_i[0]]
    d2[d2_i[0]], d2[d2_i[1]] = d2[d2_i[1]], d2[d2_i[0]]

    return d1, d2, True


def assign_routes(ch, C, D_t, D, E, T, g_start,
                    times, t2i, cache={}, granularity=1.0):
    """
    Assign a single-route chromosome into multiple sub-chromosomes
        of smaller routes consisting of a set of customers.
    These sub-chromosomes are used for fitness evaluation,
        and may represent a solution to the overall problem.
    Problem constraints are utilized here.

    Parameters:
    - ch: 1D np.ndarray (chromosome)
    - C: float (truck capacity; assume it is the same for each truck)
    - D_t: 1D np.ndarray (delivery times for each location)
    - D: 1D np.ndarray (demand for each location)
    - E: 1D np.ndarray (end time for each location)
    - T: 3D np.ndarray (time-dependent travel time matrix)
        - axes: time, source, destination (respectively)
    - g_start: float (global start time, aka departure from depot)
    - times: list (times of indices of axis 0 of T)
    - t2i: dict (map time to index of T on axis 0)
    - cache: dict
    - granularity: float

    Returns:
    - a tuple containing:
        - routes: list of lists (list of routes; each val = one truck)
        - amounts: list of 1ists (list of delivery amounts)
        - arrival_times: list of lists (arrival times to each customer)
        - departure_times: list of lists (departure times from each customer)
        - depot_arrivals: list (final arrival time at depot for each truck)
    """

    # Output variables
    routes = []
    arrival_times = []
    depot_arrivals = []
    amounts = []
    departure_times = []

    # Temporary variables
    curr_route = [0]  # locations visited so far
    curr_arrival = []  # arrival times so far
    curr_amounts = []  # amounts delivered to each location
    curr_departure = [g_start]  # departure times so far
    curr_deliv = 0.0  # total amount delivered so far

    # Assign customers iteratively, adding more trucks if needed
    cust_idx = 0
    while cust_idx < len(ch):
        next_cust = ch[cust_idx]

        # Retrieve travel time from current to next location
        t_next = take_closest(times,
                              curr_departure[-1],
                              cache=cache,
                              granularity=granularity)
        i_next = t2i[t_next]
        travel_time_next = T[i_next, curr_route[-1], next_cust]

        next_arrival = curr_departure[-1] + travel_time_next

        # Retrieve travel time from next location to depot
        next_departure = next_arrival + D_t[next_cust]
        t_depot = take_closest(times,
                               next_departure,
                               cache=cache,
                               granularity=granularity)
        i_depot = t2i[t_depot]
        travel_time_depot = T[i_depot, next_cust, 0]

        # Check customer end time, depot end time, and capacity constraints
        over_time_cust = (next_arrival > E[next_cust])
        over_time_depot = (next_departure + travel_time_depot > E[0])
        over_capacity = (curr_deliv + D[next_cust] > C)

        if not over_time_cust and not over_time_depot and not over_capacity:
            # Add to existing route information
            curr_route += [next_cust]
            curr_arrival += [curr_departure[-1] + travel_time_next]
            curr_amounts += [D[next_cust]]
            curr_departure += [next_departure]
            curr_deliv += D[next_cust]

            cust_idx += 1

            if cust_idx < len(ch):
                continue

        # Conclude current route; drive truck back to depot
        routes += [curr_route[1:]]
        arrival_times += [curr_arrival]
        amounts += [curr_amounts]
        departure_times += [curr_departure[1:]]
        depot_arrivals += [curr_departure[-1] + T[i_next, curr_route[-1], 0]]

        # Start next route; reset temporary variables
        curr_route, curr_arrival, curr_deliv = [0], [], 0.0
        curr_amounts, curr_departure = [], [g_start]

    return (routes, amounts, arrival_times, departure_times, depot_arrivals)


def eval_fitness(obj_func, g_start, routes=None, departure_times=None,
                    dist_matrix=None, times=None, t2i=None, cache=None,
                    granularity=None, depot_arrivals=None, w_t=1):
    """
    "FITNESS EVALUATION" MODULE OF GENETIC ALGORITHM.
    Evaluates fitness function with respect to distance/time/vehicles.

    Parameters:
    - obj_func: str
    - routes: list of lists
    - departure_times: list of lists
    - dist_matrix: 3D np.ndarray
    - times: list
    - t2i: dict
    - cache: dict
    - granularity: float
    - depot_arrivals: list
    - g_start: int
    - w_t: float

    Returns:
    - score: float (representing fitness score)
    """
    assert obj_func in ['v', 'd', 't', 'dt']
    
    if obj_func == 'v':  # number of vehicles
        return len(routes)

    assert g_start is not None

    score = 0.0

    if 'd' in obj_func:  # distance
        assert routes is not None and dist_matrix is not None
        assert departure_times is not None
        assert times is not None and t2i is not None
        assert cache is not None and granularity is not None

        for x, route in enumerate(routes):
            # Depot to first customer
            t = take_closest(times,
                             g_start,
                             cache=cache,
                             granularity=granularity)
            score += dist_matrix[t2i[t], 0, route[0]]

            # Customer to customer
            for y in range(len(route) - 1):
                t = take_closest(times,
                                 departure_times[x][y],
                                 cache=cache,
                                 granularity=granularity)
                score += dist_matrix[t2i[t], route[y], route[y + 1]]
            
            # Last customer (back) to depot
            t = take_closest(times,
                             departure_times[x][-1],
                             cache=cache,
                             granularity=granularity)
            score += dist_matrix[t2i[t], route[-1], 0]

    if 't' in obj_func:  # time
        assert depot_arrivals is not None

        time_score = sum(depot_arrivals) - g_start * len(depot_arrivals)
        
        mult = w_t if 'd' in obj_func else 1.0
        score += (mult * time_score)

    return score


def replace_in_pop(L, L_scores, L_res, i, o, o_score, o_res):
    """
    "REPLACE" MODULE OF GENETIC ALGORITHM.
    Replace parent chromosome with offspring if offspring has
        better fitness score than parent.

    Parameters:
    - L: list of np.ndarray (population)
    - L_scores: list of float (population scores)
    - L_res: list of tuple (population detailed info)
    - i: float (parent chromosome index)
    - o: np.ndarray (offspring chromosome)
    - o_score: float (offspring chromosome score)
    - o_res: tuple (offspring detailed info)

    Returns:
    - None
    """

    if o_score < L_scores[i]:
        L[i] = o
        L_scores[i] = o_score
        L_res[i] = o_res
