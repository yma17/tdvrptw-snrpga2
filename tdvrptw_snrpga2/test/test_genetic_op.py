"""
Contains unit tests for functions in src/genetic_op.py.
Use pytest to run tests.
"""

import math
import numpy as np

from tdvrptw_snrpga2.test.cases import *
from tdvrptw_snrpga2.src.compute_inputs import *
from tdvrptw_snrpga2.src.genetic_op import *

np.random.seed(0)


def test_subpop_selection_1():
    scores = [6,3,7,1,3,8]
    res = subpop_selection(scores, 0.3, 'prop')
    assert np.array_equal(res, np.array([3,1]))
    res = subpop_selection(scores, 4, 'top')
    assert np.array_equal(res, np.array([3,1,4,0]))


# def test_tournament_selection_1():
#     """Always select top 2 scores in each tournament (p = 1)"""
#     S_ind = [3,1,4,0,9]
#     n, k, p = 100, 3, 1
#     pairs = tournament_selection(S_ind, n, k, p)
#     counts = {}
#     for pair in pairs:
#         counts[pair[0]] = counts.get(pair[0], 0) + 1
#         counts[pair[1]] = counts.get(pair[1], 0) + 1
    
#     assert 9 not in counts.keys()
#     for i in range(len(S_ind) - 2):
#         # this should hold true statistically
#         assert counts[S_ind[i]] >= counts[S_ind[i + 1]]


# def test_tournament_selection_2():
#     """Always select top 2 scores in each tournament (p = 0.75)"""
#     S_ind = [3,1,4,0,9]
#     n, k, p = 1000, 3, 0.75
#     pairs = tournament_selection(S_ind, n, k, p)
#     counts = {}
#     for pair in pairs:
#         counts[pair[0]] = counts.get(pair[0], 0) + 1
#         counts[pair[1]] = counts.get(pair[1], 0) + 1
    
#     assert len(counts) == len(S_ind)
#     for i in range(len(S_ind) - 1):
#         # this should hold true statistically
#         assert counts[S_ind[i]] >= counts[S_ind[i + 1]]


def test_crossover_1():
    c1 = np.array([4,7,3,5,1,2,8,6])
    c2 = np.array([5,2,3,7,4,8,6,1])
    d1, d2, _ = crossover(c1, c2, 1.0)
    # Check if every location is visited exactly once
    assert len(np.unique(c1)) == len(c1)
    assert len(np.unique(c2)) == len(c2)


def test_mutation_1():
    d1_orig = np.array([2, 4, 1, 3])
    d2_orig = np.array([3, 2, 1, 4])
    d1_new, d2_new, _ = mutation(d1_orig.copy(), d2_orig.copy(), 1.0)
    # Check d1
    assert len(np.unique(d1_new)) == len(d1_orig) 
    diff = 0
    for i in range(len(d1_new)):
        if d1_orig[i] != d1_new[i]:
            diff += 1
    assert diff == 2
    # Check d2
    assert len(np.unique(d2_new)) == len(d2_orig) 
    diff = 0
    for i in range(len(d2_new)):
        if d2_orig[i] != d2_new[i]:
            diff += 1
    assert diff == 2


def test_assign_routes_1():
    """
    caseA
    
    check:
    - computation, evaluation of constraints
    - assignment of new routes
    - computation of correct return values
    """
    x, y, B, E, D_t, window_size = caseA()
    ch = np.array([2,4,3,1])
    C = 20
    D = np.array([0,15,10,10,10])

    t2i = compute_t2i(B[0], E[0], window_size)

    D_r = compute_dist_matrix(x, y)
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, compute_i2t(t2i))

    res = assign_routes(ch, C, D_t, D, E, T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 3
    assert res[0][0][0] == 2
    assert res[0][1][0] == 4
    assert res[0][1][1] == 3
    assert res[0][2][0] == 1
    # Check amounts
    assert len(res[1]) == len(res[0])
    assert res[1][0][0] == 10
    assert res[1][1][0] == 10
    assert res[1][1][1] == 10
    assert res[1][2][0] == 15
    # Check arrival_times
    assert len(res[2]) == len(res[0])
    assert abs(res[2][0][0] - math.sqrt(2)) <= 1e-4
    assert abs(res[2][1][0] - math.sqrt(2)) <= 1e-4
    assert abs(res[2][1][1] - math.sqrt(2) - 2) <= 1e-4
    assert abs(res[2][2][0] - math.sqrt(2)) <= 1e-4
    # Check departure times
    assert len(res[3]) == len(res[0])
    assert res[2][0][0] == res[3][0][0]
    assert res[2][1][0] == res[3][1][0]
    assert res[2][1][1] == res[3][1][1]
    assert res[2][2][0] == res[3][2][0]
    # Check depot arrivals
    assert len(res[4]) == len(res[0])
    assert abs(res[4][0] - 2 * math.sqrt(2)) <= 1e-4
    assert abs(res[4][1] - 2 * math.sqrt(2) - 2) <= 1e-4
    assert abs(res[4][2] - 2 * math.sqrt(2)) <= 1e-4


def test_assign_routes_2():
    """
    caseB
    
    check:
    -correct handling of waiting time wrt time windows
    -correct handling of delivery times
    """
    x, y, B, E, D_t, window_size = caseB()
    ch = np.array([1,2,3])
    C = 10
    D = np.array([0,5,4,6])

    t2i = compute_t2i(B[0], E[0], window_size)

    D_r = compute_dist_matrix(x, y)
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, compute_i2t(t2i))

    res = assign_routes(ch, C, D_t, D, E, T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 2
    assert res[0][0][0] == 1
    assert res[0][1][0] == 2
    assert res[0][1][1] == 3
    # Check arrival, departure times for truck 1
    assert res[2][0][0] == 20
    assert res[3][0][0] == 30
    assert res[4][0] == 35
    # Check arrival, departure times for truck 2
    assert abs(res[2][1][0] - 9.4868) <= 1e-4
    assert abs(res[3][1][0] - 14.4868) <= 1e-4
    assert abs(res[2][1][1] - 24.4868) <= 1e-4
    assert abs(res[3][1][1] - 29.4868) <= 1e-4
    assert abs(res[4][1] - 32.6491) <= 1e-4


def test_assign_routes_3():
    """
    caseC

    check:
    - correct handling of speeds across times
    - correct handling of waiting time wrt time windows
    - correct handling of delivery times
    - correct handling of end times for locations
    """
    x, y, B, E, D_t, window_size, C, S = caseC()
    ch = np.array([1, 2])
    capacity = 50
    D = np.array([0, 25, 25])

    t2i = compute_t2i(B[0], E[0], window_size)
    i2t = compute_i2t(t2i)
    D_r = compute_dist_matrix(x, y)
    T_r = compute_raw_time_matrix(D_r, scen=0, C=C, S=S, i2t=i2t)
    T_m = compute_time_matrix(T_r, B, i2t)
    
    res = assign_routes(ch,capacity,D_t,D,E, T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 2
    # Check truck 1 arrival/departure times
    assert res[2][0][0] == 100
    assert res[3][0][0] == 111
    assert abs(res[4][0] - 112) <= 1e-4
    # Check truck 2 arrival/departure times
    assert abs(res[2][1][0] - 0.8) <= 1e-4
    assert abs(res[3][1][0] - 22.8) <= 1e-4
    assert abs(res[4][1] - 23.6) <= 1e-4


def test_assign_routes_4():
    """
    caseD
    
    check:
    - correct handling of speeds across scenarios
    - correct handling of waiting time wrt time windows
    - correct handling of delivery times
    """
    x, y, B, E, D_t, window_size, C, S = caseD()
    ch = np.array([1, 2])
    capacity = 50
    D = np.array([0, 25, 25])

    t2i = compute_t2i(B[0], E[0], window_size)
    i2t = compute_i2t(t2i)

    D_r = compute_dist_matrix(x, y)

    #
    # Scenario 0 (fastest)
    #
    T_r = compute_raw_time_matrix(D_r, scen=0, C=C, S=S, i2t=i2t)
    T_m = compute_time_matrix(T_r, B, i2t)
    res = assign_routes(ch,capacity,D_t,D,E,T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 1
    # Check truck 1 arrival/departure times
    assert res[2][0][0] == 1.0
    assert res[3][0][0] == 1.05
    assert res[2][0][1] == 2.25  # should be 2.5 (but too large of window_size)
    assert res[3][0][1] == 2.35  # etc.
    assert abs(res[4][0] - 3.15) <= 1e-4

    #
    # Scenario 1
    #
    T_r = compute_raw_time_matrix(D_r, scen=1, C=C, S=S, i2t=i2t)
    T_m = compute_time_matrix(T_r, B, i2t)
    res = assign_routes(ch,capacity,D_t,D,E,T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 1
    # Check truck 1 arrival/departure times
    assert res[2][0][0] == 1.0
    assert res[3][0][0] == 1.05
    assert res[2][0][1] == 2.85
    assert res[3][0][1] == 2.95
    assert res[4][0] == 3.95

    #
    # Scenario 2 (slowest)
    #
    T_r = compute_raw_time_matrix(D_r, scen=2, C=C, S=S, i2t=i2t)
    T_m = compute_time_matrix(T_r, B, i2t)
    res = assign_routes(ch,capacity,D_t,D,E,T_m, t2i, window_size, B[0], E[0])
    # Check routes
    assert len(res[0]) == 2
    # Check truck 1 arrival/departure times
    assert res[2][0][0] == 1.3
    assert res[3][0][0] == 1.35
    assert abs(res[4][0] - 2.65) <= 1e-4
    # Check truck 2 arrival/departure times
    assert res[2][1][0] == 2.5
    assert res[3][1][0] == 2.6
    assert res[4][1] == 4.0


def test_eval_fitness_1():
    """Objective function: distance"""
    x, y, _, _, _, _ = caseA()

    D_r = compute_dist_matrix(x, y) 

    routes = [[1, 2], [3, 4]]
    score = eval_fitness('d', routes=routes, dist_matrix=D_r)
    assert abs(score - 2 * (2 + 2 * math.sqrt(2))) <= 1e-4

    routes = [[1, 2], [3], [4]]
    score = eval_fitness('d', routes=routes, dist_matrix=D_r)
    assert abs(score - (2 + 2 * math.sqrt(2)) - 4 * math.sqrt(2)) <= 1e-4


def test_eval_fitness_2():
    """Objective function: time"""

    g_start = 6
    depot_arrivals = [9, 14, 15, 18, 12]
    score = eval_fitness('t', depot_arrivals=depot_arrivals, g_start=g_start)
    assert score == 38


def test_eval_fitness_3():
    """Objective function: weighted sum of distance and time"""
    x, y, _, _, _, _ = caseA()

    D_r = compute_dist_matrix(x, y) 

    g_start = 6
    depot_arrivals = [9, 14, 15, 18, 12]
    routes = [[1, 2], [3, 4]]
    score = eval_fitness('dt', routes=routes, dist_matrix=D_r,
                depot_arrivals=depot_arrivals, g_start=g_start)
    assert abs(score - 2 * (2 + 2 * math.sqrt(2)) - 38) <= 1e-4


def test_replace_in_pop_1():
    """Replace"""
    L = [np.array([1,2,3]),np.array([4,5,6])]
    L_scores = [3.0, 2.0]
    L_res = [('a', 'b'), ('c', 'd')]
    replace_in_pop(L, L_scores, L_res, 1, np.array([10,11,12]), 1.0,('e', 'f'))
    assert np.array_equal(L[1], np.array([10,11,12]))
    assert L_scores[1] == 1.0
    assert L_res[1][0] == 'e' and L_res[1][1] == 'f'


def test_replace_in_pop_2():
    """Replace"""
    L = [np.array([1,2,3]),np.array([4,5,6])]
    L_scores = [3.0, 2.0]
    L_res = [('a', 'b'), ('c', 'd')]
    replace_in_pop(L, L_scores, L_res, 1, np.array([10,11,12]), 4.0,('e', 'f'))
    assert np.array_equal(L[1], np.array([4,5,6]))
    assert L_scores[1] == 2.0
    assert L_res[1][0] == 'c' and L_res[1][1] == 'd'
