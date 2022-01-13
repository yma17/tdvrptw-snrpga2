"""
Contains unit tests for the high-level function in src/genetic_alg.py.
These are much smaller than the SOLOMON test cases, allowing for more   
    easy debugging and more predictable results.
Use pytest to run tests.
"""

import math
import numpy as np

from cases import *
from tdvrptw_snrpga2.compute_inputs import *
from tdvrptw_snrpga2.genetic_alg import *


def test_snrpga2_0():
    """Test assertion checks"""
    params = {"init_method": "invalid"}

    b = False
    D_m = np.zeros((2, 2))
    try:
        snrpga2(D_m,None,None,None,None,None,None,None,None,params)
    except AssertionError:
        b = True
    assert b

    params = {"init_method": "random_sample", "cache_gran": 0.01}

    b = False
    D_m = np.zeros((1, 2, 2))
    try:
        snrpga2(D_m,None,None,None,None,None,None,None,None,params)
    except AssertionError:
        b = True
    assert b

    D, C = np.array([30]), 20
    D_m = np.zeros((1, 3, 3))
    b = False
    try:
        snrpga2(D_m,None,None,D,None,None,None,None,C,params)
    except AssertionError:
        b = True
    assert b

    D, C = np.array([0, 10]), 20
    b = False
    T_m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    g_start = 10
    E = np.array([18, 18])
    t2i = {10:0, 15:1, 20:2}
    times = [10, 15, 20]
    D_t = np.array([0, 3])
    D_m = np.zeros((3, 3, 3))
    try:
        snrpga2(D_m,T_m,D_t,D,E,g_start,t2i,times,C,params)
    except AssertionError:
        b = True
    assert b


def test_snrpga2_1():
    """caseA - no demand limitations"""
    x, y, B, E, D_t, window_size = caseA()
    C = 20
    D = np.array([0,10,10,10,10])

    t2i = compute_t2i(B[0], E[0], window_size)
    i2t = compute_i2t(t2i)
    D_r = compute_dist_matrix(x, y)
    D_m = np.repeat(D_r[np.newaxis, :, :], len(i2t), axis=0)
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, i2t)

    params = {"init_method": "random_sample", "cache_gran": 0.01,
                "ts_prob": 0.9, "x_prob": 0.7, "m_prob": 0.3, "w_t": 1.0,
                "mng": 1000, "pop_size":100, "obj_func":'d'}

    res, score = snrpga2(D_m, T_m, D_t, D, E, B[0], t2i, i2t, C, params)
    assert abs(score - (2 * (2 + 2 * math.sqrt(2)))) <= 1e-4
    assert len(res[0]) == 2
    assert len(res[0][0]) == 2 and len(res[1][0]) == 2


def test_snrpga2_2():
    """caseA - demand limitations"""
    x, y, B, E, D_t, window_size = caseA()
    C = 20
    D = np.array([0,15,10,10,10])

    t2i = compute_t2i(B[0], E[0], window_size)
    i2t = compute_i2t(t2i)
    D_r = compute_dist_matrix(x, y)
    D_m = np.repeat(D_r[np.newaxis, :, :], len(i2t), axis=0)
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, i2t)

    params = {"init_method": "random_sample", "cache_gran": 0.01,
                "ts_prob": 0.9, "x_prob": 0.7, "m_prob": 0.3, "w_t": 1.0,
                "mng": 1000, "pop_size":100, "obj_func":'d'}

    _, score = snrpga2(D_m, T_m, D_t, D, E, B[0], t2i, i2t, C, params)
    assert abs(score - (2 + 2 * math.sqrt(2)) - (4 * math.sqrt(2))) <= 1e-4
