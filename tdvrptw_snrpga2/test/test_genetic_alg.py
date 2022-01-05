"""
Contains unit tests for the high-level function in src/genetic_alg.py.
These are much smaller than the SOLOMON test cases, allowing for more   
    easy debugging and more predictable results.
Use pytest to run tests.
"""

import math
import numpy as np

from tdvrptw_snrpga2.test.cases import *
from tdvrptw_snrpga2.src.compute_inputs import *
from tdvrptw_snrpga2.src.genetic_alg import *


def test_snrpga2_0():
    """Test assertion checks"""
    b = False
    D_m = np.zeros((2, 2))
    try:
        snrpga2(D_m, None, None, None, None, None, None, None, None, None)
    except AssertionError:
        b = True
    assert b

    D, C = np.array([30]), 20
    D_m = np.zeros((3, 3))
    b = False
    try:
        snrpga2(D_m, None, None, D, None, None, None, None, None, C)
    except AssertionError:
        b = True
    assert b

    D, C = np.array([0, 10]), 20
    b = False
    T_m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    g_start, g_end = 10, 18
    window_size = 5
    t2i = {10:0, 15:1, 20:2}
    D_t = np.array([0, 3])
    try:
        snrpga2(D_m, T_m, D_t, D, None, t2i, window_size, g_start, g_end, C)
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
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, i2t)

    res, score = snrpga2(D_r, T_m, D_t, D, E, t2i, window_size, B[0], E[0], C,
                            obj_func='d')
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
    T_r = compute_raw_time_matrix(D_r)
    T_m = compute_time_matrix(T_r, B, i2t)

    _, score = snrpga2(D_r, T_m, D_t, D, E, t2i, window_size, B[0], E[0], C,
                        obj_func='d')
    assert abs(score - (2 + 2 * math.sqrt(2)) - (4 * math.sqrt(2))) <= 1e-4
