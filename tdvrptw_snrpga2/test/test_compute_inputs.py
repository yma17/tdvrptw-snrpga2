"""
Contains unit tests for functions in src/compute_inputs.py.
Use pytest to run tests.
"""

import math

from tdvrptw_snrpga2.test.cases import *
from tdvrptw_snrpga2.src.compute_inputs import *


def test_compute_t2i_and_i2t_1():
    b_d, e_d, window_size = 0, 40, 10
    
    t2i = compute_t2i(b_d, e_d, window_size)
    assert len(t2i) == 5
    assert t2i[0] == 0
    assert t2i[20] == 2
    assert t2i[40] == 4

    i2t = compute_i2t(t2i)
    assert len(i2t) == 5
    assert i2t[0] == 0
    assert i2t[2] == 20
    assert i2t[4] == 40


def test_compute_t2i_and_i2t_2():
    b_d, e_d, window_size = 9, 41, 10
    
    t2i = compute_t2i(b_d, e_d, window_size)
    assert len(t2i) == 6
    assert t2i[0] == 0
    assert t2i[50] == 5
    
    i2t = compute_i2t(t2i)
    assert i2t[0] == 0
    assert i2t[5] == 50


def test_compute_dist_matrix_1():
    x, y, _, _, _, _ = caseA()
    
    D_r = compute_dist_matrix(x, y)
    for i in range(len(x)):
        assert D_r[i, i] == 0  # diagonal should be 0
    for p, q in [(1, 3), (2, 4)]:
        assert D_r[p, q] == D_r[q, p]  # should be symmetrical
        assert D_r[p, q] == 2 * math.sqrt(2)  # distance check
    for p, q in [(1, 4), (2, 3)]:
        assert D_r[p, q] == 2  # distance check


def test_compute_raw_time_matrix_1():
    """function returns a 3D matrix"""
    x, y, B, E, _, window_size, C, S = caseC()

    i2t = compute_i2t(compute_t2i(B[0], E[0], window_size))
    D_r = compute_dist_matrix(x, y)

    T_r = compute_raw_time_matrix(D_r, scen=0, C=C, S=S, i2t=i2t)
    # Check shape
    assert T_r.shape == (3, 3, 3)
    # Check travel times from 1 to 2 (category 0)
    assert T_r[0, 1, 2] == 1.2
    assert T_r[1, 1, 2] == 1.8
    assert T_r[2, 1, 2] == 2.4
    # Check if travel times from 2 to 1 are identical
    assert T_r[0, 2, 1] == T_r[0, 1, 2]
    # Check travel times from 0 to 1 (category 1)
    assert T_r[0, 0, 1] == 0.7
    # Check travel times from 0 to 2 (category 2)
    assert T_r[0, 2, 0] == 0.8


def test_compute_time_matrix_1():
    """Raw travel times are not time-dependent"""
    x, y, B, E, _, window_size = caseB()

    i2t = compute_i2t(compute_t2i(B[0], E[0], window_size))
    D_r = compute_dist_matrix(x, y)
    T_r = compute_raw_time_matrix(D_r)
    
    T_m = compute_time_matrix(T_r, B, i2t)
    # Check shape
    assert T_m.shape == (len(i2t), D_r.shape[0], D_r.shape[1])
    # Check times from 0 to 1
    assert T_m[0, 0, 1] == 20
    assert T_m[1, 0, 1] == 10
    for i in range(2, 5):
        assert T_m[i, 0, 1] == 5
    # Check times from 1 to 2
    for i in range(5):
        assert abs(T_m[i, 1, 2] - 13.8924) < 1e-4
    # Check times from 2 to 3
    assert T_m[0, 2, 3] == 20
    for i in range(1, 5):
        assert T_m[i, 2, 3] == 10


def test_compute_time_matrix_2():
    """Raw travel times are time dependent"""
    x, y, B, E, _, window_size, C, S = caseC()

    i2t = compute_i2t(compute_t2i(B[0], E[0], window_size))
    D_r = compute_dist_matrix(x, y)
    T_r = compute_raw_time_matrix(D_r, scen=0, C=C, S=S, i2t=i2t)
    
    T_m = compute_time_matrix(T_r, B, i2t)
    # Check shape
    assert T_m.shape == (len(i2t), D_r.shape[0], D_r.shape[1])
    # Check times from 0 to 1
    assert T_m[0, 0, 1] == 100
    assert T_m[1, 0, 1] == 1.0
    assert T_m[2, 0, 1] == 1.3
    # Check times from 0 to 2
    assert T_m[0, 0, 2] == 0.8
    assert T_m[1, 0, 2] == 1.1
    assert T_m[2, 0, 2] == 1.4
    # Check times from 1 to 2
    assert T_m[0, 1, 2] == 1.2
    assert T_m[1, 1, 2] == 1.8
    assert T_m[2, 1, 2] == 2.4
