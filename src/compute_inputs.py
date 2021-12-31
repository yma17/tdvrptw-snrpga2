"""
Contains helper functions for computation of values for algorithm input.
"""

import numpy as np
from scipy.spatial import distance_matrix

from .utils import round_down, round_up


def compute_t2i(b_d, e_d, window_size):
    """
    Construct mapping from raw time to time window index.
    Parameters:
    - b_d: float (beginning depot time, >= 0)
    - e_d: float (ending depot time, >= 0)
    - window_size: int (amount of unit time for each window)
    Returns:
    - t2i: dict
    """
    assert b_d >= 0
    assert e_d >= 0

    t2i = {}
    i = 0
    window_begin = round_down(b_d, window_size)
    window_end = round_up(e_d, window_size)
    while window_begin <= window_end:
        t2i[window_begin] = i
        window_begin += window_size
        i += 1

    return t2i


def compute_i2t(t2i):
    """
    Construct mapping from time window index to raw time.
    Parameters:
    - t2i: dict
    Returns:
    - i2t: list (keys-values from t2i reversed)
    """
    i2t = [0 for _ in range(len(t2i))]
    for k, v in t2i.items():
        i2t[v] = k
    return i2t


def compute_raw_dist_matrix(x, y):
    """
    Compute (symmetric) raw distance matrix given x and y coords.
    Parameters:
    - x, y: list (representing coordinates)
    Returns:
    - dist_matrix: np.ndarray
    """
    assert len(x) == len(y)
    pts = np.vstack([x, y]).T
    dist_matrix = distance_matrix(pts, pts)
    return dist_matrix


def compute_dist_matrix(D_r, B, E, D, i2t, window_size, mmp=2):
    """
    Compute time-dependent distance matrix.
    Introduce penalties based on time intervals for each location.
    If penalties are incurred, the matrix will be non-symmetric.

    Parameters:
    - D_r: np.ndarray  (raw distance matrix)
    - B: 1D np.ndarray (begin times for each location)
    - E: 1D np.ndarray (end times for each location)
    - D: 1D np.ndarray (delivery times for each location)
    - i2t: list (map time window index to raw time)
    - window_size: int (amount of unit time for each window)
    - mmp: float (minimum multiplicative penalty)

    Returns:
    - D_m: 3D np.ndarray
        - D_m[i,s,d] = distance to travel from s to d
            if arrival time is at time i2t[i]
    """

    D_m = np.repeat(D_r[np.newaxis, :, :], len(i2t), axis=0)

    # For arrivals a location outside of the time interval,
    #  use this value as a penalty for the objective function.
    # This can be used to bring the arrival times within the
    #  input time windows ("soft" TW constraint).
    penalty_mltpr = 100 / window_size

    for d in range(D_r.shape[0]):
        for i, t in enumerate(i2t):
            if t < B[d]:  # too early
                D_m[i, :, d] *= max(penalty_mltpr * (B[d] - t), mmp)
            elif t + D[d] > E[d]:  # too late
                D_m[i, :, d] *= max(penalty_mltpr * (t + D[d] - E[d]), mmp)

    return D_m


def compute_raw_time_matrix(D_r, scen=None, C=None, S=None, i2t=None):
    """
    Construct raw travel time matrix from distances and speeds
        for each time window. It will be symmetrical.

    Parameters:
    - D_r: 2D np.ndarray  (raw distance matrix)
    - if time-dependent matrix returned, these variables are not None:
        - scen: int  (scenario id, determines degree of time dep)
        - C: 2D np.ndarray  (upper triangular category matrix)
        - S: 3D np.ndarray  (speed matrix, scenario on axis 0)
        - i2t: list

    Returns:
    - T_r: np.ndarray
        - 2D if not time-dependent
        - 3D if time-dependent (departure time window on axis 0) 
    """

    if scen is None:  # non time dependent, all speeds are trivially 1
        assert C is None and S is None and i2t is None
        return D_r  # 2D
    
    assert scen is not None and C is not None
    assert S is not None and i2t is not None
    assert S.shape[1] == len(i2t)
    assert D_r.shape[0] <= C.shape[0]

    T_r = np.repeat(D_r[np.newaxis, :, :], len(i2t), axis=0)
    
    for p in range(D_r.shape[0] - 1):
        for q in range(p + 1, D_r.shape[1]):
            T_r[:, p, q] *= S[scen, :, C[p, q]]
            T_r[:, q, p] *= S[scen, :, C[p, q]]
            
    return T_r  # 3D


def compute_time_matrix(T, B, i2t):
    """
    Construct travel time matrix that uses (possibly time-dependent)
        travel times, begin and end times for each location, and
        delivery times for each location.

    Parameters:
    - T: np.ndarray (raw travel times between each location)
        - 2D if not time-dependent
        - 3D if time-dependent (departure time window on axis 0)
    - B: 1D np.ndarray (begin times for each location)
    - i2t: list (map time window index to raw time)

    Returns:
    - T_m: 3D np.ndarray
        - T_m[i,s,d] = time to travel from s to d if departure time is i2t[i]
        - may include waiting time
        - does not include delivery time
    """
    
    T_m = np.zeros((len(i2t), T.shape[0], T.shape[1]))

    if T.ndim == 2:  # need to add time-dimension    
        for d in range(T.shape[1]):
            for s in range(T.shape[0]):
                for i, t in enumerate(i2t):
                    if t + T[s, d] < B[d]:  # arrival before start time
                        T_m[i, s, d] = B[d] - t  # waiting time
                    else:  # no need to wait
                        T_m[i, s, d] = T[s, d]
    else:  # T.ndim == 3
        for d in range(T.shape[1]):
            for s in range(T.shape[0]):
                for i, t in enumerate(i2t):
                    if t + T[i, s, d] < B[d]:  # arrival before start time
                        T_m[i, s, d] = B[d] - t  # waiting time
                    else:  # no need to wait
                        T_m[i, s, d] = T[i, s, d]

    return T_m