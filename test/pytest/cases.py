"""
File containing test cases that can be reused across unit tests.
"""

import numpy as np


def caseA():
    """
    Uniform time windows that fully span depot time
    Zero delivery time
    """
    x, y = [0, -1, -1, 1, 1], [0, -1, 1, 1, -1]
    B, E = np.array([0,0,0,0,0]), np.array([5,5,5,5,5])
    D = np.array([0, 0, 0, 0, 0])
    window_size = 0.5
    
    return x, y, B, E, D, window_size


def caseB():
    """
    Non-uniform time windows that fully span depot time
    Nonzero delivery time
    Speed is not dependent on time or category
    """
    x, y = [4, 8, 1, 7], [1, 4, -8, 0]
    B, E = np.array([0, 20, 5, 20]), np.array([40, 30, 25, 40])
    D = np.array([0, 10, 5, 5])
    window_size = 10

    return x, y, B, E, D, window_size


def caseC():
    """
    Non-uniform time windows that fully span depot time
    Nonzero delivery time
    Speed is dependent on time and category
    Only one scenario
    Loose time windows
    """
    x, y = [0, 1, -1], [-10, -10, -10]
    B, E = np.array([0, 100, 0]), np.array([200, 200, 100])
    D = np.array([0, 11, 22])
    window_size = 100
    C = np.array([[0,1,2],[-1,2,0],[-1,-1,1]])
    S = np.array([[[0.6, 0.7, 0.8], [0.9, 1.0, 1.1], [1.2, 1.3, 1.4]]])

    return x, y, B, E, D, window_size, C, S


def caseD():
    """
    Non-uniform time windows that fully span depot time
    Nonzero delivery time
    Speed is dependent on category, but not time
    Multiple scenarios
    More tight time windows
    """
    x, y = [0, 1, -1], [-10, -10, -10]
    B, E = np.array([0, 1, 2.5]), np.array([4, 4, 4])
    D = np.array([0, 0.05, 0.1])
    window_size = 2
    C = np.array([[0,1,2],[-1,2,0],[-1,-1,1]])
    S = np.array([[[0.6, 0.7, 0.8], [0.6, 0.7, 0.8], [0.6, 0.7, 0.8]],
                [[0.9, 1.0, 1.0], [0.9, 1.0, 1.0], [0.9, 1.0, 1.0]],
                [[1.2, 1.3, 1.4], [1.2, 1.3, 1.4], [1.2, 1.3, 1.4]]])

    return x, y, B, E, D, window_size, C, S