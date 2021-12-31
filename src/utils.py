"""
Contains miscellaneous utility functions.
"""

import math


def round_down(val, base):
    """
    Round val down to multiple of base.
    """
    return base * math.floor(val / base)


def round_up(val, base):
    """
    Round val up to multiple of base.
    """
    return base * math.ceil(val / base)


def round_nearest(val, base):
    """
    Round val to nearest multiple of base.
    """
    return base * round(val / base)