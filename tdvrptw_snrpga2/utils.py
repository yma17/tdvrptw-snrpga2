"""
Contains miscellaneous utility functions.
"""

import math
from bisect import bisect_left


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


def take_closest(myList, myNumber, cache={}, granularity=1.0):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    Additionally, use cache to conserve runtime.
    Use granularity to round myNumber to cache key.
    Source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    myNumberGran = round_nearest(myNumber, granularity)

    if myNumberGran in cache:
        return cache[myNumberGran]

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        cache[myNumberGran] = after   # add to cache
        return after
    else:
        cache[myNumberGran] = before   # add to cache
        return before