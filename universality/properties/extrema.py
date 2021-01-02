"""a module housing logic to identify cruves' extrema
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality.utils import io

#-------------------------------------------------

def data2extrema(d, Ncol, static_ranges=None, dynamic_minima=None, dynamic_maxima=None):
    truth = np.ones(len(d), dtype=bool)

    if static_ranges is not None:
        for j, (m, M) in static_ranges:
            truth *= (m<=d[:,j])*(d[:,j]<=M)

    if dynamic_minima is not None:
        for j, minima in dynamic_minima:
            truth *= d[:,j] >= minima

    if dynamic_maxima is not None:
        for j, maxima in dynamic_maxima:
            truth *= d[:,j] <= maxima

    if not np.any(truth):
        raise RuntimeError('could not find any samples within all specified ranges!')
    d = d[truth]

    ans = np.empty(2*Ncol, dtype=float)
    for j in range(Ncol):
        ans[2*j] = np.max(d[:,j])
        ans[2*j+1] = np.min(d[:,j])

    return ans

MAX_TEMPLATE = 'max(%s)'
MIN_TEMPLATE = 'min(%s)'
def outputcolumns(columns, custom_names=None):
    if custom_names is None:
        custom_names = dict()

    outcols = []
    for column in columns:
        outcols += custom_names.get(column, [MAX_TEMPLATE%column, MIN_TEMPLATE%column])

    return outcols

def process2extrema(
        data,
        tmp,
        mod,
        columns,
        static_ranges=None,
        dynamic_minima=None,
        dynamic_maxima=None,
        verbose=False,
    ):
    """manages I/O and extracts max, min for the specified columns
    """
    N = len(data)
    Ncol = len(columns)
    loadcolumns = columns[:]

    if static_ranges is not None:
        loadcolumns += [key for key in static_ranges.keys() if key not in loadcolumns] ### avoid duplicates
        static_ranges = [(loadcolumns.index(column), val) for column, val in static_ranges.items()]
    else:
        static_ranges = []

    if dynamic_minima is not None:
        for val in dynamic_minima.values():
            assert len(val)==N, 'dynamic minima must have the same length as data'
        loadcolumns += [key for key in dynamic_minima.keys() if key not in loadcolumns]
        dynamic_minima = [(loadcolumns.index(key), val) for key, val in dynamic_minima.items()]
    else:
        dynamic_minima = []

    if dynamic_maxima is not None:
        for val in dynamic_maxima.values():
            assert len(val)==N, 'dynamic minima must have the same length as data'
        loadcolumns += [key for key in dynamic_maxima.keys() if key not in loadcolumns]
        dynamic_maxima = [(loadcolumns.index(key), val) for key, val in dynamic_maxima.items()]
    else:
        dynamic_maxima = []

    ans = np.empty((N, 2*len(columns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    %d/%d %s'%(i+1, N, path))
        d, _ = io.load(path, loadcolumns)

        minima = [(j, val[i]) for j, val in dynamic_minima]
        maxima = [(j, val[i]) for j, val in dynamic_maxima]

        ans[i] = data2extrema(d, Ncol, static_ranges=static_ranges, dynamic_minima=minima, dynamic_maxima=maxima)

    return ans
