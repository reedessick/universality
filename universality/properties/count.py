"""a module housing logic to count the number of stable branches, etc
"""
__author__ = "Reed Essick (reed.essick@gmai.com)"

#-------------------------------------------------

import numpy as np

from universality.utils import io

#-------------------------------------------------

def required_columns(
        reference,
        equals=[]
        greater_than=[],
        less_than=[],
        overlaps=[],
    ):
    columns = [reference]
    columns += [k for k, _ in equals + greater_than + less_than]
    for k, K, _, _ in overlaps:
        columns += [k, K]
    return columns

#------------------------

def data2count(data, cols, reference, greater_than=[], less_than=[], overlaps=[], equals=[]):
    """counts the number of unique elements in a reference column subject to some selection criteria
currently, all criteria are applied with AND logic. That is, all criteria must be passed in order for a row to count.
    """
    truth = np.ones(len(data), dtype=bool)

    for col, thr in greater_than:
        truth *= data[:,cols.index(col)] > thr

    for col, thr in less_than:
        truth *= data[:,cols.index(col)] < thr

    for start, stop, low, high in overlaps:
        truth *= (data[:,cols.index(start)] <= high) * (low <= data[:,cols.index(stop)])

    for col, val in equals:
        truth *= np.isclose(data[:,cols.index(col)], val) # use default rtol and atol

    return len(np.unique(data[:,cols.index(reference)][truth]))

#------------------------

COUNT_TEMPLATE = 'num_%s'

def process2count(
        data,
        tmp,
        mod,
        reference_column,
        static_greater_than=[],
        dynamic_greater_than=[],
        static_less_than=[],
        dynamic_less_than=[],
        static_overlaps=[],
        dynamic_overlaps=[],
        static_equals=[],
        dynamic_equals=[],
        verbose=False,
    ):
    """handles I/O and extracts the number of lines in a particular file (like the number of branches, etc)
    """
    N = len(data)
    ans = np.empty(N, dtype=int)

    # figure out which columns we need to read from each file
    columns = required_columns(
        reference_column,
        greater_than=static_greater_than+dynamic_greater_than,
        less_than=static_less_than+dynamic_less_than,
        overlaps=static_overlaps+dynamic_overlaps,
        equals=static_equals+dynamic_equals,
    )

    # iterate and extract data
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    %d/%d %s'%(i+1, N, path))
        d, c = io.load(path, columns)
        ans[i] = data2count(
            d,
            c,
            reference_column,
            greater_than=static_greater_than+[(key, val[i]) for key, val in dynamic_greater_than],
            less_than=static_less_than+[(key, val[i]) for key, val in dynamic_less_than],
            overlaps=static_overlaps+[(k, K, v[i], V[i]) for k, K, v, V in dynamic_overlaps],
            equals=static_equals+[(key, val[i]) for key, val in dynamic_equals],
        )

    return ans, COUNT_TEMPLATE%reference_column
