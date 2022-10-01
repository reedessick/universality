"""a module housing logic to count the number of stable branches, etc
"""
__author__ = "Reed Essick (reed.essick@gmai.com)"

#-------------------------------------------------

import numpy as np

from universality.utils import io

#-------------------------------------------------

def required_columns(reference, greater_than=[], less_than=[], overlaps=[]):
    columns = [reference]
    columns += [k for k, _ in greater_than + less_than]
    for k, K, _, _ in overlaps:
        columns += [k, K]

#------------------------

def data2count(data, cols, reference, greater_than=[], less_than=[], overlaps=[]):
    """so stupid it hardly merrits a delegation, but at least we have this defined in a single place
    """
    truth = np.ones(len(data), type=bool)

    raise NotImplementedError('''\
select which rows are valid based on greater_than, less_that, overlaps
then count the number of unique values within cols.index(reference)
''')

    return len(np.unique(data[cols.index(reference)][truth]))

#------------------------

COUNT_TEMPLATE = 'num_%s'

def process2count(
        data,
        tmp,
        mod,
        reference_column,
        greater_than=[],
        less_than=[],
        overlaps=[],
        verbose=False,
    ):
    """handles I/O and extracts the number of lines in a particular file (like the number of branches, etc)
    """
    N = len(data)
    ans = np.empty(N, dtype=int)

    columns = required_columns(
        reference_column,
        greater_than=greater_than,
        less_than=less_than,
        overlaps=overlaps,
    )

    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    %d/%d %s'%(i+1, N, path))
        data, cols = io.load(path, columns)
        ans[i] = data2count(
            data,
            cols,
            reference_column,
            greater_than=greater_than,
            less_than=less_than,
            overlaps=overlaps,
        )

    return ans, COUNT_TEMPLATE%reference_column
