"""a module housing logic to count the number of stable branches, etc
"""
__author__ = "Reed Essick (reed.essick@gmai.com)"

#-------------------------------------------------

import numpy as np

from universality.utils import io

#-------------------------------------------------

def data2count(d):
    """so stupid it hardly merrits a delegation, but at least we have this defined in a single place
    """
    return len(d)

COUNT_TEMPLATE = 'num%s'

def process2count(
        data,
        tmp,
        mod,
        reference_column,
        verbose=False,
    ):
    """handles I/O and extracts the number of lines in a particular file (like the number of branches, etc)
    """
    ans = np.empty(len(data), dtype=int)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    '+path)
        d, _ = io.load(path, [reference_column])
        ans[i] = data2count(d)

    return ans, COUNT_TEMPLATE%reference_column
