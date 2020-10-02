"""a module housing logic to identify points when curves first cross a certain value
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

def data2crossing(d, ref):
    '''returns the indecies of the 1D array 'data' that braket the first time it crosses ref'''
    N = len(d)
    above = d[0] > ref
    ind = 1 ### we already checked the case of i=0
    while ind < N: # direct iteration over samples to see when we cross the reference value
        if above != (d[ind] > ref): ### we crossed
            break
        ind += 1
    else: # ind == N and we haven't crossed
        raise RuntimeError('reference_column_value=%f never crossed by eos=%d'%(ref, eos))

    return ind-1, ind

def process2crossing(
    data,
    tmp,
    mod,
    reference_column,
    reference_column_value,
    columns,
    static_ranges=None,
    dynamic_minima=None,
    dynamic_maxima=None,
    verbose=False,
    ):
    """manages I/O and extracts the first, last crossings of specified columns
    """
    N = len(data)
    loadcolumns = [reference_column] + columns

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

    if isinstance(reference_column_value, (int, float)):
        reference_column_value = np.ones(len(data), dtype=float)*reference_column_value
    else:
        assert len(reference_column_value)==len(data), 'reference_column_value must be a int, float or must have the same length as data'

    ans = np.empty((N, 2*len(columns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    '+path)
        d, _ = load(path, loadcolumns)

        truth = np.ones(len(d), dtype=bool)
        for j, (m, M) in static_ranges:
            truth *= (m<=d[:,j])*(d[:,j]<=M)

        for j, minima in dynamic_minima:
            truth *= d[:,j] >= minima[i]

        for j, maxima in dynamic_maxima:
            truth *= d[:,j] <= maxima[i]

        if not np.any(truth):
            raise RuntimeError('could not find any samples within all specified ranges!')
        d = d[truth]
        num = len(d)

        ref = reference_column_value[i] ### pull this out only once

        ### direct iteration to find the first and last crossings
        first_before, first_after = _data2crossing(d[:,0], ref)
        
        last_before, last_after = _data2crossing(d[::-1,0], ref) ### iterate from the back to the front...
        last_before, last_after = num-1-last_after, num-1-last_before

        # linearly interpolate to find values at the crossing
        first_a = (ref - d[first_before,0])/(d[first_after,0] - d[first_before,0])
        first_b = (d[first_after,0] - ref)/(d[first_after,0] - d[first_before,0])

        last_a = (ref - d[last_before,0])/(d[last_after,0] - d[last_before,0])
        last_b = (d[last_after,0] - ref)/(d[last_after,0] - d[last_before,0])

        for j, column in enumerate(columns):
            ans[i,2*j] = d[first_after,j+1]*first_a + d[first_before,j+1]*first_b ### store the value at the first crossing
            ans[i,2*j+1] = d[last_after,j+1]*last_a + d[last_before,j+1]*last_b ### store the value at the last crossing

    return ans
