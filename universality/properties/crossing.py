"""a module housing logic to identify points when curves first cross a certain value
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

def _data2crossing(d, ref):
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

def data2crossing(x, d, ref, Ncol, static_ranges=None, dynamic_minima=None, dynamic_maxima=None):

    truth = np.ones(len(d), dtype=bool)
    if static_ranges is not None:
        for j, (m, M) in static_ranges:
            truth *= (m<=d[:,j])*(d[:,j]<=M)

    if dynamic_minima is not None:
        for j, minima in dynamic_minima:
            truth *= d[:,j] >= minima[i]

    if dynamic_maxima is not None:
        for j, maxima in dynamic_maxima:
            truth *= d[:,j] <= maxima[i]

    if not np.any(truth):
        raise RuntimeError('could not find any samples within all specified ranges!')
    d = d[truth]
    num = len(d)

    ### direct iteration to find the first and last crossings
    first_before, first_after = _data2crossing(x, ref)

    last_before, last_after = _data2crossing(x[::-1], ref) ### iterate from the back to the front...
    last_before, last_after = num-1-last_after, num-1-last_before

    # linearly interpolate to find values at the crossing
    first_a = (ref - x[first_before])/(x[first_after] - x[first_before])
    first_b = (x[first_after] - ref)/(x[first_after] - x[first_before])

    last_a = (ref - x[last_before])/(x[last_after] - x[last_before])
    last_b = (x[last_after] - ref)/(x[last_after] - x[last_before])

    ans = np.empty(2*Ncol, dtype=float)
    for j in range(Ncol):
        ans[2*j] = d[first_after,j]*first_a + d[first_before,j]*first_b ### store the value at the first crossing
        ans[2*j+1] = d[last_after,j]*last_a + d[last_before,j]*last_b ### store the value at the last crossing

    return ans

def outputcolumns(columns, reference, reference_value, reference_is_column=False, custom_names=None):
    if custom_names is None:
        custom_names = dict()

    if reference_column_value_is_column:
        col_first = '(first_%s@%s)'
        col_last = '(last_%s@%s)'
    else:
        col_first = '(first_%s=%s)'
        col_last = '(last_%s=%s)'

    col_first = col_first%(reference, reference_value)
    col_last = col_last%(reference, reference_value)

    for column in args.columns:
        outcols += custom_names.get(column, [column+col_first, column+col_last])

    return outcols

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
    Ncol = len(columns)
    loadcolumns = [reference_column] + columns

    if static_ranges is not None:
        loadcolumns += [key for key in static_ranges.keys() if key not in loadcolumns] ### avoid duplicates
        static_ranges = [(loadcolumns.index(column)-1, val) for column, val in static_ranges.items()] ### NOTE: assumes data will be loaded in the same order as loadcolumns
    else:
        static_ranges = []

    if dynamic_minima is not None:
        for val in dynamic_minima.values():
            assert len(val)==N, 'dynamic minima must have the same length as data'
        loadcolumns += [key for key in dynamic_minima.keys() if key not in loadcolumns]
        dynamic_minima = [(loadcolumns.index(key)-1, val) for key, val in dynamic_minima.items()]
    else:
        dynamic_minima = []

    if dynamic_maxima is not None:
        for val in dynamic_maxima.values():
            assert len(val)==N, 'dynamic minima must have the same length as data'
        loadcolumns += [key for key in dynamic_maxima.keys() if key not in loadcolumns]
        dynamic_maxima = [(loadcolumns.index(key)-1, val) for key, val in dynamic_maxima.items()]
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
        ref = reference_column_value[i] ### pull this out only once

        ans[i] = data2crossing(d[:,0], d[:,1:], ref, Ncol, static_ranges=static_ranges, dynamic_minima=dynamic_minima, dynamic_maxima=dynamic_maxima)

    return ans
