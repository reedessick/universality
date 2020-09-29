__doc__ = "a module housing logic to identify separate stable branches from sequences of solutions to the TOV equations"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

def data2extrema(*args, **kwargs):
    raise NotImplementedError

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

        for j, column in enumerate(columns):
            ans[i,2*j] = np.max(d[:,j])
            ans[i,2*j+1] = np.min(d[:,j])

    return ans
