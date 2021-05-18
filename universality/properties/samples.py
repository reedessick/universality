"""a module housing logic to manipulate samples within large sets of EoS realizations
"""
__author__ = "Reed Essick (reed.essick@gmai.com)"

#-------------------------------------------------

import os
import sys

import numpy as np

from universality.utils import (io, utils)
from universality import stats

#-------------------------------------------------

KNOWN_SELECTION_RULES = [
    'random',
    'min',
    'max',
    'nearest_neighbor',
]

DEFAULT_SELECTION_RULE = KNOWN_SELECTION_RULES[0]

#------------------------

DEFAULT_COLUMN_NAME = {
    'identity': '%(fcolumn)s',
    'add': '(%(fcolumn)s)+(%(xcolumn)s)',
    'subtract': '(%(fcolumn)s)-(%(xcolumn)s)',
    'multiply': '(%(fcolumn)s)*(%(xcolumn)s)',
    'divide': '(%(fcolumn)s)/(%(xcolumn)s)',
    'logarithmic differentiate': 'd(ln_%(fcolumn)s)/d(ln_%(xcolumn)s)',
    'differentiate': 'd(%(fcolumn)s)/d(%(xcolumn)s)',
    'integrate': 'int(%(fcolumn)s)d(%(xcolumn)s)',
}

FUNCTIONS = {
    'identity': (lambda x, f: f),
    'add': (lambda x, f : x+f),
    'subtract': (lambda x, f : f-x),
    'multiply': (lambda x, f : x*f),
    'divide': (lambda x, f : f/x),
    'logarithmic differentiate': (lambda x,y : utils.num_dfdx(x,f)*x/f),
    'differentiate': utils.num_dfdx,
    'integrate': utils.num_intfdx,
}

KNOWN_ACTIONS = list(DEFAULT_COLUMN_NAME.keys())

#------------------------

DEFAULT_SCALE = 1.0
DEFAULT_SHIFT = 0.0

#-------------------------------------------------

def calculus(data, cols, xcolumn, fcolumn, foo, newcolumn, scale=DEFAULT_SCALE, shift=DEFAULT_SHIFT, overwrite=False):
    """perform basic operations on a data set: scale*(foo(xcolumn, fcolumn) + shift)
    """
    npts, ncol = data.shape
    if overwrite:
        if newcolumn in cols:
            ans = data
            ind = cols.index(newcolumn)
            header = cols
        
        else:
            ans = np.empty((npts, ncol+1), dtype=float)
            ans[:,:-1] = data
            ind = -1
            header = cols+[newcolumn]
        
    else:
        assert newcolumn not in cols, "column=%s already exists!"%newcolumn
        ans = np.empty((npts, ncol+1), dtype=float)
        ans[:,:-1] = data
        ind = -1
        header = cols+[newcolumn]

    ans[:,ind] = scale*(foo(data[:,cols.index(xcolumn)], data[:,cols.index(fcolumn)]) + shift) ### compute the integral or derivative
    return ans, header

def process_calculus(
        data,
        input_tmp,
        mod,
        output_tmp,
        xcolumn,
        fcolumn,
        foo,
        newcolumn,
        scale=DEFAULT_SCALE,
        shift=DEFAULT_SHIFT,
        overwrite=False,
        verbose=False,
    ):
    """manages I/O for performing calculations on a large number of files
    """
    N = len(data)
    for ind, eos in enumerate(data):
        tmp = {'moddraw':eos//mod, 'draw':eos}
        path = input_tmp%tmp
        if verbose:
            print('    %d/%d %s'%(ind+1, N, path))
        d, c = io.load(path)

        ans, cols = calculus(d, c, xcolumn, fcolumn, foo, newcolumn, scale=scale, shift=shift, overwrite=overwrite)

        new = output_tmp%tmp
        if verbose:
            print('        writing: '+new)

        newdir = os.path.dirname(new)
        if not os.path.exists(newdir):
            try:
                os.makedirs(newdir)
            except OSError:
                pass ### directory already exists

        io.write(new, ans, cols) ### save the result to the same file

#-------------------------------------------------
# utility functions for processing process directory structures
#-------------------------------------------------

def data2samples(x, data, x_test, selection_rule=DEFAULT_SELECTION_RULE, branches=None, default_values=None):
    """logic for exactly how we extract samples from (possibly non-monotonic) data
    """
    inds = np.arange(len(x))

    # set up logic for which variables to look up
    Nref = len(x_test)

    Ndata, Ncols = data.shape
    ans = np.empty(Nref*Ncols, dtype=float)

    # set up logic surrounding stable branches
    if branches is None:
        branches = [np.ones_like(x, dtype=bool)]

    branches = [branch for branch in branches if np.any(branch)] ### NOTE: this may be fragile because we may have "sparse sampling"
                                                                 ###     that just happens to not have any points within the branch.
                                                                 ###     We may be able to improve upon this in the future...

    for branch in branches:
        assert np.all(np.diff(x[branch]) > 0), 'reference value must monotonically increase on each branch!'

    # retrieve values from data on each branch separately
    if selection_rule == 'nearest_neighbor':
        ### NOTE: we do not require default_values to be provided with this logic!

        # set up holders for the indecies used in each branch...
        ref_inds = np.empty(Nref, dtype=int)
        ref_dx = np.empty(Nref, dtype=float)
        ref_dx[:] = +np.infty

        # iterate over branches, finding the nearest neighbor from any branch
        for branch in branches:
            # find index for static x
            for i, X in enumerate(x_test): ### extract values from static and dynamic at the same time
                dX = np.abs(x[branch]-X)
                m = np.min(dX)

                print " "
                print dX
                print m, inds[branch][np.argmin(dX)]

                if m < ref_dx[i]: ### closer than we have previously seen
                    ref_dx[i] = m
                    ref_inds[i] = inds[branch][np.argmin(dX)] ### the corresponding index of x

        if np.any(ref_inds >= len(x)):
            print branches ### require a default value in case there are no stable branches?
            print ref_inds
            raise RuntimeError

        # extract values corresponding to the nearest neighbor indecies and assign to ans
        for j in range(Ncols):
            ans[j*Nref:(j+1)*Nref] = data[:,j][ref_inds]

    # the rest of the selection rules are more standard; we find all possible values
    else:

        # extract values from each branch where there is coverage
        vals = [[] for _ in range(Nref)] ### holder for values from each branch
        for branch in branches:
            minX = np.min(x[branch])
            maxX = np.max(x[branch])
            for i, X in enumerate(x_test):
                if (minX <= X) and (X <= maxX): ### we have coverage on this branch
                    datum = []
                    for j in range(Ncols):
                        datum.append(np.interp(X, x[branch], data[branch,j]))
                    vals[i].append(datum)

        ### iterate through vals and pick based on selection rule

        print "\n"
        print len(branches)
        print vals

        # fill in default values as needed
        for i, val in enumerate(vals): ### one for each x_test
            if len(val) == 0: ### x_test not found on any branch!
                assert (default_values is not None) and (len(default_values) == Ncols), 'default_values must be specified if x_test is not on any branch!'
                vals[i] = [default_values]

        # pick from the branches at random (independently for each x_test)
        if selection_rule == 'random': ### select from multivalued at random
            vals = [val[np.random.randint(len(val), size=1)] for val in vals]

        elif selection_rule == 'min': ### pick the minimum, requires Ncols==1
            assert Ncols == 1, 'cannot use selection_rul="min" with more than one column simultaneously!'
            vals = [[np.min(val)] for val in vals]

        elif selection_rule == 'max': ### pick the max, requires Ncol==1
            assert Ncols == 1, 'cannot use selection_rul="max" with more than one column simultaneously!'
            vals = [[np.max(val)] for val in vals]

        else:
            raise ValueError('selection_rule=%s not understood!'%selection_rule)

        print vals
        print "\n"

        ### iterate again to map results into ans
        vals = np.transpose(vals) ### map from Nref*Ncol --> Ncol*Nref
        for j in range(Ncols):
            ans[j*Nref:(j+1)*Nref] = vals[j,:]

    return ans

#------------------------

COL_TEMPLATE = '%s(%s=%s)' ### add columns corresponding to a specific value
REF_TEMPLATE = '%s(%s@%s)' ### add columns corresponding to reference values read dynamically from columns

def outputcolumns(columns, reference, reference_values=[], reference_columns=[]):
    outcols = []
    for column in columns:
        outcols += [COL_TEMPLATE%(column, reference, val) for val in reference_values]
        outcols += [REF_TEMPLATE%(column, reference, col) for col in reference_columns]
    return outcols

def process2samples(
        data,
        tmp,
        mod,
        xcolumn,
        ycolumns,
        static_x_test=None,
        dynamic_x_test=None,
        x_multiplier=1.,
        verbose=False,
        selection_rule=DEFAULT_SELECTION_RULE,
        branches_mapping=None,
        default_values=None,
    ):
    """manages I/O and extracts samples at the specified places
    returns an array that is ordered as follows
        for each ycolumn: for each static x_test
    """
    loadcolumns = [xcolumn] + ycolumns

    if static_x_test is None:
        static_x_test = []
    static_x_test = list(static_x_test) ### make sure this is a list
    Nref = len(static_x_test)

    if dynamic_x_test is not None:
        assert len(dynamic_x_test)==len(data)
        Ndyn = np.shape(dynamic_x_test)[1]
    else:
        Ndyn = 0

    Ntot = Nref+Ndyn
    assert Ntot > 0, 'must provide at least one static_x_test or dynamic_x_test'

    if branches_mapping is not None:
        if selection_rule != 'nearest_neighbor':
            assert default_values is not None, 'must specify default_values when branches_mapping is not None'
            assert len(default_values) == len(ycolumns), 'must specify exactly 1 default value for each ycolumn!'

        branches_tmp, affine, affine_start, affine_stop = branches_mapping
        loadcolumns.append(affine)

    else:
        branches_tmp = None

    N = len(data)
    ans = np.empty((N, (Nref+Ndyn)*len(ycolumns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            sys.stdout.write('\r    %d/%d %s'%(i+1, N, path))
            sys.stdout.flush()

        d, c = io.load(path, loadcolumns)
        if branches_mapping is not None:
            a = d[:,c.index(affine)]

        x = d[:,c.index(xcolumn)] * x_multiplier
        d = d[:,[c.index(col) for col in ycolumns]]

        if branches_tmp is not None:
            branches_path = branches_tmp%{'moddraw':eos//mod, 'draw':eos}
            if verbose:
                sys.stdout.write('\r    %d/%d %s'%(i+1, N, branches_path))
                sys.stdout.flush()

            b, _ = io.load(branches_path, [affine_start, affine_stop])
            branches = [(start <= a)*(a <= stop) for start, stop in b] ### define booleans to represent branches

        else:
            branches = None

        ans[i] = data2samples(
            x,
            d,
            static_x_test+list(dynamic_x_test[i]),
            selection_rule=selection_rule,
            branches=branches,
            default_values=default_values,
        )

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return ans

#-------------------------------------------------

def process2quantiles(
        data,
        tmp,
        mod,
        xcolumn,
        ycolumn,
        x_test,
        quantiles,
        quantile_type='sym',
        x_multiplier=1.,
        y_multiplier=1.,
        weights=None,
        selection_rule=DEFAULT_SELECTION_RULE,
        branches_mapping=None,
        default_y_value=None,
        verbose=False,
    ):
    """manages I/O and extracts quantiles at the specified places
    """
    num_points = len(x_test)

    truth = np.empty(num_points, dtype=bool) ### used to extract values
    N = len(data)
    columns = [xcolumn, ycolumn]
    if weights is None:
        weights = np.ones(N, dtype=float) / N

    ### compute the quantiles
    Nquantiles = len(quantiles)
    if quantile_type=='hpd':
        Nquantiles *= 2 ### need twice as many indicies for this

    qs = np.empty((Nquantiles, num_points), dtype=float)
    med = np.empty(num_points, dtype=float)

    ### delegate to process2samples to actually extract data from files
    if default_y_value is None:
        default_values = [np.nan] ### flag that we want to exclude this from the quantile calculation
    else:
        default_values = [default_y_value]

    # returns an array of shape: (len(data), len(x_test))
    samps = process2samples(
        data,
        tmp,
        mod,
        xcolumn,
        [ycolumn],
        static_x_test=x_test,
        verbose=verbose,
        selection_rule=selection_rule,
        branches_mapping=branches_mapping,
        default_values=default_values,
    )

    ### iterate over the corresponding samps, computing quantile for each x_test
    for i in xrange(num_points):

        _y = samps[:,i] ### extract the values from this particular x_test
        truth = _y==_y
        _y = _y[truth] ### only keep things that are not nan
        _w = weights[truth]

        if quantile_type=="sym":
            qs[:,i] = stats.quantile(_y, quantiles, weights=_w)     ### compute quantiles

        elif quantile_type=="hpd":

            ### FIXME: the following returns bounds on a contiguous hpd interval, which is not necessarily what we want...

            bounds = stats.samples2crbounds(_y, quantiles, weights=_w) ### get the bounds
            qs[:,i] = np.array(bounds).flatten()

        else:
            raise ValueError('did not understand --quantile-type=%s'%quantile_type)

        med[i] = stats.quantile(_y, [0.5], weights=_w)[0] ### compute median

    return qs, med
