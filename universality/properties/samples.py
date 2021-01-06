"""a module housing logic to manipulate samples within large sets of EoS realizations
"""
__author__ = "Reed Essick (reed.essick@gmai.com)"

#-------------------------------------------------

import os
import glob

import numpy as np

from universality.utils import (io, utils)
from universality import stats

#-------------------------------------------------

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

def data2samples(x, data, static, dynamic, nearest_neighbor=False):
    """logic for exactly how we extract samples from (possibly non-monotonic) data
    """
    Nref, static_x_test = static
    Ndyn, dynamic_x_test = dynamic

    Ndata, Ncols = data.shape
    ans = np.empty((Nref+Ndyn)*Ncols, dtype=float)

    if nearest_neighbor:
        if Nref > 0:
            inds = np.empty(Nref, dtype=int)
            for i, X in enumerate(static_x_test):
                inds[i] = np.argmin(np.abs(x-X)) ### which indecies to look up

        if Ndyn > 0:
            dyn_inds = np.empty(Ndyn, dtype=int)
            for i, X in enumerate(dynamic_x_test):
                dyn_inds[i] = np.argmin(np.abs(x-X))

    if (Nref > 0) and (Ndyn > 0):
        for j in range(Ncols):
            if nearest_neighbor:
                ans[j*Ntot:j*Ntot+Nref] = data[:,j][inds]
                ans[j*Ntot+Nref:(j+1)*Ntot] = data[:,j][dyn_inds]
            else:
                ans[j*Ntot:j*Ntot+Nref] = np.interp(static_x_test, x, data[:,j])
                ans[j*Ntot+Nref:(j+1)*Ntot] = np.interp(dynamic_x_test, x, data[:,j])

    elif Nref > 0:
        for j in range(Ncols):
            if nearest_neighbor:
                ans[j*Nref:(j+1)*Nref] = data[:,j][inds]
            else:
                ans[j*Nref:(j+1)*Nref] = np.interp(static_x_test, x, data[:,j])

    else: ### Ndyn > 0
        for j in range(Ncols):
            if nearest_neighbor:
                ans[j*Ndyn:(j+1)*Ndyn] = data[:,j][dyn_inds]
            else:
                ans[j*Ndyn:(j+1)*Ndyn] = np.interp(dynamic_x_test, x, data[:,j])

    return ans

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
        verbose=False,
        nearest_neighbor=False,
    ):
    """manages I/O and extracts samples at the specified places
    returns an array that is ordered as follows
        for each ycolumn: for each static x_test
    """
    loadcolumns = [xcolumn] + ycolumns

    if static_x_test is not None:
        Nref = len(static_x_test)
    else:
        Nref = 0

    if dynamic_x_test is not None:
        assert len(dynamic_x_test)==len(data)
        Ndyn = np.shape(dynamic_x_test)[1]
    else:
        Ndyn = 0

    Ntot = Nref+Ndyn
    assert Ntot > 0, 'must provide at least one static_x_test or dynamic_x_test'

    N = len(data)
    ans = np.empty((N, (Nref+Ndyn)*len(ycolumns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    %d/%d %s'%(i+1, N, path))
        d, c = io.load(path, loadcolumns)
        x = d[:,c.index(xcolumn)]
        d = d[:,1:]

        ans[i] = data2samples(x, d, (Nref, static_x_test), (Ndyn, dynamic_x_test[i]), nearest_neighbor=nearest_neighbor)

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
        default_y_value=None,
        verbose=False,
    ):
    """manages I/O and extracts quantiles at the specified places
    """
    y_test = [] ### keep this as a list because we don't know how many stable branches there are
    w_test = []
    num_points = len(x_test)

    truth = np.empty(num_points, dtype=bool) ### used to extract values
    N = len(data)
    columns = [xcolumn, ycolumn]
    if weights is None:
        weights = np.ones(N, dtype=float) / N

    for ind, (eos, weight) in enumerate(zip(data, weights)): ### iterate over samples and compute weighted moments
        paths = sorted(glob.glob(tmp%{'moddraw':eos//mod, 'draw':eos}))
        for eos_path in paths:
            if verbose:
                print('    %d/%d %s'%(ind+1, N, eos_path))
            d, _ = io.load(eos_path, columns)

            d[:,0] *= x_multiplier
            d[:,1] *= y_multiplier

            _y = np.empty(num_points, dtype=float)
            _y[:] = np.nan ### signal that nothing was available at this x-value

            truth[:] = (np.min(d[:,0])<=x_test)*(x_test<=np.max(d[:,0])) ### figure out which x-test values are contained in the data
            _y[truth] = np.interp(x_test[truth], d[:,0], d[:,1]) ### fill those in with interpolated values

            y_test.append( _y ) ### add to the total list
            w_test.append( weight )

        if len(paths) and (default_y_value is not None) and np.any(x_test > np.max(d[:,0])):
            _y = np.empty(num_points, dtype=float)
            _y[:] = np.nan ### signal that nothing was available at this x-value
            _y[x_test >= np.max(d[:,0])] = default_y_value
            y_test.append( _y ) ### add to the total list
            w_test.append( weight )

    if len(y_test)==0:
        raise RuntimeError('could not find any files matching "%s"'%tmp)

    y_test = np.array(y_test) ### cast to an array
    w_test = np.array(w_test)

    ### compute the quantiles
    Nquantiles = len(quantiles)
    if quantile_type=='hpd':
        Nquantiles *= 2 ### need twice as many indicies for this

    qs = np.empty((Nquantiles, num_points), dtype=float)
    med = np.empty(num_points, dtype=float)

    for i in xrange(num_points):

        _y = y_test[:,i]
        truth = _y==_y
        _y = _y[truth] ### only keep things that are not nan
        _w = w_test[truth]

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
