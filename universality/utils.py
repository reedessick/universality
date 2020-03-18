__doc__ = "a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples of a univeral relation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import glob

from collections import defaultdict
import numpy as np

import multiprocessing as mp

from . import eos
from . import stats

#-------------------------------------------------

G = 6.674e-8        # newton's constant in (g^-1 cm^3 s^-2)
c = (299792458*100) # speed of light in (cm/s)
c2 = c**2
Msun = 1.989e33     # mass of the sun in (g)

#------------------------

DEFAULT_NUM_PROC = min(max(mp.cpu_count()-1, 1), 15) ### reasonable bounds for parallelization...

DEFAULT_BANDWIDTH = 0.1
DEFAULT_MAX_NUM_SAMPLES = np.infty

KNOWN_CUMULATIVE_INTEGRAL_DIRECTIONS = [
    'increasing',
    'decreasing',
]
DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION = 'increasing'

DEFAULT_WEIGHT_COLUMN = 'logweight'

#-------------------------------------------------
# basic utilities for simulating samples
#-------------------------------------------------

def draw(mean, std, size=1, bounds=None):
    '''
    draw samples from a normal distribution. between bounds=[min, max], if supplied
    '''
    if bounds!=None:
        m, M = bounds
        ans = []
        while len(ans) < size:
            sample = np.random.normal(mean, std)
            if m <= sample <= M:
                ans.append(sample)
        return np.array(ans)

    else:
        return np.random.normal(mean, std, size)

def draw_from_weights(weights, size=1):
    """
    return indecies corresponding to the weights chosen herein
    we draw with replacement
    """
    N = len(weights)
    weights = np.array(weights)
    order = weights.argsort()
    weights = weights[order]

    ### make sure weights are normalized
    weights /= np.sum(weights)

    # compute a cdf and draw from it
    return order[np.ceil(np.interp(np.random.random(size), np.cumsum(weights), np.arange(N))).astype(int)]

def models2combinations(models):
    return zip(*[_.flatten() for _ in np.meshgrid(*[range(len(model)) for model in models])])

def logLike2weights(logLike):

    truth = logLike==logLike ### only keep things that are not nan

    weights = np.zeros_like(logLike, dtype=float)
    weights[truth] = np.exp(logLike[truth]-np.max(logLike[truth]))
    weights[truth] /= np.sum(weights[truth])

    truth[truth] = weights[truth]>1e-5*np.max(weights[truth]) ### only keep things that have a big enough weight

    return truth, weights

#-------------------------------------------------
# basic utilities for manipulating existing sapmles
#-------------------------------------------------

def column2logcolumn(name):
    return 'log(%s)'%name

def load(inpath, columns=[], logcolumns=[], max_num_samples=DEFAULT_MAX_NUM_SAMPLES):
    data = []
    with open(inpath, 'r') as obj:
        cols = obj.readline().strip().split(',')
        if columns:
            check_columns(cols, columns)
        else:
            columns = cols

        inds = [cols.index(col) for col in columns]

        count = 0
#        strmap = dict() ### the commented lines are an attempt to keep the data array as a float but support mappings to strings
#        strind = 0      ### this would require a big overhaul to the entire repo, though, so we're not wedded to the idea just yet
        for line in obj:
            if line[0]=='#':
                continue
            if count >= max_num_samples:
                break
            fields = line.strip().split(',')
            ans = [] ### downselect to what we actually want and cast to float
            for ind in inds:
                try: ### try casting everything to a float
                    ans.append(float(fields[ind]))
                except ValueError: ### could not cast to a float
                    ans.append(fields[ind])
#                    if fields[ind] not in strmap:
#                        strmap[fields[ind]] = strind
#                        strind += 1
#                    ans.append(strmap[fields[ind]])
            data.append(ans)
            count += 1

    data = np.array(data) ### cast as an array

    cols = [] ### figure out column names and map to logs as requested
    for i, col in enumerate(columns):
        if col in logcolumns:
            data[:,i] = np.log(data[:,i])
            cols.append(column2logcolumn(col))
        else:
            cols.append(col)

    return data, cols #, dict((strind, col) for col, strind in strmap.items())

#    data = np.genfromtxt(inpath, names=True, delimiter=',') ### assumes standard CSV format
#    if data.size==1:
#        data = np.array([data], dtype=data.dtype)
#
#    # check that all requested columns are actually in the data
#    if columns:
#        check_columns(data.dtype.fields.keys(), columns)
#    else:
#        columns = data.dtype.fields.keys()
#
#    # downselect data to what we actually want
#    if len(data) > max_num_samples: ### downsample if requested
#         data = data[:max_num_samples]
#
#    return \
#        np.transpose([np.log(data[column]) if column in logcolumns else data[column] for column in columns]), \
#        ['log(%s)'%column if column in logcolumns else column for column in columns]

def load_weights(*args, **kwargs):
    """loads and returns weights from multiple columns via  delegation to load_logweights
    normalizes the weights while it's at it
    """
    normalize = kwargs.pop('normalize', True)
    return exp_weights(load_logweights(*args, **kwargs), normalize=normalize)

def exp_weights(logweights, normalize=True):
    """exponentiate logweights and normalize them (if desired)
    """
    if normalize:
        logweights -= np.max(logweights)
        weights = np.exp(logweights)
        weights /= np.sum(weights)
    else:
        weights = np.exp(logweights)
    return weights

def load_logweights(inpath, weight_columns, logweightcolumns=[], invweightcolumns=[], max_num_samples=DEFAULT_MAX_NUM_SAMPLES):
    """loads and returns logweights from multiple columns
    """
    data, columns = load(inpath, columns=weight_columns, max_num_samples=max_num_samples) ### load the raw data

    for i, column in enumerate(columns): ### iterate through columns, transforming as necessary
        if column in logweightcolumns:
            if column in invweightcolumns:
                data[:,i] *= -1

        else:
            if column in invweightcolumns:
                data[:,i] = 1./data[:,i]

            data[:,i] = np.log(data[:,i])

    # multiply weights across all samples, which is the same as adding the logs
    return np.sum(data, axis=1)

def marginalize(data, logweights, columns):
    """marginalize to get equivalent weights over unique sets of columns
    """
    logmargweight = defaultdict(float)
    logmargweight2 = defaultdict(float)
    counts = defaultdict(int)
    for sample, logweight in zip(data, logweights):
        tup = tuple(sample)
        logmargweight[tup] = sum_log((logmargweight.get(tup, -np.infty), logweight))
        logmargweight2[tup] = sum_log((logmargweight2.get(tup, -np.infty), 2*logweight)) ### track the variance
        counts[tup] += 1

    num_columns = len(columns)
    ### store the columns requested, the marginalized weight, and the number of elements included in the set for this particular tuple
    columns = columns+['logmargweight', 'logvarmargweight', 'num_elements']

    results = np.empty((len(logmargweight.keys()), len(columns)), dtype=float)
    for i, key in enumerate(logmargweight.keys()):
        results[i,:num_columns] = key

        ### compute the log of the variance of the maginalized weight
        cnt = counts[key]
        lmw = logmargweight[key]
        lmw2 = logmargweight2[key]

        logvar = lmw2 + np.log(1. - np.exp(2*lmw - lmw2 - np.log(cnt)))

        results[i,num_columns:] = lmw, logvar, cnt

    return results, columns

def sum_log(logweights):
    """returns the log of the sum of the weights, retaining high precision
    """
    if np.any(logweights==np.infty):
        return np.infty
    m = np.max(logweights)
    if m==-np.infty:
        return -np.infty
    return np.log(np.sum(np.exp(logweights-m))) + m

def check_columns(present, required, logcolumns=[]):
    required = [column2logcolumn(column) if column in logcolumns else column for column in required]
    for column in required:
        assert column in present, 'required column=%s is missing!'%column

def whiten(data, verbose=False, outlier_stdv=np.infty):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    data -= means
    data /= stds

    # adjust stds to reject outliers
    if outlier_stdv < np.infty:
        for i in xrange(data.shape[1]):
            truth = np.abs(data[:,i]) < outlier_stdv
            refactor = np.std(data[truth,i])

            stds[i] *= refactor
            data[:,i] /= refactor

    if verbose:
        print('whitening marginal distributions')
        if len(data.shape)==1:
            print('  mean = %+.3e'%(means))
            print('  stdv = %+.3e'%(stds))

        else:
            for i, (m, s) in enumerate(zip(means, stds)):
                print('  mean(%01d) = %+.3e'%(i, m))
                print('  stdv(%01d) = %+.3e'%(i, s))

    return data, means, stds

def iqr_whiten(data, verbose=False, low=0.16, high=0.84):
    medians = np.median(data, axis=0)
    iqrs = np.percentile(data, 100*high, axis=0) - np.percentile(data, 100*low, axis=0)

    data -= medians
    data /= iqrs

    if verbose:
        print('whitening marginal distributions')
        if len(data.shape)==1:
            print('  median = %+.3e'%(medians))
            print('  IQR[%.2f,%.2f] = %+.3e'%(low, high, iqrs))

        else:
            for i, (m, s) in enumerate(zip(medians, iqrs)):
                print('  median(%01d) = %+.3e'%(i, m))
                print('  IQR[%.2f,%.2f](%01d) = %+.3e'%(low, high, i, s))

    return data, medians, iqrs

def upsample(x, y, n):
    """compute the path length along the curve and return an equivalent curve with "n" points
    this is done to avoid possible issues with np.interp if the curve is not monotonic or 1-to-1
    """
    X = np.linspace(np.min(x), np.max(x), n)
    return X, np.interp(X, x, y)

#    dx = x[1:]-x[:-1]
#    dy = y[1:]-y[:-1]
#    ds = (dx**2 + dy**2)**0.5
#    cs = np.cumsum(ds)
#    s = np.linspace(0, 1, n)*cs[-1] ### the total path-length associated with each point
#
#    bot = np.floor(np.interp(np.linspace(0, cs[-1], n), cs, np.arange(len(x)-1))) ### the lower index bounding the linear segment containing each new point
#
#    ### fill in the resulting array
#    X, Y = np.empty((2,n), dtype=float)
#    X[0] = x[0] ### end points are the same
#    Y[0] = y[0]
#    X[-1] = x[-1]
#    Y[-1] = y[-1]
#
#    ### fill in the intermediate values
#    for i, b in enumerate(bot):
#        X[i] = x[b-1] + ((s[i]-s[b])/ds[b])*dx[b]
#        Y[i] = y[b-1] + ((s[i]-s[b])/ds[b])*dy[b]
#
#    return X, Y

def downsample(data, n):
    """return a random subset of the data
    """
    N = len(data)
    assert n>0 and n<=N, 'cannot downsample size=%d to size=%d'%(N, n)

    truth = np.zeros(N, dtype=bool)
    while np.sum(truth) < n:
        truth[np.random.randint(0, N-1)] = True ### FIXME: this could be pretty wasteful...

    return data[truth], truth

def logaddexp(logx):
    '''
    assumes we have more than one axis and sums over axis=-1
    '''
    N = logx.shape[-1]
    max_logx = np.max(logx, axis=-1)

    return max_logx + np.log( np.sum(np.exp(logx - np.outer(max_logx, np.ones(N)).reshape(logx.shape)) , axis=-1) )

#-------------------------------------------------
# 1D CDF estimation
#-------------------------------------------------

def logcdf(samples, data, prior_bounds, weights=None, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION, num_proc=DEFAULT_NUM_PROC):
    """estimates the log(cdf) at all points in samples based on data and integration in "direction".
    Does this directly by estimating the CDF from the weighted samples WITHOUT building a KDE"""

    ### this should be relatively quick (just an ordered summation), so we do it once
    data, cweights = stats.samples2cdf(data, weights=weights)
    if direction=='increasing':
        pass ### we already integrate up from the lower values to higher values
    elif direction=='decreasing':
        cweights = 1. - cweights ### reverse the order of the integral
    else:
        raise ValueError('direction=%s not understood!'%direction)

    logcdfs = np.empty(len(samples), dtype=float)
    if num_proc==1: ### do everything on this one core
        logcdfs[:] = _logcdf_worker(samples, data, cweights, prior_bounds)

    else: ### parallelize
        # partition work amongst the requested number of cores
        Nsamp = len(samples)
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logcdf_worker, args=(samples[truth], data, cweights, prior_bounds), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logcdfs[truth] = conni.recv()

    return logcdfs

def _logcdf_worker(samples, data, cweights, bounds, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION, conn=None):
    ### we have to account for prior volume differences between different models to properly do the model-selection integral
    ### this is handled by explicitly passing the bounds for the overall prior that may or may not be truncated by samples
    local_samples = samples[:] ### make a copy so I can modify it in-place
    local_samples[local_samples<bounds[0]] = bounds[0]
    local_samples[local_samples>bounds[1]] = bounds[1]

    ###                   approx to the cumulative integral within the prior bounds
    logcdfs = np.interp(local_samples, data, cweights) - np.interp(bounds[0], data, cweights)
    truth = logcdfs > 0
    logcdfs[truth] = np.log(logcdfs[truth])
    logcdfs[np.logical_not(truth)] = -np.infty

    ### add the prior volume correction
    ### NOTE: we assume flat priors implicitly! really, this should be an integral over the (non-trivial) prior distribution
    if direction=='increasing':
        truth = bounds[0] < local_samples
        logcdfs[truth] -= np.log(local_samples[truth] - bounds[0])
        logcdfs[np.logical_not(truth)] = -np.infty ### these sample shave zero support in the prior, so we assign them zero weight

    elif direction=='decreasing':
        truth = bounds[1] > local_samples
        logcdfs[truth] -= np.log(bounds[1] - local_samples[truth])
        logcdfs[np.logical_not(truth)] = -np.infty ### this is the same thing as above

    else:
        raise ValueError('direction=%s not understood!'%direction)

    if conn is not None:
        conn.send(logcdfs)
    return logcdfs

def logcumkde(samples, data, variance, bounds=None, weights=None, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION):
    """estimates the log(cdf) at all points in samples based on data and integration in "direction"
    This is done with a 1D KDE-based CDF estimate between bounds
    """
    raise NotImplementedError('This is a non-trivial thing to do, and I have not done it yet')

def process2samples(
        data,
        tmp,
        mod,
        xcolumn,
        ycolumns,
        x_test,
        verbose=False,
    ):
    """manages I/O and extracts samples at the specified places
    """
    loadcolumns = [xcolumn] + ycolumns
    Nref = len(x_test)

    ans = np.empty((len(data), Nref*len(ycolumns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    '+path)
        d, c = load(path, loadcolumns)

        for j, column in enumerate(c[1:]):
            ans[i,j*Nref:(j+1)*Nref] = np.interp(x_test, d[:,0], d[:,1+j])

    return ans

def process2extrema(
        data,
        tmp,
        mod,
        columns,
        ranges,
        verbose=False,
    ):
    """manages I/O and extracts max, min for the specified columns
    """
    ref = ranges.keys()
    loadcolumns = columns + ref
    ranges = dict((loadcolumns.index(column), val) for column, val in ranges.items())

    ans = np.empty((len(data), 2*len(columns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    '+path)
        d, _ = load(path, loadcolumns)

        truth = np.ones(len(d), dtype=bool)
        for j, (m, M) in ranges.items():
            truth *= (m<=d[:,j])*(d[:,j]<=M)

        if not np.any(truth):
            raise RuntimeError('could not find any samples within all specified ranges!')
        d = d[truth]

        for j, column in enumerate(columns):
            ans[i,2*j] = np.max(d[:,j])
            ans[i,2*j+1] = np.min(d[:,j])

    return ans

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
        verbose=False,
    ):
    """manages I/O and extracts quantiles at the specified places
    """
    y_test = [] ### keep this as a list because we don't know how many stable branches there are
    w_test = []
    num_points = len(x_test)

    truth = np.empty(num_points, dtype=bool) ### used to extract values

    columns = [xcolumn, ycolumn]
    if weights is None:
        weights = np.ones(len(data), dtype=float) / len(data)

    for eos, weight in zip(data, weights): ### iterate over samples and compute weighted moments
        for eos_path in glob.glob(tmp%{'moddraw':eos//mod, 'draw':eos}):
            if verbose:
                print('    '+eos_path)
            d, _ = load(eos_path, columns)

            d[:,0] *= x_multiplier
            d[:,1] *= y_multiplier

            _y = np.empty(num_points, dtype=float)
            _y[:] = np.nan ### signal that nothing was available at this x-value

            truth[:] = (np.min(d[:,0])<=x_test)*(x_test<=np.max(d[:,0])) ### figure out which x-test values are contained in the data
            _y[truth] = np.interp(x_test[truth], d[:,0], d[:,1]) ### fill those in with interpolated values

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

#-------------------------------------------------
# KDE and cross-validation likelihood
#-------------------------------------------------

def data2range(data, pad=0.1):
    m = np.min(data)
    M = np.max(data)
    delta = (M-m)*pad
    return (m-delta, M+delta)

def vects2flatgrid(*vects):
    return np.transpose([_.flatten() for _ in np.meshgrid(*vects, indexing='ij')])

def prune(data, bounds, weights=None):
    """
    downselect data to only contain samples that lie within bounds
    """
    Nsamp, Ndim = data.shape
    truth = np.ones(Nsamp, dtype=bool)
    for i, bound in enumerate(bounds):
        if bound is None:
            continue
        else:
            m, M = bound
            truth *= (m<=data[:,i])*(data[:,i]<=M)

    if weights is not None:
        truth *= weights > 0 ### only return weights that actually matter
        return data[truth], weights[truth]
    else:
        return data[truth]

def reflect(data, bounds, weights=None):
    """
    expect
        data.shape = (Nsamp, Ndim)
        bounds.shape = (Ndim, 2)
    returns a large array with data reflected across bounds for each dimension as one would want for reflecting boundary conditions in a KDE
    """
    Ndim = len(bounds)

    d = data[...]
    for i in xrange(Ndim): # by iterating through dimensions, we effectivly reflect previously reflected samples in other directions as needed
        if bounds[i] is None:
            continue
        
        # figure out how big the new array will be and create it
        Nsamp = len(d)
        twoNsamp = 2*Nsamp
        new = np.empty((3*Nsamp, Ndim), dtype=float)

        # fill everything in as just a copy of what we already had
        new[:Nsamp,...] = d
        new[Nsamp:twoNsamp,...] = d
        new[twoNsamp:,...] = d

        # now reflect 2 of the copies around the bounds for this dimension only
        m, M = bounds[i]
        new[Nsamp:twoNsamp,i] = 2*m - d[:,i]
        new[twoNsamp:,i] = 2*M - d[:,i]

        ### update reference to be the new array, then proceed to the next epoch
        d = new
        if weights is not None:
            weights = np.concatenate((weights, weights, weights))

    if weights is not None:
        return d, weights

    else:
        return d

def logkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around actually computing the KDE estimate at a collection of samples

    estimates kde as sum_i[weight_i * K(sample, data_i)]

    returns log(kde(samples))
    """
    shape = samples.shape
    if len(shape) not in [1, 2]:
        raise ValueError('bad shape for samples')

    if len(shape)==1:
        Nsamp = shape[0]
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))
    else:
        Nsamp, Ndim = samples.shape

    if np.any(weights==None): ### needed because modern numpy performs element-wise comparison here
        Ndata = len(data)
        weights = np.ones(Ndata, dtype='float')/Ndata

    logkdes = np.empty(Nsamp, dtype=float)
    if num_proc == 1: ### do everything on this one core
        logkdes[:] = _logkde_worker(samples, data, variances, weights)

    else: ### parallelize
        # partition work amongst the requested number of cores
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logkde_worker, args=(samples[truth], data, variances, weights), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logkdes[truth] = conni.recv()

    return logkdes

def _define_sets(Nsamp, num_proc):
    sets = [np.zeros(Nsamp, dtype=bool) for i in xrange(num_proc)]
    for i in xrange(Nsamp):
        sets[i%num_proc][i] = True
    return [_ for _ in sets if np.any(_)]

def _logkde_worker(samples, data, variances, weights, conn=None):
    Nsamp, Ndim = samples.shape
    Ndata = len(data)

    logkdes = np.empty(Nsamp, dtype='float')
    twov = -0.5/variances

    z = np.empty(Ndata, dtype=float)
    for i in xrange(Nsamp):
        sample = samples[i]
        z[:] = np.sum((data-sample)**2 * twov, axis=1)       ### shape: (Ndata, Ndim) -> (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        logkdes[i] = np.log(np.sum(weights*np.exp(z-m))) + m    

    logkdes += -0.5*Ndim*np.log(2*np.pi) - 0.5*np.sum(np.log(variances))

    if conn is not None:
        conn.send(logkdes)

    return logkdes

def grad_logkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    Nsamp, Ndim = samples.shape
    returns the gradient of the logLikelihood based on (data, variances, weights) at each sample (shape=Nsamp, Ndim)
    """
    shape = samples.shape
    if len(shape) not in [1, 2]:
        raise ValueError('bad shape for samples')

    if len(shape)==1:
        Nsamp = shape[0]
        Ndim = 1
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))

    else:
        Nsamp, Ndim = samples.shape

    grad_logkdes = np.empty(Nsamp, dtype=float)

    Ndata = len(data)
    if np.any(weights==None): ### needed because modern numpy performs element-wise comparison here
        weights = np.ones(Ndata, dtype='float')/Ndata

    if num_proc==1:
        grad_logkdes[:] = _grad_logkde_worker(samples, data, variances, weights)

    else:
        # divide the work
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_grad_logkde_worker, args=(samples[truth], data, variances, weights), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            grad_logkdes[truth] = conni.recv()

    return grad_logkdes

def _grad_logkde_worker(samples, data, variances, weights, conn=None):
    Nsamp, Ndim = samples.shape
    Ndata = len(data)

    grad_logkdes = np.empty(Nsamp, dtype='float')

    v = variances[0]
    assert np.all(variances==v), 'we only support a single variance at this time, even though it must be repeated Ndim times within "variances"'
    twov = -0.5/v

    z = np.empty(Ndata, dtype=float)
    for i in xrange(Nsamp):
        sample = samples[i]
        z[:] = np.sum((data-sample)**2 * twov, axis=1)  ### shape: (Ndata, Ndim) -> (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        y = np.sum(weights*np.exp(z-m))
        x = np.sum(weights*np.exp(z-m)*(-z/v))

        if y==0:
           if np.all(x==0):
                grad_logkdes[i] = 0 ### this is the appropriate limit here
           else:
                raise Warning('something bad happened with your estimate of the gradient in logleave1outLikelihood')
        else:
            grad_logkdes[i] = Ndim*twov + x/y

    if conn is not None:
        conn.send(grad_logkdes)
    return grad_logkdes

def logvarkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around computing bootstrapped estimates of the variance of the kde
    delegates to logcovkde
    """
    return logcovkde((samples, samples), data, variances, weights=weights, num_proc=num_proc)

def logcovkde(samples1, samples2, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around computing bootstrapped estimates of the covariance of the kde (btwn points defined in samples1, samples2)

    estimates covariance as sum_i[ weight_i**2 * K(sample1, data_i) * K(sample2, data_i) ] - (1/Ndata)*sum_i[weight_i * K(sample1, data_i)]*sum_j[weight_j * K(sample2, data_j)]

    return logcovkde(samples1, samples2), logkdes(samples1), logkdes(samples2)
    """
    assert samples1.shape==samples2.shape, 'samples1 and samples2 must have the same shape!'

    Nsamp = len(samples1)
    if samples1.ndim not in [1, 2]:
        raise ValueError('bad shape for samples1')
    elif samples1.ndim==1:
        Ndim = 1
        samples1 = samples1.reshape(Nsamp,1)
        samples2 = samples2.reshape(Nsamp,1)
        data = data.reshape((len(data),1))
    else:
        Nsamp, Ndim = samples1.shape

    Ndata = len(data)
    if np.any(weights==None):
        weights = np.ones(Ndata, dtype=float)/Ndata

    # compute first moments
    samples = np.array(set(list(samples1)+list(samples2)))
    logfirst = dict(zip(samples, logkde(samples, data, variances, weights=weights, num_proc=num_proc)))

    # compute second moments
    logseconds = np.empty((Nsamp, 2), dtype=float)
    if num_proc==1: # do everything on this core
        logseconds[:,:] = _logsecond_worker(samples1, samples2, data, variances, weights)

    else: # parallelize
        # divide the work
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logsecond_worker, args=(samples1[truth], samples2[truth], data, variances, weights, logfirst), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logsecond[truth,:] = conni.recv()

    ### manipulate the moments to get the variance
    logcovkdes = logsecond + np.log(1 - np.exp(logsecond[:,1]-logsecond[:,0]))
    return logcovkdes, np.array([logfirst[sample] for sample in samples1]), np.array([logfirst[sample] for sample in samples2])

def _logsecond_worker(samples1, samples2, data, variances, weights, logfirst, conn=None):
    Nsamp, Ndim = samples1.shape
    Ndata = len(data)

    logsecond = np.empty((Nsamp, 2), dtype=float)
    twov = -0.5/variances

    z = np.empty(Ndata, dtype=float)
    w2 = weights**2
    for i in xrange(Nsamp):
        sample1 = samples1[i]
        sample2 = samples2[i]

        ### compute first moments
        z[:] = np.sum((data-sample1)**2 * twov, axis=1) + np.sum((data-sample2)**2 * twov, axis=1) ### shape: (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        logsecond[i,0] = np.log(np.sum(w2*np.exp(z-m))) + m ### it is important that the weights are squared because we treat the samples as drawn from some prior and
                                                            ### therefore the weights are random variables that need to be included within the average
                                                            ### just like we have multiple factors of the kernel

        logsecond[i,1] = logfirst[sample1]+logfirst[sample2] ### the first-moment terms

    ### subtract off common factors
    logsecond[:,0] += -Ndim*np.log(2*np.pi) - np.sum(np.log(variances))
    logsecond[:,1] -= np.log(Ndata)

    if conn is not None:
        conn.send(logsecond)
    return logsecond

def logleave1outLikelihood(data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    computes the logLikelihood for how well this bandwidth produces a KDE that reflects p(data|B) using samples drawn from p(B|data)=p(data|B)*p(B)

    assumes data's shape is (Nsamples, Ndim)
    assumes variances's shape is (Ndim,) -> a diagonal covariance matrix

    assumes weights' shape is (Nsamples,) and must be normalized so sum(weights)=1

    returns mean(logL), var(logL), mean(grad_logL), covar(dlogL/dvp)
    """
    return logleavekoutLikelihood(data, variances, k=1, weights=weights, num_proc=num_proc)

def logleavekoutLikelihood(data, variances, k=1, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    implements a leave-k-out cross validation estimate of the logLikelihood
    returns mean(logL), var(logL), mean(dlogL/dv), var(dlogL/dv)
    """
    Nsamples, Ndim = data.shape

    if weights==None:
        weights = np.ones(Nsamples, dtype='float')/Nsamples

    sets = [[] for _ in range(max(int(np.ceil(Nsamples/k)),2))]
    Nsets = len(sets)
    for i in xrange(Nsamples):
        sets[i%Nsets].append(i)

    logL = np.empty(Nsets, dtype='float')
    grad_logL = np.empty(Nsets, dtype='float')

    twov = -0.5/variances
    truth = np.ones(Nsamples, dtype=bool)

    for i in xrange(Nsets):
        sample = data[sets[i]]

        truth[:] = True
        truth[sets[i]] = False

        ### compute logLikelihood
        ### take the joint kde (sum of logs) and weight it by the joint weights of all points within that set (product of weights)
        logL[i] = np.sum(logkde(sample, data[truth], variances, weights=weights[truth], num_proc=num_proc)) - np.log(np.sum(weights[truth]))

        ### compute gradient of logLikelihood
        grad_logL[i] = np.sum(grad_logkde(sample, data[truth], variances, weights=weights[truth], num_proc=num_proc))

    ### compute statistics
    set_weights = np.array([np.sum(np.log(weights[sets[i]])) for i in xrange(Nsets)], dtype=float) ### compute the cumulative weights for each set
    set_weights = np.exp(set_weights-np.max(set_weights))
    set_weights /= np.sum(set_weights)

    mlogL = np.sum(set_weights*logL) # scalar
    vlogL = (np.sum(set_weights*logL**2) - mlogL**2) / Nsets # scalar

    mglogL = np.sum(set_weights*grad_logL)  # scalar
    vglogL = (np.sum(set_weights*grad_logL**2) - mglogL**2) / Nsets ### scaler

    return mlogL, vlogL, mglogL, vglogL

#-------------------------------------------------

def set_crust(crust_eos=eos.DEFAULT_CRUST_EOS):
    global CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY
    CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY = \
        load(eos.eospaths.get(crust_eos, crust_eos), columns=['pressurec2', 'energy_densityc2', 'baryon_density'])[0].transpose()

def crust_energy_densityc2(pressurec2):
    """
    return energy_densityc2 for the crust from Douchin+Haensel, arXiv:0111092
    this is included in our repo within "sly.csv", taken from Ozel's review.
    """
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2)

def crust_baryon_density(pressurec2):
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_BARYON_DENSITY)

def stitch_below_reference_pressure(energy_densityc2, pressurec2, reference_pressurec2):
    """reutrn energy_densityc2, pressurec2"""
    sly_truth = CRUST_PRESSUREC2 <= reference_pressurec2
    eos_truth = reference_pressurec2 <= pressurec2
    return np.concatenate((CRUST_ENERGY_DENSITYC2[sly_truth], energy_densityc2[eos_truth])), np.concatenate((CRUST_PRESSUREC2[sly_truth], pressurec2[eos_truth]))

### integration routines
def dedp2e(denergy_densitydpressure, pressurec2, reference_pressurec2):
    """
    integrate to obtain the energy density
    if stitch=True, map the lower pressures onto a known curst below the reference pressure instead of just matching at the reference pressure
    """
    energy_densityc2 = np.empty_like(pressurec2, dtype='float')
    energy_densityc2[0] = 0 # we start at 0, so handle this as a special case

    # integrate in the bulk via trapazoidal approximation
    energy_densityc2[1:] = np.cumsum(0.5*(denergy_densitydpressure[1:]+denergy_densitydpressure[:-1])*(pressurec2[1:] - pressurec2[:-1]))

    ### match at reference pressure
    energy_densityc2 += crust_energy_densityc2(reference_pressurec2) - np.interp(reference_pressurec2, pressurec2, energy_densityc2)

    return energy_densityc2

def e_p2rho(energy_densityc2, pressurec2, reference_pressurec2):
    """
    integrate the first law of thermodynamics
        dvarepsilon = rho/(varepsilon+p) drho
    """
    baryon_density = np.ones_like(pressurec2, dtype='float')

    integrand = 1./(energy_densityc2+pressurec2)
    baryon_density[1:] *= np.exp(np.cumsum(0.5*(integrand[1:]+integrand[:-1])*(energy_densityc2[1:]-energy_densityc2[:-1]))) ### multiply by this factor

    ### FIXME: match baryon density to energy density at reference pressure
    #baryon_density *= ec2 / np.interp(ref_pc2, pressurec2, baryon_density)

    ### match at the lowest allowed energy density
    baryon_density *= crust_baryon_density(reference_pressurec2)/np.interp(reference_pressurec2, pressurec2, baryon_density)

    return baryon_density

#-------------------------------------------------

### load the sly EOS for stitching logic
set_crust() ### use this as the crust!
