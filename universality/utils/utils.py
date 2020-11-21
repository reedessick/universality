"""a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from collections import defaultdict
import numpy as np

import multiprocessing as mp

#-------------------------------------------------

DEFAULT_NUM_PROC = min(max(mp.cpu_count()-1, 1), 15) ### reasonable bounds for parallelization...

#------------------------

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

#-------------------------------------------------
# basic utilities for manipulating weights
#-------------------------------------------------

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

#-------------------------------------------------
# basic utilities for manipulating existing samples
#-------------------------------------------------

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

        ### this is the variance of the sum, not the mean
        ### V[sum] = N*V[w] = N*E[w**2] - N*(E{w})**2
        ###                 ~ lmw2 - N*(lmw/N)**2 = lmw2 * (1 - lmw**2/(lmw2*N))
        logvar = lmw2 + np.log(1. - np.exp(2*lmw - lmw2 - np.log(cnt)))

        results[i,num_columns:] = lmw, logvar, cnt

    return results, columns

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

#-------------------------------------------------
# utilities for maintaing high precision within sums
#-------------------------------------------------

def sum_log(logweights):
    """returns the log of the sum of the weights, retaining high precision
    """
    if np.any(logweights==np.infty):
        return np.infty
    m = np.max(logweights)
    if m==-np.infty:
        return -np.infty
    return np.log(np.sum(np.exp(logweights-m))) + m

def logaddexp(logx):
    '''
    assumes we have more than one axis and sums over axis=-1
    '''
    N = logx.shape[-1]
    max_logx = np.max(logx, axis=-1)

    return max_logx + np.log( np.sum(np.exp(logx - np.outer(max_logx, np.ones(N)).reshape(logx.shape)) , axis=-1) )

#-------------------------------------------------
# convenience functions for sanity checking
#-------------------------------------------------

def num_dfdx(x_obs, f_obs):
    '''
    estimate the derivative numerically
    '''
    df = f_obs[1:] - f_obs[:-1]
    dx = x_obs[1:] - x_obs[:-1]

    dfdx = np.empty_like(f_obs, dtype=float)

    dfdx[0] = df[0]/dx[0]   # handle boundary conditions as special cases
    dfdx[-1] = df[-1]/dx[-1]

    dfdx[1:-1] = 0.5*(df[:-1]/dx[:-1] + df[1:]/dx[1:]) ### average in the bulk
    ### NOTE: this is different than what numpy.gradient will yield...

    return dfdx

def num_intfdx(x_obs, f_obs):
    '''
    estimate the definite integral numerically
    '''
    F = np.empty_like(f_obs, dtype=float)

    F[0] = 0 ### initial value is a special case
    F[1:] = np.cumsum(0.5*(f_obs[1:] + f_obs[:-1]) * (x_obs[1:] - x_obs[:-1])) ### trapazoidal approximation

    return F
