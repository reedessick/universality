"""a module that houses utilities that compute statistics based directly on samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def quantile(x, quantiles, weights=None):
    if weights is None:
        return np.percentile(x, np.array(quantiles)*100)

    else:
        order = x.argsort()
        x = x[order]
        csum = np.cumsum(weights[order])
        csum /= csum[-1]

        return np.interp(quantiles, csum, x)
#        return np.array([x[[csum<=q]][-1] for q in quantiles])

def samples2cdf(data, weights=None):
    """estimate a CDF (integrating from small values to large values in data) based on weighted samples
    returns data, cweights (data is sorted from smallest to largest values)
    """
    if weights is None:
        N = len(data)
        weights = np.ones(N, dtype=float)/N

    order = data.argsort()
    data = data[order]
    weights = weights[order]
    cweights = np.cumsum(weights)/np.sum(weights)

    return data, cweights

def samples2range(data, pad=0.1):
    m = np.min(data)
    M = np.max(data)
    delta = (M-m)*pad
    return (m-delta, M+delta)

def samples2median(data, weights=None):
    data, cweights = samples2cdf(data, weights=weights)
    return np.interp(0.5, cweights, data) ### find the median via interpolation

def samples2mean(data, weights=None):
    if weights is None:
        weights = np.ones_like(data, dtype=float)
    return np.sum(weights*data)/np.sum(weights)

def samples2crbounds(data, levels, weights=None):
    """
    expects 1D data and returns the smallest contiguous confidence region that contains a certain amount of the cumulative weight
    returns a contiguous confidence region for level
    does this by trying all possible regions (defined by data's sampling) that have at least as much cumulative weight as each level and selecting the smallest
    """
    N = len(data)
    stop = N-1
    if weights is None:
        weights = np.ones(N, dtype=float)/N

    data, cweights = samples2cdf(data, weights=weights)

    return cdf2crbounds(data, cweights, levels)

def cdf2crbounds(data, cweights, levels):
    N = len(data)
    stop = N-1
    bounds = []
    for level in levels:
        i = 0
        j = 0
        best = None
        best_size = np.infty
        while (i < stop):
            while (j < stop) and (cweights[j]-cweights[i] < level): ### make the region big enough to get the confidence we want
                j += 1

            if cweights[j] - cweights[i] < level: ### too small!
                break

            size = data[j]-data[i]
            if size < best_size:
                best_size = size
                best = (data[i], data[j])
            i += 1 # increment starting point and then repeat

        bounds.append( best )

    return bounds
