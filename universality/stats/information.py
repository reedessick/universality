"""a module that houses utilities that compute estimates of the information
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality.kde import kde
from universality.utils import DEFAULT_NUM_PROC
from universality.stats.kde import (logkde2entropy, logkde2information, kldiv, sym_kldiv)

#-------------------------------------------------
# basic statistical quantities for discrete distributions
#-------------------------------------------------

def entropy(weights, base=2.):
    """compute the entropy of the distribution"""
    weights = np.array(weights)
    truth = weights > 0
    weights /= np.sum(weights)
    return -np.sum(weights[truth]*np.log(weights[truth])) / np.log(base)

def information(weights, base=2.):
    """compute the information in the distribution"""
    return np.log(len(weights))/np.log(base) - entropy(weights, base=base)

#-------------------------------------------------
# numeric integrals over pre-computed KDEs on grids
#-------------------------------------------------

# imported from universality.stats.kde

#-------------------------------------------------
# monte carlo estimates of integrals over samples that involve KDEs
#-------------------------------------------------

def montecarloentropy(samples, weights=None, variances=None, num_proc=DEFAULT_NUM_PROC, verbose=False):
    """
    computes an estimate of the entropy in a distribution via a Monte Carlo sum over the KDE evaluated at the sample points
    H = - \int dx P(x) ln[P(x)]
      ~ - \sum_i ln[KDE(x_i)] | x_i ~ P(x)
    """
    if weights is None:
        weights = np.ones(len(samples), dtype=float)
    weights = weights/np.sum(weights) ### make sure weights are normalized (don't modify these in place)

    # compute kde based on samples at the sample locations
    if verbose:
        print('computing KDE at %d points using %d processes'%(len(samples), num_proc))
    logkde = kde.logkde(samples, samples, variances, weights=weights, num_proc=num_proc)

    # now compute the monte carlo sum
    truth = weights > 0
    h = - np.sum(logkde[truth]*weights[truth]) # handle limit_{p->0} of p*ln(p) by hand

    # return
    return h
