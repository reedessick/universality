"""a module that houses utilities that compute estimates of the information
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality import kde
from universality.utils import DEFAULT_NUM_PROC

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

def logkde2entropy(vects, logkde):
    """
    computes the entropy of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    vol = vects2vol(vects)
    truth = logkde > -np.infty
    return -vects2vol(vects)*np.sum(np.exp(logkde[truth])*logkde[truth])

def logkde2information(vects, logkde):
    """
    computes the information of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    vol = vects2vol(vects)
    return np.log(len(logkde.flatten())*vol) - logkde2entropy(vects, logkde)

def kldiv(vects, logkde1, logkde2):
    """
    computes the KL divergence from kde1 to kde2
        Dkl(k1||k2) = sum(k1*log(k1/k2))
    """
    truth = logkde1 > -np.infty
    return vects2vol(vects)*np.sum(np.exp(logkde1[truth]*(logkde1[truth] - logkde2[truth])))

def sym_kldiv(vects, logkde1, logkde2):
    """
    Dkl(k1||k2) + Dkl(k2||k1)
    """
    return kldiv(vects, logkde1, logkde2) + kldiv(vects, logkde2, logkde1)

#-------------------------------------------------
# monte carlo estimates of integrals over samples that involve KDEs
#-------------------------------------------------

def montecarloentropy(samples, weights=None, variances=None, num_proc=DEFAULT_NUM_PROC, verbose=False):
    """
    computes an estimate of the entropy in a distribution via a Monte Carlo sum over the KDE evaluated at the sample points
    H = - \int dx P(x) ln[P(x)]
      ~ - \sum_i ln[KDE(x_i)] | x_i ~ P(x)
    """
    if verbose:
        print('computing KDE at %d points using %d processes'%(len(samples), num_proc))

    raise NotImplementedError('''

MORE IMPORTANTLY, universality.kde depends on universality.stats, so I can't import KDE here...

  * kde.logkde(samples, data, variances, weights, num_proc=DEFAULT)
  * stats.logkde2entropy(vects, logkde) ### can handle multi-dimensional data
    -> requires KDE to be computed on a grid and then numerically integrates over that grid
    -> we instead want to do a monte carlo sum over the values of the KDE evaulated at the sample points
''')

