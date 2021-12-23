"""a module that houses utilities that compute basic statistical quantities about convergence of monte-carlo integrals
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def nkde(weights):
    """the number of samples that determine the scaling of the variance of our KDE estimates"""
    weights /= np.sum(weights)
    return 1./np.sum(weights**2)

def neff(weights):
    """the effective number of samples based on a set of weights"""
    return np.exp(entropy(weights, base=np.exp(1)))

def entropy(weights, base=2.):
    """compute the entropy of the distribution"""
    weights = np.array(weights)
    truth = weights > 0
    weights /= np.sum(weights)
    return -np.sum(weights[truth]*np.log(weights[truth])) / np.log(base)

def information(weights, base=2.):
    """compute the information in the distribution"""
    return np.log(len(weights))/np.log(base) - entropy(weights, base=base)
