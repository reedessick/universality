__doc__ = "a module that houses utilities that compute statistics based on KDEs"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def kde2levels(kde, levels):
    kde = kde.flatten()

    order = kde.argsort()[::-1] ### largest to smallest
    ckde = np.cumsum(kde[order]) ### cumulative distribution
    ckde /= np.sum(kde)

    ans = []
    for level in levels: ### iterate through levels, returning the kde value associated with that confidence
                         ### assume kde spacing is close enough that interpolation isn't worth while...
        ans.append(kde[order[ckde<=level][-1]])

    return ans

def kde2cr(vects, logkde, levels):
    """
    compute the volumes of confidence regions associated with levels
    """
    kde = np.exp(logkde-np.max(logkde))
    size = [np.sum(kde>=thr) for thr in kde2levels(kde, levels)]
    raise NotImplementedError('figure out how much volume is associated with each grid point, multiply that by size and call it a day')

def kde2entropy(vects, kde):
    """
    computes the entropy of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    raise NotImplementedError

def kde2information(vects, kde):
    """
    computes the information of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    raise NotImplementedError

def kldiv(vects, kde1, kde2):
    """
    computes the KL divergence from kde1 to kde2
        Dkl(k1||k2) = sum(k1*log(k1/k2))
    """
    raise NotImplementedError

def dlogkde(point, vects, kde):
    """
    returns the difference in loglikelihood between a point and the argmax of kde
    """
    raise NotImplementedError

def kde2argmax(vects, kde):
    """
    returns the argmax of the kde
    """
    raise NotImplementedError
