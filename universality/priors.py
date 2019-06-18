__doc__ = "a module for generating prior weights based on known sampling priors within lalinference"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

KNOWN_PRIORS = [
    'uniform',
    'Dsqrd',
    'mdet_dsqrd',
    'msrc_Vcov',
]

#-------------------------------------------------

def logprior(prior_type, data, **kwargs):
    """compute the prior weight associated with each sample in data. This is a general routing function that should delegate to different types of priors, as specified in KNOWN_PRIORS"""
    assert prior_type in KNOWN_PRIORS, 'prior=%s not understood! must be one of %s'%(prior_type, ', '.join(KNOWN_PRIORS))

    if prior_type=='uniform':
        return uniform(data, **kwargs)

    elif prior_type=='Dsqrd':
        return dsqrd(data, **kwargs)

    elif prior_type=="mdet_dsqrd":
        return mdet_dsqrd(data, **kwargs)

    elif prior_type=="msrc_Vcov":
        return msrc_Vcov(data, **kwargs)

    else:
        raise ValueError('prior_type=%s not understood'%prior_type)

#-------------------------------------------------

def uniform(data, **kwargs):
    """return a flat prior regardless of the data (ie, just zeros of the correct length
    """
    return np.zeros(len(data), dtype=float)

def dsqrd(data, distance='dist', **kwargs):
    """assumes a uniform in "volume" prior
    p(D) dD ~ D**2 dD
    """
    return data[distance]**2

def mdet_dsqrd(data, distance='dist', m1='m1', m2='m2', **kwargs):
    """assumes a uniform in "volume" prior
    p(D) dD ~ D**2 dD
and a uniform in component detector-frame component mass prior
    p(m1, m2) ~ constant | m1 > m2
    """
    logp = dsqrd(data, distance=distance)
    logp[data[m1] < data[m2]] -= np.infty ### zero out weights corresponding to bad combos of weights
    return logp

def msrc_Vcov(data, distance='dist', m1='m1', m2='m2', **kwargs):
    """assumes a uniform prior in source-frame component masses and a uniform in co-moving volume prior
    """
    raise NotImplementedError
