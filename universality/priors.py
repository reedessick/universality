__doc__ = "a module for generating prior weights based on known sampling priors within lalinference"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

KNOWN_PRIORS = [
    'uniform',
]

#-------------------------------------------------

def logprior(prior_type, data):
    """compute the prior weight associated with each sample in data. This is a general routing function that should delegate to different types of priors, as specified in KNOWN_PRIORS"""
    assert prior_type in KNOWN_PRIORS, 'prior=%s not understood! must be one of %s'%(prior_type, ', '.join(KNOWN_PRIORS))

    if prior_type=='uniform':
        return uniform(data)

#-------------------------------------------------

def uniform(data):
    """return a flat prior regardless of the data (ie, just zeros of the correct length
    """
    return np.zeros(len(data), dtype=float)
