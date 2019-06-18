__doc__ = """a module for generating prior weights based on known sampling priors within lalinference. 
**NOTE**, we assume LALInference samples uniformly in *detector-frame* component masses (m1, m2) assuming m1>=m2, with a prior on the luminosity distance p(D) ~ D**2, and uniformly in the tidal deformabilities (Lambda1, Lambda2). All priors computed herein are the analytic prior weights in the requested parameters induced by this sampling procedure. Dividing by these prior weights in a monte-carlo sampling procedure will therefore induce uniform priors on the specified parameters (e.g., weighing each sample by the inverse of msrc_Vcov will produce samples drawn with a uniform prior in the source-frame masses and a uniform prior in the comoving volume).
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import cosmology as cosmo

#-------------------------------------------------

KNOWN_PRIORS = [
    'uniform',
    'Dsqrd',
    'mdet_dsqrd',
    'msrc_Vcov',
]

#------------------------

DEFAULT_DISTANCE_NAME = 'dist'
DEFAULT_M1_NAME = 'm1'
DEFAULT_M2_NAME = 'm2'

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
    """return a flat logprior regardless of the data (ie, just zeros of the correct length)
    p(data) ~ constant
    """
    return np.zeros(len(data), dtype=float)

def d(data, distance_name=DEFAULT_DISTANCE_NAME, **kwargs):
    """returns the induced logprior over luminosity distance
    p(d) ~ d**2
    """
    return 2*np.log(data[distance_name])

def mdet_d(data, distance_name=DEFAULT_DISTANCE_NAME, m1_name=DEFAULT_M1_NAME, m2_name=DEFAULT_M2_NAME, **kwargs):
    """returns the induced logprior over detector-frame component masses and luminosity distance
    p(m1, m2, d) ~ d**2 * Theta(m1 > m2)
    """
    logp = d(data, distance_name=distance_name)
    logp[data[m1_name] < data[m2_name]] -= np.infty ### zero out weights corresponding to bad combos of weights
    return logp

def msrc_Vcov(data, distance_name=DEFAULT_DISTANCE_NAME, m1_name=DEFAULT_M1_NAME, m2_name=DEFAULT_M2_NAME, cosmology=cosmo.DEFAULT_COSMOLOGY, **kwargs):
    """returns induced logprior over source-frame component masses and comoving volume
    p(m1_source, m2_source, Vc) ~ (1+z)**2 * d**2 * (dd/dz) * (dVc/dz)**-1
    """
    d = data[distance_name]
    z = cosmology.DL2z(d)
    return np.log(d) + 2*np.log(1+z) + np.log( (1+z)**2/d + cosmology.z2E(z)/cosmology.c_over_Ho )
