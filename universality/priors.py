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
    'uniform_in_log',
    'pareto',
    'ordered_joint_pareto',
    'comoving_volume_DL',
    'comoving_volume_z',
    'lalinf_DL',
    'lalinf_mdet_DL',
    'lalinf_msrc_Vc',
]

#------------------------

DEFAULT_SAMPLE_SIZE = 1000

#------------------------

DEFAULT_DL_NAME = 'dist'
DEFAULT_M1_NAME = 'm1'
DEFAULT_M2_NAME = 'm2'

#-------------------------------------------------

def logprior(prior_type, data, **kwargs):
    """compute the prior weight associated with each sample in data. This is a general routing function that should delegate to different types of priors, as specified in KNOWN_PRIORS"""
    assert prior_type in KNOWN_PRIORS, 'prior=%s not understood! must be one of %s'%(prior_type, ', '.join(KNOWN_PRIORS))

    #--- general priors; useful when applying weights to a set of samples

    if prior_type=='uniform':
        return uniform(data, **kwargs)

    elif prior_type=='uniform_in_log':
        return uniform_in_log(data, **kwargs)

    elif prior_type=='pareto':
        return pareto(data, **kwargs)

    elif prior_type=='ordered_joint_pareto':
        return ordered_joint_pareto(data, **kwargs)

    elif prior_type=='comoving_volume_DL':
        return comoving_volume_DL(data, **kwargs)

    elif prior_type=='comoving_volume_z':
        return comoving_volume_z(data, **kwargs)

    #--- priors induced by lalinference's scalings; useful if inverted to un-weight samples

    elif prior_type=='lalinf_DL':
        return lalinf_DL(data, **kwargs)

    elif prior_type=="lalinf_mdet_DL":
        return lalinf_mdet_DL(data, **kwargs)

    elif prior_type=="lalinf_msrc_Vc":
        return lalinf_msrc_Vc(data, **kwargs)

    #---

    else:
        raise ValueError('prior_type=%s not understood'%prior_type)

#-------------------------------------------------

### generic priors, both evaluation and sampling

#--- UNIFORM

def uniform(data, name=None, minval=None, maxval=None, **kwargs):
    """the uniform distribution | minval <= val < maxval
    """
    assert (minval is not None) and (maxval is not None), \
        'must supply minval and maxval!' ### done to make command-line arguments easier in add-prior-weights
    if name is not None:
        data = data[name]
    ans = np.zeros(len(data), dtype=float)
    ans[np.logical_not((minval<=data[name])*(data[name]<maxval))] = -np.infty
    return ans

def sample_uniform(minval=None, maxval=None, size=DEFAULT_SAMPLE_SIZE, **kwargs):
    """sample from the uniform distribution
    """
    assert (minval is not None) and (maxval is not None), 'must supply both minval and maxval!' ### done to make command-line arguments easier in add-prior-weights
    return minval + np.random.random(size)*(maxval-minval)

#--- UNIFORM IN LOG

def uniform_in_log(data, name=None, minval=None, maxval=None, **kwargs):
    """the uniform-in-natural log distribution | minval <= val < maxval
    """
    if name is not None:
        data = data[name]
    return uniform(np.log(data), np.log(minval), np.log(maxval))

def sample_uniform_in_log(minval, maxval, size=DEFAULT_SAMPLE_SIZE):
    """sample from the uniform-in-natural log distribution
    """
    return np.exp(sample_uniform(np.log(minval), np.log(maxval), size=size))

#--- PARETO DISTRIBUTION

def pareto(data, name=None, exp=None, minval=None, maxval=None, **kwargs):
    """the pareto distribution: val ~ val**exp | minval <= val < maxval
    """
    assert (exp is not None) and (minval is not None) and (maxval is not None), \
        'must supply exp, minval, and maxval!' ### done to make command-line arguments easier in add-prior-weights
    if name is not None:
        data = data[name]
    ans = exp*np.log(data)
    ans[np.logical_not((minval<=val)*(val<maxval))] = -np.infty
    return ans

def sample_pareto(exp=None, minval=None, maxval=None, size=DEFAULT_SAMPLE_SIZE, **kwargs):
    """sample from the pareto distribution
    """
    assert (exp is not None) and (minval is not None) and (maxval is not None), \
        'must supply exp, minval, and maxval!' ### done to make command-line arguments easier in add-prior-weights
    rand = np.random.random(size)
    if exp == -1:
        return minval * np.exp(rand*np.log(maxval/minval))
    else:
        exp += 1
        return ((maxval**exp - minval**exp)*rand + minval**exp)**(1./exp)

#--- ORDERED JOINT PARETO DISTRIBUTION

def ordered_joint_pareto(data, name1=None, name2=None, exp=None, minval=None, maxval=None, **kwargs):
    """the pareto distribution for 2 values: val1, val2 ~ val1**exp * val2**exp | minval <= val2 <= val1 < maxval
    """
    assert (exp is not None) and (minval is not None) and (maxval is not None), \
        'must supply exp, minval, and maxval!' ### done to make command-line arguments easier in add-prior-weights

    if name1 is None:
        data1 = data[:,0]
    else:
        data1 = data[name1]

    if name2 is None:
        data2 = data[:,1]
    else:
        data2 = data[name2]

    ans = exp*(np.log(data1) + np.log(data2))
    ans[np.logical_not((minval<=data2)*(data2<=data1)*(data1<maxval))] = -np.infty
    return ans

def sample_ordered_join_pareto(exp=None, minval=None, maxval=None, size=DEFAULT_SAMPLE_SIZE, **kwargs):
    """sample from the ordered joint pareto distribution
    """
    assert (exp is not None) and (minval is not None) and (maxval is not None), \
        'must supply exp, minval, and maxval!' ### done to make command-line arguments easier in add-prior-weights

    if exp==-1: ### just do rejection sampling here 'cause the analytic expression is transcendental...
        data1 = []
        data2 = []
        N = len(data1)
        f = np.log(maxval/minval)
        while N<size:
            rand1, rand2 = np.random.rand(2*(size-N)) ### expect to throw some away

            d1 = minval * np.exp(rand1*f)
            d2 = minval * np.exp(rand2*f)

            truth = d2 <= d1
            data1 += list(d1[truth][:size-N]) ### discard extras
            data2 += list(d2[truth][:size-N])

    else:
        rand1, rand2 = np.random.random((2,size))
        exp += 1

        data2 = (maxval**exp - (1-rand2)**0.5 * (maxval**exp - minval**exp))**(1./exp) # based on marginal distrib for data2 < data1
        data1 = (rand1*(maxval**exp - minval**exp) + data2**exp)**(1./exp)             # based on conditioned distrib for data1 | data2

    return np.array(zip(data1, data2))

#--- UNIFORM IN COMOVING VOLUME (in terms of luminosity distance)

def comoving_volume_DL(data, DL=None, minDL=None, maxDL=None, cosmology=cosmo.DEFAULT_COSMOLOGY, **kwargs):
    """evaluate the prior value for luminosity distance given a cosmology and a uniform-in-comoving volume distribution
    """
    assert (minDL is not None) and (maxDL is not None), \
        'must supply minDL and maxDL!'

    if DL is not None:
        data = data[DL]

    ans = np.empty(len(data), type=float)
    ans[:] = -np.infty

    truth = (minDL<=data)*(data<maxDL)

    z = cosmology.DL2z(data[truth])
    ans[truth] = np.log(cosmology.dVcdz(z)) - np.log(cosmology.dDLdz(z))

    return ans

def sample_comoving_volume_DL(minDL=None, maxDL=None, cosmology=cosmo.DEFAULT_COSMOLOGY, size=DEFAULT_SAMPLE_SIZE, **kwargs):
    """sample from the distribution over luminosity distance given a uniform distribution in comoving volume
    """
    assert (minDL is not None) and (maxDL is not None), \
        'must supply minDL and maxDL!'
    cosmology._extend(max_DL=maxDL)
    minz = cosmology.DL2z(minDL)
    maxz = cosmology.DL2z(maxDL)
    return sample_comoving_volume_z(minz, maxz, cosmology=cosmology, size=size)

#--- UNIFORM IN COMOVING VOLUME (in terms of redshift)

def comoving_volume_z(data, z=None, minz=None, maxz=None, cosmology=cosmo.DEFAULT_COSMOLOGY, **kwargs):
    """evaluate the prior for redshift given a cosmology and uniform-in-comoving volume distribution
    """
    assert (minz is not None) and (maxz is not None) and (cosmology is not None), \
        'must supply minz, maxz, and cosmology!'
    if z is not None:
        data = data[z]
    ans = np.empty(len(data), dtype=float)
    ans[:] = -np.infty
    truth = (minz<=data)*(data<maxz)
    ans[truth] = np.log(cosmology.dVcdz(data[truth]))
    return ans

def sample_comoving_volume_z(minz=None, maxz=None, cosmology=cosmo.DEFAULT_COSMOLOGY, size=DEFAULT_SAMPLE_SIZE, **kwargs):
    """sample from the distribution over redshift given a uniform distribution in comoving volume
    """
    assert (minz is not None) and (maxz is not None) and (cosmology is not None), \
        'must supply minz, maxz, and cosmology!'
    cosmology._extend(max_z=maxz)
    minVc = cosmology.z2Vc(minz)
    maxVc = cosmology.z2Vc(maxz)
    return cosmology.Vc2z(sample_uniform(minVc, maxVc, size=size))

#-------------------------------------------------

### priors induced by LALInference's sampling

def lalinf_DL(data, DL=DEFAULT_DL_NAME, **kwargs):
    """returns the induced logprior over luminosity distance
    p(d) ~ d**2
    """
    return 2*np.log(data[DL])

def lalinf_mdet_DL(data, DL=DEFAULT_DL_NAME, m1=DEFAULT_M1_NAME, m2=DEFAULT_M2_NAME, **kwargs):
    """returns the induced logprior over detector-frame component masses and luminosity distance
    p(m1, m2, d) ~ d**2 * Theta(m1 > m2)
    """
    logp = d(data, DL=distance_name)
    logp[data[m1] < data[m2]] -= np.infty ### zero out weights corresponding to bad combos of weights
    return logp

def lalinf_msrc_Vc(data, DL=DEFAULT_DL_NAME, m1=DEFAULT_M1_NAME, m2=DEFAULT_M2_NAME, cosmology=cosmo.DEFAULT_COSMOLOGY, **kwargs):
    """returns induced logprior over source-frame component masses and comoving volume
    p(m1_source, m2_source, Vc) ~ d**2 * [ (1+z)**2 * (dd/dz) * (dVc/dz)**-1 ]
This is the prior LALInf assumes for detector-frame component masses and luminosity distance multiplied by the appropriate Jacobian to change coordinates to source-frame component masses and (enclosed) comoving volume
    """
    d = data[DL] * cosmo.cm_per_Mpc ### convert Mpc (LALInference's units) to cm (our units)
    z = cosmology.DL2z(d)
    ans = np.log(d) + 2*np.log(1+z) + np.log( (1+z)**2/d + cosmology.z2E(z)/cosmology.c_over_Ho )
    ans[data[m1] < data[m2]] = -np.infty
    return ans
