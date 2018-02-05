__doc__ = "a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples of a univeral relation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

c = (299792458*100) # speed of light in (cm/s)
c2 = c**2

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
# basic utilities for manipulating existing sapmles
#-------------------------------------------------

def load(inpath, columns=[], logcolumns=[]):
    data = np.genfromtxt(inpath, names=True, delimiter=',') ### assumes standard CSV format

    # check that all requested columns are actually in the data
    if columns:
        check_columns(data.dtype.fields.keys(), columns)
    else:
        columns = data.dtype.fields.keys()

    # downselect data to what we actually want
    return \
        np.transpose([np.log(data[column]) if column in logcolumns else data[column] for column in columns]), \
        ['log(%s)'%column if column in logcolumns else column for column in columns]

def check_columns(present, required):
    for column in required:
        assert column in present, 'required column=%s is missing!'%column

def whiten(data, verbose=False):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    if verbose:
        print('whitening marginal distributions')
        for i, (m, s) in enumerate(zip(means, stds)):
            print('  mean(%01d) = %+.3e'%(i, m))
            print('  stdv(%01d) = %+.3e'%(i, s))
    data -= means
    data /= stds

    return data, means, stds

def downsample(data, n):
    N = len(data)
    assert n>0 and n<=N, 'cannot downsample size=%d to size=%d'%(N, n)

    truth = np.zeros(N, dtype=bool)
    while np.sum(truth) < n:
        truth[np.random.randint(0, N-1)] = True ### FIXME: this could be pretty wasteful...

    return data[truth], truth

def logaddexp(logx):
    '''
    assumes we have more than one axis and sums over axis=-1
    '''
    N = logx.shape[-1]
    max_logx = np.max(logx, axis=-1)

    return max_logx + np.log( np.sum(np.exp(logx - np.outer(max_logx, np.ones(N)).reshape(logx.shape)) , axis=-1) )

#-------------------------------------------------
# cross-validation likelihood
#-------------------------------------------------

def logkde(samples, data, variances, weights=None):
    """
    a wrapper around actually computing the KDE estimate at a collection of samples
    """
    shape = samples.shape
    if len(shape) in [1, 2]:

        if len(shape)==1:
            Nsamp = shape[0]
            Ndim = 1
            samples = samples.reshape((Nsamp,1))
            data = data.reshape((len(data),1))
           
        else:
            Nsamp, Ndim = samples.shape

        if np.any(weights==None): ### needed because modern numpy performs element-wise comparison here
            Ndata = len(data)
            weights = np.ones(Ndata, dtype='float')/Ndata

        logkdes = np.empty(Nsamp, dtype='float')
        twov = -0.5/variances
        for i in xrange(Nsamp):
            sample = samples[i]

            zi = (data-sample)**2 * twov ### shape: (Ndata, Ndim)
            z = np.sum(zi, axis=1)       ### shape: (Ndata)

            ### do this backflip to preserve accuracy
            m = np.max(z)
            logkdes[i] = np.log(np.sum(weights*np.exp(z-m))) + m 

        ### subtract off common factors
        logkdes += -0.5*Ndim*np.log(2*np.pi) - 0.5*np.sum(np.log(variances))

    else:
        raise ValueError, 'bad shape for samples'

    return logkdes

def logleave1outLikelihood(data, variances, weights=None):
    """
    computes the logLikelihood for how well this bandwidth produces a KDE that reflects p(data|B) using samples drawn from p(B|data)=p(data|B)*p(B)

    assumes data's shape is (Nsamples, Ndim)
    assumes variances's shape is (Ndim,) -> a diagonal covariance matrix

    assumes weights' shape is (Nsamples,) and must be normalized so sum(weights)=1

    returns mean(logL), var(logL), mean(grad_logL), covar(dlogL/dvp)
    """
    Nsamples, Ndim = data.shape

    if weights==None:
        weights = np.ones(Nsamples, dtype='float')/Nsamples

    logL = np.empty(Nsamples, dtype='float')
    grad_logL = np.empty((Nsamples, Ndim), dtype='float')

    twov = -0.5/variances
    truth = np.ones(Nsamples, dtype=bool)

    for i in xrange(Nsamples):
        sample = data[i]

        truth[i-1] = True
        truth[i] = False

        ### compute logLikelihood
        logL[i] = logkde(np.array([data[i]]), data[truth], variances, weights=weights[truth])[0]

        ### compute gradient of logLikelihood
        ### NOTE: this repeats some work that's done within delegation to logkde, but that's probably fine
        zi = (data[truth]-sample)**2 * twov ### shape: (Nsamples, Ndim)
        z = np.sum(zi, axis=1)              ### shape: (Nsamples)

        m = np.max(z)
        z = weights[truth]*np.exp(z-m)
        y = np.sum(z)
        x = np.sum(z*(-zi/variances).transpose(), axis=1)

        if y==0:
            if np.all(x==0):
                grad_logL[i,:] = 0 ### this is the appropriate limit here
            else:
                raise Warning, 'something bad happened with your estimate of the gradient in logleave1outLikelihood'
        else:
            grad_logL[i,:] = twov + x/y

    ### add in constant terms to logL
    logL -= np.log(Nsamples-1) ### take the average as needed

    ### compute statistics
    mlogL = np.mean(weights*logL) # scalar
    vlogL = np.var(weights*logL)  # scalar

    mglogL = np.mean(weights*grad_logL.transpose(), axis=1)  # vector: (Ndim,)
    vglogL = np.cov(weights*grad_logL.transpose(), rowvar=1) # matrix: (Ndim, Ndim)

    return mlogL, vlogL, mglogL, vglogL
