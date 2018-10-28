__doc__ = "a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples of a univeral relation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

c = (299792458*100) # speed of light in (cm/s)
c2 = c**2

DEFAULT_BANDWIDTH = 0.1
DEFAULT_MAX_NUM_SAMPLES = np.infty

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

def load(inpath, columns=[], logcolumns=[], max_num_samples=DEFAULT_MAX_NUM_SAMPLES):
    data = np.genfromtxt(inpath, names=True, delimiter=',') ### assumes standard CSV format

    # check that all requested columns are actually in the data
    if columns:
        check_columns(data.dtype.fields.keys(), columns)
    else:
        columns = data.dtype.fields.keys()

    if len(data) > max_num_samples: ### downsample if requested
         data = data[:max_num_samples]

    # downselect data to what we actually want
    return \
        np.transpose([np.log(data[column]) if column in logcolumns else data[column] for column in columns]), \
        ['log(%s)'%column if column in logcolumns else column for column in columns]

def check_columns(present, required):
    for column in required:
        assert column in present, 'required column=%s is missing!'%column

def whiten(data, verbose=False, outlier_stdv=np.infty):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    data -= means
    data /= stds

    # adjust stds to reject outliers
    if outlier_stdv < np.infty:
        for i in xrange(data.shape[1]):
            truth = np.abs(data[:,i]) < outlier_stdv
            refactor = np.std(data[truth,i])

            stds[i] *= refactor
            data[:,i] /= refactor

    if verbose:
        print('whitening marginal distributions')
        if len(data.shape)==1:
            print('  mean = %+.3e'%(means))
            print('  stdv = %+.3e'%(stds))

        else:
            for i, (m, s) in enumerate(zip(means, stds)):
                print('  mean(%01d) = %+.3e'%(i, m))
                print('  stdv(%01d) = %+.3e'%(i, s))

    return data, means, stds

def iqr_whiten(data, verbose=False, low=0.16, high=0.84):
    medians = np.median(data, axis=0)
    iqrs = np.percentile(data, 100*high, axis=0) - np.percentile(data, 100*low, axis=0)

    data -= medians
    data /= iqrs

    if verbose:
        print('whitening marginal distributions')
        if len(data.shape)==1:
            print('  median = %+.3e'%(medians))
            print('  IQR[%.2f,%.2f] = %+.3e'%(low, high, iqrs))

        else:
            for i, (m, s) in enumerate(zip(medians, iqrs)):
                print('  median(%01d) = %+.3e'%(i, m))
                print('  IQR[%.2f,%.2f](%01d) = %+.3e'%(low, high, i, s))

    return data, medians, iqrs

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

def data2range(data, pad=0.1):
    m = np.min(data)
    M = np.max(data)
    delta = (M-m)*0.1
    return (m-delta, M+delta)

def vects2flatgrid(*vects):
    return np.transpose([_.flatten() for _ in np.meshgrid(*vects, indexing='ij')])

def quantile(x, quantiles, weights=None):
    if weights is None:
        return np.percentile(x, np.array(quantiles)*100)

    else:
        order = x.argsort()
        x = x[order]
        csum = np.cumsum(weights[order])
        return np.array([x[[csum<=q]][-1] for q in quantiles])

def neff(weights):
    """the effective number of samples based on a set of weights"""
    truth = weights > 0
    weights /= np.sum(weights)
    return np.exp(-np.sum(weights[truth]*np.log(weights[truth])))

def prune(data, bounds, weights=None):
    """
    downselect data to only contain samples that lie within bounds
    """
    Nsamp, Ndim = data.shape
    truth = np.ones(Nsamp, dtype=bool)
    for i, (m, M) in enumerate(bounds):
        truth *= (m<=data[:,i])*(data[:,i]<=M)

    if weights is not None:
        data = data[truth]
        weights = weights[truth]
        truth = weights > 0 ### only keep samples that actually matter
        return data[truth], weights[truth]
    else:
        return data[truth]

def reflect(data, bounds, weights=None):
    """
    expect
        data.shape = (Nsamp, Ndim)
        bounds.shape = (Ndim, 2)
    returns a large array with data reflected across bounds for each dimension as one would want for reflecting boundary conditions in a KDE
    """
    Ndim = len(bounds)

    d = data[...]
    for i in xrange(Ndim): # by iterating through dimensions, we effectivly reflect previously reflected samples in other directions as needed
        # figure out how big the new array will be and create it
        Nsamp = len(d)
        twoNsamp = 2*Nsamp
        new = np.empty((3*Nsamp, Ndim), dtype=float)

        # fill everything in as just a copy of what we already had
        new[:Nsamp,...] = d
        new[Nsamp:twoNsamp,...] = d
        new[twoNsamp:,...] = d

        # now reflect 2 of the copies around the bounds for this dimension only
        m, M = bounds[i]
        new[Nsamp:twoNsamp,i] = 2*m - d[:,i]
        new[twoNsamp:,i] = 2*M - d[:,i]

        ### update reference to be the new array, then proceed to the next epoch
        d = new
        if weights is not None:
            weights = np.concatenate((weights, weights, weights))

    if weights is not None:
        return d, weights

    else:
        return d

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

def grad_logkde(samples, data, variances, weights=None):
    """
    Nsamp, Ndim = samples.shape
    returns the gradient of the logLikelihood based on (data, variances, weights) at each sample (shape=Nsamp, Ndim)
    """
    shape = samples.shape
    grad_logkdes = np.empty(shape, dtype=float)
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

        grad_logkdes = np.empty(Nsamp, dtype='float')
        twov = -0.5/variances
        for i in xrange(Nsamp):
            sample = samples[i]

            zi = (data-sample)**2 * twov ### shape: (Ndata, Ndim)
            z = np.sum(zi, axis=1)       ### shape: (Ndata)

            ### do this backflip to preserve accuracy
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

    else:
        raise ValueError, 'bad shape for samples'

    return grad_logkdes

def logvarkde(samples, data, variances, weights=None):
    """
    a wrapper around computing bootstrapped estimates of the variance of the kde
    """
    raise NotImplementedError

def logcovkde((samples1, samples2), data, variances, weights=None):
    """
    a wrapper around computing bootstrapped estimates of the covariance of the kde (btwn points defined in samples1, samples2)
    """
    raise NotImplementedError

def logleave1outLikelihood(data, variances, weights=None):
    """
    computes the logLikelihood for how well this bandwidth produces a KDE that reflects p(data|B) using samples drawn from p(B|data)=p(data|B)*p(B)

    assumes data's shape is (Nsamples, Ndim)
    assumes variances's shape is (Ndim,) -> a diagonal covariance matrix

    assumes weights' shape is (Nsamples,) and must be normalized so sum(weights)=1

    returns mean(logL), var(logL), mean(grad_logL), covar(dlogL/dvp)
    """
    return logleavekoutLikelihood(data, variances, k=1, weights=weights)

def logleavekoutLikelihood(data, variances, k=1, weights=None):
    """
    implements a leave-k-out cross validation estimate of the logLikelihood
    """
    Nsamples, Ndim = data.shape

    if weights==None:
        weights = np.ones(Nsamples, dtype='float')/Nsamples

    sets = [[] for _ in xrange(np.ceil(Nsamples/k))]
    Nsets = len(segs)
    for i in xrange(Nsamples):
        sets[i%Nsets].append(i)

    logL = np.empty(Nsets, dtype='float')
    grad_logL = np.empty((Nsets, Ndim), dtype='float')

    twov = -0.5/variances
    truth = np.ones(Nsamples, dtype=bool)

    for i in xrange(Nsets):
        sample = data[sets[i]]

        truth[:] = True
        truth[sets[i]] = False

        ### compute logLikelihood
        logL[i] = np.sum(logkde(sample, data[truth], variances, weights=weights[truth]))

        ### compute gradient of logLikelihood
        grad_logL[i,:] = np.sum(grad_logkde(sample, data[truth], variances, weights=weights[truth]), axis=0)

    ### add in constant terms to logL
    logL -= np.log(Nsets-1) ### take the average as needed

    ### compute statistics
    mlogL = np.mean(weights*logL) # scalar
    vlogL = np.var(weights*logL)  # scalar

    mglogL = np.mean(weights*grad_logL.transpose(), axis=1)  # vector: (Ndim,)
    vglogL = np.cov(weights*grad_logL.transpose(), rowvar=1) # matrix: (Ndim, Ndim)

    return mlogL, vlogL, mglogL, vglogL
