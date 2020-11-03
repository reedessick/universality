"""a module for custom (Gaussian) kernel density estimation (KDE)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import erf

import multiprocessing as mp

from universality import stats
from universality.utils import DEFAULT_NUM_PROC

#-------------------------------------------------

DEFAULT_BANDWIDTH = 0.1

KNOWN_CUMULATIVE_INTEGRAL_DIRECTIONS = [
    'increasing',
    'decreasing',
]
DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION = 'increasing'

#-------------------------------------------------
# 1D CDF estimation
#-------------------------------------------------

def logcdf(samples, data, prior_bounds, weights=None, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION, num_proc=DEFAULT_NUM_PROC):
    """estimates the log(cdf) at all points in samples based on data and integration in "direction".
    Does this directly by estimating the CDF from the weighted samples WITHOUT building a KDE"""

    ### this should be relatively quick (just an ordered summation), so we do it once
    data, cweights = stats.samples2cdf(data, weights=weights)
    if direction=='increasing':
        pass ### we already integrate up from the lower values to higher values
    elif direction=='decreasing':
        cweights = 1. - cweights ### reverse the order of the integral
    else:
        raise ValueError('direction=%s not understood!'%direction)

    logcdfs = np.empty(len(samples), dtype=float)
    if num_proc==1: ### do everything on this one core
        logcdfs[:] = _logcdf_worker(samples, data, cweights, prior_bounds)

    else: ### parallelize
        # partition work amongst the requested number of cores
        Nsamp = len(samples)
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logcdf_worker, args=(samples[truth], data, cweights, prior_bounds), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logcdfs[truth] = conni.recv()

    return logcdfs

def _logcdf_worker(samples, data, cweights, bounds, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION, conn=None):
    ### we have to account for prior volume differences between different models to properly do the model-selection integral
    ### this is handled by explicitly passing the bounds for the overall prior that may or may not be truncated by samples
    local_samples = samples[:] ### make a copy so I can modify it in-place
    local_samples[local_samples<bounds[0]] = bounds[0]
    local_samples[local_samples>bounds[1]] = bounds[1]

    ###                   approx to the cumulative integral within the prior bounds
    logcdfs = np.interp(local_samples, data, cweights) - np.interp(bounds[0], data, cweights)
    truth = logcdfs > 0
    logcdfs[truth] = np.log(logcdfs[truth])
    logcdfs[np.logical_not(truth)] = -np.infty

    ### add the prior volume correction
    ### NOTE: we assume flat priors implicitly! really, this should be an integral over the (non-trivial) prior distribution
    if direction=='increasing':
        truth = bounds[0] < local_samples
        logcdfs[truth] -= np.log(local_samples[truth] - bounds[0])
        logcdfs[np.logical_not(truth)] = -np.infty ### these sample shave zero support in the prior, so we assign them zero weight

    elif direction=='decreasing':
        truth = bounds[1] > local_samples
        logcdfs[truth] -= np.log(bounds[1] - local_samples[truth])
        logcdfs[np.logical_not(truth)] = -np.infty ### this is the same thing as above

    else:
        raise ValueError('direction=%s not understood!'%direction)

    if conn is not None:
        conn.send(logcdfs)
    return logcdfs

def logcumkde(samples, data, variance, bounds=None, weights=None, direction=DEFAULT_CUMULATIVE_INTEGRAL_DIRECTION):
    """estimates the log(cdf) at all points in samples based on data and integration in "direction"
    This is done with a 1D KDE-based CDF estimate between bounds
    computes I = \sum_i w_i \int_0^samples dx K(x, data_i; variance) / \sum_i w_i
    This corresponds to integrating up to the value passed as samples for a Gaussian kernel centered at data_i
    if direction == 'increasing', we just return this. If direction == 'decreasing', we return 1 - I
    """
    ### sanity-check the input argumants
    assert len(np.shape(samples))==1, 'samples must be a 1D array'
    assert len(np.shape(data))==1, 'data must be a 1D array'
    assert isinstance(variance, (int, float)), 'variance must be an int or a float'

    ### set up variables for computation
    ans = np.empty(len(samples), dtype=float)
    frac = variance**-0.5

    if weights is None:
        N = len(data)
        weights = np.ones(N, dtype=float)/N

    ### set up bounds
    if bounds is None:
        lower = 0
        norm = 1
    else:
        m, M = bounds ### assumes all samples are between these bounds, but data need not be...
        lower = _cumulative_gaussian_distribution((m - data)*frac)
        norm = _cumulative_gaussian_distribution((M - data)*frac) - lower

    ### iterate and compute the cumuative integrals
    for i, sample in enumerate(samples):
        ### NOTE: it is important that we pass "sample - data" so that data is the mean
        ans[i] = np.sum(weights * (_cumulative_gaussian_distribution((sample - data)*frac) - lower))
    ans /= np.sum(weights * norm)

    ### return based on the requested direction
    if direction == 'increasing':
        return np.log(ans)
    elif direction == 'decreasing':
        return np.log(1 - ans)
    else:
        raise RuntimeError('direction=%s not understood!'%direction)

def _cumulative_gaussian_distribution(z):
    """standard cumulative Gaussian distribution"""
    return 0.5*(1 + erf(z/2**0.5))

#-------------------------------------------------
# KDE and cross-validation likelihood
#-------------------------------------------------

def vects2flatgrid(*vects):
    return np.transpose([_.flatten() for _ in np.meshgrid(*vects, indexing='ij')])

def logkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around actually computing the KDE estimate at a collection of samples

    estimates kde as sum_i[weight_i * K(sample, data_i)]

    returns log(kde(samples))
    """
    shape = samples.shape
    if len(shape) not in [1, 2]:
        raise ValueError('bad shape for samples')

    if len(shape)==1:
        Nsamp = shape[0]
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))
    else:
        Nsamp, Ndim = samples.shape

    if weights is None:
        Ndata = len(data)
        weights = np.ones(Ndata, dtype='float')/Ndata

    logkdes = np.empty(Nsamp, dtype=float)
    if num_proc == 1: ### do everything on this one core
        logkdes[:] = _logkde_worker(samples, data, variances, weights)

    else: ### parallelize
        # partition work amongst the requested number of cores
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logkde_worker, args=(samples[truth], data, variances, weights), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logkdes[truth] = conni.recv()

    return logkdes

def _define_sets(Nsamp, num_proc):
    sets = [np.zeros(Nsamp, dtype=bool) for i in xrange(num_proc)]
    for i in xrange(Nsamp):
        sets[i%num_proc][i] = True
    return [_ for _ in sets if np.any(_)]

def _logkde_worker(samples, data, variances, weights, conn=None):
    Nsamp, Ndim = samples.shape
    Ndata = len(data)

    logkdes = np.empty(Nsamp, dtype='float')
    twov = -0.5/variances

    z = np.empty(Ndata, dtype=float)
    for i in xrange(Nsamp):
        sample = samples[i]
        z[:] = np.sum((data-sample)**2 * twov, axis=1)       ### shape: (Ndata, Ndim) -> (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        logkdes[i] = np.log(np.sum(weights*np.exp(z-m))) + m    

    logkdes += -0.5*Ndim*np.log(2*np.pi) - 0.5*np.sum(np.log(variances))

    if conn is not None:
        conn.send(logkdes)

    return logkdes

def grad_logkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    Nsamp, Ndim = samples.shape
    returns the gradient of the logLikelihood based on (data, variances, weights) at each sample (shape=Nsamp, Ndim)
    """
    shape = samples.shape
    if len(shape) not in [1, 2]:
        raise ValueError('bad shape for samples')

    if len(shape)==1:
        Nsamp = shape[0]
        Ndim = 1
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))

    else:
        Nsamp, Ndim = samples.shape

    grad_logkdes = np.empty(Nsamp, dtype=float)

    Ndata = len(data)
    if weights is None: ### needed because modern numpy performs element-wise comparison here
        weights = np.ones(Ndata, dtype='float')/Ndata

    if num_proc==1:
        grad_logkdes[:] = _grad_logkde_worker(samples, data, variances, weights)

    else:
        # divide the work
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_grad_logkde_worker, args=(samples[truth], data, variances, weights), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            grad_logkdes[truth] = conni.recv()

    return grad_logkdes

def _grad_logkde_worker(samples, data, variances, weights, conn=None):
    Nsamp, Ndim = samples.shape
    Ndata = len(data)

    grad_logkdes = np.empty(Nsamp, dtype='float')

    v = variances[0]
    assert np.all(variances==v), 'we only support a single variance at this time, even though it must be repeated Ndim times within "variances"'
    twov = -0.5/v

    z = np.empty(Ndata, dtype=float)
    for i in xrange(Nsamp):
        sample = samples[i]
        z[:] = np.sum((data-sample)**2 * twov, axis=1)  ### shape: (Ndata, Ndim) -> (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        y = np.sum(weights*np.exp(z-m))
        x = np.sum(weights*np.exp(z-m)*(-z/v))

        if y==0:
           if np.all(x==0):
                grad_logkdes[i] = 0 ### this is the appropriate limit here
           else:
                raise Warning('something bad happened with your estimate of the gradient in logleave1outLikelihood')
        else:
            grad_logkdes[i] = Ndim*twov + x/y

    if conn is not None:
        conn.send(grad_logkdes)
    return grad_logkdes

def logvarkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around computing bootstrapped estimates of the variance of the kde
    delegates to logcovkde
    """
    return logcovkde((samples, samples), data, variances, weights=weights, num_proc=num_proc)

def logcovkde(samples1, samples2, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around computing bootstrapped estimates of the covariance of the kde (btwn points defined in samples1, samples2)

    estimates covariance as sum_i[ weight_i**2 * K(sample1, data_i) * K(sample2, data_i) ] - (1/Ndata)*sum_i[weight_i * K(sample1, data_i)]*sum_j[weight_j * K(sample2, data_j)]

    return logcovkde(samples1, samples2), logkdes(samples1), logkdes(samples2)
    """
    assert samples1.shape==samples2.shape, 'samples1 and samples2 must have the same shape!'

    Nsamp = len(samples1)
    if samples1.ndim not in [1, 2]:
        raise ValueError('bad shape for samples1')
    elif samples1.ndim==1:
        Ndim = 1
        samples1 = samples1.reshape(Nsamp,1)
        samples2 = samples2.reshape(Nsamp,1)
        data = data.reshape((len(data),1))
    else:
        Nsamp, Ndim = samples1.shape

    Ndata = len(data)
    if weights is None:
        weights = np.ones(Ndata, dtype=float)/Ndata

    # compute first moments
    samples = np.array(set(list(samples1)+list(samples2)))
    logfirst = dict(zip(samples, logkde(samples, data, variances, weights=weights, num_proc=num_proc)))

    # compute second moments
    logseconds = np.empty((Nsamp, 2), dtype=float)
    if num_proc==1: # do everything on this core
        logseconds[:,:] = _logsecond_worker(samples1, samples2, data, variances, weights)

    else: # parallelize
        # divide the work
        sets = _define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logsecond_worker, args=(samples1[truth], samples2[truth], data, variances, weights, logfirst), kwargs={'conn':conn2})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            logsecond[truth,:] = conni.recv()

    ### manipulate the moments to get the variance
    logcovkdes = logsecond + np.log(1 - np.exp(logsecond[:,1]-logsecond[:,0]))
    return logcovkdes, np.array([logfirst[sample] for sample in samples1]), np.array([logfirst[sample] for sample in samples2])

def _logsecond_worker(samples1, samples2, data, variances, weights, logfirst, conn=None):
    Nsamp, Ndim = samples1.shape
    Ndata = len(data)

    logsecond = np.empty((Nsamp, 2), dtype=float)
    twov = -0.5/variances

    z = np.empty(Ndata, dtype=float)
    w2 = weights**2
    for i in xrange(Nsamp):
        sample1 = samples1[i]
        sample2 = samples2[i]

        ### compute first moments
        z[:] = np.sum((data-sample1)**2 * twov, axis=1) + np.sum((data-sample2)**2 * twov, axis=1) ### shape: (Ndata)

        ### do this backflip to preserve accuracy
        m = np.max(z)
        logsecond[i,0] = np.log(np.sum(w2*np.exp(z-m))) + m ### it is important that the weights are squared because we treat the samples as drawn from some prior and
                                                            ### therefore the weights are random variables that need to be included within the average
                                                            ### just like we have multiple factors of the kernel

        logsecond[i,1] = logfirst[sample1]+logfirst[sample2] ### the first-moment terms

    ### subtract off common factors
    logsecond[:,0] += -Ndim*np.log(2*np.pi) - np.sum(np.log(variances))
    logsecond[:,1] -= np.log(Ndata)

    if conn is not None:
        conn.send(logsecond)
    return logsecond

def logleave1outLikelihood(data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    computes the logLikelihood for how well this bandwidth produces a KDE that reflects p(data|B) using samples drawn from p(B|data)=p(data|B)*p(B)

    assumes data's shape is (Nsamples, Ndim)
    assumes variances's shape is (Ndim,) -> a diagonal covariance matrix

    assumes weights' shape is (Nsamples,) and must be normalized so sum(weights)=1

    returns mean(logL), var(logL), mean(grad_logL), covar(dlogL/dvp)
    """
    return logleavekoutLikelihood(data, variances, k=1, weights=weights, num_proc=num_proc)

def logleavekoutLikelihood(data, variances, k=1, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    implements a leave-k-out cross validation estimate of the logLikelihood
    returns mean(logL), var(logL), mean(dlogL/dv), var(dlogL/dv)
    """
    Nsamples, Ndim = data.shape

    if weights is None:
        weights = np.ones(Nsamples, dtype='float')/Nsamples

    sets = [[] for _ in range(max(int(np.ceil(Nsamples/k)),2))]
    Nsets = len(sets)
    for i in xrange(Nsamples):
        sets[i%Nsets].append(i)

    logL = np.empty(Nsets, dtype='float')
    grad_logL = np.empty(Nsets, dtype='float')

    twov = -0.5/variances
    truth = np.ones(Nsamples, dtype=bool)

    for i in xrange(Nsets):
        sample = data[sets[i]]

        truth[:] = True
        truth[sets[i]] = False

        ### compute logLikelihood
        ### take the joint kde (sum of logs) and weight it by the joint weights of all points within that set (product of weights)
        logL[i] = np.sum(logkde(sample, data[truth], variances, weights=weights[truth], num_proc=num_proc)) - np.log(np.sum(weights[truth]))

        ### compute gradient of logLikelihood
        grad_logL[i] = np.sum(grad_logkde(sample, data[truth], variances, weights=weights[truth], num_proc=num_proc))

    ### compute statistics
    set_weights = np.array([np.sum(np.log(weights[sets[i]])) for i in xrange(Nsets)], dtype=float) ### compute the cumulative weights for each set
    set_weights = np.exp(set_weights-np.max(set_weights))
    set_weights /= np.sum(set_weights)

    mlogL = np.sum(set_weights*logL) # scalar
    vlogL = (np.sum(set_weights*logL**2) - mlogL**2) / Nsets # scalar

    mglogL = np.sum(set_weights*grad_logL)  # scalar
    vglogL = (np.sum(set_weights*grad_logL**2) - mglogL**2) / Nsets ### scaler

    return mlogL, vlogL, mglogL, vglogL

#-------------------------------------------------
# automatic bandwidth selection
#-------------------------------------------------

def silverman_bandwidth(data, weights=None):
    """approximate rule of thumb for bandwidth selection"""
    if weights is None:
        std = np.std(data)
        num = len(data)
    else: ### account for weights when computing std
        N = np.sum(weights)
        std = (np.sum(weights*data**2)/N - (np.sum(weights*data)/N)**2)**0.5
        num = neff(weights/np.sum(weights)) ### approximate number of samples that matter
    return 0.9 * std * num**(-0.2)

def optimize_bandwidth(
        data,
        bandwidth_range,
        rtol=1e-3,
        k=1,
        weights=None,
        num_proc=DEFAULT_NUM_PROC,
        minb_result=None,
        maxb_result=None,
        verbose=False,
    ):
    """optimize the bandwidth by finding the zero-crossing of the derivative via a bisection search"""
    Nsamp, Ncol = data.shape
    v = np.empty(Ncol, dtype=float)
    minb, maxb = bandwidth_range

    midb = (maxb*minb)**0.5 ### take geometric mean as guess so that we can scan large swaths of bandwidth values efficiently

    if (maxb-minb) < rtol*midb: ### basic termination condition, note that we still use the difference (rather than the ratio)
        if verbose:
            print('bandwidths agree to within %.6e, processing midb=%.6e and returning'%(rtol, midb))
        ### Note, we probably don't need to evaluate this at the mid point (most users won't care)
        ### however, making sure we do simplifies the return formatting (we always will return the logleavekout result)
        ### and taking the mid point will give us strictly smaller errors than rtol, which is desired
        v[:] = midb**2
        return midb, logleavekoutLikelihood(data, v, k=k, weights=weights, num_proc=num_proc)

    else: ### may need to recurse

        ### check boundaries to see if we can terminate early
        if minb_result is None:
            if verbose:
                print('processing minb=%.6e'%minb)
            v[:] = minb**2
            minb_result = logleavekoutLikelihood(data, v, k=k, weights=weights, num_proc=num_proc)

        if minb_result[2] < 0: ### convex function, if this is already decreasing at the min(b), then we should just return that
            if verbose:
                print('logL decreasing at minb, so returning edge case')
            return minb, minb_result

        if maxb_result is None:
            if verbose:
                print('processing maxb=%.6e'%maxb)
            v[:] = maxb**2
            maxb_result = logleavekoutLikelihood(data, v, k=k, weights=weights, num_proc=num_proc)

        if maxb_result[2] > 0: ### convex function, if this is still increasing at max(b), then we should just return that
            if verbose:
                print('logL increasing at maxb, so returning edge case')
            return maxb, maxb_result

        ### we need to recurse
        if verbose:
            print('processing midb=%.6e'%midb)
        v[:] = midb**2
        midb_result = logleavekoutLikelihood(data, v, k=k, weights=weights, num_proc=num_proc)

        if midb_result[2] == 0: ### vanishing, this is the optimum
            if verbose:
                print('    vanishing derivative at midb=%.6e; returning'%midb)
            return mid, midb_result

        elif midb_result[2] > 0: # increasing at midb, so that's the new minimum
            if verbose:
                print('    increasing at midb=%.6e, recursing with (minb=%.6e, maxb=%.6e)'%(midb, midb, maxb))

            return optimize_bandwidth(
                data,
                (midb, maxb),
                minb_result=midb_result,
                maxb_result=maxb_result,
                rtol=rtol,
                k=k,
                weights=weights,
                num_proc=num_proc,
                verbose=verbose,
            )

        else: # decreasing at midb, so that's the new maximum
            if verbose:
                print('    decreasing at midb=%.6e, recursing with (minb=%.6e, maxb=%.6e)'%(midb, minb, midb))

            return optimize_bandwidth(
                data,
                (minb, midb),
                minb_result=minb_result,
                maxb_result=midb_result,
                rtol=rtol,
                k=k,
                weights=weights,
                num_proc=num_proc,
                verbose=verbose,
            )
