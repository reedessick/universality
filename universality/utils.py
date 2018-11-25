__doc__ = "a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples of a univeral relation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

from collections import defaultdict
import numpy as np

import multiprocessing as mp

from . import eos

#-------------------------------------------------

DEFAULT_NUM_PROC = min(max(mp.cpu_count()-1, 1), 15) ### reasonable bounds for parallelization...

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

def draw_from_weights(weights, size=1):
    """
    return indecies corresponding to the weights chosen herein
    we draw with replacement
    """
    N = len(weights)
    weights = np.array(weights)
    order = weights.argsort()
    weights = weights[order]

    # compute a cdf and draw from it
    return order[np.ceil(np.interp(np.random.random(size), np.cumsum(weights), np.arange(N))).astype(int)]

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
    delta = (M-m)*pad
    return (m-delta, M+delta)

def vects2flatgrid(*vects):
    return np.transpose([_.flatten() for _ in np.meshgrid(*vects, indexing='ij')])

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

def logkde(samples, data, variances, weights=None, num_proc=DEFAULT_NUM_PROC):
    """
    a wrapper around actually computing the KDE estimate at a collection of samples

    estimates kde as sum_i[weight_i * K(sample, data_i)]

    returns log(kde(samples))
    """
    shape = samples.shape
    if len(shape) not in [1, 2]:
        raise ValueError, 'bad shape for samples'

    if len(shape)==1:
        Nsamp = shape[0]
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))
    else:
        Nsamp, Ndim = samples.shape

    if np.any(weights==None): ### needed because modern numpy performs element-wise comparison here
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
    return sets

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
    grad_logkdes = np.empty(shape, dtype=float)
    if len(shape) not in [1, 2]:
        raise ValueError, 'bad shape for samples'

    if len(shape)==1:
        Nsamp = shape[0]
        Ndim = 1
        samples = samples.reshape((Nsamp,1))
        data = data.reshape((len(data),1))

    else:
        Nsamp, Ndim = samples.shape

    Ndata = len(data)
    if np.any(weights==None): ### needed because modern numpy performs element-wise comparison here
        weights = np.ones(Ndata, dtype='float')/Ndata

    if num_proc==1:
        grad_logkdes[:,:] = _grad_log_kde_worker(samples, data, variances, weights)

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
    twov = -0.5/variances
    z = np.empty(Ndata, dtype=float)
    for i in xrange(Nsamp):
        sample = samples[i]
        z[:] = np.sum((data-sample)**2 * twov, axis=1)  ### shape: (Ndata, Ndim) -> (Ndata)

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

    if conn is not None:
        conn.send(grad_logL)
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
    if np.any(weights==None):
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

#-------------------------------------------------

def set_crust(crust_eos=eos.DEFAULT_CRUST_EOS):
    global CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY
    CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY = \
        load(eos.eospaths.get(crust_eos, crust_eos), columns=['pressurec2', 'energy_densityc2', 'baryon_density'])[0].transpose()

def crust_energy_densityc2(pressurec2):
    """
    return energy_densityc2 for the crust from Douchin+Haensel, arXiv:0111092
    this is included in our repo within "sly.csv", taken from Ozel's review.
    """
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2)

def crust_baryon_density(pressurec2):
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_BARYON_DENSITY)

def stitch_below_reference_pressure(energy_densityc2, pressurec2, reference_pressurec2):
    """reutrn energy_densityc2, pressurec2"""
    sly_truth = CRUST_PRESSUREC2 <= reference_pressurec2
    eos_truth = reference_pressurec2 <= pressurec2
    return np.concatenate((CRUST_ENERGY_DENSITYC2[sly_truth], energy_densityc2[eos_truth])), np.concatenate((CRUST_PRESSUREC2[sly_truth], pressurec2[eos_truth]))

### integration routines
def dedp2e(denergy_densitydpressure, pressurec2, reference_pressurec2):
    """
    integrate to obtain the energy density
    if stitch=True, map the lower pressures onto a known curst below the reference pressure instead of just matching at the reference pressure
    """
    energy_densityc2 = np.empty_like(pressurec2, dtype='float')
    energy_densityc2[0] = 0 # we start at 0, so handle this as a special case

    # integrate in the bulk via trapazoidal approximation
    energy_densityc2[1:] = np.cumsum(0.5*(denergy_densitydpressure[1:]+denergy_densitydpressure[:-1])*(pressurec2[1:] - pressurec2[:-1]))

    ### match at reference pressure
    energy_densityc2 += crust_energy_densityc2(reference_pressurec2) - np.interp(reference_pressurec2, pressurec2, energy_densityc2)

    return energy_densityc2

def e_p2rho(energy_densityc2, pressurec2, reference_pressurec2):
    """
    integrate the first law of thermodynamics
        dmu = rho/(mu+p) drho
    """
    baryon_density = np.ones_like(pressurec2, dtype='float')

    integrand = 1./(energy_densityc2+pressurec2)
    baryon_density[1:] *= np.exp(np.cumsum(0.5*(integrand[1:]+integrand[:-1])*(energy_densityc2[1:]-energy_densityc2[:-1]))) ### multiply by this factor

    ### FIXME: match baryon density to energy density at reference pressure
    #baryon_density *= ec2 / np.interp(ref_pc2, pressurec2, baryon_density)

    ### match at the lowest allowed energy density
    baryon_density *= crust_baryon_density(reference_pressurec2)/np.interp(reference_pressurec2, pressurec2, baryon_density)

    return baryon_density

#-------------------------------------------------

### load the sly EOS for stitching logic
set_crust() ### use this as the crust!
