__doc__ = "a module for general utility functions when applying \"rapid sampling\" based on monte carlo samples of a univeral relation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------
# basic utilities
#-------------------------------------------------

def load(inpath, columns, logcolumns=[]):
    data = np.genfromtxt(inpath, names=True, delimiter=',') ### assumes standard CSV format

    # check that all requested columns are actually in the data
    for column in columns:
        assert column in data.dtype.fields, 'column=%s not in %s'%(column, inpath)

    # downselect data to what we actually want
    return \
        np.transpose([np.log(data[column]) if column in logcolumns else data[column] for column in columns]), \
        ['log(%s)'%column if column in logcolumns else column for column in columns]

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

    return data[truth]

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

def logkde(samples, data, variances, verbose=False, weights=None):
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

        if weights==None:
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
        z = np.sum(zi, axis=1)       ### shape: (Nsamples)

        x = np.sum((weights[truth]*np.exp(z))*(-zi/variances).transpose(), axis=1)
        y = np.exp(logL[i])

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

#-------------------------------------------------
# methods for computing marginal likelihoods
#-------------------------------------------------

def compute_marginalLikelihood(post_data, prior_data):
    """
    expects post_data to be a structured array with (at least) the following fields
        lambda1
        labmda1
        m1
        m2

    expects prior_data to be a structured array with (at least) the following fields
        lambda1
        lambda2
        m1
        m2
        p1
        gamma1
        gamma2
        gamma3

    should return a 1D array with lenght=len(prior_data)

    assumes a ***flat prior*** in the GW analysis over
        m1, m2, lambda1, lambda2
    separately. Really, prior on m2 is conditional on m1, but it is flat regardless

    Defining
        A = p1, gamma1, gamma2, gamma3
        B = m1, m2, lambda1, lambda2

    computes
        p(data|A_i) = \int dB [ p(data|B) * p(B|A) ]
                    = sum_j [ KDE(B_j, B(A_i); bandwidth) / pgw(B_j) ]

    where
        KDE(B_j, B(A_i); bandwidth) is the KDE approximation of the conditional probability associated with the mapping A->B
        pgw(B_j) is the value of the prior used in the GW analysis
        B_j are drawn from the GW posterior, which is proportional to p(data|B)*p(B)

    performs some cross validation to determine an optimal value of the bandwidth used within the Gaussian KDE
    """
    bandwidth = np.array([0.1, 0.1, 0.1, 0.1]) ### FIXME pick something better, or do shit adaptively
    params = ['lambda1', 'lambda2', 'm1', 'm2']

    # record sizes
    Nparams = len(params)
    Npost = len(post_data)
    Nprior = len(prior_data)

    # extract interesting parameters
    post_vec = np.transpose([post_data[param] for param in params]) # shape = (Npost, Nparams)
    prior_vec = np.transpose([prior_data[param] for param in params]) # shape = (Nprior, Nparams)
    logpgw_vec = logpgw(post_data['lambda1'], post_data['labmda2'], post_data['m1'], post_data['m2']) # shape = (Npost,)

    # compute big ol' arrays with shape = (Nprior, Npost, Nparams)
    post_vec = np.outer(np.ones(Nprior), post_vec.flatten()).reshape((Nprior, Npost, Nparams)) # shape = (Nprior, Npost, Nparams)
    prior_vec = np.outer(prior_vec.flatten(), np.ones(Npost)).reshape((Nprior, Nparams, Npost)).transpose(0,2,1) # shape = (Nprior, Npost, Nparams)
    pgw_vec = np.outer(np.ones(Nprior), pgw_vec).reshape((Nprior, Npost)) # shape = (Nprior, Npost)

    ### compute the difference, take the 
    return logaddexp(-0.5*np.sum((post_vec - prior_vec)**2/bandwidth, axis=-1) - (Nparams/2.)*np.log(2*np.pi) - 0.5*np.sum(np.log(bandwidth)) - logpgw_vec) - np.log(Npost) # shape = (Nprior,)

    ### this should be a less precise but possibly easier to debug version
#    return np.log(np.sum(np.exp(-0.5*np.sum(post_vec - prior_vec)**2/bandwidth, axis=-1) / (2*np.pi*bandwidth)**(Nparams/2.) / pgw_vec, axis=-1) / Npost) # shape = (Nprior,)

#---

def logpgw(lambda1, lambda2, m1, m2):
    """
    proportional to the prior assigned in the GW analysis to the parameters
        lambda1, lambda2, m1, m2
    """
    return 0
