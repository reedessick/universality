__doc__ = "a module to help select hyper paramters. Delegates a lot to gaussianprocess.py"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

try:
    import emcee
except ImportError:
    emcee = None

try:
    from scipy import optimize
except ImportError:
    optimize = None

import multiprocessing as mp

### non-standard libraries
from universality import gaussianprocess as gp
from universality import utils

#-------------------------------------------------

DEFAULT_NUM_WALKERS = 50

DEFAULT_NUM_MCMC = 101
DEFAULT_NUM_STRIP = 0

DEFAULT_METHOD = None
DEFAULT_TOL = None

SAMPLES_DTYPE = [('sigma','float'), ('l','float'), ('sigma_obs','float'), ('logLike','float')]
CVSAMPLES_DTYPE = [('sigma','float'), ('l','float'), ('sigma_obs','float'), ('model_multiplier', 'float'), ('logLike','float')]

DEFAULT_MIN_SIGMA = 1.e-4
DEFAULT_MAX_SIGMA = 1.0

DEFAULT_MIN_L = 0.1
DEFAULT_MAX_L = 5.0

DEFAULT_MIN_M = 0.1
DEFAULT_MAX_M = 10.0

DEFAULT_SIGMA_PRIOR = 'log'
DEFAULT_L_PRIOR = 'lin'
DEFAULT_M_PRIOR = 'lin'

DEFAULT_TEMPERATURE = 1.

#-------------------------------------------------
### methods useful for a basic squared-exponential kernel
#-------------------------------------------------

def samples2gpr_f_dfdx(
        x_tst,
        f_obs,
        x_obs,
        samples,
        weights=None,
    ):
    """
    estimate the covariances needed for inference of f, dfdx based on samples of hyperparameters
    same as gaussianprocess.gpr_f_dfdx() but with marginalization over hyperparameters
    """
    Ntst = len(x_tst)
    NTST = 2*Ntst
    Nobs = len(x_obs)

    # covariance between test points
    cov_tst_tst = np.zeros((NTST,NTST), dtype='float')
    # covariance between test and observation points
    cov_tst_obs = np.zeros((NTST,Nobs), dtype='float')
    # covariance between observation and test points
    cov_obs_tst = np.zeros((Nobs,NTST), dtype='float')
    # covariance between observation points
    cov_obs_obs = np.zeros((Nobs,Nobs), dtype='float')

    N = 1.*len(samples)
    if weights is None:
        weights = np.ones(N, dtype='float')
    weights /= np.sum(weights)

    for sample, weight in zip(samples, weights):
        ### pull out hyperparameters from this sample
        sigma2 = sample['sigma']**2
        l2 = sample['l']**2
        sigma2_obs = sample['sigma_obs']**2

        # covariance between test points
        cov_tst_tst[:Ntst,:Ntst] += weight*gp.cov_f1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
        cov_tst_tst[:Ntst,Ntst:] += weight*gp.cov_f1_df2dx2(x_tst, x_tst, sigma2=sigma2, l2=l2)
        cov_tst_tst[Ntst:,:Ntst] += weight*gp.cov_df1dx1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
        cov_tst_tst[Ntst:,:Ntst] += weight*gp.cov_df1dx1_df2dx2(x_tst, x_tst, sigma2=sigma2, l2=l2)

        # covariance between test and observation points
        cov_tst_obs[:Ntst,:] += weight*gp.cov_f1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
        cov_tst_obs[Ntst:,:] += weight*gp.cov_df1dx1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)

        # covariance between observation and test points
        cov_obs_tst[:,Ntst:] += weight*gp.cov_f1_df2dx2(x_obs, x_tst, sigma2=sigma2, l2=l2)
        cov_obs_tst[:,:Ntst] += weight*gp.cov_f1_f2(x_obs, x_tst, sigma2=sigma2, l2=l2)

        # covariance between observation points
        cov_obs_obs += weight*gp._cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    # delegate to perform GPR
    mean, cov, logweight = gp.gpr(f_obs, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)

    ### slice the resulting arrays and return
    ### relies on the ordering we constructed within our covariance matricies!
    #            mean_f             mean_dfdx            cov_f_f          cov_f_dfdx        cov_dfdx_f      cov_dfdx_dfdx
    return mean[:Ntst], mean[Ntst:], cov[:Ntst,:Ntst], cov[:Ntst,Ntst:], cov[Ntst:,:Ntst], cov[:Ntst,:Ntst]

def samples2resample_f_dfdx(
        x_tst,
        f_obs,
        x_obs,
        samples,
        weights=None,
        degree=1,
    ):
    """
    same as gaussianprocess.gpr_resample_f_dfdx() but with marginalization over hyperparameters
    """
    # compute poly fit
    f_fit, f_tst, dfdx_tst = gp.poly_model_f_dfdx(x_tst, f_obs, x_obs, degree=degree)

    # delegate to perform GPR
    mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx = samples2gpr_f_dfdx(x_tst, f_obs-f_fit, x_obs, samples, weights=weights)

    mean_f += f_tst
    mean_dfdx += dfdx_tst

    return mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx

#------------------------

def param_grid(minimum, maximum, size=gp.DEFAULT_NUM, prior='log'):
    if minimum==maximum:
        param = [minimum]
    else:
        if prior=='log':
            minimum = np.log(minimum)
            maximum = np.log(maximum)
            param = np.exp(minimum + (maximum-minimum)*np.random.rand(size))
        elif prior=='lin':
            param = minimum + (maximum-minimum)*np.random.rand(size)
        else:
            raise ValueError('unknown prior='+prior)
    return param

def param_mc(minimum, maximum, size=gp.DEFAULT_NUM, prior='log'):
    if minimum==maximum:
        param = np.ones(size, dtype=float)*minimum
    else:
        if prior=='log':
            minimum = np.log(minimum)
            maximum = np.log(maximum)
            param = np.exp(minimum+np.random.rand(size)*(maximum-minimum))
        elif prior=='lin':
            param = minimum + np.random.rand(size)*(maximum-minimum)
        else:
            raise ValueError('unknown prior='+prior)
    return param

#-------------------------------------------------
# Marginal Likelihood routines used within investigate-gpr-resample-hyperparams
#-------------------------------------------------

def _logLike_worker(f_obs, x_obs, SIGMA, L, SIGMA_NOISE, degree, temperature=DEFAULT_TEMPERATURE, conn=None):
    beta = 1./temperature
    if conn is not None:
        conn.send(
            np.array([(s, l, sn, gp.logLike(f_obs, x_obs, sigma2=s**2, l2=l**2, sigma2_obs=sn**2, degree=degree)*beta) for s, l, sn in zip(SIGMA, L, SIGMA_NOISE)])
        )
    else:
        return np.array(
            [(s, l, sn, gp.logLike(f_obs, x_obs, sigma2=s**2, l2=l**2, sigma2_obs=sn**2, degree=degree)*beta) for s, l, sn in zip(SIGMA, L, SIGMA_NOISE)],
            dtype=SAMPLES_DTYPE,
        )

def logLike_grid(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        num_sigma=gp.DEFAULT_NUM,
        num_l=gp.DEFAULT_NUM,
        num_sigma_obs=gp.DEFAULT_NUM,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        degree=1,
        num_proc=utils.DEFAULT_NUM_PROC,
        temperature=DEFAULT_TEMPERATURE,
    ):
    '''
    compute logLike on a grid and return "samples" with associated logLike values corresponding to each grid point
    We space points logarithmically for sigma and sigma_obs, linearly for l
    '''
    ### compute grid
    sigma = param_grid(min_sigma, max_sigma, size=num_sigma, prior=sigma_prior)
    l = param_grid(min_l, max_l, size=num_l, prior=l_prior)
    sigma_obs = param_grid(min_sigma_obs, max_sigma_obs, size=num_sigma_obs, prior=sigma_obs_prior)

    SIGMA, L, SIGMA_NOISE = np.meshgrid(sigma, l, sigma_obs, indexing='ij')

    # flatten for ease of iteration
    SIGMA = SIGMA.flatten()
    L = L.flatten()
    SIGMA_NOISE = SIGMA_NOISE.flatten()

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _logLike_worker(f_obs, x_obs, SIGMA, L, SIGMA_NOISE, degree, temperature=temperature)

    else: ### divide up work and parallelize

        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, 4), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logLike_worker, args=(f_obs, x_obs, SIGMA[truth], L[truth], SIGMA_NOISE[truth], degree), kwargs={'conn':conn2, 'temperature':temperature})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            ans[truth,:] = conni.recv()

        # cast ans to the correct structured array
        ans = np.array(zip(ans[:,0], ans[:,1], ans[:,2], ans[:,3]), dtype=SAMPLES_DTYPE) ### do this because numpy arrays are stupid and don't cast like I want

    return ans

def logLike_mc(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        num_samples=gp.DEFAULT_NUM,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        degree=1,
        num_proc=utils.DEFAULT_NUM_PROC,
        temperature=DEFAULT_TEMPERATURE,
    ):
    '''
    draw samples from priors and return associated logLikes
    '''
    ### draw hyperparameters from hyperpriors
    SIGMA = param_mc(min_sigma, max_sigma, size=num_samples, prior=sigma_prior)
    L = param_mc(min_l, max_l, size=num_samples, prior=l_prior)
    SIGMA_NOISE = param_mc(min_sigma_obs, max_sigma_obs, size=num_samples, prior=sigma_obs_prior)

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _logLike_worker(f_obs, x_obs, SIGMA, L, SIGMA_NOISE, degree, temperature=temperature)

    else: ### divide up work and parallelize
        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, 4), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_logLike_worker, args=(f_obs, x_obs, SIGMA[truth], L[truth], SIGMA_NOISE[truth], degree), kwargs={'conn':conn2, 'temperature':temperature})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            ans[truth,:] = conni.recv()

        # cast ans to the correct structured array
        ans = np.array(zip(ans[:,0], ans[:,1], ans[:,2], ans[:,3]), dtype=SAMPLES_DTYPE) ### do this because numpy arrays are stupid and don't cast like I want

    return ans

def logLike_mcmc(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        num_samples=gp.DEFAULT_NUM,
        num_walkers=DEFAULT_NUM_WALKERS,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        degree=1,
        temperature=DEFAULT_TEMPERATURE,
    ):
    '''
    draw samples from the target distribution defined by logLike using emcee
    return samples with associated values of logLike
    '''
    if emcee is None:
        raise ImportError('could not import emcee')

    if sigma_prior=='log':
        ln_sigma_prior = lambda sigma: 1./sigma if min_sigma<sigma<max_sigma else -np.infty
    elif sigma_prior=='lin':
        ln_sigma_prior = lambda sigma: 0 if min_sigma<sigma<max_sigma else -np.infty
    else: 
        raise ValueError, 'unkown sigma_prior='+sigma_prior

    if l_prior=='log':
        ln_l_prior = lambda l: 1./l if min_l<l<max_l else -np.infty
    elif l_prior=='lin':
        ln_l_prior = lambda l: 0 if min_l<l<max_l else -np.infty
    else:
        raise ValueError, 'unkown l_prior='+l_prior

    if sigma_obs_prior=='log':
        ln_sigma_obs_prior = lambda sigma_obs: 1./sigma_obs if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    elif sigma_obs_prior=='lin':
        ln_sigma_obs_prior = lambda sigma_obs: 0 if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    else:
        raise ValueError, 'unkown sigma_obs_prior='+sigma_obs_prior

    beta = 1./temperature
    foo = lambda args: gp.logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=1)*beta \
        + ln_sigma_prior(args[0]) \
        + ln_l_prior(args[1]) \
        + ln_sigma_obs_prior(args[2])

    x0 = np.array([(min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_obs*max_sigma_obs)**0.5])
    x0 = emcee.utils.sample_ball(
        x0,
        x0*0.1,
        size=num_walkers,
    )

    sampler = emcee.EnsembleSampler(num_walkers, 3, foo)
    sampler.run_mcmc(x0, num_samples)

    sigma = sampler.chain[:,:,0]
    l = sampler.chain[:,:,1]
    sigma_obs = sampler.chain[:,:,2]
    lnprob = sampler.lnprobability

    return np.array(
        [(sigma[i,j], l[i,j], sigma_obs[i,j], lnprob[i,j]) for j in xrange(num_samples) for i in xrange(num_walkers)],
        dtype=SAMPLES_DTYPE,
    )

def logLike_maxL(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        method=DEFAULT_METHOD,
        tol=DEFAULT_TOL,
        degree=1,
    ):
    '''
    find the maximum logLikelihood as a function of hyperparamters.
    return a single "sample" with associated logLike
    '''
    if optimize is None:
        raise ImportError('could not import scipy.optimize')

    min_sigma2 = min_sigma**2
    max_sigma2 = max_sigma**2
    min_l2 = min_l**2
    max_l2 = max_l**2
    min_sigma2_obs = min_sigma_obs**2
    max_sigma2_obs = max_sigma_obs**2

    x0 = (min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_obs*max_sigma_obs)**0.5

    foo = lambda args: -gp.logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=degree)
    jac = lambda args: -np.array(gp.grad_logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=degree))

    kwargs = dict(
        jac=jac,
        bounds=[
            (min_sigma2, max_sigma2),
            (min_l2, max_l2),
            (min_sigma2_obs, max_sigma2_obs),
        ],
        constraints=[
            {'type':'eq', 'fun':lambda args:min_sigma2<=args[0]<=max_sigma2},
            {'type':'eq', 'fun':lambda args:min_l2<=args[1]<=max_l2},
            {'type':'eq', 'fun':lambda args:min_sigma2_obs<=args[2]<=max_sigma2_obs},
        ],
        tol=tol,
    )
    if method is not None:
        kwargs['method'] = method

    res = optimize.minimize(foo, x0, **kwargs)

    if not res.success:
        print('\n>>> WARNING: failed to converge: %s\n'%res.message)

    sigma, l, sigma_obs = res.x
    return np.array(
        [(sigma, l, sigma_obs, -foo(res.x))],
        dtype=SAMPLES_DTYPE,
    )

#-------------------------------------------------
# cross-validation likelihoods used within investigate-gpr-gpr-hyperparameters
#-------------------------------------------------

def _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MODEL_MULTIPLIER, degree, temperature=DEFAULT_TEMPERATURE, conn=None):

    ### initialize output array
    ans = np.empty((len(SIGMA), 5), dtype=float)
    ans[:,0] = SIGMA
    ans[:,1] = L
    ans[:,2] = SIGMA_NOISE
    ans[:,3] = MODEL_MULTIPLIER
    ans[:,4] = -np.infty 

    inds = utils.models2combinations(models)

    beta = 1./temperature
    SIGMA2 = SIGMA**2
    L2 = L**2
    SIGMA_NOISE2 = SIGMA_NOISE**2

    for indecies in inds: ### iterate over combinatorics to marginalize and compute logLike
        _models = [model[i] for model, i in zip(models, indecies)]
        logw = np.sum(np.log(model['weight']) for model in _models) ### the weight prefactor for this combination

        x_obs, f_obs, covs, covs_model = gp.cov_altogether_noise(_models, stitch) ### compute this once per combination
        Nobs = len(x_obs)

        ### make boolean arrays labeling where each model lives in the bigger matrix
        keep_bools = []
        start = 0
        for model in _models:
            end = start+len(model['x'])
            t = np.zeros(Nobs, dtype=bool)
            t[start:end] = True
            keep_bools.append(t)
            start = end

        for ind, (s2, l2, S2, m) in enumerate(zip(SIGMA2, L2, SIGMA_NOISE2, MODEL_MULTIPLIER)):
            cov = gp.cov_altogether_obs_obs(x_obs, covs, covs_model, sigma2=s2, l2=l2, sigma2_obs=S2, model_multiplier=m)
            invcov = np.linalg.inv(cov) ### invert this exactly once! still expensive, but we save with the iteration over models

            ### compute cross validation likelihood for this combination
            logprob = _cvlogprob(keep_bools, x_obs, f_obs, invcov, degree) + logw # include overall weight from combinatorics

            ### add to overall marginalization
            M = max(ans[ind,4], logprob)
            ans[ind,4] = np.log(np.exp(ans[ind,4]-M) + np.exp(logprob-M))+M

    if conn is not None:
        conn.send(ans)

    else:
        return np.array([tuple(_) for _ in ans], dtype=CVSAMPLES_DTYPE)

def _cvlogprob(keep_bools, x_obs, f_obs, invcov, degree):
    logprob = -np.infty
    inds = np.arange(len(invcov))

    tmp_f_obs = np.empty_like(f_obs, dtype=float)
    tmp_x_obs = np.empty_like(x_obs, dtype=float)
    tmp_invcov = np.empty_like(invcov, dtype=float) ### used so we can re-order in place

    for keep in keep_bools:
        Nkeep = np.sum(keep)

        tmp_f_obs[:] = f_obs
        tmp_invcov[:] = invcov ### re-set this

        ### re-order
        _cvlogprob_reorder(keep, tmp_x_obs, tmp_f_obs, tmp_invcov)

        ### compute poly-model fits
        mu_rst, mu_tst = gp.poly_model(x_obs[:Nkeep], f_obs[Nkeep:], x_obs[Nkeep:], degree=degree) ### poly_model fit

        ### extract things from invcov
        P = tmp_invcov[:Nkeep,:Nkeep]
        df = (tmp_f_obs[:Nkeep]-mu_tst) + np.dot(np.linalg.inv(P), np.dot(tmp_invcov[:Nkeep,Nkeep:], (tmp_f_obs[Nkeep:]-mu_rst)))

        ### compute loglike for this cross-validation
        _logprob = gp._logLike(df, P)

        ### add to overall counter
        m = max(logprob, _logprob)
        logprob = np.log(np.exp(logprob-m)+np.exp(_logprob-m))+m

    return logprob

def _cvlogprob_reorder(keep_bool, x_obs, f_obs, invcov):
    for i, j in zip(np.arange(np.sum(keep_bool)), np.arange(len(x_obs))[keep_bool]): ### mapping of the indecies that need to switch
        if i!=j:
            ### interchange the vector in place
            _ = x_obs[i]
            x_obs[i] = x_obs[j]
            x_obs[j] = _

            _ = f_obs[i]
            f_obs[i] = f_obs[j]
            f_obs[j] = _

            ### interchange the matrix in place
            gp._interchange(i, j, invcov)

def cvlogLike_grid(
        models,
        stitch,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        (min_model_multiplier, max_model_multiplier),
        num_sigma=gp.DEFAULT_NUM,
        num_l=gp.DEFAULT_NUM,
        num_sigma_obs=gp.DEFAULT_NUM,
        num_model_multiplier=gp.DEFAULT_NUM,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        model_multiplier_prior='log',
        degree=1,
        num_proc=utils.DEFAULT_NUM_PROC,
        temperature=DEFAULT_TEMPERATURE,
    ):
    ### compute grid
    sigma = param_grid(min_sigma, max_sigma, size=num_sigma, prior=sigma_prior)
    l = param_grid(min_l, max_l, size=num_l, prior=l_prior)
    sigma_obs = param_grid(min_sigma_obs, max_sigma_obs, size=num_sigma_obs, prior=sigma_obs_prior)
    model_multiplier = param_grid(min_model_multiplier, max_model_multiplier, size=num_model_multiplier, prior=model_multiplier_prior)

    SIGMA, L, SIGMA_NOISE, MODEL_MULTIPLIER = np.meshgrid(sigma, l, sigma_obs, model_multiplier, indexing='ij')

    # flatten for ease of iteration
    SIGMA = SIGMA.flatten()
    L = L.flatten()
    SIGMA_NOISE = SIGMA_NOISE.flatten()
    MODEL_MULTIPLIER = MODEL_MULTIPLIER.flatten()

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MODEL_MULTIPLIER, degree, temperature=temperature)

    else: ### divide up work and parallelize

        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, 4), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_cvlogLike_worker, args=(models, stitch, SIGMA[truth], L[truth], SIGMA_NOISE[truth], MODEL_MULTIPLIER[truth], degree), kwargs={'conn':conn2, 'temperature':temperature})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            ans[truth,:] = conni.recv()

        # cast ans to the correct structured array
        ans = np.array(zip(ans[:,0], ans[:,1], ans[:,2], ans[:,3], ans[:,4]), dtype=CVSAMPLES_DTYPE) ### do this because numpy arrays are stupid and don't cast like I want

    return ans

def cvlogLike_mc(
        models,
        stitch,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        (min_model_multiplier, max_model_multiplier),
        num_samples=gp.DEFAULT_NUM,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        model_multiplier_prior='log',
        degree=1,
        num_proc=utils.DEFAULT_NUM_PROC,
        temperature=DEFAULT_TEMPERATURE,
    ):
    ### draw hyperparameters from hyperpriors
    SIGMA = param_mc(min_sigma, max_sigma, size=num_samples, prior=sigma_prior)
    L = param_mc(min_l, max_l, size=num_samples, prior=l_prior)
    SIGMA_NOISE = param_mc(min_sigma_obs, max_sigma_obs, size=num_samples, prior=sigma_obs_prior)
    MM = param_mc(min_model_multiplier, max_model_multiplier, size=num_samples, prior=model_multiplier_prior)

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MM, degree, temperature=temperature)

    else: ### divide up work and parallelize

        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, 4), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_cvlogLike_worker, args=(models, stitch, SIGMA[truth], L[truth], SIGMA_NOISE[truth], MM[truth], degree), kwargs={'conn':conn2, 'temperature':temperature})
            proc.start()
            procs.append((proc, conn1))
            conn2.close()

        # read in results from process
        for truth, (proci, conni) in zip(sets, procs):
            proci.join() ### should clean up child...
            ans[truth,:] = conni.recv()

        # cast ans to the correct structured array
        ans = np.array(zip(ans[:,0], ans[:,1], ans[:,2], ans[:,3], ans[:,4]), dtype=CVSAMPLES_DTYPE) ### do this because numpy arrays are stupid and don't cast like I want

    return ans
