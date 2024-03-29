"""a module to help select hyper paramters. Delegates a lot to gaussianprocess.py
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import time
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
from . import gaussianprocess as gp
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
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
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
    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs

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
        ans = np.empty((Nsamp, len(SAMPLES_DTYPE)), dtype=float)

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
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
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
    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs

    ### draw hyperparameters from hyperpriors
    SIGMA = param_mc(min_sigma, max_sigma, size=num_samples, prior=sigma_prior)
    L = param_mc(min_l, max_l, size=num_samples, prior=l_prior)
    SIGMA_NOISE = param_mc(min_sigma_obs, max_sigma_obs, size=num_samples, prior=sigma_obs_prior)

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _logLike_worker(f_obs, x_obs, SIGMA, L, SIGMA_NOISE, degree, temperature=temperature)

    else: ### divide up work and parallelize
        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, len(SAMPLES_DTYPE)), dtype=float)

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
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
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
    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs

    if emcee is None:
        raise ImportError('could not import emcee')

    if sigma_prior=='log':
        ln_sigma_prior = lambda sigma: 1./sigma if min_sigma<sigma<max_sigma else -np.infty
    elif sigma_prior=='lin':
        ln_sigma_prior = lambda sigma: 0 if min_sigma<sigma<max_sigma else -np.infty
    else: 
        raise ValueError('unkown sigma_prior='+sigma_prior)

    if l_prior=='log':
        ln_l_prior = lambda l: 1./l if min_l<l<max_l else -np.infty
    elif l_prior=='lin':
        ln_l_prior = lambda l: 0 if min_l<l<max_l else -np.infty
    else:
        raise ValueError('unkown l_prior='+l_prior)

    if sigma_obs_prior=='log':
        ln_sigma_obs_prior = lambda sigma_obs: 1./sigma_obs if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    elif sigma_obs_prior=='lin':
        ln_sigma_obs_prior = lambda sigma_obs: 0 if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    else:
        raise ValueError('unkown sigma_obs_prior='+sigma_obs_prior)

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
        [(sigma[i,j], l[i,j], sigma_obs[i,j], lnprob[i,j]) for j in range(num_samples) for i in range(num_walkers)],
        dtype=SAMPLES_DTYPE,
    )

def logLike_maxL(
        f_obs,
        x_obs,
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
        method=DEFAULT_METHOD,
        tol=DEFAULT_TOL,
        degree=1,
    ):
    '''
    find the maximum logLikelihood as a function of hyperparamters.
    return a single "sample" with associated logLike
    '''
    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs

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

def _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MODEL_MULTIPLIER, degree, temperature=DEFAULT_TEMPERATURE, slow=False, diagonal_model_covariance=False, conn=None, verbose=False):

    ### initialize output array
    ans = np.empty((len(SIGMA), 5), dtype=float)
    ans[:,0] = SIGMA
    ans[:,1] = L
    ans[:,2] = SIGMA_NOISE
    ans[:,3] = MODEL_MULTIPLIER
    ans[:,4] = -np.infty 

    inds = utils.models2combinations(models)

    SIGMA2 = SIGMA**2
    L2 = L**2
    SIGMA_NOISE2 = SIGMA_NOISE**2
    MODEL_MULTIPLIER2 = MODEL_MULTIPLIER**2

    if not slow:
        print("WARNING: the 'fast' cvlogprob algorithm is known to be numerically unstable! Please proceed with extreme caution!")

    for indecies in inds: ### iterate over combinatorics to marginalize and compute logLike
        _models = [model[i] for model, i in zip(models, indecies)]
        logw = np.sum(np.log(model['weight']) for model in _models) ### the weight prefactor for this combination

        ### compute this once per combination
        x_obs, f_obs, covs, covs_model, Nstitch = gp.cov_altogether_noise(_models, stitch, diagonal_model_covariance=diagonal_model_covariance)
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

        for ind, (s2, l2, S2, m2) in enumerate(zip(SIGMA2, L2, SIGMA_NOISE2, MODEL_MULTIPLIER2)):
            ### compute cross validation likelihood for this combination
            if slow:
                logscore = _slow_cvlogprob(keep_bools, x_obs, f_obs, covs, covs_model, Nstitch, s2, l2, S2, m2, degree, verbose=verbose) + logw
            else:
                invcov = np.linalg.inv(gp.cov_altogether_obs_obs(x_obs, covs, covs_model, Nstitch, sigma2=s2, l2=l2, sigma2_obs=S2, model_multiplier2=m2))
                logscore = _cvlogprob(keep_bools, x_obs, f_obs, covs, covs_model, invcov, S2, m2, degree, verbose=verbose) + logw

            ### add to overall marginalization
            M = max(ans[ind,4], logscore)
            ans[ind,4] = np.log(np.exp(ans[ind,4]-M) + np.exp(logscore-M))+M

    ans[:,4] /= temperature

    if conn is not None:
        conn.send(ans)

    else:
        return np.array([tuple(_) for _ in ans], dtype=CVSAMPLES_DTYPE)

def _slow_cvlogprob(keep_bools, x_obs, f_obs, covs, covs_model, Nstitch, s2, l2, S2, m2, degree, verbose=False):
    logscore = 0.
    for keep in keep_bools:
        lose = np.logical_not(keep)

        mean, cov, logweight = gp.gpr_altogether( ### do expensive conditioning
            x_obs[keep],
            f_obs[lose],
            x_obs[lose],
            gp._extract_subset_from_matrix(lose, covs),
            gp._extract_subset_from_matrix(lose, covs_model),
            Nstitch,
            degree=degree,
            guess_sigma2=s2,
            guess_l2=l2,
            guess_sigma2_obs=S2,
            guess_model_multiplier2=m2,
        )

        # compute the probability of seeing the held out thing given everyrhing else
        logscore += gp._logprob(
            f_obs[keep],
            mean,
            np.linalg.inv(cov),
            invcov_obs=np.linalg.inv(gp._extract_subset_from_matrix(keep, covs)),
            verbose=verbose,
        )

    return logscore

def _cvlogprob(keep_bools, x_obs, f_obs, covs, covs_model, invcov, S2, m2, degree, verbose=False):
    logscore = 0

    for keep in keep_bools:
        lose = np.logical_not(keep)
        Nkeep = np.sum(keep)

        ### compute poly-model fits
        mu_obs, mu_tst = gp.poly_model(x_obs[keep], f_obs[lose], x_obs[lose], degree=degree) ### poly_model fit

        ### extract things from invcov
        f_prb, invcov_prb = gp._extract_invconditioned_mean_from_invcov(keep, f_obs[lose]-mu_obs, invcov)
        f_prb += mu_tst

        ### subtract out the bits we don't want...
        cov_prb = np.linalg.inv(invcov_prb)
        cov_prb -= np.ones(Nkeep)*S2 + gp._extract_subset_from_matrix(keep, covs + m2*covs_model)
        cov_prb = gp.posdef(cov_prb)
        invcov_prb = np.linalg.inv(cov_prb)

        ### compute loglike for this cross-validation
        logscore += gp._logprob(
            f_obs[keep], ### the mean from the held-out observations
            f_prb,       ### conditioned mean
            invcov_prb,  ### invcov for conditioned process
            invcov_obs=np.linalg.inv(gp._extract_subset_from_matrix(keep, covs)),  ### invcov for the held-out process
            verbose=verbose,
        )

    return logscore

def cvlogLike_grid(
        models,
        stitch,
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
        min_model_multiplierANDmax_model_multiplier,
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
        slow=False,
        diagonal_model_covariance=False,
        verbose=False,
    ):
    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs
    min_model_multiplier, max_model_multiplier = min_model_multiplierANDmax_model_multiplier

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
        ans = _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MODEL_MULTIPLIER, degree, temperature=temperature, slow=slow, diagonal_model_covariance=diagonal_model_covariance, verbose=verbose)

    else: ### divide up work and parallelize

        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, len(CVSAMPLES_DTYPE)), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_cvlogLike_worker, args=(models, stitch, SIGMA[truth], L[truth], SIGMA_NOISE[truth], MODEL_MULTIPLIER[truth], degree), kwargs={'conn':conn2, 'temperature':temperature, 'slow':slow, 'diagonal_model_covariance':diagonal_model_covariance, 'verbose':verbose})
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
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
        min_model_multiplierANDmax_model_multiplier,
        num_samples=gp.DEFAULT_NUM,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        model_multiplier_prior='log',
        degree=1,
        num_proc=utils.DEFAULT_NUM_PROC,
        temperature=DEFAULT_TEMPERATURE,
        slow=False,
        diagonal_model_covariance=False,
        verbose=False,
    ):

    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs
    min_model_multiplier, max_model_multiplier = min_model_multiplierANDmax_model_multiplier

    ### draw hyperparameters from hyperpriors
    SIGMA = param_mc(min_sigma, max_sigma, size=num_samples, prior=sigma_prior)
    L = param_mc(min_l, max_l, size=num_samples, prior=l_prior)
    SIGMA_NOISE = param_mc(min_sigma_obs, max_sigma_obs, size=num_samples, prior=sigma_obs_prior)
    MM = param_mc(min_model_multiplier, max_model_multiplier, size=num_samples, prior=model_multiplier_prior)

    ### iterate over grid points and copmute logLike for each
    if num_proc==1: ### do this on a single core
        ans = _cvlogLike_worker(models, stitch, SIGMA, L, SIGMA_NOISE, MM, degree, temperature=temperature, slow=slow, diagonal_model_covariance=diagonal_model_covariance, verbose=verbose)

    else: ### divide up work and parallelize

        Nsamp = len(SIGMA)
        ans = np.empty((Nsamp, len(CVSAMPLES_DTYPE)), dtype=float)

        # partition work amongst the requested number of cores
        sets = utils._define_sets(Nsamp, num_proc)

        # set up and launch processes.
        procs = []
        for truth in sets:
            conn1, conn2 = mp.Pipe()
            proc = mp.Process(target=_cvlogLike_worker, args=(models, stitch, SIGMA[truth], L[truth], SIGMA_NOISE[truth], MM[truth], degree), kwargs={'conn':conn2, 'temperature':temperature, 'slow':slow, 'diagonal_model_covariance':diagonal_model_covariance, 'verbose':verbose})
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

def cvlogLike_mcmc(
        models,
        stitch,
        min_sigmaANDmax_sigma,
        min_lANDmax_l,
        min_sigma_obsANDmax_sigma_obs,
        min_model_multiplierANDmax_model_multiplier,
        num_samples=gp.DEFAULT_NUM,
        num_walkers=DEFAULT_NUM_WALKERS,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        model_multiplier_prior='log',
        degree=1,
        temperature=DEFAULT_TEMPERATURE,
        slow=False,
        diagonal_model_covariance=False,
    ):

    min_sigma, max_sigma = min_sigmaANDmax_sigma
    min_l, max_l = min_lANDmax_l
    min_sigma_obs, max_sigma_obs = min_sigma_obsANDmax_sigma_obs
    min_model_multiplier, max_model_multiplier = min_model_multiplierANDmax_model_multiplier

    if emcee is None:
        raise ImportError('could not import emcee')

    if sigma_prior=='log':
        ln_sigma_prior = lambda sigma: 1./sigma if min_sigma<sigma<max_sigma else -np.infty
    elif sigma_prior=='lin':
        ln_sigma_prior = lambda sigma: 0 if min_sigma<sigma<max_sigma else -np.infty
    else:
        raise ValueError('unkown sigma_prior='+sigma_prior)

    if l_prior=='log':
        ln_l_prior = lambda l: 1./l if min_l<l<max_l else -np.infty
    elif l_prior=='lin':
        ln_l_prior = lambda l: 0 if min_l<l<max_l else -np.infty
    else:
        raise ValueError('unkown l_prior='+l_prior)

    if sigma_obs_prior=='log':
        ln_sigma_obs_prior = lambda sigma_obs: 1./sigma_obs if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    elif sigma_obs_prior=='lin':
        ln_sigma_obs_prior = lambda sigma_obs: 0 if min_sigma_obs<sigma_obs<max_sigma_obs else -np.infty
    else:
        raise ValueError('unkown sigma_obs_prior='+sigma_obs_prior)

    if model_multiplier_prior=='log':
        ln_model_multiplier_prior = lambda m: 1./m if min_model_multiplier<m<max_model_multiplier else -np.infty
    elif model_multiplier_prior=='lin':
        ln_model_multiplier_prior = lambda m: 0 if min_model_multiplier<m<max_model_multiplier else -np.infty
    else:
        raise ValueError('unknown model_multiplier_prior='+model_multiplier_prior)

    beta = 1./temperature
    foo = lambda args: _cvlogLike_worker(
            models,
            stitch,
            [args[0]**2],
            [l2args[1]**2],
            [args[2]**2],
            [args[3]],
            [degree],
            temperature=temperature,
            slow=slow,
            diagonal_model_covariance=diagonal_model_covariance
        )[0,-1] \
        + ln_sigma_prior(args[0]) \
        + ln_l_prior(args[1]) \
        + ln_sigma_obs_prior(args[2]) \
        + ln_model_multiplier_prior(args[3])

    x0 = np.array([(min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_obs*max_sigma_obs)**0.5, (min_model_multiplier*max_model_multiplier)**0.5])
    x0 = emcee.utils.sample_ball(
        x0,
        x0*0.1,
        size=num_walkers,
    )

    sampler = emcee.EnsembleSampler(num_walkers, 4, foo)
    sampler.run_mcmc(x0, num_samples)

    sigma = sampler.chain[:,:,0]
    l = sampler.chain[:,:,1]
    sigma_obs = sampler.chain[:,:,2]
    model_multiplier = sampler.chain[:,:,3]
    lnprob = sampler.lnprobability

    return np.array(
        [(sigma[i,j], l[i,j], sigma_obs[i,j], model_multiplier[i,j], lnprob[i,j]) for j in range(num_samples) for i in range(num_walkers)],
        dtype=CVSAMPLES_DTYPE,
    )
