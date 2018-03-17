__doc__ = "a module to help select hyper paramters. Delegates a lot to gaussianprocess.py"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np
from universality import gaussianprocess as gp

import emcee
from scipy import optimize

#-------------------------------------------------

__default_num__ = 101
__default_num_walkers__ = 50
__default_method__ = 'BFGS'
__default_tol__ = None

__samples_dtype__ = [('sigma','float'), ('l','float'), ('sigma_obs','float'), ('logLike','float')]

#-------------------------------------------------
### methods useful for a basic squared-exponential kernel
#-------------------------------------------------

def logLike_grid(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        num_sigma=__default_num__,
        num_l=__default_num__,
        num_sigma_obs=__default_num__,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        degree=1,
    ):
    '''
    compute logLike on a grid and return "samples" with associated logLike values corresponding to each grid point
    We space points logarithmically for sigma and sigma_obs, linearly for l
    '''
    ### compute grid
    if sigma_prior=='log':
        sigma = np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma)
    elif sigma_prior=='lin':
        sigma = np.linspace(min_sigma, max_sigma, num_sigma)
    else:
        raise ValueError, 'unknown sigma_prior='+sigma_prior

    if l_prior=='log':
        l = np.logspace(np.log10(min_l), np.log10(max_l), num_l)
    elif l_prior=='lin':
        l = np.linspace(min_l, max_l, num_l)
    else:
        raise ValueError, 'unkown l_prior='+l_prior

    if sigma_obs_prior=='log':
        sigma_obs = np.logspace(np.log10(min_sigma_obs), np.log10(max_sigma_obs), num_sigma_obs)
    elif sigma_obs_prior=='lin':
        sigma_obs = np.linspace(min_sigma_obs, max_sigma_obs, num_sigma_obs)
    else:
        raise ValueError, 'unkown sigma_obs_prior='+sigma_obs_prior

    SIGMA, L, SIGMA_NOISE = np.meshgrid(sigma, l, sigma_obs, indexing='ij')

    # flatten for ease of iteration
    SIGMA = SIGMA.flatten()
    L = L.flatten()
    SIGMA_NOISE = SIGMA_NOISE.flatten()

    ### iterate over grid points and copmute logLike for each
    return np.array(
        [(s, l, sn, gp.logLike(f_obs, x_obs, sigma2=s**2, l2=l**2, sigma2_obs=sn**2, degree=degree)) for s, l, sn in zip(SIGMA, L, SIGMA_NOISE)],
        dtype=__samples_dtype__,
    )

def logLike_mcmc(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        num_samples=__default_num__,
        num_walkers=__default_num_walkers__,
        sigma_prior='log',
        sigma_obs_prior='log',
        l_prior='lin',
        degree=1,
    ):
    '''
    draw samples from the target distribution defined by logLike using emcee
    return samples with associated values of logLike
    '''
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

    foo = lambda args: gp.logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=1) \
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
        dtype=__samples_dtype__,
    )

def logLike_maxL(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_obs, max_sigma_obs),
        method=__default_method__,
        tol=__default_tol__,
        degree=1,
    ):
    '''
    find the maximum logLikelihood as a function of hyperparamters.
    return a single "sample" with associated logLike
    '''
    min_sigma2 = min_sigma**2
    max_sigma2 = max_sigma**2
    min_l2 = min_l**2
    max_l2 = max_l**2
    min_sigma2_obs = min_sigma_obs**2
    max_sigma2_obs = max_sigma_obs**2

    x0 = (min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_obs*max_sigma_obs)**0.5

    foo = lambda args: -gp.logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=degree)
    jac = lambda args: -np.array(gp.grad_logLike(f_obs, x_obs, sigma2=args[0]**2, l2=args[1]**2, sigma2_obs=args[2]**2, degree=degree))

    res = optimize.minimize(
        foo,
        x0,
        method=method,
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

    sigma, l, sigma_obs = res.x
    return np.array(
        [(sigma, l, sigma_obs, -foo(res.x))],
        dtype=__samples_dtype__,
    )
