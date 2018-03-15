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

__samples_dtype__ = [('sigma','float'), ('l','float'), ('sigma_noise','float'), ('logLike','float')]

#-------------------------------------------------
### methods useful for a basic squared-exponential kernel
#-------------------------------------------------

def logLike_grid(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_noise, max_sigma_noise),
        num_sigma=__default_num__,
        num_l=__default_num__,
        num_sigma_noise=__default_num__,
    ):
    '''
    compute logLike on a grid and return "samples" with associated logLike values corresponding to each grid point
    We space points logarithmically for sigma and sigma_noise, linearly for l
    '''
    ### compute grid
    sigma = np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma)
    l = np.linspace(min_l, max_l, num_l)
    sigma_noise = np.logspace(np.log10(min_sigma_noise), np.log10(max_sigma_noise), num_sigma_noise)

    SIGMA, L, SIGMA_NOISE = np.meshgrid(sigma, l, sigma_noise, indexing='ij')

    # flatten for ease of iteration
    SIGMA = SIGMA.flatten()
    L = L.flatten()
    SIGMA_NOISE = SIGMA_NOISE.flatten()

    ### iterate over grid points and copmute logLike for each
    return np.array(
        [(s, l, sn, gp.logLike(f_obs, x_obs, sigma2=s**2, l=l**2, sigma2_noise=sn**2)) for s, l, sn in zip(SIGMA, L, SIGMA_NOISE)],
        dtype=__samples_dtype__,
    )

def logLike_mcmc(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_noise, max_sigma_noise),
        num_samples=__default_num__,
        num_walkers=__default_num_walkers__,
    ):
    '''
    draw samples from the target distribution defined by logLike using emcee
    return samples with associated values of logLike
    '''
    x0 = (min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_noise*max_sigma_noise)**0.5

    sampler = emcee.EnsembleSampler(
        num_walkers,
        3, ### ndim
        lambda args: gp.logLike(f_obs, x_obs, sigma2=args[0], l2=args[1], sigma2_noise=args[2]),
    )
    sampler.run_mcmc(x0, num_samples)
    return np.array(
        [(sigma2**0.5, l2**0.5, sigma2_noise**2, logL) for (sigma2, l2, sigma2_noise), logL in zip(sampler.flatchain, sampler.lnprobability)],
        dtype=__damples_dtype__,
    )

def logLike_maxL(
        f_obs,
        x_obs,
        (min_sigma, max_sigma),
        (min_l, max_l),
        (min_sigma_noise, max_sigma_noise),
        method=__default_method__,
        tol=__default_tol__,
    ):
    '''
    find the maximum logLikelihood as a function of hyperparamters.
    return a single "sample" with associated logLike
    '''
    min_sigma2 = min_sigma**2
    max_sigma2 = max_sigma**2
    min_l2 = min_l**2
    max_l2 = max_l**2
    min_sigma2_noise = min_sigma_noise**2
    max_sigma2_noise = max_sigma_noise**2

    x0 = (min_sigma*max_sigma)**0.5, (min_l+max_l)*0.5, (min_sigma_noise*max_sigma_noise)**0.5

    res = optimize.minimize(
        lambda args: -gp.logLike(f_obs, x_obs, sigma2=args[0], l=args[1], sigma2_noise=args[2]), 
        x0,
        method=method,
        jac=lambda args: -gp.grad_logLike(f_obs, x_obs, sigma2=args[0], l=args[1], sigma2_noise=args[2]),
        bounds=[
            (min_sigma2, max_sigma2),
            (min_l2, max_l2),
            (min_sigma2_noise, max_sigma2_noise),
        ],
        constraints=[
            {'type':'eq', 'fun':lambda args:min_sigma2<=args[0]<=max_sigma2},
            {'type':'eq', 'fun':lambda args:min_l2<=args[1]<=max_l2},
            {'type':'eq', 'fun':lambda args:min_sigma2_noise<=args[2]<=max_sigma2_noise},
        ],
        tol=tol,
    )
    sigma2, l2, sigma2_noise = res.x
    return np.array(
        [(sigma2**0.5, l2**0.5, sigma2_noise**0.5, gp.logLike(f_obs, x_obs, sigma2=sigma2, l2=l2, sigma2_noise=sigma2_noise))],
        dtype=__samples_dtype__,
    )