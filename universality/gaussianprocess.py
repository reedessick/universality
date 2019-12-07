__doc__ = "a module that houses simple Gaussian Process routines"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import sys
import numpy as np
import h5py
import pickle

import time

from universality import utils

#-------------------------------------------------

### defaults
DEFAULT_SIGMA = 0.1
DEFAULT_L = 0.1

DEFAULT_SIGMA2 = DEFAULT_SIGMA**2
DEFAULT_L2 = DEFAULT_L**2

DEFAULT_POLY_DEGREE = 1
DEFAULT_MODEL_MULTIPLIER = 1

DEFAULT_NUM = 51

#-------------------------------------------------
# i/o specific for storing processes
#-------------------------------------------------

def pkldump(path, xlabel, ylabel, x_tst, mean, cov):
    with open(path, 'w') as file_obj:
        pickle.dump(xlabel, file_obj) 
        pickle.dump(ylabel, file_obj) 
        pickle.dump(x_tst, file_obj) 
        pickle.dump(mean, file_obj) 
        pickle.dump(cov, file_obj) 

def pklload(path):
    with open(path, 'r') as file_obj:
        xlabel = pickle.load(file_obj)
        ylabel = pickle.load(file_obj)
        x_tst = pickle.load(file_obj)
        mean = pickle.load(file_obj)
        cov = pickle.load(file_obj)
    return xlabel, ylabel, x_tst, mean, cov

#-------------------------------------------------
# hdf5 process files
#-------------------------------------------------

def create_process_group(group, poly_degree, sigma, length_scale, sigma_obs, x_tst, f_tst, cov_f_f, xlabel='xl', flabel='f', weight=1., model_multiplier=None):
    """helper funtion to record the data about a process in a mixture model"""
    group.attrs.create('weight', weight)
    group.attrs.create('poly_degree', poly_degree)
    group.attrs.create('sigma', sigma)
    group.attrs.create('length_scale', length_scale)
    group.attrs.create('sigma_obs', sigma_obs)
    if model_multiplier is not None:
        group.attrs.create('model_multiplier', model_multiplier)

    group.attrs.create('xlabel', xlabel)
    group.attrs.create('flabel', flabel)

    means = group.create_dataset('mean', data=np.array(zip(x_tst, f_tst), dtype=[(xlabel, 'float'), (flabel, 'float')]))
    cov = group.create_dataset('cov', data=cov_f_f)

def hdf5load(path):
    model = []
    with h5py.File(path, 'r') as obj:
        for key in obj.keys(): ### iterate over groups
            weight, x, f, cov, (xlabel, flabel), (p, s, l, S, m) = parse_process_group(obj[key])
            model.append({
                'weight':weight,
                'x':x,
                'f':f,
                'cov':posdef(cov),
                'labels':{'xlabel':xlabel, 'flabel':flabel},
                'hyperparams':{
                    'poly_degree':p,
                    'sigma':s,
                    'length_scale':l,
                    'sigma_obs':S,
                    'model_multiplier':m,
                },
            })
    return model

def parse_process_group(group):
    """helper function to read stuff out of our hdf5 data structures
    return weight, x_tst, f_tst, cov_f_f, (xlabel, flabel), (poly_degree, sigma, length_scale, sigma_obs)
    """
    attrs = group.attrs
    weight = attrs['weight']

    xlabel = attrs['xlabel']
    flabel = attrs['flabel']

    poly_degree = attrs['poly_degree']
    sigma = attrs['sigma']
    length_scale = attrs['length_scale']
    sigma_obs = attrs['sigma_obs']
    m = attrs['model_multiplier'] if ('model_multiplier' in attrs.keys()) else None

    x_tst = group['mean'][xlabel]
    f_tst = group['mean'][flabel]

    cov_f_f = posdef(group['cov'][...]) ### run this through our algorithm to guarantee it is postive definite

    return weight, x_tst, f_tst, cov_f_f, (xlabel, flabel), (poly_degree, sigma, length_scale, sigma_obs, m)

#-------------------------------------------------
# convenience functions for sanity checking
#-------------------------------------------------

def num_dfdx(x_obs, f_obs):
    '''
    estimate the derivative numerically
    '''
    df = f_obs[1:] - f_obs[:-1]
    dx = x_obs[1:] - x_obs[:-1]

    dfdx = np.empty_like(f_obs, dtype='float')

    dfdx[0] = df[0]/dx[0]   # handle boundary conditions as special cases
    dfdx[-1] = df[-1]/dx[-1]

    dfdx[1:-1] = 0.5*(df[:-1]/dx[:-1] + df[1:]/dx[1:]) ### average in the bulk

    return dfdx

#-------------------------------------------------
# covariance kernels
#-------------------------------------------------

def cov_f1_f2(x1, x2, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2):
    '''
    cov(f1, f2) = sigma2 * np.exp(-(x1-x2)**2/(2*l2))
    '''
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return sigma2 * np.exp(-0.5*(X1-X2)**2/l2)

def cov_df1dx1_f2(x1, x2, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2):
    '''
    cov(df1/dx1, f2) = -sigma2 * (x1-x2)/l**2 * np.exp(-(x1-x2)**2/(2*l2))
    '''
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return -(X1-X2)/l2 * cov_f1_f2(x1, x2, sigma2=sigma2, l2=l2)

def cov_f1_df2dx2(x1, x2, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2):
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return (X1-X2)/l2 * cov_f1_f2(x1, x2, sigma2=sigma2, l2=l2)

def cov_df1dx1_df2dx2(x1, x2, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2):
    '''
    cov(df1/dx1, df2/dx2) = -sigma2 (x1-x2)**2/l2**2 * np.exp(-(x1-x2)**2/(2*l2)) + sigma2 / l2 * np.exp(-(x1-x2)**2/(2*l2))
    '''
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return (l2 - (X1-X2)**2)/l2**2 * cov_f1_f2(x1, x2, sigma2=sigma2, l2=l2)

#-------------------------------------------------
# GPR likelihoods and jacobians thereof
#-------------------------------------------------

def _target_in_source(target, source):
    """returns a boolean array of the same length as target corresponding to the values of target that are in source
    """
    return np.array([_ in source for _ in target], dtype=bool)

def _intersect_models(x_obs, f_obs, invcov_obs, x_prb, f_prb, invcov_prb):
    """helper function that returns the correct (x, f, invcov) for each model corresponding to the x values both share
    useful for model_logprob, and draw_logprob

    return f_obs, invcov_obs, f_prb, invcov_prb
    """
    t_obs = _target_in_source(x_obs, x_prb)
    f_obs = f_obs[t_obs]
    if invcov_obs is not None:
        invcov_obs = _extract_invsubset_from_invcov(t_obs, invcov_obs)

    t_prb = _target_in_source(x_prb, x_obs)
    f_prb = f_prb[t_prb]
    invcov_prb = _extract_invsubset_from_invcov(t_prb, invcov_prb)

    return f_obs, invcov_obs, f_prb, invcov_prb

def _extract_subset_from_matrix(bool_keep, matrix):
    if np.all(bool_keep):
        return matrix
    else:
        n = np.sum(bool_keep)
        return matrix[np.outer(bool_keep, bool_keep)].reshape((n,n))

def _extract_invsubset_from_invcov(bool_keep, invcov):
    """return the inverse of a subset of cov by extracting it from invcov

    we first re-order invcov so that the indecies we want to keep are in the lower-right block-diagonal
    we then use the identity that if
      Cov = [[p, q], [r, s]]
    and
     inv(Cov) = [[P, Q], [R, S]]
    then inv(p) = P - Qinv(S)R
    """
    if np.all(bool_keep): ### nothing to do
        return invcov
    else:
        n = np.sum(bool_keep)
        N = len(bool_keep)-n
        bool_lose = np.logical_not(bool_keep)
        return invcov[np.outer(bool_keep,bool_keep)].reshape((n,n)) \
               - np.dot(
                   invcov[np.outer(bool_keep,bool_lose)].reshape((n,N)), \
                   np.dot(
                       np.linalg.inv(invcov[np.outer(bool_lose,bool_lose)].reshape(N,N)),
                       invcov[np.outer(bool_lose, bool_keep)].reshape((N,n))
                   )
               )

def _extract_invconditioned_from_invcov(bool_keep, invcov):
    return _extract_subset_from_matrix(bool_keep, invcov)

def _extract_conditioned_from_invcov(bool_keep, invcov):
    return np.linalg.inv(_extract_invconditioned_from_invcov(bool_keep, invcov))

def _extract_invconditioned_mean_from_invcov(bool_keep, mean, invcov):
    if np.all(bool_keep):
        return mean, invcov
    else:
        n = np.sum(bool_keep)
        N = len(bool_keep)-n
        P = invcov[np.outer(bool_keep, bool_keep)].reshape((n,n)) ### do not delegate to _extract_invconditioned_from_invcov to avoid repeated conditional
        Q = invcov[np.outer(bool_keep, np.logical_not(bool_keep))].reshape((n,N))
        return -np.dot(np.linalg.inv(P), np.dot(Q, mean)), P

def logprob(x_obs, f_obs, x_prb, f_prb, cov_prb, cov_obs=None):
    '''
    compute the probablity of seeing f_obs (measurement uncertainty encapsulated in cov_obs) given a process described by (f_prob, cov_prob)
    assumes f_obs and f_prob are sampled at the same abscissa

    note, the default assumes f_obs is perfectly measured (cov_obs=None). If this is not the case, specify the covariance matrix for f_obs via the cov_obs kwarg
    '''
    ### make sure everything corresponds to consistent x values
    if len(x_obs)!=len(x_prb) or (not np.all(x_obs!=x_prb)):
        t_obs = _target_in_source(x_obs, x_prb)
        f_obs = f_obs[t_obs]
        if cov_obs is not None:
            cov_obs = cov_obs[np.outer(t_obs,t_obs)].reshape((len(f_obs),)*2)

        t_prb = _target_in_source(x_prb, x_obs)
        f_prb = f_prb[t_prb]
        cov_prb = cov_prb[np.outer(t_prb, t_prb)].reshape((len(f_prb),)*2)

    return _logprob(
        f_obs,
        f_prb,
        np.linalg.inv(cov_prb),
        invcov_obs=None if cov_obs is None else np.linalg.inv(cov_obs),
    )

def _logprob(f_obs, f_prb, invcov_prb, invcov_obs=None, verbose=False):
    if invcov_obs is None:
        invcov = invcov_prb
    elif np.all(invcov_prb==invcov_obs): ### take a shortcut to avoid matrix inverses
        invcov = 0.5*invcov_prb
    else:
        invcov = posdef(np.dot(invcov_obs, np.dot(np.linalg.inv(invcov_obs + invcov_prb), invcov_prb)))
    return _logLike(f_obs-f_prb, invcov, verbose=verbose)

def model_logprob(model_obs, model_prb):
    '''
    a convenience function to do the combinatorics stuff to marginalize over the weights stored in the models

    NOTE: assumes models have properly normalized weights
    '''
    # invert all matricies only once
    obs = [(obs['x'], obs['f'], np.linalg.inv(obs['cov']) if obs['cov'] is not None else None, np.log(obs['weight'])) for obs in model_obs]
    prb = [(prb['x'], prb['f'], np.linalg.inv(prb['cov']), np.log(prb['weight'])) for prb in model_prb]

    logscore = -np.infty
    for x_obs, f_obs, invcov_obs, logw_obs in obs:
        for x_prb, f_prb, invcov_prb, logw_prb in prb:
            fo, ico, fp, icp = _intersect_models(x_obs, f_obs, invcov_obs, x_prb, f_prb, invcov_prb) 
            _logscore = _logprob(fo, fp, icp, invcov_obs=ico) + logw_obs + logw_prb ### the conditioned probability multiplied by the 2 weights
            m = max(logscore, _logscore)
            logscore = np.log(np.exp(logscore-m) + np.exp(_logscore-m)) + m

    return logscore

def draw_logprob(model, size=1, return_realizations=False, verbose=False):
    '''
    a convenience function to compute the probability of a bunch of realizations from a model
    '''
    logscores = []
    if return_realizations:
        realizations = []

    prb = [(prb['x'], prb['f'], np.linalg.inv(prb['cov']), np.log(prb['weight'])) for prb in model] ### only invert the matricies once
    if verbose:
        tmp = '\r %'+'%d'%(np.int(np.log10(size))+1)+'d / '+'%d'%size
        i = 1
    for ind in utils.draw_from_weights(np.array([m['weight'] for m in model]), size=size): ### draw a realization from the set of weights
        if verbose:
            sys.stdout.write(tmp%i)
            sys.stdout.flush()
            i += 1

        m = model[ind]
        f_obs = np.random.multivariate_normal(m['f'], m['cov']) ### draw the realization
        x_obs = m['x']
        if return_realizations:
            realizations.append((x_obs, f_obs))

        ### compute the score
        logscore = -np.infty
        for x_prb, f_prb, invcov_prb, logw_prb in prb:
            fo, _, fp, icp = _intersect_models(x_obs, f_obs, None, x_prb, f_prb, invcov_prb) 
            _logscore = _logprob(fo, fp, icp) + logw_prb ### the conditioned probability multiplied by the 2 weights
            m = max(logscore, _logscore)
            logscore = np.log(np.exp(logscore-m) + np.exp(_logscore-m)) + m

        logscores.append(logscore)

    if verbose:
        sys.stdout.write('\n')

    logscores = np.array(logscores)
    if return_realizations:
        return logscores, realizations
    else:
        return logscores

def logLike(f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2, degree=1):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    if (sigma2<=0) or (l2<=0) or (sigma2_obs<=0):
        return np.infty

    cov = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    f_fit, _ = poly_model(np.array([]), f_obs, x_obs, degree=degree)

    ### compute the logLikelihood
    N = len(x_obs)

    # compute components separately because they each require different techniques to stabilize them numerically
    return _logLike(f_obs-f_fit, np.linalg.inv(cov))

def _logLike(f_obs, invcov, verbose=False):
    # because the covariances might be so small, and there might be a lot of data points, we need to handle the determinant with care
    sign, det = np.linalg.slogdet(invcov)
    if sign<0: # do this first so we don't waste time computing other stuff that won't matter
        return -np.infty ### rule this out by setting logLike -> -infty
    det *= 0.5

    # now compute inner product in the diagonal basis
    obs = -0.5*np.dot(f_obs, np.dot(invcov, f_obs))
    if obs > 0:
        raise ValueError('unphysical value for inner product: %.6e (s=%d logdet=%.6e)'%(obs, sign, det))

    n = len(f_obs)
    nrm = -0.5*n*np.log(2*np.pi)

    if verbose:
        print((obs, det, nrm))

    return obs + det + nrm ### assemble components and return

def grad_logLike(f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2, degree=1):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    if (sigma2<=0) or (l2<=0) or (sigma2_obs<=0):
        raise ValueError('unphysical hyperparameters!')

    f_fit, _ = poly_model(np.array([]), f_obs, x_obs, degree=degree) 

    invcov_obs_obs = np.linalg.inv(_cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs))
    a = np.dot(invcov_obs_obs, f_obs-f_fit)

    ### compute the gradient with each hyperparameter
    m = np.outer(a,a) - invcov_obs_obs
    dlogL_dsigma2 = 0.5*np.trace(np.dot(m, _dcov_dsigma2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dl2     = 0.5*np.trace(np.dot(m, _dcov_dl2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dsigma2_obs = 0.5*np.trace(np.dot(m, _dcov_dsigma2_obs(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))

    return dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs

#-------------------------------------------------
# GPR via conditioning given a set of observation and a covariance matrix
#-------------------------------------------------

def posdef(cov, epsilon=1e-6):
    '''
    identifies the nearest positive semi-definite symmetric matrix and returns it
    '''
    ### NOTE: the following is based on Hingham (1988)
    cov[:] = 0.5*(cov+cov.T) # make sure this is symmetric

    nug = 0.
    eye = np.diag(np.ones(len(cov), dtype=float))
    try:
        np.linalg.cholesky(cov)

    except np.linalg.linalg.LinAlgError: ### not pos-semi definite
        ### find smallest nugget that renders this positive semi-definite
        nug = 1e-10
        while True:
            try:
                np.linalg.cholesky(cov+nug*eye)
                break
            except np.linalg.linalg.LinAlgError:
                nug *= 2

        # perform a bisection search to find the smallest possible nug
        low = nug/2
        while (nug-low) > 0.5*(nug+low)*epsilon:
            mid = (nug*low)**0.5
            try:
                np.linalg.cholesky(cov+mid*eye)
                nug = mid
            except np.linalg.linalg.LinAlgError:
                low = mid

    finally: ### return the "corrected" maxtrix
        return cov + nug*eye
        
    ### NOTE: the following tends to introduce nans because eigh assumes things about cov that are not true when it's wonky
#    cov[:] = 0.5*(cov+cov.T) ### re-use memory and symmetrize this thing to try to average away numerical errors
#
#    val, vec = np.linalg.eigh(cov)                     ### take spectral decomposition
#    epsilon *= np.max(val)                             ### normalize this
#    val[val<epsilon] = epsilon                         ### regularlize eigenvalue
#    cov[:] = np.dot(vec, np.dot(np.diag(val), vec.T))  ### compute the matrix with the regularlize eigenvalues
#
#    sign, det = np.linalg.slogdet(cov)
#    if sign!=1:
#        print('***WARNING*** non-physical conditioned covariance matrix detected! sign=%s; log|det|=%s'%(sign, det))
#
#    return cov

def gpr(f_obs, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs):
    '''
    constructs the parameters for the conditional distribution: f_tst|f_obs,x_obs,x_text based on cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs
        cov_tst_tst : (N_tst, N_tst) the covariance matrix between test samples in the joint distribution
        cov_tst_obs : (N_tst, N_obs) the covariance matrix between test samples and observed samples in the joint distribution
        cov_obs_obs : (N_obs, N_obs) the covariance matrix between observed samples in the joint distribution
    returns the mean_tst, cov_tst_tst
    '''
    ### invert matix only once. This is the expensive part
    invcov_obs_obs = np.linalg.inv(cov_obs_obs)

    ### do some matrix multiplcation here
    mean = np.dot(cov_tst_obs, np.dot(invcov_obs_obs, f_obs))
    # NOTE: may not be positive semidefinite due to floating point errors! so we run it through a routine to clean this up
    cov  = posdef(cov_tst_tst - np.dot(cov_tst_obs, np.dot(invcov_obs_obs, cov_obs_tst)))
    logweight = _logLike(f_obs, invcov_obs_obs) ### the logweight associated with the observed data assuming this covariance matrix

    return mean, cov, logweight

def _cov(x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    a helper function that computes the covariance matrix for observed points
    '''
    cov_obs_obs = cov_f1_f2(x_obs, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_obs += np.diag(np.ones(len(x_obs)))*sigma2_obs
    return cov_obs_obs

def _dcov_dsigma2(x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    a helper function for the derivative of cov_obs_obs with respect to sigma2
    '''
    X1, X2 = np.meshgrid(x_obs, x_obs, indexing='ij')
    return np.exp(-0.5*(X1-X2)**2/l2) ### simple derivative of cov_f1_f2

def _dcov_dl2(x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    a helper function for the derivative of cov_obs_obs with respect to l2
    '''
    X1, X2 = np.meshgrid(x_obs, x_obs, indexing='ij')
    Z = 0.5*(X1-X2)**2/l2
    return sigma2*np.exp(-Z)*Z/l2 ### simple derivative of cov_f1_f2

def _dcov_dsigma2_obs(x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    a helper function for the derivative of cov_obs_obs with respect to sigma2_obs
    '''
    return np.diag(np.ones(len(x_obs))) ### derivative of diagonal, white noise

def gpr_f(x_tst, f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    constructs covariance for f_tst|f_obs,x_obs,x_tst
    returns mean_tst, cov_tst_tst
    '''
    ### compute covariances
    cov_tst_tst = cov_f1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_obs = cov_f1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_obs = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### delegate
    return gpr(f_obs, cov_tst_tst, cov_tst_obs, np.transpose(cov_tst_obs), cov_obs_obs)

def gpr_dfdx(x_tst, f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    constructs covariance needed for df_tst/dx_tst|f_obs,x_obs,x_tst
    return mean_tst, cov_tst_tst
    '''
    ### compute covariances
    cov_tst_tst = cov_df1dx1_df2dx2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_obs = cov_df1dx1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_tst = cov_f1_df2dx2(x_obs, x_tst, sigma2=sigma2, l2=l2)
    cov_obs_obs = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### delegate
    return gpr(f_obs, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)

def gpr_f_dfdx(x_tst, f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2):
    '''
    constructs covariance needed for f_tst,dfdx_tst|f_obs,x_obs,x_tst
    return mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx
    '''
    ### compute covariances
    Ntst = len(x_tst)
    NTST = 2*Ntst
    Nobs = len(x_obs)

    # covariance between test points
    cov_tst_tst = np.empty((NTST,NTST), dtype='float')
    cov_tst_tst[:Ntst,:Ntst] = cov_f1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_tst[:Ntst,Ntst:] = cov_f1_df2dx2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_tst[Ntst:,:Ntst] = cov_df1dx1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_tst[Ntst:,:Ntst] = cov_df1dx1_df2dx2(x_tst, x_tst, sigma2=sigma2, l2=l2)

    # covariance between test and observation points
    cov_tst_obs = np.empty((NTST,Nobs), dtype='float')
    cov_tst_obs[:Ntst,:] = cov_f1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
    cov_tst_obs[Ntst:,:] = cov_df1dx1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)

    # covariance between observation and test points
    cov_obs_tst = np.empty((Nobs,NTST), dtype='float')
    cov_obs_tst[:,:Ntst] = cov_f1_f2(x_obs, x_tst, sigma2=sigma2, l2=l2)
    cov_obs_tst[:,Ntst:] = cov_f1_df2dx2(x_obs, x_tst, sigma2=sigma2, l2=l2)

    # covariance between observation points
    cov_obs_obs = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### delegate to compute conditioned process
    mean, cov, logweight = gpr(f_obs, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)

    ### slice the resulting arrays and return
    ### relies on the ordering we constructed within our covariance matricies!
    #        mean_f      mean_dfdx       cov_f_f          cov_f_dfdx        cov_dfdx_f      cov_dfdx_dfdx
    return mean[:Ntst], mean[Ntst:], cov[:Ntst,:Ntst], cov[:Ntst,Ntst:], cov[Ntst:,:Ntst], cov[:Ntst,:Ntst], logweight

#-------------------------------------------------
# specific utilities for "one-stop shop" scripts
#-------------------------------------------------

def poly_model(x_tst, f_obs, x_obs, degree=1):
    '''
    fit a polynomial model to the data
    return f_fit, f_tst
    '''
    f_fit = np.zeros_like(x_obs, dtype='float')
    f_tst = np.zeros_like(x_tst, dtype='float')
    poly = np.polyfit(x_obs, f_obs, degree)
    for i in xrange(degree+1):
        f_fit += poly[-1-i]*x_obs**i
        f_tst += poly[-1-i]*x_tst**i
    return f_fit, f_tst

def poly_model_f_dfdx(x_tst, f_obs, x_obs, degree=1):
    '''
    like poly_model, except
    return f_fit, dfdx_tst
    '''
    f_fit = np.zeros_like(x_obs, dtype='float')
    f_tst = np.zeros_like(x_tst, dtype='float')
    dfdx_tst = np.zeros_like(x_tst, dtype='float')
    poly = np.polyfit(x_obs, f_obs, degree)
    for i in xrange(degree+1):
        f_fit += poly[-1-i]*x_obs**i
        f_tst += poly[-1-i]*x_tst**i
        dfdx_tst += i*poly[-1-i]*x_tst**(i-1)
    return f_fit, f_tst, dfdx_tst

def gpr_resample(x_tst, f_obs, x_obs, degree=1, guess_sigma2=DEFAULT_SIGMA2, guess_l2=DEFAULT_L2, guess_sigma2_obs=DEFAULT_SIGMA2):
    '''
    resample the data via GPR to samples along x_tst
    performs automatic optimization to find the best hyperparameters along with subtracting out f_fit from a polynomial model
    '''
    ### pre-condition the data
    f_fit, f_tst = poly_model(x_tst, f_obs, x_obs, degree=degree)

    ### perform GPR in an optimization loop to find the best logLike
    ### FIXME: use this to figure out best hyper-parameters, but for now just take some I know work reasonably well
    sigma2 = guess_sigma2
    l2 = guess_l2
    sigma2_obs = guess_sigma2_obs

    ### perform GPR with best hyperparameters to infer the function at x_tst
    mean, cov, logweight = gpr_f(x_tst, f_obs-f_fit, x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)
    mean += f_tst ### add the polyfit model back in 

    return mean, cov, logweight

def gpr_resample_f_dfdx(x_tst, f_obs, x_obs, degree=1, guess_sigma2=DEFAULT_SIGMA2, guess_l2=DEFAULT_L2, guess_sigma2_obs=DEFAULT_SIGMA2):
    '''
    the same as gpr_resample, except we regress out both the function and the derivative of the function instead of the just the function
    resample the data via GPR to samples along x_tst
    performs automatic optimization to find the best hyperparameters along with subtracting out f_fit from a polynomial model
    '''
    ### pre-condition the data
    f_fit, f_tst, dfdx_tst = poly_model_f_dfdx(x_tst, f_obs, x_obs, degree=degree)

    ### perform GPR in an optimization loop to find the best logLike
    ### FIXME: use this to figure out best hyper-parameters, but for now just take some I know work reasonably well
    sigma2 = guess_sigma2
    l2 = guess_l2
    sigma2_obs = guess_sigma2_obs

    ### perform GPR with best hyperparameters to infer the function at x_tst
    mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx, logweight = gpr_f_dfdx(x_tst, f_obs-f_fit, x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    # add the polyfit model back in
    mean_f += f_tst
    mean_dfdx += dfdx_tst

    return mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx, logweight

def mean_phi(x_tst, mean_f, mean_dfdx):
    '''
    compute the mean of the process for phi =  log(de/dp-1) = log((exp(f)/x)*dfdx - 1) by assuming phi can be approximated by a 1st order Taylor expansion in the neighborhood of each x_tst. 
    This assumption makes the resulting process on phi Gaussian with a straightforward covariance matrix

    NOTE: this is pretty fine-tuned for what I'm doing in this project, so it may not be useful elsewhere, but that's probably fine...

    return mean_phi
    '''
    ans = np.exp(mean_f)/np.exp(x_tst) * mean_dfdx - 1 ### NOTE: there might be issues with numerical stability here...
    truth = ans>0 ### FIXME? only these points are causal, so we return this boolean array for filtering on the user's side
    return np.log(ans), truth

def cov_phi_phi(x_tst, mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx):
    '''
    compute the covariance matrix for the process for phi = log(de/dp-1) = log((exp(f)/x)*dfdx - 1) by assuming phi can be approximated by a 1st order Taylor expansion in the neighborhood of each x_tst. 
    This assumption makes the resulting process on phi Gaussian with a straightforward covariance matrix

    NOTE: this is pretty fine-tuned for what I'm doing in this project, so it may not be useful elsewhere, but that's probably fine...

    return cov_phi_phi
    '''
    ### compute partial derivatives of phi evaluated at the means
    # compute some useful values from the means
    expf = np.exp(mean_f)
    expx = np.exp(x_tst)

    ratio = expf/expx
    denom = (ratio*mean_dfdx - 1)

    # construct the actual partial dervatives
    dphidf = (denom+1) / denom
    dphiddfdx = ratio / denom

    ### construct the covariance matrix
    # this is the product of outer-products between the partials and the covariance matricies
    # I break the computation up like this for clarity within the code, not speed
    Ntst = len(x_tst)
    cov = np.zeros((Ntst,Ntst), dtype='float')

    cov += np.outer(dphidf, dphidf)*cov_f_f              ### contribution from Cov(f,f) 
    cov += np.outer(dphidf, dphiddfdx)*cov_f_dfdx        ### contribution from Cov(f,df/dx)
    cov += np.outer(dphiddfdx, dphidf)*cov_dfdx_f        ### contribution from Cov(df/dx,f)
    cov += np.outer(dphiddfdx, dphiddfdx)*cov_dfdx_dfdx  ### contribution from Cov(df/dx,df/dx)

    return cov

def gpr_altogether(x_tst, f_obs, x_obs, cov_noise, cov_models, Nstitch, degree=1, guess_sigma2=DEFAULT_SIGMA2, guess_l2=DEFAULT_L2, guess_sigma2_obs=DEFAULT_SIGMA2, guess_model_multiplier2=1):
    '''
    a delegation function useful when I've already got a bunch of "noise" covariances known for f_obs(x_obs)
    performs automatic optimization ot find the best hyperparameters along with subtracting out f_fit from a polynomial model
    '''
    ### pre-condition the data
    f_fit, f_tst = poly_model(x_tst, f_obs, x_obs, degree=degree)

    ### perform GPR in an optimization loop to find the best logLike
    ### FIXME: use this to figure out best hyper-parameters, but for now just take some I know work reasonably well
    sigma2 = guess_sigma2
    l2 = guess_l2
    sigma2_obs = guess_sigma2_obs
    model_multiplier2 = guess_model_multiplier2

    ### perform GPR with best hyperparameters to infer function at x_tst
    ### note, we don't delegate to gpr_f here because we want to build the covariance functions ourself
    cov_tst_tst = cov_f1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_obs = cov_f1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_tst = cov_f1_f2(x_obs, x_tst, sigma2=sigma2, l2=l2)
    ### NOTE, we just add the known "noise" along with the GPR kernel
    cov_obs_obs = cov_altogether_obs_obs(x_obs, cov_noise, cov_models, Nstitch, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs, model_multiplier2=model_multiplier2)

    ### and now we delgeate
    mean, cov, logweight = gpr(f_obs-f_fit, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)
    mean += f_tst ### add the polyfit model back in

    return mean, cov, logweight

def cov_phi_phi_stitch(x_stitch, stitch_mean, stitch_pressure, stitch_index):
    n = len(x_stitch)
    f_stitch = np.ones(n, dtype=float)*stitch_mean
    cov_stitch = np.diag(np.exp(x_stitch - np.log(stitch_pressure/utils.c2))**stitch_index) ### the stitching white-noise kernel
    return f_stitch, cov_stitch

def cov_altogether_obs_obs(x_obs, cov_noise, cov_models, Nstitch, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2, model_multiplier2=1):

    ### we add the diagonal component for the models in separate from the stitch
    ans = cov_noise + model_multiplier2*cov_models + cov_f1_f2(x_obs, x_obs, sigma2=sigma2, l2=l2)
    N = len(x_obs)-Nstitch
    ans[:N,:N] += np.diag(np.ones(N, dtype=float)*sigma2_obs)

    ##############################################################################################
    ### FOR TESTING PURPOSES: want to visualize the covariance between models, etc
    ##############################################################################################
#    import matplotlib
#    matplotlib.use("Agg")
#    from matplotlib import pyplot as plt
#
#    fig = plt.figure(figsize=(5,4))
#    ax1 = plt.subplot(1,1,1)
#    cb1 = fig.colorbar(
#        ax1.imshow(np.tanh(ans/0.1), cmap='RdGy_r', origin='lower', aspect='equal', vmin=-1, vmax=+1),
#        orientation='vertical',
#        shrink=0.90,
#    )
#    cb1.set_label('tanh(covs/0.1)')
#
#    fig.savefig('TEST.png')
#    plt.close(fig)
    ##############################################################################################

    return ans

def cov_altogether_noise(models, stitch, diagonal_model_covariance=False):
    """compute the big-ol covariance matrix for gpr_altogether
    NOTE: we expect only a single element in each model at this point!
    """
    x_obs = []
    f_obs = []
    for model in models:
        x_obs.append(model['x'])
        f_obs.append(model['f'])
    x_obs = np.concatenate(x_obs)
    f_obs = np.concatenate(f_obs)

    ### compute the big uncertainty estimate between all the models
    Nobs = len(x_obs)

    ### add in "theory model noise" as diagonal components based on variance of means at each pressure
    if stitch:
        x_stitch = []
        f_stitch = []
        for s in stitch:
            x_stitch.append(s['x'])
            f_stitch.append(s['f'])
        x_stitch = np.concatenate(x_stitch)
        f_stitch = np.concatenate(f_stitch)

        num_stitch = len(x_stitch)
        covs = np.zeros((Nobs+num_stitch,)*2, dtype=float) ### include space for the stitching conditions

        x_obs = np.concatenate((x_obs, x_stitch)) ### add in stitching data
        f_obs = np.concatenate((f_obs, f_stitch))
        start = Nobs
        for s in stitch:
            stop = start+len(s['x'])
            covs[start:stop,start:stop] = s['cov']
            start = stop
    else:
        covs = np.zeros((Nobs,Nobs), dtype=float)
        num_stitch = 0

    ### add block-diagonal components
    start = 0
    for model in models:
        stop = start+len(model['x']) ### the number of points in this model
        covs[start:stop,start:stop] = model['cov'] ### fill in block-diagonal component
        start = stop

    ### iterate through pressure samples and compute theory variance of each
    ### NOTE: the following iteration may not be the most efficient thing in the world, but it should get the job done...

    # compute means of means (average over models)
    x_set = np.array(sorted(set([x for model in models for x in model['x']])), dtype=float)
    n_set = len(x_set)
    mu_set = np.empty(n_set, dtype=float)
    for ind, x in enumerate(x_set): # iterate over all included x-points
        sample = []
        for model in models:
            i = x==model['x'] ### should be either 1 or 0 matches
            if np.any(i):
                sample.append(model['f'][i])

        mu_set[ind] = np.mean(sample)

    # compute the average of the covariances and the 2nd moment of the means
    cov_set = np.zeros((n_set, n_set), dtype=float)
    for ind, x in enumerate(x_set):
        for IND, X in enumerate(x_set[ind:]):
            IND += ind ### correct index for the big set

            if diagonal_model_covariance and (ind!=IND): ### only include diagonal components
                continue

            sample = []
            for model in models:
                i = x==model['x'] ### either 1 or 0 matches
                j = X==model['x']
                if np.any(i) and np.any(j): ### both abscissa are present in this model
                    sample.append(model['f'][i]*model['f'][j] + model['cov'][i,j]) ### add both these things together for convenience

            if sample: ### there is something to add here, which is not guaranteed
                cov_set[ind,IND] = np.mean(sample) - mu_set[ind]*mu_set[IND] ### NOTE:
                if ind!=IND:
                    cov_set[IND,ind] = cov_set[ind,IND]  ###   this is equivalent to the average (over models) of the covariance of each model
                                                         ###   plus the covariance between the mean of each model (with respect to the models)
                else:
                    cov_set[ind,ind] = max(cov_set[ind,ind],0) ### minimum allowable

    cov_set = posdef(cov_set) ### regularize the result to make sure it's positive definite (for numerical stability)

    # map cov_set into the appropriate elements of model_covs
    model_covs = np.zeros_like(covs, dtype=float)
    start = 0
    ind_set = np.arange(n_set)
    truth_set = np.empty(n_set, dtype=bool)
    for model in models:

        # identify which abscissa from this model correspond to which indecies in x_set
        truth_set[:] = False
        truth_set[np.array([ind_set[x==x_set] for x in model['x']])] = True

        # map these indecies into model_covs. ASSUMES ABSCISSA ARE ORDERED WITHIN model_covs
        n = len(model['x'])
        model_covs[start:start+n,start:start+n] = cov_set[np.outer(truth_set,truth_set)].reshape((n,n))

        # bump starting index
        start += n

    ##############################################################################################
    ### FOR TESTING PURPOSES: want to visualize the covariance between models, etc
    ##############################################################################################
#    import matplotlib
#    matplotlib.use("Agg")
#    from matplotlib import pyplot as plt
#
#    fig = plt.figure(figsize=(10,4))
#    ax1 = plt.subplot(1,2,1)
#    cb1 = fig.colorbar(
#        ax1.imshow(np.tanh(covs/0.01), cmap='RdGy_r', origin='lower', aspect='equal', vmin=-1, vmax=+1),
#        orientation='vertical',
#        shrink=0.90,
#    )
#    cb1.set_label('tanh(covs/0.01)')
#
#    ax2 = plt.subplot(1,2,2)
#    cb2 = fig.colorbar(
#        ax2.imshow(np.tanh(model_covs/0.01), cmap='RdGy_r', origin='lower', aspect='equal', vmin=-1, vmax=+1),
#        orientation='vertical',
#        shrink=0.90,
#    )
#    cb2.set_label('tanh(model covs/0.01)')
#
#    fig.savefig('TEST.png')
#    plt.close(fig)
#    ##############################################################################################

    return x_obs, f_obs, covs, model_covs, num_stitch
