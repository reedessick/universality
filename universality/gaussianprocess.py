__doc__ = "a module that houses simple Gaussian Process routines"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np
import pickle

import time

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

def create_process_group(group, poly_degree, sigma, length_scale, sigma_obs, x_tst, f_tst, cov_f_f, xlabel='xl', flabel='f', weight=1.):
    """helper funtion to record the data about a process in a mixture model"""
    group.attrs.create('weight', weight)
    group.attrs.create('poly_degree', poly_degree)
    group.attrs.create('sigma', sigma)
    group.attrs.create('length_scale', length_scale)
    group.attrs.create('sigma_obs', sigma_obs)
    group.attrs.create('xlabel', xlabel)
    group.attrs.create('flabel', flabel)

    means = group.create_dataset('mean', data=np.array(zip(x_tst, f_tst), dtype=[(xlabel, 'float'), (flabel, 'float')]))
    cov = group.create_dataset('cov', data=cov_f_f)

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

    x_tst = group['mean'][xlabel]
    f_tst = group['mean'][flabel]

    cov_f_f = group['cov'][...]

    return weight, x_tst, f_tst, cov_f_f, (xlabel, flabel), (poly_degree, sigma, length_scale, sigma_obs)

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
# GPR via conditioning given a set of observation and a covariance matrix
#-------------------------------------------------

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
    cov  = cov_tst_tst - np.dot(cov_tst_obs, np.dot(invcov_obs_obs, cov_obs_tst))

    return mean, cov

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

def logLike(f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2, timeit=False, degree=1):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    if timeit:
        t0 = time.time()

    if (sigma2<=0) or (l2<=0) or (sigma2_obs<=0):
        return np.infty

    cov = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    f_fit, _ = poly_model(np.array([]), f_obs, x_obs, degree=degree)

    ### compute the logLikelihood
    N = len(x_obs)

    # compute components separately because they each require different techniques to stabilize them numerically
    obs = -0.5*np.dot(f_obs-f_fit, np.dot(np.linalg.inv(cov), f_obs-f_fit))

    nrm = -0.5*N*np.log(2*np.pi)

    # because the covariances might be so small, and there might be a lot of data points, we need to handle the determinant with care
    sign, det = np.linalg.slogdet(cov)
    if sign<0:
        det = -np.infty ### rule this out by setting logLike -> -infty
#        raise ValueError, 'unphysical covariance matrix!'
    else:
        det *= -0.5

#    print 'obs: %.3e\nnrm: %.3e\ndet: %.3e\nsgn: %.1f\nlnL: %.3e'%(obs, nrm, det, sign, obs+nrm+det)

    if timeit:
        print time.time()-t0

    return obs + det + nrm ### assemble components and return

def grad_logLike(f_obs, x_obs, sigma2=DEFAULT_SIGMA2, l2=DEFAULT_L2, sigma2_obs=DEFAULT_SIGMA2, degree=1):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    if (sigma2<=0) or (l2<=0) or (sigma2_obs<=0):
        raise ValueError, 'unphysical hyperparameters!'

    f_fit, _ = poly_model(np.array([]), f_obs, x_obs, degree=degree) 

    invcov_obs_obs = np.linalg.inv(_cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs))
    a = np.dot(invcov_obs_obs, f_obs-f_fit)

    ### compute the gradient with each hyperparameter
    m = np.outer(a,a) - invcov_obs_obs
    dlogL_dsigma2 = 0.5*np.trace(np.dot(m, _dcov_dsigma2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dl2     = 0.5*np.trace(np.dot(m, _dcov_dl2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dsigma2_obs = 0.5*np.trace(np.dot(m, _dcov_dsigma2_obs(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))

    return dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs

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
    mean, cov = gpr(f_obs, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)

    ### slice the resulting arrays and return
    ### relies on the ordering we constructed within our covariance matricies!
    #        mean_f      mean_dfdx       cov_f_f          cov_f_dfdx        cov_dfdx_f      cov_dfdx_dfdx
    return mean[:Ntst], mean[Ntst:], cov[:Ntst,:Ntst], cov[:Ntst,Ntst:], cov[Ntst:,:Ntst], cov[:Ntst,:Ntst]

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
    mean, cov = gpr_f(x_tst, f_obs-f_fit, x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)
    mean += f_tst ### add the polyfit model back in 

    return mean, cov

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
    mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx = gpr_f_dfdx(x_tst, f_obs-f_fit, x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    # add the polyfit model back in
    mean_f += f_tst
    mean_dfdx += dfdx_tst

    return mean_f, mean_dfdx, cov_f_f, cov_f_dfdx, cov_dfdx_f, cov_dfdx_dfdx

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

def gpr_altogether(x_tst, f_obs, x_obs, cov_noise, degree=1, guess_sigma2=DEFAULT_SIGMA2, guess_l2=DEFAULT_L2, guess_sigma2_obs=DEFAULT_SIGMA2):
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

    ### perform GPR with best hyperparameters to infer function at x_tst
    ### note, we don't delegate to gpr_f here because we want to build the covariance functions ourself
    cov_tst_tst = cov_f1_f2(x_tst, x_tst, sigma2=sigma2, l2=l2)
    cov_tst_obs = cov_f1_f2(x_tst, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_tst = cov_f1_f2(x_obs, x_tst, sigma2=sigma2, l2=l2)
    cov_obs_obs = cov_noise + _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs) ### NOTE, we just add the know "noise" along with the GPR kernel

    ### and now we delgeate
    mean, cov = gpr(f_obs-f_fit, cov_tst_tst, cov_tst_obs, cov_obs_tst, cov_obs_obs)
    mean += f_tst ### add the polyfit model back in

    return mean, cov
