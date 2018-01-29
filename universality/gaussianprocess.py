__doc__ = "a module that houses simple Gaussian Process routines"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

### defaults
__default_sigma__ = 0.1
__default_l__ = 0.1

__default_sigma2__ = __default_sigma__**2
__default_l2__ = __default_l__**2

#-------------------------------------------------
# covariance kernels
#-------------------------------------------------

def cov_f1_f2(x1, x2, sigma2=__default_sigma2__, l2=__default_l2__):
    '''
    cov(f1, f2) = sigma2 * np.exp(-(x1-x2)**2/(2*l2))
    '''
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return sigma2 * np.exp(-0.5*(X1-X2)**2/l2)

def cov_df1dx1_f2(x1, x2, sigma2=__default_sigma2__, l2=__default_l2__):
    '''
    cov(df1/dx1, f2) = -sigma2 * (x1-x2)/l**2 * np.exp(-(x1-x2)**2/(2*l2))
    '''
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return -(X1-X2)/l2 * cov_f1_f2(x1, x2, sigma2=sigma2, l2=l2)

def cov_f1_df2dx2(x1, x2, sigma2=__default_sigma2__, l2=__default_l2__):
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return (X1-X2)/l2 * cov_f1_f2(x1, x2, sigma2=sigma2, l2=l2)

def cov_df1dx1_df2dx2(x1, x2, sigma2=__default_sigma2__, l2=__default_l2__):
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

def _cov(x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    a helper function that computes the covariance matrix for observed points
    '''
    cov_obs_obs = cov_f1_f2(x_obs, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_obs += np.diag(np.ones(len(x_obs)))*sigma2_obs
    return cov_obs_obs

def _dcov_dsigma2(x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    a helper function for the derivative of cov_obs_obs with respect to sigma2
    '''
    X1, X2 = np.meshgrid(x_obs, x_obs, indexing='ij')
    return np.exp(-0.5(X1-X2)**2/l2) ### simple derivative of cov_f1_f2

def _dcov_dl2(x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    a helper function for the derivative of cov_obs_obs with respect to l2
    '''
    X1, X2 = np.meshgrid(x_obs, x_obs, indexing='ij')
    Z = 0.5*(X1-X2)**2/l2
    return sigma2*np.exp(-Z)*Z/l2 ### simple derivative of cov_f1_f2

def _dcov_dsigma2_obs(x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    a helper function for the derivative of cov_obs_obs with respect to sigma2_obs
    '''
    N = len(x_obs)
    return np.diag(N,N) ### derivative of diagonal, white noise

def logLike(f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    cov = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### compute the logLikelihood
    N = len(x_obs)
    # compute components separately because they each require different techniques to stabilize them numerically
    obs = -0.5*np.dot(f_obs, np.dot(np.linalg.inv(cov), f_obs))
    nrm = -0.5*N*np.log(2*np.pi)
    # because the covariances might be so small, and there might be a lot of data points, we need to handle the determinant with care
    det = -0.5*np.sum(np.log(np.linalg.eigvals(cov).real)) ### FIXME: this may be fragile
#    m = np.max(cov)
#    det = -0.5*(np.log(np.linalg.det(cov/m)) + N*np.log(m))

    return obs + det + nrm ### assemble components and return

def grad_logLike(f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    computes the logLikelihood and jacobian thereof based on the covariance between observation points
    the covariance is constructed given the hyper parameters just as it is for gpr_f and gpr_dfdx

    return logL, (dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs)
    '''
    invcov_obs_obs = np.linalg.inv(_cov_obs_obs(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs))
    a = np.dot(invcov_obs_obs, f_obs)

    ### compute the gradient with each hyperparameter
    m = np.outer(a,a) - invcov_obs_obs
    dlogL_dsigma2 = 0.5*np.trace(np.dot(m, _dcov_dsigma2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dl2     = 0.5*np.trace(np.dot(m, _dcov_dl2(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))
    dlogL_dsigma2_obs = 0.5*np.trace(np.dot(m, _dcov_dsigma2_obs(sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)))

    return dlogL_dsigma2, dlogL_dl2, dlogL_dsigma2_obs

def gpr_f(x_tst, f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
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

def gpr_dfdx(x_tst, f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
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

def poly_model_dfdx(x_tst, f_obs, x_obs, degree=1):
    '''
    like poly_model, except
    return f_fit, dfdx_tst
    '''
    f_fit = np.zeros_like(x_obs, dtype='float')
    f_tst = np.zeros_like(x_tst, dtype='float')
    poly = np.polyfit(x_obs, f_obs, degree)
    for i in xrange(degree+1):
        f_fit += poly[-1-i]*x_obs**i
        f_tst += i*poly[-1-i]*x_tst**(i-1)
    return f_fit, f_tst

def gpr_resample(x_tst, f_obs, x_obs, degree=1, guess_sigma2=__default_sigma2__, guess_l2=__default_l2__, guess_sigma2_obs=__default_sigma2__):
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

def gpr_resample_dfdx(x_tst, f_obs, x_obs, degree=1, guess_sigma2=__default_sigma2__, guess_l2=__default_l2__, guess_sigma2_obs=__default_sigma2__):
    '''
    the same as gpr_resample, except we regress out the derivative of the function instead of the function
    resample the data via GPR to samples along x_tst
    performs automatic optimization to find the best hyperparameters along with subtracting out f_fit from a polynomial model
    '''
    ### pre-condition the data
    f_fit, f_tst = poly_model_dfdx(x_tst, f_obs, x_obs, degree=degree)

    ### perform GPR in an optimization loop to find the best logLike
    ### FIXME: use this to figure out best hyper-parameters, but for now just take some I know work reasonably well
    sigma2 = guess_sigma2
    l2 = guess_l2
    sigma2_obs = guess_sigma2_obs

    ### perform GPR with best hyperparameters to infer the function at x_tst
    mean, cov = gpr_dfdx(x_tst, f_obs-f_fit, x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)
    mean += f_tst ### add the polyfit model back in 

    return mean, cov


def gpr_altogether(x_tst, f_obs, x_obs, cov_noise, degree=1, guess_sigma2=__default_sigma2__, guess_l2=__default_l2__, guess_sigma2_obs=__default_sigma2__):
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
