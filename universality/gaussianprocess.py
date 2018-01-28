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

def gpr(f_obs, cov_test_test, cov_test_obs, cov_obs_test, cov_obs_obs):
    '''
    constructs the parameters for the conditional distribution: f_test|f_obs,x_obs,x_text based on cov_test_test, cov_test_obs, cov_obs_obs
        cov_test_test : (N_test, N_test) the covariance matrix between test samples in the joint distribution
        cov_test_obs  : (N_test, N_obs)  the covariance matrix between test samples and observed samples in the joint distribution
        cov_obs_obs   : (N_obs, N_obs)   the covariance matrix between observed samples in the joint distribution
    returns the mean_test, cov_test
    '''
    ### invert matix only once. This is the expensive part
    invcov_obs_obs = np.linalg.inv(cov_obs_obs)

    ### do some matrix multiplcation here
    mean = np.dot(cov_test_obs, np.dot(invcov_obs_obs, f_obs))
    cov  = cov_test_test - np.dot(cov_test_obs, np.dot(invcov_obs_obs, cov_obs_test))

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

def gpr_f(x_test, f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    constructs covariance for f_test|f_obs,x_obs,x_test
    returns mean_test, cov_test
    '''
    ### compute covariances
    cov_test_test = cov_f1_f2(x_test, x_test, sigma2=sigma2, l2=l2)
    cov_test_obs  = cov_f1_f2(x_test, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_obs = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### delegate
    return gpr(f_obs, cov_test_test, cov_test_obs, np.transpose(cov_test_obs), cov_obs_obs)

def gpr_dfdx(x_test, f_obs, x_obs, sigma2=__default_sigma2__, l2=__default_l2__, sigma2_obs=__default_sigma2__):
    '''
    constructs covariance needed for df_test/dx_test|f_obs,x_obs,x_test
    return mean_test, cov_test
    '''
    ### compute covariances
    cov_test_test = cov_df1dx1_df2dx2(x_test, x_test, sigma2=sigma2, l2=l2)
    cov_test_obs  = cov_df1dx1_f2(x_test, x_obs, sigma2=sigma2, l2=l2)
    cov_obs_test  = cov_f1_df2dx2(x_obs, x_test, sigma2=sigma2, l2=l2)
    cov_obs_obs = _cov(x_obs, sigma2=sigma2, l2=l2, sigma2_obs=sigma2_obs)

    ### delegate
    return gpr(f_obs, cov_test_test, cov_test_obs, cov_obs_test, cov_obs_obs) 
