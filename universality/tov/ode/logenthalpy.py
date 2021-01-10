"""a module that houses TOV solvers based on the log(enthalpy per unit rest mass)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

import numpy as np
from scipy.integrate import odeint

from universality.utils import utils
from universality.utils.units import (G, c2, Msun)

from .standard import (eta2lambda, omega2i)
from .standard import (initial_m, initial_mb, initial_eta, initial_omega)
from .standard import (dmdr, dmbdr, detadr, domegadr)

#-------------------------------------------------

DEFAULT_INITIAL_FRAC = 1e-8 ### the initial change in pressure we allow when setting the intial conditions

DEFAULT_RTOL = 1e-6

#------------------------

TWOPI = 2*np.pi
FOURPI = 2*TWOPI

Gc2 = G/c2
c2G = 1./Gc2

#-------------------------------------------------
### Formulation of the TOV equations in terms of the log(enthalpy per unit rest mass) = log( (eps+p)/rho )
#-------------------------------------------------

def eos2logh(pc2, ec2):
    return utils.num_intfdx(np.log(pc2), pc2/(ec2+pc2)) ### thought to be more numerically stable given sparse samples of pc2 in the crust

#------------------------

def drdlogh(r, m, pc2):
    return - r * (r*c2G - 2*m) / (m + FOURPI * r**3 * pc2)

def dmdlogh(r, epsc2, dr_dlogh):
    return dmdr(r, epsc2) * dr_dlogh

def dmbdlogh(r, m, rho, dr_dlogh):
    return dmbdr(r, rho, m) * dr_dlogh

def detadlogh(r, pc2, m, eta, epsc2, cs2c2, dr_dlogh):
    return detadr(r, pc2, m, eta, epsc2, cs2c2) * dr_dlogh

def domegadlogh(r, pc2, m, omega, epsc2, dr_dlogh):
    return domegadr(r, pc2, m, omega, epsc2) * dr_dlogh

#-------------------------------------------------
# initial conditions
#-------------------------------------------------

def initial_logh(loghi, frac):
    return (1. - frac)*loghi ### assume a constant slope over a small change in the pressure

def initial_r(loghi, pc2i, ec2i, frac):
    return ( 3.*frac*loghi*c2G / (TWOPI*(ec2i + 3.*pc2i)) )**0.5

#-------------------------------------------------
# central loop that solves the TOV equations given a set of coupled ODEs
#-------------------------------------------------

def engine(
        logh,
        vec,
        eos,
        dvecdlogh_func,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    ### integrate out until we hit termination condition
    return odeint(
        dvecdlogh_func,
        vec,
        (logh, 0.),
        args=(eos,),
        rtol=rtol,
        mxstep=10000, ### empirically found to be sufficient, the default is 500
        mxhnil=1, ### maximum number of messages to print
    )[-1,:]

#-------------------------------------------------

### solver that yields all known macroscopic quantities
MACRO_COLS = ['M', 'R', 'Lambda', 'I', 'Mb'] ### the column names for what we compute

def dvecdlogh(vec, logh, eos):
    eos0 = eos[0]
    pc2 = np.interp(logh, eos0, eos[1])
    ec2 = np.interp(logh, eos0, eos[2])
    rho = np.interp(logh, eos0, eos[3])
    cs2c2 = np.interp(logh, eos0, eos[4])

    m, r, eta, omega, mb = vec
    dr_dlogh = drdlogh(r, m, pc2)

    return \
        dmdlogh(r, ec2, dr_dlogh), \
        dr_dlogh, \
        detadlogh(r, pc2, m, eta, ec2, cs2c2, dr_dlogh), \
        domegadlogh(r, pc2, m, omega, ec2, dr_dlogh), \
        dmbdlogh(r, m, rho, dr_dlogh)

def initial_condition(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    '''analytically solve for the initial condition around the divergence at r=0
    '''
    eos1 = eos[1]
    loghi = np.interp(pc2i, eos1, eos[0])
    ec2i = np.interp(pc2i, eos1, eos[2])
    rhoi = np.interp(pc2i, eos1, eos[3])
    cs2c2i = np.interp(pc2i, eos1, eos[4])

    logh = initial_logh(loghi, frac)

    r = initial_r(loghi, ec2i, pc2i, frac)
    m = initial_m(r, ec2i)
    mb = initial_mb(r, ec2i)
    eta = initial_eta(r, pc2i, ec2i, cs2c2i)
    omega = initial_omega(r, pc2i, ec2i)

    return logh, (m, r, eta, omega, mb)

def integrate(
        pc2i,
        eos,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    ### define initial condition
    logh, vec = initial_condition(pc2i, eos, frac=initial_frac)

    m, r, eta, omega, mb = engine(
        logh,
        vec,
        eos,
        dvecdlogh,
        rtol=rtol,
    )

    # compute tidal deformability
    l = eta2lambda(r, m, eta)

    # compute  moment of inertia
    i = omega2i(r, omega)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    mb /= Msun
    r *= 1e-5 ### convert from cm to km
    i /= 1e45 ### normalize this to a common value but still in CGS

    return m, r, l, i, mb

#-------------------------------------------------

## lightweight solver that only computes M, R
MACRO_COLS_MR = ['M', 'R']

def dvecdlogh_MR(vec, logh, eos):
    eos0 = eos[0]
    pc2 = np.interp(logh, eos0, eos[1])
    ec2 = np.interp(logh, eos0, eos[2])

    m, r = vec
    dr_dlogh = drdlogh(r, m, pc2)

    return \
        dmdlogh(r, ec2, dr_dlogh), \
        dr_dlogh

def initial_condition_MR(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    '''analytically solve for the initial condition around the divergence at r=0
    '''
    eos1 = eos[1]
    loghi = np.interp(pc2i, eos1, eos[0])
    ec2i = np.interp(pc2i, eos1, eos[2])

    logh = initial_logh(loghi, frac)

    r = initial_r(loghi, ec2i, pc2i, frac) ### NOTE: this is good enough for the M-R integrals
    m = initial_m(r, ec2i)                 ### but we have to do something more complicated for the other perturbation equations

    return logh, (m, r)

def integrate_MR(
        pc2i,
        eos,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    ### define initial condition
    logh, vec = initial_condition_MR(pc2i, eos, frac=initial_frac)

    m, r = engine(
        logh,
        vec,
        eos,
        dvecdlogh_MR,
        rtol=rtol,
    )

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    r *= 1e-5 ### convert from cm to km

    return m, r

#-------------------------------------------------

### lightweight solver that yields M, R, Lambda
### solver that yields all known macroscopic quantities
MACRO_COLS_MRLambda = ['M', 'R', 'Lambda'] ### the column names for what we compute

def dvecdlogh_MRLambda(vec, logh, eos):
    eos0 = eos[0]
    pc2 = np.interp(logh, eos0, eos[1])
    ec2 = np.interp(logh, eos0, eos[2])
    rho = np.interp(logh, eos0, eos[3])
    cs2c2 = np.interp(logh, eos0, eos[4])

    m, r, eta = vec
    dr_dlogh = drdlogh(r, m, pc2)

    return \
        dmdlogh(r, ec2, dr_dlogh), \
        dr_dlogh, \
        detadlogh(r, pc2, m, eta, ec2, cs2c2, dr_dlogh)

def initial_condition_MRLambda(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    '''analytically solve for the initial condition around the divergence at r=0
    '''
    eos1 = eos[1]
    loghi = np.interp(pc2i, eos1, eos[0])
    ec2i = np.interp(pc2i, eos1, eos[2])
    rhoi = np.interp(pc2i, eos1, eos[3])
    cs2c2i = np.interp(pc2i, eos1, eos[4])

    logh = initial_logh(loghi, frac)

    r = initial_r(loghi, ec2i, pc2i, frac)
    m = initial_m(r, ec2i)
    eta = initial_eta(r, pc2i, ec2i, cs2c2i)

    return logh, (m, r, eta)

def integrate_MRLambda(
        pc2i,
        eos,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    ### define initial condition
    logh, vec = initial_condition_MRLambda(pc2i, eos, frac=initial_frac)

    m, r, eta = engine(
        logh,
        vec,
        eos,
        dvecdlogh_MRLambda,
        rtol=rtol,
    )

    # compute tidal deformability
    l = eta2lambda(r, m, eta)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    r *= 1e-5 ### convert from cm to km

    return m, r, l
