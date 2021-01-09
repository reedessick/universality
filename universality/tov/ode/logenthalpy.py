"""a module that houses TOV solvers based on the log(enthalpy per unit rest mass)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

import numpy as np
from scipy.integrate import odeint

from universality.utils import utils
from universality.utils.units import (G, c2, Msun)
from .standard import (dpc2dr, eta2lambda, omega2i, initial_m, initial_mb, initial_eta, initial_omega)

#-------------------------------------------------

DEFAULT_MIN_DLOGH = 1e-3
DEFAULT_MAX_DLOGH = 1e-1 ### maximum step size allowed within the integrator (dimensionless)

DEFAULT_INITIAL_FRAC = 1e-6 ### the initial change in pressure we allow when setting the intial conditions

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
    return - r * (r*c2 - 2*G*m) / (G*(m + FOURPI * r**3 * pc2))

def dmdlogh(r, epsc2, dr_dlogh):
    return FOURPI * r**2 * epsc2 * dr_dlogh

def dmbdlogh(r, m, rho, dr_dlogh):
    return FOURPI * r**2 * rho * dr_dlogh / (1 - 2*Gc2*m/r)**0.5

def detadlogh(r, pc2, m, eta, epsc2, cs2c2):
    f = 1. - 2.*Gc2*m/r
    A = 2. * r * (c2G - 3.*m/r - TWOPI*r**2*(epsc2 + 3.*pc2))
    B = r*(6.*c2G - FOURPI*r**2*(epsc2 + pc2)*(3. + 1./cs2c2)) ### NOTE: the inverse sound speed can do bad things here...
    return (eta*(eta - 1.)*r*f*c2G + A*eta - B)/(m + FOURPI*r**3*pc2) # from Landry+Poisson PRD 89 (2014)

def domegadlogh(r, pc2, m, omega, epsc2):
    f = 1. - 2.*Gc2*m/r
    F = FOURPI*r**3*(epsc2 + pc2)/(r*c2G - 2*m)
    return (omega*(omega + 3.) - F*(omega + 4.))*r*f*c2G/(m + FOURPI*r**3*pc2)

#-------------------------------------------------
# initial conditions
#-------------------------------------------------

def initial_logh(loghi, frac):
    return (1. - frac)*loghi ### assume a constant slope over a small change in the pressure

def initial_r(loghi, pc2i, ec2i, frac):
    return ( 3.*frac*loghi*c2 / (TWOPI*G*(ec2i + 3.*pc2i)) )**0.5

#-------------------------------------------------
# central loop that solves the TOV equations given a set of coupled ODEs
#-------------------------------------------------

def engine(
        logh,
        vec,
        eos,
        dvecdlogh_func,
        min_dlogh=DEFAULT_MIN_DLOGH,
        max_dlogh=DEFAULT_MAX_DLOGH,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    ### integrate out until we hit termination condition
    return odeint(dvecdlogh_func, vec, (logh, 0), args=(eos,), rtol=rtol)[-1,:]

#    vec = np.array(vec, dtype=float) # make sure this is an array
#
#    while logh > 0:
#        logh0 = logh
#        vec0 = vec[:]
#
#        ### guess the next step
#        dvec_dlogh = np.array(dvecdlogh_func(vec0, logh, eos))
#        logh = logh0 - max(min_dlogh, min(np.min(vec/dvec_dlogh), max_dlogh)) ### we might be able to speed this up by guessing...
#        logh = max(0., logh) ### make sure we never go past zero
#
#        vec[:] = odeint(dvecdlogh_func, vec0, (logh0, logh), args=(eos,), rtol=rtol)[-1,:]
#
#    ### extract final values at the surface
#    return vec

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
        detadlogh(r, pc2, m, eta, epsc2, cs2c2), \
        domegadlogh(r, pc2, m, omega, epsc2), \
        dmbdlogh(r, rho, dr_dlogh)

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
        min_dlogh=DEFAULT_MIN_DLOGH,
        max_dlogh=DEFAULT_MAX_DLOGH,
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
        min_dlogh=min_dlogh,
        max_dlogh=max_dlogh,
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

    r = initial_r(loghi, ec2i, pc2i, frac)
    m = initial_m(r, ec2i)

    return logh, (m, r)

def integrate_MR(
        pc2i,
        eos,
        min_dlogh=DEFAULT_MIN_DLOGH,
        max_dlogh=DEFAULT_MAX_DLOGH,
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
        min_dlogh=min_dlogh,
        max_dlogh=max_dlogh,
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
        detadlogh(r, pc2, m, eta, ec2, cs2c2)

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
        min_dlogh=DEFAULT_MIN_DLOGH,
        max_dlogh=DEFAULT_MAX_DLOGH,
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
        min_dlogh=min_dlogh,
        max_dlogh=max_dlogh,
        rtol=rtol,
    )

    # compute tidal deformability
    l = eta2lambda(r, m, eta)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    r *= 1e-5 ### convert from cm to km

    return m, r, l
