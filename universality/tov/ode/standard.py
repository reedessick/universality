"""a module that houses TOV solvers in the "standard" formulation
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.integrate import odeint
from scipy.special import hyp2f1

from universality.utils.units import (G, c2, Msun)

#-------------------------------------------------

#DEFAULT_MAX_DR = 1e5 ### maximum step size allowed within the integrator (in standard units, which should be in cm)
DEFAULT_MAX_DR = 1e6
DEFAULT_MIN_DR = 1.0 ### the smallest step size we allow (in standard units, which should be cm)

DEFAULT_GUESS_FRAC = 0.1 ### how much of the way to the vanishing pressure we guess via Newton's method

DEFAULT_INITIAL_FRAC = 1e-3 ### the initial change in pressure we allow when setting the intial conditions

DEFAULT_RTOL = 1e-4

#------------------------

TWOPI = 2*np.pi
FOURPI = 2*TWOPI

Gc2 = G/c2

#-------------------------------------------------
### Standard formulation of the TOV equations
#-------------------------------------------------

### basic evolutionary equations

def dmdr(r, epsc2):
    return FOURPI * r**2 * epsc2

def dmbdr(r, m, dm_dr):
    return dm_dr * (1 - 2*Gc2*m/r)**-0.5

def dpc2dr(r, pc2, m, epsc2):
    return - Gc2 * (epsc2 + pc2)*(m + FOURPI * r**3 * pc2)/(r * (r - 2*Gc2*m))

def detadr(r, pc2, m, eta, epsc2, cs2c2):
    invf = (1. - 2.*Gc2*m/r)**-1
    A = 2. * invf * (1. -3.*Gc2*m/r - TWOPI * G * r**2 * (epsc2 + 3.*pc2))
    B = invf * (6. - FOURPI*G*r**2 * (epsc2 + pc2)*(3. + 1./cs2c2))
    return -1.*(eta*(eta - 1.) + A*eta - B)/r

def domegadr(r, pc2, m, omega, epsc2):
    P = FOURPI * G * r**2 * (epsc2 + pc2)/ (1. - 2.*Gc2*m/r)
    return -(omega*(omega + 3.) - P*(omega + 4.))/r

#-------------------------------------------------
# functions for values at the stellar surface
#-------------------------------------------------

def eta2lambda(r, m, eta): ### dimensionless tidal deformability
    C = Gc2*m/r # compactness
    fR = 1.-2.*C
    F = hyp2f1(3., 5., 6., 2.*C) # a hypergeometric function

    z = 2.*C
    dFdz = (5./(2.*z**6.)) * (z*(z*(z*(3.*z*(5. + z) - 110.) + 150.) - 60.) / (z - 1.)**3 + 60.*np.log(1. - z))
    RdFdr = -2.*C*dFdz # log derivative of hypergeometric function

    k2el = 0.5*(eta - 2. - 4.*C/fR) / (RdFdr -F*(etaR + 3. - 4.*C/fR)) # gravitoelectric quadrupole Love number
    return (2./3.)*(k2el/C**5)

def omega2i(r, omega): ### moment of inertia
    return (omega/(3. + omega)) * r**3/(2.*Gc2)

#-------------------------------------------------
# initial conditions
#-------------------------------------------------

def initial_pc2(pc2i, frac):
    return (1. - frac)*pc2i ### assume a constant slope over a small change in the pressure

def initial_r(pc2i, ec2i, frac):
    return (frac*pc2i / ( G * (ec2i + pc2i) * (ec2i/3. + pc2i) * TWOPI ) )**0.5 ### solve for the radius that corresponds to that small change

def initial_m(r, ec2i):
    return FOURPI * r**3 * ec2i / 3.  # gravitational mass

def initial_mb(r, rhoi):
    return FOURPI * r**3 * rhoi / 3.  # gravitational mass

def intitial_eta(r, pc2i, ec2i, cs2c2i):
    return 2. + FOURPI * G * r**2 * (9.*pc2i + 13.*ec2i + 3.*(pc2i+ec2i)/cs2c2i)/21. # intial perturbation for dimensionless tidal deformability

def intitial_omega(r, pc2i, ec2i):
    return 16.*np.pi * G * r**2 * (pc2i + ec2i)/5. # initial frame-dgragging function

#-------------------------------------------------
# central loop that solves the TOV equations given a set of coupled ODEs
#-------------------------------------------------

def engine(
        r,
        vec,
        eos,
        dvdr_func,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc" and equation of state described by energy density "eps" and pressure "p"
    expects eos = (pressure, energy_density)
    """
    while vec[0] > 0: ### continue until pressure vanishes
        vec0 = vec[:] # store the current location as the old location
        r0 = r

        ### estimate the radius at which this p will vanish via Newton's method
        r = r0 + max(min_dr, min(max_dr, guess_frac * abs(vec[0]/dvdr_func(vec, r, eos)[0])))

        ### integrate out until we hit that estimate
        vec = odeint(ddr_func, vec0, (r0, r), args=(eos,), rtol=rtol, hmax=max_dr)[-1,:] ### retain only the last point

    ### return to client, who will then interpolate to find the surface
    ### interpolate to find stellar surface
    p = [vec0[0], vec[0]]

    # radius
    r = np.interp(0, p, [r0, r])
    # the rest of the macro properties
    vals = [np.interp(0, p, [vec0[i], vec[i]]) for i in range(1, len(vec))]

    return r, vals

#-------------------------------------------------

### the solver that yields all known macroscopic quantites
MACRO_COLS = ['M', 'R', 'Lambda', 'I', 'Mb'] ### the column names for what we compute

def dvecdr(vec, r, eos):
    pc2, m, eta, omega, mb = vec
    epsc2 = np.interp(pc2, eos[0], eos[1])
    rho = np.interp(pc2, eos[0], eos[2])
    cs2c2 = np.interp(pc2, eos[0], eos[3])

    return \
        dpc2dr(r, pc2, m, epsc2), \
        dmdr(r, epsc2), \
        detadr(r, pc2, m, eta, epsc2, cs2c2), \
        domegadr(r, pc2, m, omega, epsc2), \
        dmbdr(r, m, dmdr(r, rho))

def initial_condition(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    """determines the initial conditions for a stellar model with central pressure pc
    this is done by analytically integrating the TOV equations over very small radii to avoid the divergence as r->0
    """
    ec2i = np.interp(pc2i, eos[0], eos[1])
    rhoi = np.interp(pc2i, eos[0], eos[2])
    cs2c2i = np.interp(pc2i, eos[0], eos[3])

    pc2 = initial_pc2(pc2i, frac)
    r = initial_r(pc2i, ec2i, frac)
    m = initial_m(r, ec2i)
    mb = initial_mb(r, rhoi)
    eta = initial_eta(r, pc2i, ec2i, cs2c2i)
    omega = initial_omega(r, pc2i, ec2i)

    return r, (pc2, m, eta, omega, mb)

def integrate_all(
        pc2i,
        eos,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc" and equation of state described by energy density "eps" and pressure "p"
    expects eos = (pressure, energy_density, baryon_density, cs2c2)
    """
    r, vec = initial_conditions(pc2i, eos, frac=initial_frac)
    if vec[0] < 0: ### guarantee that we enter the loop
        raise RuntimeError('bad initial condition!')

    r, (m, eta, omega, mb) = engine(
        r,
        vec,
        eos,
        dvecdr,
        min_dr=min_dr,
        max_dr=max_dr,
        guess_frac=guess_frac,
        rtol=rtol,
    )

    # compute tidal deformability
    l = eta2labmda(r, m, eta)

    # compute  moment of inertia
    i = omega2i(r, omega)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    mb /= Msun
    r *= 1e-5 ### convert from cm to km
    i /= 1e45 ### normalize this to a common value but still in CGS

    return m, r, l, i, mb

#-------------------------------------------------

### light-weight solver that only includes M and R
MACRO_COLS_MR = ['M', 'R']

def dvecdr_MR(vec, r, eos):
    '''returns d(p, m)/dr
    expects: pressurec2, energy_densityc2 = eos
    '''
    pc2, m, eta, omega, mb = vec
    epsc2 = np.interp(pc2, eos[0], eos[1])
    rho = np.interp(pc2, eos[0], eos[2])

    return \
        dpc2dr(r, pc2, m, epsc2), \
        dmdr(r, epsc2)

def initial_condition_MR(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    """determines the initial conditions for a stellar model with central pressure pc
    this is done by analytically integrating the TOV equations over very small radii to avoid the divergence as r->0
    """
    ec2i = np.interp(pc2i, eos[0], eos[1])
    rhoi = np.interp(pc2i, eos[0], eos[2])

    pc2 = initial_pc2(pc2i, frac)
    r = initial_r(pc2i, ec2i, frac)
    m = initial_m(r, ec2i)

    return r, (pc2, m)

def integrate_MR(
        pc2i,
        eos,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc" and equation of state described by energy density "eps" and pressure "p"
    expects eos = (pressure, energy_density, baryon_density, cs2c2)
    """
    r, vec = initial_conditions_MR(pc2i, eos, frac=initial_frac)
    if vec[0] < 0: ### guarantee that we enter the loop
        raise RuntimeError('bad initial condition!')

    r, (m,) = engine(
        r,
        vec,
        eos,
        dvecdr_MR,
        min_dr=min_dr,
        max_dr=max_dr,
        guess_frac=guess_frac,
        rtol=rtol,
    )

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    r *= 1e-5 ### convert from cm to km

    return m, r

#-------------------------------------------------

### light-weight solver that only includes M, R, and Lambda 
MACRO_COLS_MRLambda = ['M', 'R', 'Lambda']

def dvecdr_MRLambda(vec, r, eos):
    '''returns d(p, m)/dr
    expects: pressurec2, energy_densityc2 = eos
    '''
    pc2, m, eta, omega, mb = vec
    epsc2 = np.interp(pc2, eos[0], eos[1])
    rho = np.interp(pc2, eos[0], eos[2])
    cs2c2 = np.interp(pc2, eos[0], eos[3])

    return \
        dpc2dr(r, pc2, m, epsc2), \
        dmdr(r, epsc2), \
        detadr(r, pc2, m, eta, epsc2, cs2c2)

def initial_condition_MRLambda(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    """determines the initial conditions for a stellar model with central pressure pc
    this is done by analytically integrating the TOV equations over very small radii to avoid the divergence as r->0
    """
    ec2i = np.interp(pc2i, eos[0], eos[1])
    rhoi = np.interp(pc2i, eos[0], eos[2])
    cs2c2i = np.interp(pc2i, eos[0], eos[3])

    pc2 = initial_pc2(pc2i, frac)
    r = initial_r(pc2i, ec2i, frac)
    m = initial_m(r, ec2i)
    eta = initial_eta(r, pc2i, ec2i, cs2c2i)

    return r, (pc2, m, eta)

def integrate_MRLambda(
        pc2i,
        eos,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc" and equation of state described by energy density "eps" and pressure "p"
    expects eos = (pressure, energy_density, baryon_density, cs2c2)
    """ 
    r, vec = initial_conditions_MRLambda(pc2i, eos, frac=initial_frac)
    if vec[0] < 0: ### guarantee that we enter the loop
        raise RuntimeError('bad initial condition!')

    r, (m, eta) = engine(
        r,
        vec,
        eos,
        dvecdr_MRLambda,
        min_dr=min_dr,
        max_dr=max_dr,
        guess_frac=guess_frac,
        rtol=rtol,
    )

    # compute tidal deformability
    l = eta2labmda(r, m, eta)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    r *= 1e-5 ### convert from cm to km

    return m, r, l
