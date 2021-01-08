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
    return dm_dr * (1 - 2*G*m/(r*c2))**-0.5

def dpc2dr(r, pc2, m, epsc2):
    return - G * (epsc2 + pc2)*(m + FOURPI * r**3 * pc2)/(r * (r*c2 - 2*G*m))

def detadr(r, pc2, m, eta, epsc2, cs2c2):
    f = 1. - 2.*Gc2*m/r
    A = 2. * r * (1/Gc2 - 3.*m/r - TWOPI*r**2*(epsc2 + 3.*pc2))
    B = r*(6./Gc2 - FOURPI*r**2*(epsc2 + pc2)*(3. + 1./cs2c2)) ### NOTE: the inverse sound speed can do bad things here...
    return (eta*(eta - 1.)*r*f/Gc2 + A*eta - B)/(m + FOURPI*r**3*pc2) # from Landry+Poisson PRD 89 (2014)

def domegadr(r, pc2, m, omega, epsc2):
    f = 1. - 2.*Gc2*m/r
    F = FOURPI*r**3*(epsc2 + pc2)/(r/Gc2 - 2*m)
    return (omega*(omega + 3.) - F*(omega + 4.))*r*f/(Gc2*m + FOURPI*Gc2*r**3*pc2)

#------------------------

def dvecdr(vec, r, eos):
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
        detadr(r, pc2, m, eta, epsc2, cs2c2), \
        domegadr(r, pc2, m, omega, epsc2), \
        dmbdr(r, m, dmdr(r, rho))

#-------------------------------------------------

def initial_condition(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    """determines the initial conditions for a stellar model with central pressure pc
    this is done by analytically integrating the TOV equations over very small radii to avoid the divergence as r->0
    """
    ec2i = np.interp(pc2i, eos[0], eos[1])
    rhoi = np.interp(pc2i, eos[0], eos[2])
    cs2c2i = np.interp(pc2i, eos[0], eos[3])

    pc2 = (1. - frac)*pc2i ### assume a constant slope over a small change in the pressure
    r = (frac*pc2i / ( G * (ec2i + pc2i) * (ec2i/3. + pc2i) * TWOPI ) )**0.5 ### solve for the radius that corresponds to that small change

    m = FOURPI * r**3 * ec2i / 3.  # gravitational mass
    mb = FOURPI * r**3 * rhoi / 3. # baryon mass

    eta = 2. + FOURPI * G * r**2 * (9.*pc2i + 13.*ec2i + 3.*(pc2i+ec2i)/cs2c2i)/21. # intial perturbation for dimensionless tidal deformability
    omega = 16.*np.pi * G * r**2 * (pc2i + ec2i)/5. # initial frame-dgragging function

    return r, (pc2, m, eta, omega, mb)

#------------------------

### the solver that yields macroscopic quantites
MACRO_COLS = ['M', 'R', 'Lambda', 'I', 'Mb'] ### the column names for what we compute

def integrate(
        pc2i,
        eos,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc" and equation of state described by energy density "eps" and pressure "p"
    expects eos = (pressure, energy_density)
    """
    ### define initial condition
    r, vec = initial_condition(pc2i, eos, frac=initial_frac)
    if vec[0] < 0: ### guarantee that we enter the loop
        raise RuntimeError('bad initial condition!')
    
    while vec[0] > 0: ### continue until pressure vanishes
        vec0 = vec[:] # store the current location as the old location
        r0 = r

        ### estimate the radius at which this p will vanish via Newton's method
        r = r0 + max(min_dr, min(max_dr, guess_frac * abs(vec[0]/dvecdr(vec, r, eos)[0])))

        ### integrate out until we hit that estimate
        vec = odeint(dvecdr, vec0, (r0, r), args=(eos,), rtol=rtol, hmax=max_dr)[-1,:] ### retain only the last point

    ### interpolate to find stellar surface
    p0, m0, eta0, omega0, mb0 = vec0
    p1, m1, eta1, omeag1, mb1 = vec

    p = [p0, p1]

    # radius
    r = np.interp(0, p, [r0, r])

    # gravitational mass and baryonic mass
    m = np.interp(0, p, [m0, m1])
    mb = np.interp(0, p, [mb0, mb1])

    # tidal deformability
    eta = np.interp(0, p, [eta0, eta1])
    C = Gc2*m/r # compactness
    fR = 1.-2.*C
    F = hyp2f1(3., 5., 6., 2.*C) # a hypergeometric function

    z = 2.*C
    dFdz = (5./(2.*z**6.)) * (z*(z*(z*(3.*z*(5. + z) - 110.) + 150.) - 60.) / (z - 1.)**3 + 60.*np.log(1. - z))
    RdFdr = -2.*C*dFdz # log derivative of hypergeometric function

    k2el = 0.5*(eta - 2. - 4.*C/fR) / (RdFdr -F*(etaR + 3. - 4.*C/fR)) # gravitoelectric quadrupole Love number
    l = (2./3.)*(k2el/C**5)

    # moment of inertia
    omega = np.interp(0, p, [omega0, omega1])
    i = (omega/(3. + omega)) * R**3/(2.*Gc2)

    # convert to "standard" units
    m /= Msun ### reported in units of solar masses, not grams
    mb /= Msun
    r *= 1e-5 ### convert from cm to km
    i /= 1e45 ### normalize this to a common value but still in CGS

    return m, r, l, i, mb
