"""a module that houses TOV solvers in the "standard" formulation
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.integrate import odeint

from universality.utils.units import (G, c2)

#-------------------------------------------------

DEFAULT_MAX_DR = 1e4 ### maximum step size allowed within the integrator (in standard units, which should be in cm)
DEFAULT_MIN_DR = 1.0 ### the smallest step size we allow (in standard units, which should be cm)
DEFAULT_GUESS_FRAC = 0.1 ### how much of the way to the vanishing pressure we guess via Newton's method

DEFAULT_INITIAL_FRAC = 1e-8 ### the initial change in pressure we allow when setting the intial conditions

DEFAULT_METHOD = 'RK45' ### integrator parameters
DEFAULT_RTOL = 1e-6

#------------------------

TWOPI = 2*np.pi
FOURPI = 2*TWOPI

#-------------------------------------------------
### Standard formulation of the TOV equations
#-------------------------------------------------

### basic evolutionary equations

def dmdr(r, epsc2):
    return FOURPI * r**2 * epsc2

def dpc2dr(r, pc2, m, epsc2):
    return - G * (epsc2 + pc2)*(m + FOURPI * r**3 * pc2)/(r * (r*c2 - 2*G*m))

def dvecdr(r, vec, eos):
    '''returns d(p, m)/dr
    expects: pressurec2, energy_densityc2 = eos
    '''
    pc2, m = vec
    epsc2 = np.interp(pc2, *eos)
    return dpc2dr(r, pc2, m, epsc2), dmdr(r, epsc2)

def initial_condition(pc2i, eos, frac=DEFAULT_INITIAL_FRAC):
    """determines the initial conditions for a stellar model with central pressure pc
    this is done by analytically integrating the TOV equations over very small radii to avoid the divergence as r->0
    """
    ec2i = np.interp(pc2i, *eos)
    
    pc2 = (1. - frac)*pc2i ### assume a constant slope over a small change in the pressure
    r = (frac*pc2i / ( G * (ec2i + pc2i) * (ec2i/3. + pc2i) * TWOPI ) )**0.5 ### solve for the radius that corresponds to that small change
    m = FOURPI * r**3 * ec2i / 3.

    return r, [pc2, m]

#------------------------

### the solver that yields macroscopic quantites
MACRO_COLS = ['M', 'R'] ### the column names for what we compute
def integrate(
        pc2i,
        eos,
        min_dr=DEFAULT_MIN_DR,
        max_dr=DEFAULT_MAX_DR,
        guess_frac=DEFAULT_GUESS_FRAC,
        initial_frac=DEFAULT_INITIAL_FRAC,
        method=DEFAULT_METHOD,
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
        r = r0 + max(min_dr, guess_frac * vec[0]/dvecdr(r, vec, eos)[0])

        ### integrate out until we hit that estimate
        vec = odeint(dvecdr, (r0, r), vec, args=(eos,), method=method, rtol=rtol, hmax=max_dr)

    ### interpolate to find stellar surface
    p = [vec0[0], vec[0]]
    m = [vec0[1], vec[1]]
    r = [r0, r]

    return np.interp(0, p, m), np.interp(0, p, r)
