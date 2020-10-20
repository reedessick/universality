"""a module that houses TOV solvers based on the log(enthalpy per unit rest mass)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

import numpy as np
from scipy.integrate import odeint

from universality.utils import utils
from universality.utils.units import (G, c2)

#-------------------------------------------------

DEFAULT_MAX_DLOGH = 0.1 ### maximum step size allowed within the integrator (dimensionless)
DEFAULT_GUESS_FRAC = 0.1 ### how much of the way to the vanishing pressure we guess via Newton's method

DEFAULT_INITIAL_FRAC = 1e-8 ### the initial change in pressure we allow when setting the intial conditions

DEFAULT_METHOD = 'RK45' ### integrator parameters
DEFAULT_RTOL = 1e-6

#------------------------

TWOPI = 2*np.pi
FOURPI = 2*TWOPI

#-------------------------------------------------
### Formulation of the TOV equations in terms of the log(enthalpy per unit rest mass) = log( (eps+p)/rho )
#-------------------------------------------------

def eos2logh(pc2, ec2):
    return utils.num_intfdx(pc2, 1./(ec2+pc2))

def drdlogh(r, m, pc2):
    return - r*(r*c2 - 2*G*m) / (G*(m + FOURPI * r**3 * pc2))

def dmdlogh(r, epsc2, dr_dlogh):
    return FOURPI * epsc2 * r**2 * dr_dlogh

def dmbdlogh(r, m, dm_dlogh):
    return dm_dlogh * (1 - 2*G*m/(r*c2))**-0.5

def dvecdlogh(logh, vec, eos):
    pc2 = np.interp(logh, eos[0], eos[1])
    ec2 = np.interp(logh, eos[0], eos[2])
    m, r = vec
    dr_dlogh = drdlogh(r, m, pc2)
    dm_dlogh = dmdlogh(r, ec2, dr_dlogh)
    return [dm_dlogh, dr_dlogh, dmbdlogh(r, m, dm_dlogh)]

def initial_condition(loghi, eos, frac=DEFAULT_INITIAL_FRAC):
    '''analytically solve for the initial condition around the divergence at r=0
    '''
    pc2 = np.interp(loghi, eos[0], eos[1])
    ec2 = np.interp(loghi, eos[0], eos[2])

    r = ( 3.*frac / (TWOPI*(ec2 + 3.*pc2)) )**0.5
    logh = loghi * (1 - frac)
    m = mb = FOURPI * ec2 * r**3 / 3.

    return logh, (m, r, mb)

#------------------------

### the solver that yields macroscopic quantites
MACRO_COLS = ['M', 'R', 'Mb'] ### the column names for what we compute

def integrate(
        pc2i,
        eos,
        max_dlogh=DEFAULT_MAX_DLOGH,
        initial_frac=DEFAULT_INITIAL_FRAC,
        method=DEFAULT_METHOD,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (pressurec2, energy_densityc2)
    """
    ### compute the log(enthalpy per rest mass)
    pc2, ec2 = eos
    logh = eos2logh(pc2, ec2)

    eos = (logh, pc2, ec2)

    ### define initial condition
    loghi, vec = initial_contitions(loghi, eos, frac=initial_frac)

    ### integrate out until we hit termination condition
    vec = odeint(dvecdlogh, (loghi, 0), vec, args=(eos,), method=method, rtol=rtol, hmax=max_dlogh)

    ### extract final values at the surface
    return vec[-1,:]
