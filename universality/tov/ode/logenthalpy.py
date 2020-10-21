"""a module that houses TOV solvers based on the log(enthalpy per unit rest mass)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

import numpy as np
from scipy.integrate import odeint

from universality.utils import utils
from universality.utils.units import (G, c2, Msun)

#-------------------------------------------------

DEFAULT_MIN_DLOGH = 1e-8
DEFAULT_MAX_DLOGH = 1e-3 ### maximum step size allowed within the integrator (dimensionless)

DEFAULT_INITIAL_FRAC = 1e-3 ### the initial change in pressure we allow when setting the intial conditions

DEFAULT_RTOL = 1e-4

#------------------------

TWOPI = 2*np.pi
FOURPI = 2*TWOPI

#-------------------------------------------------
### Formulation of the TOV equations in terms of the log(enthalpy per unit rest mass) = log( (eps+p)/rho )
#-------------------------------------------------

def eos2logh(pc2, ec2):
    return utils.num_intfdx(pc2, 1./(ec2+pc2))

def drdlogh(r, m, pc2):
    return - r * (r*c2 - 2*G*m) / (G*(m + FOURPI * r**3 * pc2))

def dmdlogh(r, epsc2, dr_dlogh):
    return FOURPI * r**2 * epsc2 * dr_dlogh

def dmbdlogh(r, m, dm_dlogh):
    return dm_dlogh * (1 - 2*G*m/(r*c2))**-0.5

def dvecdlogh(vec, logh, eos):
    eos0 = eos[0]
    pc2 = np.interp(logh, eos0, eos[1])
    ec2 = np.interp(logh, eos0, eos[2])
    rho = np.interp(logh, eos0, eos[3])

    m, r, mb = vec
    dr_dlogh = drdlogh(r, m, pc2)

    return [dmdlogh(r, ec2, dr_dlogh), dr_dlogh, dmbdlogh(r, m, dmdlogh(r, rho, dr_dlogh))]

def initial_condition(pc2, eos, frac=DEFAULT_INITIAL_FRAC):
    '''analytically solve for the initial condition around the divergence at r=0
    '''
    loghi = np.interp(pc2, eos[1], eos[0])
    ec2 = np.interp(pc2, eos[1], eos[2])
    rho = np.interp(pc2, eos[1], eos[3])

    logh = loghi * (1 - frac)
    r = ( 3.*frac*loghi*c2 / (TWOPI*(ec2 + 3.*pc2)) )**0.5
    m = FOURPI * ec2 * r**3 / 3.
    mb = FOURPI * rho * r**3 / 3.

    return logh, (m, r, mb)

#------------------------

def dvecdH(vec, H, eos):
    """defines the derivative in terms of H = -logh
    """
    logh = -H

    eos0 = eos[0]
    pc2 = np.interp(logh, eos0, eos[1])
    ec2 = np.interp(logh, eos0, eos[2])
    rho = np.interp(logh, eos0, eos[3])

    m, r, mb = vec
    dr_dlogh = - drdlogh(r, m, pc2)

    return [dmdlogh(r, ec2, dr_dlogh), dr_dlogh, dmbdlogh(r, m, dmdlogh(r, rho, dr_dlogh))]

#------------------------

### the solver that yields macroscopic quantites
MACRO_COLS = ['M', 'R', 'Mb'] ### the column names for what we compute

def integrate(
        pc2i,
        eos,
        min_dlogh=DEFAULT_MIN_DLOGH,
        max_dlogh=DEFAULT_MAX_DLOGH,
        initial_frac=DEFAULT_INITIAL_FRAC,
        rtol=DEFAULT_RTOL,
    ):
    """integrate the TOV equations with central pressure "pc2i" and equation of state described by energy density "eps/c2" and pressure "p/c2"
    expects eos = (logenthalpy, pressurec2, energy_densityc2, baryon_density)
    """
    ### define initial condition
    logh0, vec0 = initial_condition(pc2i, eos, frac=initial_frac)
    vec = vec0[:]
    logh = logh0

    ### integrate out until we hit termination condition
    while logh > 0:
        logh0 = logh
        vec0 = vec[:]

        ### guess the next step
        ### FIXME:
        ###     condition on whether (dvec = dvecdH * logh0) is << vec for termination?
        ###     when that is satisfied, just make the final approximation as a single step and be done with it?
        logh = max(0, logh0 - max(min_dlogh, min(max_dlogh, 0.1*logh)))
        vec = odeint(dvecdH, vec0, (-logh0, -logh), args=(eos,), rtol=rtol)[-1,:]

    ### extract final values at the surface
    logh = [0, logh0]

    m = np.interp(0, logh, [vec[0], vec0[0]]) / Msun
    r = np.interp(0, logh, [vec[1], vec0[1]]) * 1e-5 ### cm -> km
    mb = np.interp(0, logh, [vec[2], vec0[2]]) / Msun

    return m, r, mb
