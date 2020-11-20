"""utilities associated with the crust EoS and integrating GP draws
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality import eos
from universality.utils import utils

#-------------------------------------------------
# utilities associated with integrating the EOS realizations
#-------------------------------------------------

def stitch_below_reference_pressure(energy_densityc2, pressurec2, reference_pressurec2):
    """reutrn energy_densityc2, pressurec2"""
    crust_truth = eos.CRUST_PRESSUREC2 < reference_pressurec2
    eos_truth = reference_pressurec2 <= pressurec2
    return np.concatenate((eos.CRUST_ENERGY_DENSITYC2[crust_truth], energy_densityc2[eos_truth])), np.concatenate((eos.CRUST_PRESSUREC2[crust_truth], pressurec2[eos_truth]))

### integration routines
def dedp2e(denergy_densitydpressure, pressurec2, reference_pressurec2):
    """
    integrate to obtain the energy density
    if stitch=True, map the lower pressures onto a known curst below the reference pressure instead of just matching at the reference pressure
    """
    energy_densityc2 = utils.num_intfdx(pressurec2, denergy_densitydpressure)

    ### match at reference pressure
    energy_densityc2 += eos.crust_energy_densityc2(reference_pressurec2) - np.interp(reference_pressurec2, pressurec2, energy_densityc2)

    return energy_densityc2

def e_p2rho(energy_densityc2, pressurec2, reference_pressurec2):
    """
    integrate the first law of thermodynamics
        deps = (eps + p) (drho/rho)
    from this expression, we solve for (eps/rho) because we can guarantee eps/rho >= 1
        dlog(eps/rho) / dlog(eps) = p / (eps+p)
    """

    ''' ### OLD IMPLEMENTATION, seemed to be unstable due to numerical errors in an exponent
    baryon_density = np.ones_like(pressurec2, dtype='float')

    integrand = 1./(energy_densityc2+pressurec2)
    baryon_density[1:] *= np.exp(np.cumsum(0.5*(integrand[1:]+integrand[:-1])*(energy_densityc2[1:]-energy_densityc2[:-1]))) ### multiply by this factor

    ### FIXME: match baryon density to energy density at reference pressure
    #baryon_density *= ec2 / np.interp(ref_pc2, pressurec2, baryon_density)

    ### match at the lowest allowed energy density
    baryon_density *= eos.crust_baryon_density(reference_pressurec2)/np.interp(reference_pressurec2, pressurec2, baryon_density)
    '''

    ### NEW IMPLEMENTATION, should be more stable numerically
    # compute the numeric integratoin
    integral = utils.num_intfdx(np.log(energy_densityc2), pressurec2/(energy_densityc2+pressurec2))

    # subtract out the reference value
    integral -= np.interp(reference_pressurec2, pressurec2, integral)

    # multiply by cofactor to get the baryon density
    baryon_density = energy_densityc2 * (eos.crust_baryon_density(reference_pressurec2)/eos.crust_energy_densityc2(reference_pressurec2)) * np.exp(-integral)

    ### fix an annoying issue with numerical stability at very low pressures (wiht sparse samples)
    truth = energy_densityc2 < baryon_density
    if np.any(truth):
        print('WARNING: enforcing the requirement that baryon_density <= energy_densityc2 by hand below pressurec2=%.6e'%(pressurec2[truth][-1]))
        baryon_density[truth] = energy_densityc2[truth]

    ### return
    return baryon_density
