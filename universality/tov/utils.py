"""utilities associated with the crust EoS and integrating GP draws
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality import eos

#-------------------------------------------------
# utilities associated with integrating the EOS realizations
#-------------------------------------------------

def stitch_below_reference_pressure(energy_densityc2, pressurec2, reference_pressurec2):
    """reutrn energy_densityc2, pressurec2"""
    sly_truth = eos.CRUST_PRESSUREC2 <= reference_pressurec2
    eos_truth = reference_pressurec2 <= pressurec2
    return np.concatenate((eos.CRUST_ENERGY_DENSITYC2[sly_truth], energy_densityc2[eos_truth])), np.concatenate((eos.CRUST_PRESSUREC2[sly_truth], pressurec2[eos_truth]))

### integration routines
def dedp2e(denergy_densitydpressure, pressurec2, reference_pressurec2):
    """
    integrate to obtain the energy density
    if stitch=True, map the lower pressures onto a known curst below the reference pressure instead of just matching at the reference pressure
    """
    energy_densityc2 = np.empty_like(pressurec2, dtype='float')
    energy_densityc2[0] = 0 # we start at 0, so handle this as a special case

    # integrate in the bulk via trapazoidal approximation
    energy_densityc2[1:] = np.cumsum(0.5*(denergy_densitydpressure[1:]+denergy_densitydpressure[:-1])*(pressurec2[1:] - pressurec2[:-1]))

    ### match at reference pressure
    energy_densityc2 += eos.crust_energy_densityc2(reference_pressurec2) - np.interp(reference_pressurec2, pressurec2, energy_densityc2)

    return energy_densityc2

def e_p2rho(energy_densityc2, pressurec2, reference_pressurec2):
    """
    integrate the first law of thermodynamics
        dvarepsilon = rho/(varepsilon+p) drho
    """
    baryon_density = np.ones_like(pressurec2, dtype='float')

    integrand = 1./(energy_densityc2+pressurec2)
    baryon_density[1:] *= np.exp(np.cumsum(0.5*(integrand[1:]+integrand[:-1])*(energy_densityc2[1:]-energy_densityc2[:-1]))) ### multiply by this factor

    ### FIXME: match baryon density to energy density at reference pressure
    #baryon_density *= ec2 / np.interp(ref_pc2, pressurec2, baryon_density)

    ### match at the lowest allowed energy density
    baryon_density *= eos.crust_baryon_density(reference_pressurec2)/np.interp(reference_pressurec2, pressurec2, baryon_density)

    return baryon_density
