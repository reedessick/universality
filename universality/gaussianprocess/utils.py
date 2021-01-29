"""utilities associated with the crust EoS and integrating GP draws
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality import eos
from universality.utils import utils

#-------------------------------------------------

DEFAULT_SIGMA_LOGPRESSUREC2 = 0.0

#-------------------------------------------------
# utilities associated with integrating the EOS realizations
#-------------------------------------------------

def integrate_phi(
        pressurec2,
        phi,
        reference_pressurec2,
        sigma_logpressurec2=DEFAULT_SIGMA_LOGPRESSUREC2,
        stitch_below_reference_pressure=False,
        include_baryon_density=True,
        include_cs2c2=False,
        include_baryon_chemical_potential=False,
        verbose=False,
    ):
    '''numerically integrates the realization from a GP to get the EoS
    '''
    denergy_densitydpressure = (1 + np.exp(phi)) ### this is the definition of phi=log(de/dp - 1)

    # perform numeric integration with trapazoidal approximation
    # NOTE: this could probably be improved...
    if verbose:
        print('performing numeric integration for energy_density via trapazoidal approximation')
    reference_pressurec2 = np.exp(np.log(reference_pressurec2) + np.random.randn()*sigma_logpressurec2)
    energy_densityc2 = dedp2e(denergy_densitydpressure, pressurec2, reference_pressurec2)
    if stitch_below_reference_pressure:
        energy_densityc2, pressurec2 = stitch_below_pressure(energy_densityc2, pressurec2, reference_pressurec2)

    ### throw away pressures corresponding to negative energy densities!
    truth = energy_densityc2 >= 0

    pressurec2 = pressurec2[truth]
    energy_densityc2 = energy_densityc2[truth]

    data = [pressurec2, energy_densityc2]
    cols = ['pressurec2', 'energy_densityc2']

    #------------------------

    # perform numeric integration to obtain baryon density along with energy_density and pressure
    if include_baryon_density:
        if verbose:
            print('performing numeric integration for baryon_density via trapazoidal approximation')
        baryon_density = e_p2rho(energy_densityc2, pressurec2, reference_pressurec2, verbose=verbose)
        if stitch_below_reference_pressure:
            ### the following indecies should match up because of the call to stitch_below_pressure made above
            baryon_density[pressurec2<=reference_pressurec2] = eos.CRUST_BARYON_DENSITY[eos.CRUST_PRESSUREC2<=reference_pressurec2]

        data.append(baryon_density)
        cols.append('baryon_density')

    # take numeric derivative to estimate sound speed
    if include_cs2c2:
        if verbose:
            print('computing cs2c2 via numeric differentiation')
        data.append(utils.num_dfdx(energy_densityc2, pressurec2))
        cols.append('cs2c2')

    # take numeric derivative to estimate chemical potential
    if include_baryon_chemical_potential:
        if verbose:
            print('computing baryon_chemical_potential via numeric differentiation')
        data.append(utils.num_dfdx(baryon_density, energy_densityc2) * units.Mnuc)
        cols.append('baryon_chemical_potential')

    # return
    data = np.transpose(np.array(data, dtype=float))
    return data, cols

def stitch_below_pressure(energy_densityc2, pressurec2, reference_pressurec2):
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

def e_p2rho(energy_densityc2, pressurec2, reference_pressurec2, verbose=True):
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
        if verbose:
            print('WARNING: enforcing the requirement that baryon_density <= energy_densityc2 by hand below pressurec2=%.6e'%(pressurec2[truth][-1]))
        baryon_density[truth] = energy_densityc2[truth]

    ### return
    return baryon_density
