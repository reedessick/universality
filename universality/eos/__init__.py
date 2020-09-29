"""a module that houses a few proposed EoS from the literature
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from pkg_resources import (resource_listdir, resource_filename)

from universality.utils.io import load

#-------------------------------------------------

eospaths = dict((path[:-4], resource_filename(__name__, path)) for path in resource_listdir(__name__,'') if path[-3:]=='csv')
DEFAULT_CRUST_EOS = 'sly'

#-------------------------------------------------

def set_crust(crust_eos=DEFAULT_CRUST_EOS):
    global CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY
    CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2, CRUST_BARYON_DENSITY = \
        load(eospaths.get(crust_eos, crust_eos), columns=['pressurec2', 'energy_densityc2', 'baryon_density'])[0].transpose()

def crust_energy_densityc2(pressurec2):
    """
    return energy_densityc2 for the crust from Douchin+Haensel, arXiv:0111092
    this is included in our repo within "sly.csv", taken from Ozel's review.
    """
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_ENERGY_DENSITYC2)

def crust_baryon_density(pressurec2):
    return np.interp(pressurec2, CRUST_PRESSUREC2, CRUST_BARYON_DENSITY)

#-------------------------------------------------

### load the sly EOS for stitching logic
set_crust() ### use this as the crust!
