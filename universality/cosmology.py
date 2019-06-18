"""a simple module that implements cosmological functionality
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import utils

#-------------------------------------------------

class Cosmology(object):
    """a class that implements specific cosmological computations.
**NOTE**, we work in CGS units throughout, so Ho must be specified in s**-1
    """
    c = utils.c ### speed of light in cm/s

    def __init__(self, Ho, OmegaMatter, OmegaRadiation, OmegaLambda, OmegaKappa):
        self.Ho = Ho
        self.c_over_Ho = self.c/self.Ho

        self.OmegaMatter = OmegaMatter
        self.OmegaRatiation = OmegaRadiation
        self.OmegaLambda = OmegaLambda
        self.OmegaKappa = 1 - (OmegaMatter + OmegaRadiation + OmegaLambda)

        assert self.OmegaKappa==0, 'we only implement flat cosmologies! OmegaKappa must be 0'

    def z2E(self, z):
        """returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1+z
        return (self.OmegaLambda + self.OmegaKappa*one_plus_z**2 + self.OmegaMatter*one_plus_z**3 + self.OmegaRadiation*one_plus_z**4)**0.5

    def DL2z(self, DL):
        """
        returns redshifts for each DL specified. This is done by numerically integrating to obtain DL(z) up to the maximum required DL and then interplating to obtain z(DL)
        """
        max_DL = np.max(DL)

        raise NotImplementedError('implement numeric integration')

#-------------------------------------------------

### Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 67.32
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1.-PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.
PLANCK_2018_OmegaKappa = 0.

PLANCK_2018_Cosmology = Cosmology(PLANCK_2018_Ho, PLANCK_2018_OmegaMatter, PLANCK_2018_OmegaRadiation, PLANCK_2018_OmegaLambda, PLANCK_2018_OmegaKappa)

#------------------------

DEFAULT_COSMOLOGY = PLANCK_2018_Cosmology
