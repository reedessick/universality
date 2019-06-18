"""a simple module that implements cosmological functionality
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import utils

#-------------------------------------------------

DEFAULT_DZ = 1e-2 ### should be good enough for most things we want to do

#------------------------

lyr_per_Mpc = 3.216156*1e6
Mpc_per_lyr = 1./lyr_per_Mpc

cm_per_lyr = utils.c*86400*365
lyr_per_cm = 1./cm_per_ly

Mpc_per_cm = Mpc_per_lyr * lyr_per_cm
cm_per_Mpc = 1./Mpc_per_cm

cm_per_km = 1e5
km_per_cm = 1./cm_per_km

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

        self._init_memo() ### instantiate the memorized interpolation arrays

    def _init_memo(self):
        """instantiate things to "memorize" results and cache them
        """
        self._distances = {'z':np.array([0]), 'DL':np.array([0])}

    def z2E(self, z):
        """returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1+z
        return (self.OmegaLambda + self.OmegaKappa*one_plus_z**2 + self.OmegaMatter*one_plus_z**3 + self.OmegaRadiation*one_plus_z**4)**0.5

    def dDLdz(self, z):
        """returns (c/Ho)/E(z)
        """
        return self.c_over_Ho/self.z2E(z)

    def DL2z(self, DL, dz=DEFAULT_DZ):
        """returns redshifts for each DL specified. This is done by numerically integrating to obtain DL(z) up to the maximum required DL and then interplating to obtain z(DL)
        """
        max_DL = np.max(DL)
        if max_DL > np.max(self._distances['DL']): ### need to extend the integration
            self._extend_DL(max_DL)

        return np.interpolate(DL, self._distances['DL'], self._distances['z'])

    def z2DL(self, z, dz=DEFAULT_DZ):
        """returns luminosity distance at the specified redshifts
        """
        max_z = np.max(z)
        while max_z > np.max(self._distances['z']):
            self._extend_DL(2*np.max(self._distances['DL']), dz=dz) ### double the luminosity distance

        return np.interpolate(z, self._distances['z'], self._distances['DL'])

    def _extend_DL(self, max_DL):
        """integrates out distances until we hit max_DL
        """
        z_list = list(self._distances['z'])
        DL_list = list(self._distances['DL'])

        current_DL = DL_list[-1]
        current_z = z_list[-1]

        # initialize integration
        current_dDLdz = self.dDLdz(current_z)

        # iterate until we are far enough
        while current_DL < max_DL:
            current_z += dz                                ### increment
            dDLdz = self.dDLdz(current_z)                  ### evaluated at the next step
            current_DL += 0.5*(current_dDLdz + dDLdz) * dz ### trapazoidal approximation
            current_dDLdz = dDLdz                          ### update

            DL_list.append(current_DL)                     ### append
            z_list.append(current_z)

        self._distances['z'] = np.array(z_list, dtype=float)
        self._distances['DL'] = np.array(DL_list, dtype=float)

#-------------------------------------------------

### Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 67.32 * cm_per_Mpc * cm_per_km ### km/s/Mpc * (Mpc/lyr) * (lyr/cm) * (cm/km) = s**-1
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1. - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.
PLANCK_2018_OmegaKappa = 0.

PLANCK_2018_Cosmology = Cosmology(PLANCK_2018_Ho, PLANCK_2018_OmegaMatter, PLANCK_2018_OmegaRadiation, PLANCK_2018_OmegaLambda, PLANCK_2018_OmegaKappa)

#------------------------

DEFAULT_COSMOLOGY = PLANCK_2018_Cosmology
