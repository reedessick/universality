"""a simple module that implements cosmological functionality
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from universality.utils.units import c ### speed of light in cm/s

#-------------------------------------------------

DEFAULT_DZ = 1e-3 ### should be good enough for most things we want to do

#------------------------

lyr_per_Mpc = 3.216156*1e6
Mpc_per_lyr = 1./lyr_per_Mpc

cm_per_lyr = c*86400*365
lyr_per_cm = 1./cm_per_lyr

Mpc_per_cm = Mpc_per_lyr * lyr_per_cm
cm_per_Mpc = 1./Mpc_per_cm

cm_per_km = 1e5
km_per_cm = 1./cm_per_km

Mpc_per_km = Mpc_per_lyr * lyr_per_cm * cm_per_km
km_per_Mpc = 1./Mpc_per_km

#-------------------------------------------------

class Cosmology(object):
    """a class that implements specific cosmological computations.
**NOTE**, we work in CGS units throughout, so Ho must be specified in s**-1 and distances are specified in cm
    """

    def __init__(self, Ho, OmegaMatter, OmegaRadiation, OmegaLambda, OmegaKappa):
        self.Ho = Ho
        self.c_over_Ho = c/self.Ho

        self.OmegaMatter = OmegaMatter
        self.OmegaRadiation = OmegaRadiation
        self.OmegaLambda = OmegaLambda
        self.OmegaKappa = 1 - (OmegaMatter + OmegaRadiation + OmegaLambda)

        assert self.OmegaKappa==0, 'we only implement flat cosmologies! OmegaKappa must be 0'

        self._init_memo() ### instantiate the memorized interpolation arrays

    def _init_memo(self):
        """instantiate things to "memorize" results and cache them
        """
        self._distances = {
            'z':np.array([0]),
            'DL':np.array([0]),
            'Dc':np.array([0]),
            'Vc':np.array([0]),
        }

    def z2E(self, z):
        """returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1+z
        return (self.OmegaLambda + self.OmegaKappa*one_plus_z**2 + self.OmegaMatter*one_plus_z**3 + self.OmegaRadiation*one_plus_z**4)**0.5

    def dDcdz(self, z):
        """returns (c/Ho)/E(z)
        """
        return self.c_over_Ho/self.z2E(z)

    def dVcdz(self, z, Dc):
        """returns dVc/dz
        """
        return 4*np.pi * Dc**2 * self.dDcdz(z)

    #---

    def Dc2z(self, Dc, dz=DEFAULT_DZ):
        """return redshifts for each Dc specified.
        """
        max_Dc = np.max(Dc)
        if max_Dc > np.max(self._distances['Dc']):
            self._extend(max_Dc=max_Dc, dz=dz)

        np.interp(Dc, self._distances['Dc'], self._distances['z'])

    def z2Dc(self, z, dz=DEFAULT_DZ):
        """return Dc for each z specified
        """
        max_z = np.max(z)
        while max_z > np.max(self._distances['z']):
            self._extend(max_z=max_z, dz=dz)

        return np.interp(z, self._distances['z'], self._distances['Dc'])

    #---

    def DL2z(self, DL, dz=DEFAULT_DZ):
        """returns redshifts for each DL specified. This is done by numerically integrating to obtain DL(z) up to the maximum required DL and then interplating to obtain z(DL)
        """
        max_DL = np.max(DL)
        if max_DL > np.max(self._distances['DL']): ### need to extend the integration
            self._extend(max_DL=max_DL, dz=dz)

        return np.interp(DL, self._distances['DL'], self._distances['z'])

    def z2DL(self, z, dz=DEFAULT_DZ):
        """returns luminosity distance at the specified redshifts
        """
        max_z = np.max(z)
        while max_z > np.max(self._distances['z']):
            self._extend(max_z=max_z, dz=dz)

        return np.interp(z, self._distances['z'], self._distances['DL'])

    #---

    def Vc2z(self, Vc, dz=DEFAULT_DZ):
        max_Vc = np.max(Vc)
        if max_Vc > np.max(self._distances['Vc']):
            self._extend(max_Vc=max_Vc, dz=DEFAULT_DZ)

        return np.interp(Vc, self._distances['Vc'], self._distances['z'])

    def z2Vc(self, z, dz=DEFAULT_DZ):
        max_z = np.max(z)
        if max_z > np.max(self._distances['z']):
            self._extend(max_z=max_z, dz=DEFAULT_DZ)

        return np.interp(z, self._distances['z'], self._distances['Vc'])

    #---

    def _extend(self, max_DL=-np.infty, max_Dc=-np.infty, max_z=-np.infty, max_Vc=-np.infty, dz=DEFAULT_DZ):
        """integrate out integration objects.
NOTE, this could be slow...
        """
        z_list = list(self._distances['z'])
        Dc_list = list(self._distances['Dc'])
        Vc_list = list(self._distances['Vc'])

        current_z = z_list[-1]
        current_Dc = Dc_list[-1]
        current_DL = current_Dc * (1+current_z)
        current_Vc = Vc_list[-1]

        # initialize integration
        current_dDcdz = self.dDcdz(current_z)
        current_dVcdz = self.dVcdz(current_z, current_Dc)

        # iterate until we are far enough
        while (current_Dc < max_Dc) or (current_DL < max_DL) or (current_z < max_z) or (current_Vc < max_Vc):
            current_z += dz                                ### increment

            dDcdz = self.dDcdz(current_z)                  ### evaluated at the next step
            current_Dc += 0.5*(current_dDcdz + dDcdz) * dz ### trapazoidal approximation
            current_dDcdz = dDcdz                          ### update

            dVcdz = self.dVcdz(current_z, current_Dc)      ### evaluated at the next step
            current_Vc += 0.5*(current_dVcdz + dVcdz) * dz ### trapazoidal approximation
            current_dVcdz = dVcdz                          ### update

            current_DL = (1+current_z)*current_Dc          ### update

            Dc_list.append(current_Dc)                     ### append
            Vc_list.append(current_Vc)
            z_list.append(current_z)

        # record
        self._distances['z'] = np.array(z_list, dtype=float)
        self._distances['Dc'] = np.array(Dc_list, dtype=float)
        self._distances['Vc'] = np.array(Vc_list, dtype=float)
        self._distances['DL'] = (1+self._distances['z'])*self._distances['Dc'] ### only holds in a flat universe

#-------------------------------------------------

### Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 67.32 * Mpc_per_km ### (km/s/Mpc) * (Mpc/km) = s**-1
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1. - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.
PLANCK_2018_OmegaKappa = 0.

PLANCK_2018_Cosmology = Cosmology(PLANCK_2018_Ho, PLANCK_2018_OmegaMatter, PLANCK_2018_OmegaRadiation, PLANCK_2018_OmegaLambda, PLANCK_2018_OmegaKappa)

#------------------------

DEFAULT_COSMOLOGY = PLANCK_2018_Cosmology
