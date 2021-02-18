"""definitions of units used throughout the library. Most calculations assume CGS
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

G = 6.674e-8        # newton's constant in (g^-1 cm^3 s^-2)
c = (299792458.0*100) # speed of light in (cm/s) NOTE, this really needs to be a float, not an int
c2 = c**2

Msun = 1.989e33     # mass of the sun in (g)

Mproton = 1.67262192370e-24 ### mass of the proton (g)
Mneutron = 1.67492749804e-14 ### mass of the neutron (g)

Mnuc = Mproton ### an arbitrary choice for reference.

rho_nuc = 2.8e14 # g/cm^3
