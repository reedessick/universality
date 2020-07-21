"""description of the TOV equations in terms of the radius"""
__author__ = "reed.essick@gmail.com"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

fourpi = 4*np.pi

#-------------------------------------------------

def dp_dr(r, eps, p, m):
    """TOV equation for dp/dr"""
    return -(eps + p) * (m + fourpi * r**3 * p) / (r*(r - 2*m))

def dm_dr(r, eps):
    """TOV equation for dm/dr"""
    return fourpi * r**2 * eps

#-------------------------------------------------

def dX_dr(r, eps, p, m):
    """constructs the 1st order differential vector: [dp/dr, dm/dr]"""
    return np.array([dp_dr(r, eps, p, m), dm_dr(r, eps)])
