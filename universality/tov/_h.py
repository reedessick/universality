"""description of the TOV equations in terms of the enthalpy"""
__author__ = "reed.essick@gmail.com"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

fourpi = 4*np.pi

#-------------------------------------------------

def u(r):
    """aux variables for enthalpy description"""
    return r**2

def v(r, m):
    """aux variable for enthalpy description"""
    return m/r

#------------------------

def du_dh(u, v, eps, p):
    """du/dh for enthalpy description"""
    return -2*u * (1 - 2*v) / (fourpi * u * p + v)

def dv_dh(u, v, eps, p):
    """dv/dh for enthalpy description"""
    return - (1 - 2*v) * (fourpi * u * eps - v) / (fourpi * u * p + v)

#-------------------------------------------------

def dX_dh(u, v, eps, p):
    """constructs the 1st order differential vector: [du/dh, dv/dh]"""
    return np.array([du_dh(u, v, eps, p), dv_dh(u, v, eps, p)])
