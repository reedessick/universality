"""a module that houses plotting routines for common uses, defining common parameters for plotting logic
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
if matplotlib.__version__ < '1.3.0':
    plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    'xtick.direction':'in',
    'xtick.bottom':True,
    'xtick.top':True,
    'ytick.direction':'in',
    'ytick.left':True,
    'ytick.right':True,
})

### non-standard libraries
from universality.kde import DEFAULT_BANDWIDTH
from universality.stats import (logkde2levels, neff)

#-------------------------------------------------

DEFAULT_LEVELS=[0.5, 0.9]
DEFAULT_NUM_POINTS = 25
DEFAULT_BANDWIDTH = DEFAULT_BANDWIDTH
DEFAULT_FIGWIDTH = DEFAULT_FIGHEIGHT = 6

DEFAULT_COV_FIGWIDTH = 10
DEFAULT_COV_FIGHEIGHT = 4

DEFAULT_COLOR1 = 'k'
DEFAULT_COLOR2 = 'r'
DEFAULT_COLOR3 = 'b'
DEFAULT_COLORMAP = 'RdGy_r'

DEFAULT_TRUTH_COLOR = 'b'

DEFAULT_LINEWIDTH = 1.
DEFAULT_LINESTYLE = 'solid'
DEFAULT_MARKER = None
DEFAULT_ALPHA = 1.0
DEFAULT_LIGHT_ALPHA = 0.25

DEFAULT_FIGTYPES = ['png']
DEFAULT_DPI = 100
DEFAULT_DIRECTORY = '.'

#------------------------

CORNER_LEFT = 0.8
CORNER_RIGHT = 0.2
CORNER_TOP = 0.2
CORNER_BOTTOM = 0.8

HSPACE = 0.05
WSPACE = 0.05

#------------------------

MAIN_AXES_POSITION = [0.18, 0.39, 0.77, 0.56]
RESIDUAL_AXES_POSITION = [0.18, 0.18, 0.77, 0.2]
AXES_POSITION = [
    MAIN_AXES_POSITION[0],
    RESIDUAL_AXES_POSITION[1],
    MAIN_AXES_POSITION[2],
    MAIN_AXES_POSITION[1]+MAIN_AXES_POSITION[3]-RESIDUAL_AXES_POSITION[1]
]

#-------------------------------------------------

def setp(*args, **kwargs):
    return plt.setp(*args, **kwargs)

def figure(*args, **kwargs):
    return plt.figure(*args, **kwargs)

def subplot(*args, **kwargs):
    return plt.subplot(*args, **kwargs)

def subplots_adjust(*args, **kwargs):
    return plt.subplots_adjust(*args, **kwargs)

def save(basename, fig, figtypes=DEFAULT_FIGTYPES, directory=DEFAULT_DIRECTORY, verbose=False, **kwargs):
    template = os.path.join(directory, basename+'.%s')
    for figtype in figtypes:
        figname = template%figtype
        if verbose:
            print('saving: '+figname)
        fig.savefig(figname,  **kwargs)

def weights2color(weights, basecolor, prefact=100., minimum=1e-3):
    Nsamp = len(weights)
    scatter_color = np.empty((Nsamp, 4), dtype=float)
    scatter_color[:,:3] = matplotlib.colors.ColorConverter().to_rgb(basecolor)
    mw, Mw = np.min(weights), np.max(weights)
    if np.all(weights==Mw):
        scatter_color[:,3] = max(min(1., 1.*prefact/Nsamp), minimum) ### equal weights
    else:
        scatter_color[:,3] = weights/np.max(weights)
        scatter_color[:,3] *= prefact/neff(weights)
        scatter_color[scatter_color[:,3]>1,3] = 1 ### give reasonable bounds
        scatter_color[scatter_color[:,3]<minimum,3] = minimum ### give reasonable bounds
    return scatter_color

def close(fig):
    plt.close(fig)
