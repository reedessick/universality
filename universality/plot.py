__doc__ = "a module that houses plotting routines for common uses"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams.update({'font.family':'serif',}) # 'text.usetex':True})

from corner import corner

### non-standard libraries
from . import utils

#-------------------------------------------------

DEFAULT_BANDWIDTH = 0.1

#-------------------------------------------------

def kde_corner(data, bandwidths=None, labels=None, ranges=None, truths=None, weights=None):
    """
    should be mostly equivalent to corner.corner, except we build our own KDEs and the like
    """
    ### check data formats
    Nsamp, Ncol = data.shape

    if bandwidths is None:
        bandwidths = [DEFAULT_BANDWIDTH]*Ncol
    else:
        assert len(bandwidths)==Ncol, 'must have the same number of columns in data and bandwidths'

    if labels is None:
        labels = [str(i) for i in xrange(Ncol)]
    else:
        assert len(labels)==Ncol, 'must have the same number of columns in data and labels'

    if ranges is None:
        ranges = [(np.min(data[:,i]), np.max(data[:,i])) for i in xrange(Ncol)]
    else:
        assert len(ranges)==Ncol, 'must have the same number of columns in data and ranges'

    if truths is None:
        truths = [None]*Ncol
    else:
        assert len(truths)==Ncol, 'must have the same number of columns in data and truths'

    if weights is None:
        weights = np.ones(Nsamp, dtype=float)/Nsamp
    else:
        assert len(weights)==Nsamp, 'must have the same number of rows in data and weights'

    ### construct figure and axes objects
    raise NotImplementedError

    ### iterate over columns, building 2D KDEs as needed
    raise NotImplementedError

    ### return figure
    raise NotImplementedError

#-------------------------------------------------

### FIXME: move all the sanity-check plots from within executables to in here...
