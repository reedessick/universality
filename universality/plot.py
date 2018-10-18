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
from .stats import logkde2levels

#-------------------------------------------------

DEFAULT_LEVELS=[0.1, 0.5, 0.9]
DEFAULT_NUM_POINTS = 25
DEFAULT_BANDWIDTH = utils.DEFAULT_BANDWIDTH
DEFAULT_FIGWIDTH = DEFAULT_FIGHEIGHT = 12

DEFAULT_COLOR1 = 'k'
DEFAULT_COLOR2 = 'r'

#------------------------

CORNER_LEFT = 0.8
CORNER_RIGHT = 0.2
CORNER_TOP = 0.2
CORNER_BOTTOM = 0.8

HSPACE = 0.05
WSPACE = 0.05

#------------------------

MAIN_AXES_POSITION = [0.15, 0.46, 0.8, 0.8]
RESIDUAL_AXES_POSITION = [0.15, 0.15, 0.8, 0.3]

#-------------------------------------------------

def kde_corner(
        data,
        bandwidths=None,
        labels=None,
        ranges=None,
        truths=None,
        weights=None,
        color=DEFAULT_COLOR1,
        fig=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        num_points=DEFAULT_NUM_POINTS,
        levels=DEFAULT_LEVELS,
        hist1D=False,  ### plot a normed histogram on the 1D marginal panels in addition to the KDE estimate
        reflect=False, ### reflect data points about the boundaries when performing the KDE; may be expensive...
    ):
    """
    should be mostly equivalent to corner.corner, except we build our own KDEs and the like

    NOTE: we will skip plotting for any column for which any data is nan. In this way, we can plot multiple sets of data on the same axes even if they do not all possess the same columns (just pad with nan's as needed when calling kde_corner).
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
        ranges = [utils.data2range(data[:,i]) for i in xrange(Ncol)]
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
    if fig is None:
        fig = plt.figure(figsize=(figwidth, figheight)) ### FIXME: set figsize based on Ncol?
        plt.subplots_adjust(
            hspace=HSPACE,
            wspace=WSPACE,
            left=CORNER_LEFT/figwidth,
            right=(figwidth-CORNER_RIGHT)/figwidth,
            bottom=CORNER_BOTTOM/figheight,
            top=(figheight-CORNER_TOP)/figheight,
        )

    shape = (num_points, num_points) # used to reshape 2D sampling kdes

    truth =  np.empty(Ncol, dtype=bool) # used to index data within loop
    include = [np.all(data[:,i]==data[:,i]) for i in xrange(Ncol)] # used to determine which data we should skip

    vects = [np.linspace(m, M, num_points) for m, M in ranges] ### grid placement
    dvects = [v[1]-v[0] for v in vects]

    if hist1D:
        Nbins = max(10, int(Nsamp**0.5)/2)
        bins = [np.linspace(m, M, Nbins+1) for m, M in ranges]

    ### iterate over columns, building 2D KDEs as needed
    for row in xrange(Ncol):
        for col in xrange(row+1):
            ax = plt.subplot(Ncol, Ncol, row*Ncol+col+1)

            truth[:] = False

            if include[row] and include[col]: ### plot data for this pair
                # 1D marginal posteriors
                if row==col:
                    truth[col] = True

                    kde = utils.logkde(
                        vects[col],
                        utils.reflect(data[:,truth], ranges[truth]) if reflect else data[:,truth],
                        bandwidths[col],
                    )
                    kde = np.exp(kde - np.max(kde))

                    kde /= np.sum(kde)*dvects[col]
                    ax.plot(vects[col], kde, color=color)

                    if hist1D:
                        ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True)

                else:
                    # marginalized KDE
                    truth[row] = True
                    truth[col] = True

                    ax.scatter(
                        data[:,col],
                        data[:,row],
                        marker='.',
                        s=1,
                        alpha=min(1, 500./Nsamp), ### FIXME: change alpha to reflect weight
                        color=color,
                    )

                    kde = utils.logkde(
                        utils.vects2flatgrid(vects[col], vects[row]),
                        utils.reflect(data[:,truth], ranges[truth]) if reflect else data[:,truth],
                        bandwidths[truth],
                    )
                    thrs = np.exp(logkde2levels(kde, levels))
                    kde = np.exp(kde-np.max(kde)).reshape(shape)
                    kde /= np.sum(kde)*dvects[col]*dvects[row] # normalize kde

                    ax.contour(vects[col], vects[row], kde.transpose(), colors=color, alpha=0.5, levels=thrs)

            # decorate
            ax.grid(True, which='both')

            ax.set_xlim(ranges[col])
#            plt.setp(ax.get_xticklabels(), rotation=45)
            if row!=col:
                ax.set_ylim(ranges[row])
#                plt.setp(ax.get_yticklabels(), rotation=45)

            if row!=(Ncol-1):
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel('%s'%labels[col])

            if col!=0 or row==0: #!=0
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel('%s'%labels[row])

    ### return figure
    return fig

#-------------------------------------------------

### FIXME: move all the sanity-check plots from within executables to in here...
