"""a module that houses logic for custom corner plots
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

try:
    from corner import corner as _corner
except ImportError:
    _corner = None

### non-standard libraries
from . import utils as plt

from universality.kde import (logkde, silverman_bandwidth, vects2flatgrid)
from universality.utils import DEFAULT_NUM_PROC
from universality.utils import reflect as reflect_samples
from universality import stats

#-------------------------------------------------

def corner(*args, **kwargs):
    if _corner is None:
        raise ImportError('could not import corner')
    return _corner(*args, **kwargs)

def kde_corner(
        data,
        bandwidths=None,
        labels=None,
        ranges=None,
        alternate_units=None, # provide a shift, scale, and label such that y = scale*(x-shift)
        truths=None,
        bands=None,
        weights=None,
        num_points=plt.DEFAULT_NUM_POINTS,
        levels=plt.DEFAULT_LEVELS,
        levels1D=[],
        hist1D=False,  ### plot a normed histogram on the 1D marginal panels in addition to the KDE estimate
        reflect=False, ### reflect data points about the boundaries when performing the KDE; may be expensive...
        verbose=False,
        grid=True,
        color=plt.DEFAULT_COLOR1,
        contour_colors=None,
        truth_color=plt.DEFAULT_TRUTH_COLOR,
        truth_alpha=plt.DEFAULT_ALPHA,
        truth_linestyle=plt.DEFAULT_LINESTYLE,
        band_color=None,
        band_alpha=plt.DEFAULT_LIGHT_ALPHA,
        linewidth=plt.DEFAULT_LINEWIDTH,
        linestyle=plt.DEFAULT_LINESTYLE,
        filled=False,
        filled1D=False,
        scatter=False,
        rotate=True,
        rotate_xticklabels=0,
        rotate_yticklabels=0,
        fig=None,
        figwidth=plt.DEFAULT_FIGWIDTH,
        figheight=plt.DEFAULT_FIGHEIGHT,
        alpha=plt.DEFAULT_ALPHA,
        filled_alpha=plt.DEFAULT_ALPHA,
        num_proc=DEFAULT_NUM_PROC,
    ):
    """
    should be mostly equivalent to corner.corner, except we build our own KDEs and the like

    NOTE: we will skip plotting for any column for which any data is nan. In this way, we can plot multiple sets of data on the same axes even if they do not all possess the same columns (just pad with nan's as needed when calling kde_corner).
    """
    ### check data formats
    Nsamp, Ncol = data.shape

    if weights is None:
        weights = np.ones(Nsamp, dtype=float)/Nsamp
    else:
        assert len(weights)==Nsamp, 'must have the same number of rows in data and weights'
    weights = np.array(weights)

    if labels is None:
        labels = [str(i) for i in range(Ncol)]
    else:
        assert len(labels)==Ncol, 'must have the same number of columns in data and labels'
    labels = np.array(labels)

    if bandwidths is None:
        bandwidths = [None]*Ncol
    else:
        assert len(bandwidths)==Ncol, 'must have the same number of columns in data and bandwidths'
    variances = np.empty(Ncol, dtype=float)
    for i, b in enumerate(bandwidths):
        if b is None:
            b = silverman_bandwidth(data[:,i], weights=weights)
            if verbose:
                print('automatically selected bandwidth=%.3e for col=%s'%(b, labels[i]))
        variances[i] = b**2

    if ranges is None:
        ranges = [stats.samples2range(data[:,i]) for i in range(Ncol)]
    else:
        assert len(ranges)==Ncol, 'must have the same number of columns in data and range'
    ranges = np.array(ranges)

    if alternate_units is None:
        alternate_units = [None]*Ncol

    if truths is None:
        truths = [None]*Ncol
    else:
        assert len(truths)==Ncol, 'must have the same number of columns in data and truths'

    if bands is None:
        bands = [None]*Ncol
    else:
        assert len(bands)==Ncol, 'must have the same number of columns in data and bands'

    if band_color is None:
        band_color = truth_color

    ### construct figure and axes objects
    if fig is None:
        fig = plt.figure(figsize=(figwidth, figheight)) ### FIXME: set figsize based on Ncol?

        if any([_ is not None for _ in alternate_units]): ### we will plot alternate units
            plt.subplots_adjust(
                hspace=plt.HSPACE,
                wspace=plt.WSPACE,
                left=plt.CORNER_LEFT/figwidth,
                right=(figwidth-plt.CORNER_LEFT)/figwidth,
                bottom=plt.CORNER_BOTTOM/figheight,
                top=(figheight-plt.CORNER_BOTTOM)/figheight, # make room for labels
            )
        else:
            plt.subplots_adjust(
                hspace=plt.HSPACE,
                wspace=plt.WSPACE,
                left=plt.CORNER_LEFT/figwidth,
                right=(figwidth-plt.CORNER_RIGHT)/figwidth,
                bottom=plt.CORNER_BOTTOM/figheight,
                top=(figheight-plt.CORNER_TOP)/figheight,
            )

    shape = (num_points, num_points) # used to reshape 2D sampling kdes

    truth =  np.empty(Ncol, dtype=bool) # used to index data within loop
    include = [np.all(data[:,i]==data[:,i]) for i in range(Ncol)] # used to determine which data we should skip

    vects = [np.linspace(m, M, num_points) for m, M in ranges] ### grid placement
    dvects = [v[1]-v[0] for v in vects]

    if contour_colors is None:
        contour_colors = [color]
    ### set colors for scatter points
    scatter_color = plt.weights2color(weights, color)

    ### set up bins for 1D marginal histograms, if requested
    if hist1D:
        Nbins = max(10, int(Nsamp**0.5)/5)
        bins = [np.linspace(m, M, Nbins+1) for m, M in ranges]

    ### iterate over columns, building 2D KDEs as needed
    for row in range(Ncol):
        for col in range(row+1):
            ax = plt.subplot(Ncol, Ncol, row*Ncol+col+1)

            truth[:] = False

            if include[row] and include[col]: ### plot data for this pair
                # 1D marginal posteriors
                if row==col:
                    if verbose:
                        print('row=col='+labels[col])

                    truth[col] = True

                    d, w = reflect_samples(data[:,truth], ranges[truth], weights=weights) if reflect else (data[:,truth], weights)

                    kde = logkde(
                        vects[col],
                        d,
                        variances[col],
                        weights=w,
                        num_proc=num_proc,
                    )

                    ### figure out levels for 1D histograms
                    lines = []
                    if levels1D:     
                        for level, (m, M) in zip(levels1D, stats.logkde2crbounds(vects[col], kde, levels1D)):
                            if verbose:
                                print('    @%.3f : [%.3e, %.3e]'%(level, m, M))
                            lines.append((m, M))

                    kde = np.exp(kde - np.max(kde))
                    kde /= np.sum(kde)*dvects[col]
                    if rotate and row==(Ncol-1): ### rotate the last histogram
                        if filled1D:
                            ax.fill_betweenx(vects[col], kde, np.zeros_like(kde), color=color, linewidth=linewidth, linestyle=linestyle, alpha=filled_alpha)
                        ax.plot(kde, vects[col], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                        xmax = max(ax.get_xlim()[1], np.max(kde)*1.05)
                        if hist1D:
                            n, _, _ = ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True, weights=weights, orientation='horizontal')
                            xmax = max(xmax, np.max(n)*1.05)

                        for m, M in lines:
                            ax.plot([0, 10*xmax], [m]*2, color=color, alpha=alpha, linestyle='dashed') ### plot for a bigger range incase axes change later
                            ax.plot([0, 10*xmax], [M]*2, color=color, alpha=alpha, linestyle='dashed')

                        ax.set_xlim(xmin=0, xmax=xmax)

                    else:
                        if filled1D:
                            ax.fill_between(vects[col], kde, np.zeros_like(kde), color=color, linewidth=linewidth, linestyle=linestyle, alpha=filled_alpha)
                        ax.plot(vects[col], kde, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                        ymax = max(ax.get_ylim()[1], np.max(kde)*1.05)
                        if hist1D:
                            n, _, _ = ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True, weights=weights)
                            ymax = max(ymax, np.max(n)*1.05)

                        for m, M in lines:
                            ax.plot([m]*2, [0, 10*ymax], color=color, alpha=alpha, linestyle='dashed') ### plot for bigger range in case axes change later
                            ax.plot([M]*2, [0, 10*ymax], color=color, alpha=alpha, linestyle='dashed')

                        ax.set_ylim(ymin=0, ymax=ymax)

                else:
                    if verbose:
                        print('row=%s ; col=%s'%(labels[row], labels[col]))

                    # marginalized KDE
                    truth[row] = True
                    truth[col] = True

                    if scatter:
                        ax.scatter(
                            data[:,col],
                            data[:,row],
                            marker='o',
                            s=2,
                            color=scatter_color,
                        )

                    d, w = reflect_samples(data[:,truth], ranges[truth], weights=weights) if reflect else (data[:,truth], weights)
                    kde = logkde(
                        vects2flatgrid(vects[col], vects[row]),
                        d,
                        variances[truth],
                        weights=w,
                        num_proc=num_proc,
                    )
                    kde = np.exp(kde-np.max(kde)).reshape(shape)
                    kde /= np.sum(kde)*dvects[col]*dvects[row] # normalize kde

                    thrs = sorted(np.exp(stats.logkde2levels(np.log(kde), levels))) # smallest to largest

                    if filled:
                        ax.contourf(vects[col], vects[row], kde.transpose(), colors=contour_colors, alpha=filled_alpha, levels=thrs)
                    ax.contour(vects[col], vects[row], kde.transpose(), colors=contour_colors, alpha=alpha, levels=thrs, linewidths=linewidth, linestyles=linestyle)

            # decorate
            ax.grid(grid, which='both')
            ax.tick_params(**plt.TICK_PARAMS)

            if row!=col:
                ax.set_ylim(ranges[row])
                ax.set_xlim(ranges[col])
                plt.setp(ax.get_yticklabels(), rotation=rotate_yticklabels)
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
            elif rotate and row==(Ncol-1):
                ax.set_ylim(ranges[row])            
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
                ax.set_xticks([])
            else:
                ax.set_xlim(ranges[col])
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
                ax.set_yticks([])

            # add alternate units
            if (row == col) and (alternate_units[col] is not None):
                label, scale, shift = alternate_units[col]

                # add another set of axes that line up with the default one
                if rotate and (row == (Ncol-1)):
                    twinx = ax.twinx()
                    twinx.tick_params(labelrotation=rotate_yticklabels,**plt.TICK_PARAMS)

                    ticks = ax.get_yticks()
                    ticklabels = ['$%.1f$'%(scale*(tick-shift)) for tick in ticks]

                    twinx.set_yticks(ticks)
                    twinx.set_yticklabels(ticklabels)
                    twinx.set_ylim(ranges[col])
                    twinx.set_ylabel(label)
 
                else:
                    twiny = ax.twiny()
                    twiny.tick_params(labelrotation=rotate_xticklabels, **plt.TICK_PARAMS)

                    # compute ticklabels based on y = scale*(x-shift)
                    ticks = ax.get_xticks()
                    ticklabels = ['$%.1f$'%(scale*(tick-shift)) for tick in ticks]

                    twiny.set_xticks(ticks)
                    twiny.set_xticklabels(ticklabels)
                    twiny.set_xlim(ranges[col])
                    twiny.set_xlabel(label)

            # add Truth annotations
            if truths[col] is not None:
                for val in truths[col]:
                    if rotate and (row==(Ncol-1)) and (row==col):
                        xlim = ax.get_xlim()
                        ax.plot(xlim, [val]*2, color=truth_color, linestyle=truth_linestyle, alpha=truth_alpha)
                        ax.set_xlim(xlim)
                    else:
                        ylim = ax.get_ylim()
                        ax.plot([val]*2, ylim, color=truth_color, linestyle=truth_linestyle, alpha=truth_alpha)
                        ax.set_ylim(ylim)

            if (row!=col) and (truths[row] is not None):
                for val in truths[row]:
                    xlim = ax.get_xlim()
                    ax.plot(xlim, [val]*2, color=truth_color, linestyle=truth_linestyle, alpha=truth_alpha)
                    ax.set_xlim(xlim)

            if bands[col] is not None:
                for m, M in bands[col]:
                    if rotate and (row==(Ncol-1)) and (row==col):
                        xlim = ax.get_xlim()
                        ax.fill_between(xlim, [m]*2, [M]*2, color=band_color, alpha=band_alpha)
                        ax.set_xlim(xlim)
                    else:
                        ylim = ax.get_ylim()
                        ax.fill_between([m, M], [ylim[0]]*2, [ylim[1]]*2, color=band_color, alpha=band_alpha)
                        ax.set_ylim(ylim)

            if (row!=col) and (bands[row] is not None):
                for m, M in bands[row]:
                    xlim = ax.get_xlim()
                    ax.fill_between(xlim, [m]*2, [M]*2, color=band_color, alpha=band_alpha)
                    ax.set_xlim(xlim)

            if (row!=(Ncol-1)):
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                if (row==col) and rotate:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    ax.set_xlabel('%s'%labels[col])

            if col!=0 or row==0: #!=0
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel('%s'%labels[row])

    ### return figure
    return fig

def curve_corner(
        data,
        labels=None,
        ranges=None,
        truths=None,
        bands=None,
        color=None, ### let matplotlib pick this for you automatically
        alpha=plt.DEFAULT_ALPHA,
        grid=True,
        truth_color=plt.DEFAULT_TRUTH_COLOR,
        truth_alpha=plt.DEFAULT_ALPHA,
        truth_linestyle=plt.DEFAULT_LINESTYLE,
        band_color=None,
        band_alpha=plt.DEFAULT_LIGHT_ALPHA,
        linestyle=plt.DEFAULT_LINESTYLE,
        linewidth=plt.DEFAULT_LINEWIDTH,
        fig=None,
        figwidth=plt.DEFAULT_FIGWIDTH,
        figheight=plt.DEFAULT_FIGHEIGHT,
        verbose=False,
        rotate_xticklabels=0,
        rotate_yticklabels=0,
    ):
    """
    plot curves defined in data on the corner plot
    will mostly be used to annotate existing corner plots (through the fig kwarg)
    """
    ### check data formats
    Nsamp, Ncol = data.shape

    if labels is None:
        labels = [str(i) for i in range(Ncol)]
    else:
        assert len(labels)==Ncol, 'must have the same number of columns in data and labels'
    labels = np.array(labels)

    if ranges is None:
        ranges = [stats.stamples2range(data[:,i]) for i in range(Ncol)]
    else:
        assert len(ranges)==Ncol, 'must have the same number of columns in data and range'
    ranges = np.array(ranges)

    if truths is None:
        truths = [None]*Ncol
    else:
        assert len(truths)==Ncol, 'must have the same number of columns in data and truths'

    if bands is None:
        bands = [None]*Ncol
    else:
        assert len(bands)==Ncol, 'must have the same number of columns in data and bands'

    if band_color is None:
        band_color = truth_color

    ### construct figure and axes objects
    if fig is None:
        fig = plt.figure(figsize=(figwidth, figheight)) ### FIXME: set figsize based on Ncol?
        plt.subplots_adjust(
            hspace=plt.HSPACE,
            wspace=plt.WSPACE,
            left=plt.CORNER_LEFT/figwidth,
            right=(figwidth-plt.CORNER_RIGHT)/figwidth,
            bottom=plt.CORNER_BOTTOM/figheight,
            top=(figheight-plt.CORNER_TOP)/figheight,
        )

    ### iterate over columns, building 2D KDEs as needed
    ### NOTE: we do not plot in the 1D axes, so we don't have to worry about rotate, etc
    for row in range(Ncol):
        for col in range(row):
            ax = plt.subplot(Ncol, Ncol, row*Ncol+col+1)

            if verbose:
                print('row=%s ; col=%s'%(labels[row], labels[col]))

            # plot the curve in data
            if color is not None:
                ax.plot(data[:,col], data[:,row], color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(data[:,col], data[:,row], linestyle=linestyle, alpha=alpha, linewidth=linewidth)

            # decorate
            ax.grid(grid, which='both')
            ax.tick_params(**plt.TICK_PARAMS)

            ax.set_xlim(ranges[col])
            plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
            if row!=col:
                ax.set_ylim(ranges[row])
                plt.setp(ax.get_yticklabels(), rotation=rotate_yticklabels)

            # add Truth annotations
            if truths[col] is not None:
                for val in truths[col]:
                    ylim = ax.get_ylim()
                    ax.plot([val]*2, ylim, color=truth_color, linestyle=truth_linestyle, alpha=truth_alpha)
                    ax.set_ylim(ylim)
            if (row!=col) and (truths[row] is not None):
                for val in truths[row]:
                    xlim = ax.get_xlim()
                    ax.plot(xlim, [val]*2, color=truth_color, linestyle=truth_linestyle, alpha=truth_alpha)
                    ax.set_xlim(xlim)

            if bands[col] is not None:
                for m, M in bands[col]:
                    ylim = ax.get_ylim()
                    ax.fill_between([m, M], [ylim[0]]*2, [ylim[1]]*2, color=band_color, alpha=band_alpha)
                    ax.set_ylim(ylim)

            if (row!=col) and (bands[row] is not None):
                for m, M in bands[row]:
                    xlim = ax.get_xlim()
                    ax.fill_between(xlim, [m]*2, [M]*2, color=band_color, alpha=band_alpha)
                    ax.set_xlim(xlim)

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
