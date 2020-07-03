__doc__ = "a module that houses plotting routines for common uses"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import os

import numpy as np
from scipy.special import erfinv

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
if matplotlib.__version__ < '1.3.0':
    plt.rcParams['text.usetex'] = True

try:
    from corner import corner as _corner
except ImportError:
    _corner = None

### non-standard libraries
from . import utils
from .stats import (logkde2levels, neff)
from . import gaussianprocess as gp

#-------------------------------------------------

DEFAULT_LEVELS=[0.5, 0.9]
DEFAULT_NUM_POINTS = 25
DEFAULT_BANDWIDTH = utils.DEFAULT_BANDWIDTH
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

MAIN_AXES_POSITION = [0.18, 0.33, 0.77, 0.62]
RESIDUAL_AXES_POSITION = [0.18, 0.12, 0.77, 0.2]
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

def save(basename, fig, figtypes=DEFAULT_FIGTYPES, directory=DEFAULT_DIRECTORY, verbose=False, **kwargs):
    template = os.path.join(directory, basename+'.%s')
    for figtype in figtypes:
        figname = template%figtype
        if verbose:
            print('saving: '+figname)
        fig.savefig(figname,  **kwargs)

def weights2color(weights, basecolor, prefact=750., minimum=1e-3):
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

#-------------------------------------------------

def corner(*args, **kwargs):
    if _corner is None:
        raise ImportError('could not import corner')
    return _corner(*args, **kwargs)

def silverman_bandwidth(data, weights=None):
    """approximate rule of thumb for bandwidth selection"""
    if weights is None:
        std = np.std(data)
    else: ### account for weights when computing std
        N = np.sum(weights)
        std = (np.sum(weights*data**2)/N - (np.sum(weights*data)/N)**2)**0.5
    return 0.9 * std * len(data)**(-0.2)

def kde_corner(
        data,
        bandwidths=None,
        labels=None,
        range=None,
        truths=None,
        weights=None,
        num_points=DEFAULT_NUM_POINTS,
        levels=DEFAULT_LEVELS,
        hist1D=False,  ### plot a normed histogram on the 1D marginal panels in addition to the KDE estimate
        reflect=False, ### reflect data points about the boundaries when performing the KDE; may be expensive...
        verbose=False,
        grid=True,
        color=DEFAULT_COLOR1,
        truth_color=DEFAULT_TRUTH_COLOR,
        linewidth=DEFAULT_LINEWIDTH,
        linestyle=DEFAULT_LINESTYLE,
        filled=False,
        filled1D=False,
        scatter=False,
        rotate=True,
        rotate_xticklabels=0,
        rotate_yticklabels=0,
        fig=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        alpha=DEFAULT_ALPHA,
        num_proc=utils.DEFAULT_NUM_PROC,
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
        labels = [str(i) for i in xrange(Ncol)]
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

    if range is None:
        range = [utils.data2range(data[:,i]) for i in xrange(Ncol)]
    else:
        assert len(range)==Ncol, 'must have the same number of columns in data and range'
    range = np.array(range)

    if truths is None:
        truths = [None]*Ncol
    else:
        assert len(truths)==Ncol, 'must have the same number of columns in data and truths'
    truths = np.array(truths)

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

    vects = [np.linspace(m, M, num_points) for m, M in range] ### grid placement
    dvects = [v[1]-v[0] for v in vects]

    ### set colors for scatter points
    scatter_color = weights2color(weights, color)

    ### set up bins for 1D marginal histograms, if requested
    if hist1D:
        Nbins = max(10, int(Nsamp**0.5)/5)
        bins = [np.linspace(m, M, Nbins+1) for m, M in range]

    ### iterate over columns, building 2D KDEs as needed
    for row in xrange(Ncol):
        for col in xrange(row+1):
            ax = plt.subplot(Ncol, Ncol, row*Ncol+col+1)

            truth[:] = False

            if include[row] and include[col]: ### plot data for this pair
                # 1D marginal posteriors
                if row==col:
                    if verbose:
                        print('row=col='+labels[col])

                    truth[col] = True

                    d, w = utils.reflect(data[:,truth], range[truth], weights=weights) if reflect else (data[:,truth], weights)

                    kde = utils.logkde(
                        vects[col],
                        d,
                        variances[col],
                        weights=w,
                        num_proc=num_proc,
                    )
                    kde = np.exp(kde - np.max(kde))

                    kde /= np.sum(kde)*dvects[col]
                    if rotate and row==(Ncol-1): ### rotate the last histogram
                        if filled1D:
                            print('WARNING: filled1D only works when rotate==False')
                        ax.plot(kde, vects[col], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                        xmax = max(ax.get_xlim()[1], np.max(kde)*1.05)
                        if hist1D:
                            n, _, _ = ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True, weights=weights, orientation='horizontal')
                            xmax = max(xmax, np.max(n)*1.05)
                        ax.set_xlim(xmin=0, xmax=xmax)

                    else:
                        if filled1D:
                            ax.fill_between(vects[col], kde, np.zeros_like(kde), color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                        ax.plot(vects[col], kde, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                        ymax = max(ax.get_ylim()[1], np.max(kde)*1.05)
                        if hist1D:
                            n, _, _ = ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True, weights=weights)
                            ymax = max(ymax, np.max(n)*1.05)
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

                    d, w = utils.reflect(data[:,truth], range[truth], weights=weights) if reflect else (data[:,truth], weights)

                    kde = utils.logkde(
                        utils.vects2flatgrid(vects[col], vects[row]),
                        d,
                        variances[truth],
                        weights=w,
                        num_proc=num_proc,
                    )
                    kde = np.exp(kde-np.max(kde)).reshape(shape)
                    kde /= np.sum(kde)*dvects[col]*dvects[row] # normalize kde

                    thrs = sorted(np.exp(logkde2levels(np.log(kde), levels)), reverse=True)
                    if filled:
                        ax.contourf(vects[col], vects[row], kde.transpose(), colors=color, alpha=alpha, levels=thrs)
                    ax.contour(vects[col], vects[row], kde.transpose(), colors=color, alpha=alpha, levels=thrs, linewidths=linewidth, linestyles=linestyle)

            # decorate
            ax.grid(grid, which='both')

            if row!=col:
                ax.set_ylim(range[row])
                ax.set_xlim(range[col])
                plt.setp(ax.get_yticklabels(), rotation=rotate_yticklabels)
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
            elif rotate and row==(Ncol-1):
                ax.set_ylim(range[row])            
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
                ax.set_xticks([])
            else:
                ax.set_xlim(range[col])
                plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
                ax.set_yticks([])

            # add Truth annotations
            if truths[col] is not None:
                if rotate and (row==(Ncol-1)) and (row==col):
                    xlim = ax.get_xlim()
                    ax.plot(xlim, [truths[col]]*2, color=truth_color)
                    ax.set_xlim(xlim)
                else:
                    ylim = ax.get_ylim()
                    ax.plot([truths[col]]*2, ylim, color=truth_color)
                    ax.set_ylim(ylim)
            if (row!=col) and (truths[row] is not None):
                xlim = ax.get_xlim()
                ax.plot(xlim, [truths[row]]*2, color=truth_color)
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
        range=None,
        truths=None,
        color=None, ### let matplotlib pick this for you automatically
        alpha=DEFAULT_ALPHA,
        grid=True,
        linestyle=DEFAULT_LINESTYLE,
        linewidth=DEFAULT_LINEWIDTH,
        fig=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
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
        labels = [str(i) for i in xrange(Ncol)]
    else:
        assert len(labels)==Ncol, 'must have the same number of columns in data and labels'
    labels = np.array(labels)

    if range is None:
        range = [utils.data2range(data[:,i]) for i in xrange(Ncol)]
    else:
        assert len(range)==Ncol, 'must have the same number of columns in data and range'
    range = np.array(range)

    if truths is None:
        truths = [None]*Ncol
    else:
        assert len(truths)==Ncol, 'must have the same number of columns in data and truths'
    truths = np.array(truths)

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

    ### iterate over columns, building 2D KDEs as needed
    for row in xrange(Ncol):
        for col in xrange(row):
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

            ax.set_xlim(range[col])
            plt.setp(ax.get_xticklabels(), rotation=rotate_xticklabels)
            if row!=col:
                ax.set_ylim(range[row])
                plt.setp(ax.get_yticklabels(), rotation=rotate_yticklabels)

            # add Truth annotations
            if truths[col] is not None:
                ylim = ax.get_ylim()
                ax.plot([truths[col]]*2, ylim, color=truth_color)
                ax.set_ylim(ylim)
            if (row!=col) and (truths[row] is not None):
                xlim = ax.get_xlim()
                ax.plot(xlim, [truths[row]]*2, color=truth_color)
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

#-------------------------------------------------

def cov(model, colormap=DEFAULT_COLORMAP, figwidth=DEFAULT_COV_FIGWIDTH, figheight=DEFAULT_COV_FIGHEIGHT, tanh=False):
    """plot the covariance matrix averaged over the components of the mixture model
    """
    fig = plt.figure(figsize=(figwidth, figheight))
    eax = plt.subplot(1,2,1)
    vax = plt.subplot(1,2,2)

    ### average over mixture model
    ### NOTE:
    ###     we take special care to handle cases where the model components have different x-values
    ###     we assume that the covariance is zero for that element if the x-value is missing
    ###     practially, this should be a small edge-effect that we can probably ignore in almost all cases
    x = set()
    for m in model:
        x = x.union(m['x'])
    n = len(x)

    x = np.array(sorted(x), dtype=float)
    truth = np.empty(n*n, dtype=bool)
    c = np.zeros(n*n, dtype=float)
    c2 = np.zeros(n*n, dtype=float)
    w = 0.
    for m in model:
        contained = gp._target_in_source(x, m['x']) ### possibly expensive, but whatever
        truth[:] = np.outer(contained, contained).flatten()
        c[truth] += m['weight']*m['cov'].flatten()
        c2[truth] += m['weight']*(m['cov'].flatten()**2)
        w += m['weight']
    c /= w
    c2 /= w
    c2 = (c2 - c**2)**0.5

    c = c.reshape((n,n))
    c2 = c2.reshape((n,n))

    if tanh:
        c = np.tanh(c/tanh)
        c2 = np.tanh(c2/tanh)

    # plot average covariance
    m = np.max(np.abs(c))
    lim = -m, +m
    plt.sca(eax)
    cb = fig.colorbar(
        eax.imshow(c, cmap=colormap, aspect='equal', extent=(x[0], x[-1], x[0], x[-1]), interpolation='none', vmin=lim[0], vmax=lim[1], origin='lower'),
        orientation='vertical',
        shrink=0.90,
    )
    if tanh:
        cb.set_label('$\\tanh\left(\mu_\mathrm{Cov}/%.3f\\right)$'%tanh)
    else:
        cb.set_label('$\mu_\mathrm{Cov}$')

    # plot stdv of covariance
    lim = 0, max(np.max(c2), 0.001)
    plt.sca(vax)
    cb = fig.colorbar(
        vax.imshow(c2, cmap=colormap, aspect='equal', extent=(x[0], x[-1], x[0], x[-1]), interpolation='none', vmin=lim[0], vmax=lim[1], origin='lower'),
        orientation='vertical',
        shrink=0.90,
    )
    if tanh:
        cb.set_label('$\\tanh\left(\sigma_\mathrm{Cov}/%.3f\\right)$'%tanh)
    else:
        cb.set_label('$\sigma_\mathrm{Cov}$')

    for ax in [eax, vax]:
        ax.set_xticks(x, minor=True)
        ax.set_yticks(x, minor=True)

    plt.subplots_adjust(
        left=0.05,
        right=0.93,
        bottom=0.05,
        top=0.93,
    )

    return fig

#-------------------------------------------------

def overlay(
        curves,
        colors=None,
        alphas=None,
        linestyles=None,
        markers=None,
        xlabel=None,
        ylabel=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        reference_curve=None,
        fractions=False,
        residuals=False,
        ratios=False,
        logx=False,
        logy=False,
        grid=True,
        figtup=None,
    ):
    ### set up figure, axes
    subax = fractions or residuals or ratios
    if figtup is None:
        fig = plt.figure(figsize=(figwidth, figheight))
        if subax:
            ax = fig.add_axes(MAIN_AXES_POSITION)
            rs = fig.add_axes(RESIDUAL_AXES_POSITION)

        else:
            ax = fig.add_axes(AXES_POSITION)
    else:
        if subax:
            fig, ax, rs = figtup
        else:
            fig, ax = figtup

    N = len(curves)
    if colors is None:
        colors = [DEFAULT_COLOR2 for _ in range(N)]

    if alphas is None:
        alphas = [DEFAULT_ALPHA for _ in range(N)]

    if linestyles is None:
        linestyles = [DEFAULT_LINESTYLE for _ in range(N)]

    if markers is None:
        markers = [DEFAULT_MARKER for _ in range(N)]

    xmin = np.min([np.min(x) for x, _, _ in curves if len(x)])
    xmax = np.max([np.max(x) for x, _, _ in curves if len(x)])
    ymin = np.min([np.min(f) for _, f, _ in curves if len(f)])
    ymax = np.max([np.max(f) for _, f, _ in curves if len(f)])

    # plot the observed data
    for (x, f, label), c, l, a, m in zip(curves, colors, linestyles, alphas, markers):
        ax.plot(x, f, linestyle=l, color=c, marker=m, label=label, alpha=a)

    # plot residuals, etc
    if subax:
        if reference_curve is None:
            x_ref, f_ref = curves[0]
        else:
            x_ref, f_ref = reference_curve

        # plot the observed data
        for (x, f, _), c, l, a, m in zip(curves, colors, linestyles, alphas, markers):
            f = np.interp(x_ref, x, f)

            if fractions:
                rs.plot(x_ref, (f-f_ref)/f_ref, linestyle=l, color=c, alpha=a, marker=m)

            elif residuals:
                rs.plot(x_ref, f-f_ref, linestyle=l, color=c, alpha=a, marker=m)

            elif ratios:
                rs.plot(x_ref, f/f_ref, linestyle=l, color=c, alpha=a, marker=m)

    # decorate
    ax.grid(grid, which='both')
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xscale('log' if logx else 'linear')
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if subax:
        rs.set_xscale(ax.get_xscale())
        rs.set_xlim(ax.get_xlim())

        rs.grid(grid, which='both')
        plt.setp(ax.get_xticklabels(), visible=False)

        if xlabel is not None:
            rs.set_xlabel(xlabel)

        if fractions:
            rs.set_ylabel('$(%s - %s_{\mathrm{ref}})/%s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$'), ylabel.strip('$')))

        elif residuals:
            rs.set_ylabel('$%s - %s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$')))

        elif ratios:
            rs.set_yscale(ax.get_xscale())
            rs.set_ylabel('$%s/%s_{\mathrm{ref}$'%(ylabel.strip('$'), ylabel.strip('$')))

        return fig, ax, rs

    else:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        return fig, ax

def overlay_model(
        model,
        color=DEFAULT_COLOR1,
        alpha=DEFAULT_ALPHA,
        linestyle=DEFAULT_LINESTYLE,
        marker=DEFAULT_MARKER,
        levels=DEFAULT_LEVELS, ### the confidence levels to include in shading
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        xlabel=None,
        ylabel=None,
        label=None,
        reference_curve=None,
        fractions=False,
        residuals=False,
        ratios=False,
        logx=False,
        logy=False,
        grid=True,
        figtup=None,
    ):
    ### set up figure, axes
    subax = fractions or residuals or ratios
    if figtup is None:
        fig = plt.figure(figsize=(figwidth, figheight))
        if subax:
            ax = fig.add_axes(MAIN_AXES_POSITION)
            rs = fig.add_axes(RESIDUAL_AXES_POSITION)

        else:
            ax = fig.add_axes(AXES_POSITION)
    else:
        if subax:
            fig, ax, rs = figtup
        else:
            fig, ax = figtup

    ### plot the confidence regions
    sigmas = [2**0.5*erfinv(level) for level in levels] ### base sigmas on Guassian cumulative distribution and the desired confidence levels

    weights = [m['weight'] for m in model]
    colors = weights2color(weights, color, prefact=alpha/max(2., len(sigmas)*2.), minimum=0.002)

    xmin = +np.infty
    xmax = -np.infty
    ymin = +np.infty
    ymax = -np.infty
    x = set()
    for m, c, w in zip(model, colors, weights):
        _x = m['x']
        _f = m['f']
        _s = np.diag(m['cov'])**0.5
        for sigma in sigmas:
            ax.fill_between(m['x'], _f+_s*sigma, _f-_s*sigma, color=c, linewidth=0) ### plot uncertainty

        # book-keeping for plotting the mean
        xmin = min(xmin, np.min(_x))
        xmax = max(xmax, np.max(_x))
        ymin = min(ymin, np.min(_f))
        ymax = max(ymax, np.max(_f))
        x = x.union(set(_x))

    ### plot a representation of the overall mean
    x = np.array(sorted(x))
    f = np.zeros_like(x, dtype=float)
    for m, w in zip(model, weights):
        f += w*np.interp(x, m['x'], m['f'])
    f /= np.sum(weights) ### plot the mean of the means
    ax.plot(x, f, color=color, linestyle=linestyle, marker=marker, alpha=alpha, label=label)

    # plot residuals, etc
    if subax:
        if reference_curve is not None:
            x_ref, f_ref = reference_curve
        else:
            x_ref = x
            f_ref = f

        for m, c in zip(model, colors):
            _f = np.interp(x_ref, x, m['f'])
            _s = np.interp(x_ref, x, np.diag(m['cov'])**0.5)
            hgh = [_f+_s*sigma for sigma in sigmas]
            low = [_f-_s*sigma for sigma in sigmas]

            if fractions:
                for hgh, low in zip(hgh, low):
                    rs.fill_between(x_ref, (hgh-f_ref)/f_ref, (low-f_ref)/f_ref, color=c, linewidth=0)

            elif residuals:
                for hgh, low in zip(hgh, low):
                    rs.fill_between(x_ref, hgh-f_ref, low-f_ref, color=c, linewidth=0)

            elif ratios:
                for hgh, low in zip(hgh, low):
                    rs.fill_between(x_ref, hgh/f_ref, low/f_ref, color=c, linewidth=0)

        f = np.interp(x_ref, x, f)
        if fractions:
            rs.fill_between(x_ref, (f-f_ref)/f_ref, (f-f_ref)/f_ref, color=color, linestyle=linestyle, marker=marker, alpha=alpha)

        elif residuals:
            rs.fill_between(x_ref, f-f_ref, f-f_ref, color=color, linestyle=linestyle, marker=marker, alpha=alpha)

        elif ratios:
            rs.fill_between(x_ref, f/f_ref, color=color, linestyle=linestyle, marker=marker, alpha=alpha)

    # decorate
    ax.grid(grid, which='both')
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xscale('log' if logx else 'linear')
    ax.set_xlim(xmin=xmin, xmax=xmax)

    if logy:
        dy = (ymax/ymin)**0.05
        ax.set_ylim(ymin=ymin/dy, ymax=ymax*dy)
    else:
        dy = (ymax-ymin)*0.05
        ax.set_ylim(ymin=ymin-dy, ymax=ymax+dy)

    if xlabel is None:
        xlabel = model[0]['labels']['xlabel']
    if ylabel is None:
        ylabel = model[0]['labels']['flabel']

    ax.set_ylabel(ylabel)
    if subax:
        rs.set_xscale(ax.get_xscale())
        rs.set_xlim(ax.get_xlim())

        rs.grid(grid, which='both')
        plt.setp(ax.get_xticklabels(), visible=False)

        rs.set_xlabel(xlabel)
        if fractions:
            rs.set_ylabel('$(%s - %s_{\mathrm{ref}})/%s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$'), ylabel.strip('$')))

        elif residuals:
            rs.set_ylabel('$%s - %s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$')))

        elif ratios:
            rs.set_yscale(ax.get_xscale())
            rs.set_ylabel('$%s/%s_{\mathrm{ref}$'%(ylabel.strip('$'), ylabel.strip('$')))

        return fig, ax, rs 
    else:
        ax.set_xlabel(xlabel)
        return fig, ax
