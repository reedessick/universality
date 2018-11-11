__doc__ = "a module that houses plotting routines for common uses"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
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
DEFAULT_FIGWIDTH = DEFAULT_FIGHEIGHT = 12

DEFAULT_COLOR1 = 'k'
DEFAULT_COLOR2 = 'r'
DEFAULT_TRUTH_COLOR = 'b'

DEFAULT_LINEWIDTH = 1.
DEFAULT_LINESTYLE = '-'

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

MAIN_AXES_POSITION = [0.15, 0.33, 0.80, 0.62]
RESIDUAL_AXES_POSITION = [0.15, 0.12, 0.80, 0.2]
AXES_POSITION = [
    MAIN_AXES_POSITION[0],
    RESIDUAL_AXES_POSITION[1],
    MAIN_AXES_POSITION[2],
    MAIN_AXES_POSITION[1]+MAIN_AXES_POSITION[3]-RESIDUAL_AXES_POSITION[1]
]

#-------------------------------------------------

def save(basename, fig, figtypes=DEFAULT_FIGTYPES, directory=DEFAULT_DIRECTORY, verbose=False, **kwargs):
    template = os.path.join(directory, basename+'.%s')
    for figtype in figtypes:
        figname = template%figtype
        if verbose:
            print('saving: '+figname)
        fig.savefig(figname,  **kwargs)

def corner(*args, **kwargs):
    if _corner is None:
        raise ImportError('could not import corner')
    return _corner(*args, **kwargs)

def weights2color(weights, basecolor):
    Nsamp = len(weights)
    scatter_color = np.empty((Nsamp, 4), dtype=float)
    scatter_color[:,:3] = matplotlib.colors.ColorConverter().to_rgb(basecolor)
    mw, Mw = np.min(weights), np.max(weights)
    if np.all(weights==Mw):
        scatter_color[:,3] = max(min(1., 750./Nsamp), 0.001) ### equal weights
    else:
        scatter_color[:,3] = weights/np.max(weights)
        scatter_color[:,3] *= 750./neff(weights)
        scatter_color[scatter_color[:,3]>1,3] = 1 ### give reasonable bounds
        scatter_color[scatter_color[:,3]<0.001,3] = 0.001 ### give reasonable bounds
    return scatter_color

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
        scatter=False,
        rotate=True,
        fig=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
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
    variances = np.array(bandwidths)**2

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

    if weights is None:
        weights = np.ones(Nsamp, dtype=float)/Nsamp
    else:
        assert len(weights)==Nsamp, 'must have the same number of rows in data and weights'
    weights = np.array(weights)

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
                    )
                    kde = np.exp(kde - np.max(kde))

                    kde /= np.sum(kde)*dvects[col]
                    if rotate and row==(Ncol-1): ### rotate the last histogram
                        ax.plot(kde, vects[col], color=color, linewidth=linewidth, linestyle=linestyle)
                        xmax = max(ax.get_xlim()[1], np.max(kde)*1.05)
                        if hist1D:
                            n, _, _ = ax.hist(data[:,col], bins=bins[col], histtype='step', color=color, normed=True, weights=weights, orientation='horizontal')
                            xmax = max(xmax, np.max(n)*1.05)
                        ax.set_xlim(xmin=0, xmax=xmax)

                    else:
                        ax.plot(vects[col], kde, color=color, linewidth=linewidth, linestyle=linestyle)
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
                    )
                    kde = np.exp(kde-np.max(kde)).reshape(shape)
                    kde /= np.sum(kde)*dvects[col]*dvects[row] # normalize kde

                    thrs = sorted(np.exp(logkde2levels(np.log(kde), levels)), reverse=True)
                    if filled:
                        ax.contourf(vects[col], vects[row], kde.transpose(), colors=color, alpha=0.25, levels=thrs)
                    ax.contour(vects[col], vects[row], kde.transpose(), colors=color, alpha=0.5, levels=thrs, linewidths=linewidth, linestyles=linestyle)

            # decorate
            ax.grid(grid, which='both')

            if row!=col:
                ax.set_ylim(range[row])
#                plt.setp(ax.get_yticklabels(), rotation=45)
                ax.set_xlim(range[col])
#                plt.setp(ax.get_xticklabels(), rotation=45)
            elif rotate and row==(Ncol-1):
                ax.set_ylim(range[row])            
#                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.set_xlim(range[col])
#                plt.setp(ax.get_xticklabels(), rotation=45)

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
        alpha=1.0,
        linestyle=DEFAULT_LINESTYLE,
        linewidth=DEFAULT_LINEWIDTH,
        fig=None,
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        verbose=False,
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
            ax.grid(True, which='both')

            ax.set_xlim(range[col])
#            plt.setp(ax.get_xticklabels(), rotation=45)
            if row!=col:
                ax.set_ylim(range[row])
#                plt.setp(ax.get_yticklabels(), rotation=45)

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

def overlay(
        x,
        f,
        colors=None,
        linestyles=None,
        xlabel='x',
        ylabel='f',
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        residuals=False,
        ratios=False,
        logx=False,
        logy=False,
    ):
    if np.ndim(x)==1:
        x_obs = [x]
        f_obs = [f]
    N = len(x)

    if colors is None:
        colors = [DEFAULT_COLOR2 for _ in range(N)]

    if linestyles is None:
        linestyles = [DEFAULT_LINESTYLE for _ in range(N)]

    ### set up figure, axes
    fig = plt.figure(figsize=(figwidth, figheight))
    if residuals or ratios:
        ax = fig.add_axes(MAIN_AXES_POSITION)
        rs = fig.add_axes(RESIDUAL_AXES_POSITION)

    else:
        ax = fig.add_axes(AXES_POSITION)

    xmin = np.min([np.min(_) for _ in x])
    xmax = np.max([np.max(_) for _ in x])

    # plot the observed data
    for x, f, c, l in zip(x, f, colors, linestyles):
        ax.plot(x, f, l, color=c, alpha=0.5)

    # plot residuals, etc
    if residuals or ratios:
        x_ref = x[0]
        f_ref = f[0]

        # plot the observed data
        for x, f, c, l in zip(x, f, colors, linestyles):
            f = np.interp(x_ref, x, f)

            if residuals:
                rs.plot(x_ref, f-f_ref, l, color=c, alpha=0.5)

            elif ratios:
                rs.plot(x_ref, f/f_ref, l, color=c, alpha=0.5)

    # decorate
    ax.grid(True, which='both')
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xscale('log' if logx else 'linear')
    ax.set_xlim(xmin=xmin, xmax=xmax)

    if residuals or ratios:
        rs.set_xscale(ax.get_xscale())
        rs.set_xlim(ax.get_xlim())

        rs.grid(True, which='both')
        plt.setp(ax.get_xticklabels(), visible=False)

        rs.set_xlabel(xlabel)
        if residuals:
            rs.set_ylabel('$%s - %s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$')))
        elif ratios:
            rs.set_yscale(ax.get_xscale())
            rs.set_ylabel('$%s/%s_{\mathrm{ref}$'%(ylabel.strip('$'), ylabel.strip('$')))

    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig

def gpr_overlay(
        x_tst,
        f_tst,
        cr_tst,
        x_obs=None,
        f_obs=None,
        cr_obs=None,
        linestyle_tst=DEFAULT_LINESTYLE,
        linestyle_obs=DEFAULT_LINESTYLE,
        xlabel='x',
        ylabel='f',
        figwidth=DEFAULT_FIGWIDTH,
        figheight=DEFAULT_FIGHEIGHT,
        residuals=False,
        ratios=False,
        color_tst=DEFAULT_COLOR1,
        color_obs=None,
        logx=False,
        logy=False,
    ):
    ### set up input arguments
    if x_obs is not None:
        if np.ndim(x_obs)==1:
            x_obs = [x_obs]
            f_obs = [f_obs]
            cr_obs = [cr_obs]
        Nobs = len(x_obs)

        if color_obs is None:
            color_obs = [DEFAULT_COLOR2 for _ in range(Nobs)]

    else:
        Nobs = 0

    xmin, xmax = np.min(x_tst), np.max(x_tst)

    ### set up figure, axes
    fig = plt.figure(figsize=(figwidth, figheight))
    if (residuals or ratios) and (Nobs > 0):
        ax = fig.add_axes(MAIN_AXES_POSITION)
        rs = fig.add_axes(RESIDUAL_AXES_POSITION)

    else:
        ax = fig.add_axes(AXES_POSITION)

    # plot the test points
    ax.fill_between(x_tst, cr_tst[0], cr_tst[1], color=color_tst, alpha=0.25)
    ax.plot(x_tst, f_tst, linestyle_tst, color=color_tst)

    # plot the observed data
    if Nobs > 0:
        for x, f, cr, color in zip(x_obs, f_obs, cr_obs, color_obs):
            truth = (xmin<=x)*(x<=xmax)
            if cr is not None:
                ax.fill_between(x, cr[0], cr[1], color=color, alpha=0.25)
            ax.plot(x, f, linestyle_obs, color=color, alpha=0.25)

    # plot residuals, etc
    if (residuals or ratios) and (Nobs > 0):
        if Nobs==1:
            x_ref = x_obs[0]
            f_ref = f_obs[0]
            cr_ref = cr_obs[0]
        else:
            x_ref = x_tst
            f_ref = f_tst
            cr_ref = cr_tst
        truth = (xmin<=x_ref)*(x_ref<=xmax)

        f_tst_interp = np.interp(x_ref, x_tst, f_tst)
        hgh = np.interp(x_ref, x_tst, cr_tst[1])
        low = np.interp(x_ref, x_tst, cr_tst[0])

        if residuals:
            rs.fill_between(x_ref, hgh-f_ref, low-f_ref, color=color_tst, alpha=0.25)
            rs.plot(x_ref, f_tst_interp-f_ref, linestyle_tst, color=color_tst)

            rs.set_ylim(ymin=np.min((low-f_ref)[truth]), ymax=np.max((hgh-f_ref)[truth]))

        elif ratios:
            rs.fill_between(x_ref, hgh/f_ref, low/f_ref, color=color_tst, alpha=0.25)
            rs.plot(x_ref, f_tst_interp/f_ref, linestyle_tst, color=color_tst)

            rs.set_ylim(ymin=np.min((low/f_ref)[truth]), ymax=np.max((hgh/f_ref)[truth]))

        # plot the observed data
        for x, f, cr, color in zip(x_obs, f_obs, cr_obs, color_obs):
            f = np.interp(x_ref, x, f)

            if residuals:
                if cr is not None:
                    hgh = np.interp(x_ref, x, cr[1])
                    low = np.interp(x_ref, x, cr[0])
                    rs.fill_between(x_ref, hgh-f_ref, low-f_ref, color=color, alpha=0.25)

                rs.plot(x_ref, f-f_ref, linestyle_obs, color=color, alpha=0.25)

            elif ratios:
                if cr is not None:
                    hgh = np.interp(x_ref, x, cr[1])
                    low = np.interp(x_ref, x, cr[0])
                    rs.fill_between(x_ref, hgh/f_ref, low/f_ref, color=color, alpha=0.25)

                rs.plot(x_ref, f/f_ref, linestyle_obs, color=color_obs, alpha=0.25)

    # decorate
    ax.grid(True, which='both')
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xscale('log' if logx else 'linear')
    ax.set_xlim(xmin=xmin, xmax=xmax)

    if (residuals or ratios) and (Nobs > 0):
        rs.set_xscale(ax.get_xscale())
        rs.set_xlim(ax.get_xlim())

        rs.grid(True, which='both')
        plt.setp(ax.get_xticklabels(), visible=False)

        rs.set_xlabel(xlabel)
        if residuals:
            if Nobs==1:
                rs.set_ylabel('$%s - %s_{\\ast}$'%(ylabel.strip('$'), ylabel.strip('$')))
            else:
                rs.set_ylabel('$%s_{\\ast} - %s$'%(ylabel.strip('$'), ylabel.strip('$')))
        elif ratios:
            rs.set_yscale(ax.get_xscale())
            if Nobs==1:
                rs.set_ylabel('$%s/%s_{\\ast}$'%(ylabel.strip('$'), ylabel.strip('$')))
            else:
                rs.set_ylabel('$%s_{\\ast}/%s$'%(ylabel.strip('$'), ylabel.strip('$')))
       
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig

#-------------------------------------------------
#
### functions below here are deprecated!
#
#-------------------------------------------------

def sanity_check(x_tst, f_tst, std_tst, x_obs, f_obs, std_obs=None):
    '''
    a helper function for plotting regressions as a way to sanity check results
    '''
    xmin = np.min(x_tst)
    xmax = np.max(x_tst)

    fig = plt.figure()
    foo = fig.add_axes([0.13, 0.30, 0.85, 0.65]) ### the actual data
    res = fig.add_axes([0.13, 0.10, 0.85, 0.19]) ### residuals between data

    ### plot the regression
    foo.fill_between(x_tst, f_tst-std_tst, f_tst+std_tst, color='grey', alpha=0.5)
    foo.plot(x_tst, f_tst, color='k')

    ylim = foo.get_ylim()

    res.fill_between(x_tst, -std_tst/f_tst, +std_tst/f_tst, color='grey', alpha=0.5)
    res.plot(x_tst, np.zeros_like(x_tst, dtype='int'), color='k')

    ### iterate through observed data and overlay
    for i, (x, f) in enumerate(zip(x_obs, f_obs)):
        color = foo.plot(x, f, '.-', alpha=0.5)[0].get_color()

        if std_obs is not None:
            foo.fill_between(x, f+std_obs[i], f-std_obs[i], alpha=0.1, color=color)

        truth = (xmin<=x)*(x<=xmax)
        x = x[truth]
        f = f[truth]
        f_int = np.interp(x, x_tst, f_tst)
        res.plot(x, (f_int-f)/f_int, '.-', color=color, alpha=0.5) ### plot the residual at the observed points

        if std_obs is not None:
            res.fill_between(x, f_int-f-std_obs[i], f_int-f+std_obs[i], alpha=0.1, color=color)

    foo.set_ylim(ylim)

    ### decorate
    plt.setp(foo.get_xticklabels(), visible=False)

    for ax in [foo, res]:
      ax.set_xlim(xmin=xmin, xmax=xmax)
      ax.grid(True, which='both')

    foo.set_ylim(ylim)

    res.set_xlabel('$x$')
    res.set_ylabel(r'$(f_\ast-f)/f_\ast$')
    foo.set_ylabel('$f$')

    return fig, (foo, res)

def big_sanity_check(x_tst, f_tst, dfdx_tst, phi_tst, std_f, std_dfdx, std_phi, x_obs, f_obs):
    '''
    a basic set of plots for sanity-checking the regression
    '''
    xmin = np.min(x_tst)
    xmax = np.max(x_tst)

    zeros = np.zeros_like(x_tst, dtype='int')

    ### compute dfdx_obs and phi_obs
    dfdx_obs = []
    phi_obs = []
    for x, f in zip(x_obs, f_obs):

        # numerically estimate df/dx
        dfdx = gp.num_dfdx(x, f)
        dfdx_obs.append(dfdx) # add to list

        # compute phi = np.log(np.exp(f)/np.exp(x) * dfdx - 1)
        phi_obs.append( np.log(np.exp(f)/np.exp(x)*dfdx - 1) )

    ### set up figures and axes
    fig = plt.figure(figsize=(18,6))

    foo = plt.subplot(2,3,1) # the function itself
    res = plt.subplot(2,3,4) # the residual between GPR and the function

    doo = plt.subplot(2,3,2) # derivative of the function
    des = plt.subplot(2,3,5) # residual in derivative of the function

    phi = plt.subplot(2,3,3) # phi = log(df/dx - 1)
    pes = plt.subplot(2,3,6) # residuals of phi

    plt.subplots_adjust(
        hspace=0.02,
        wspace=0.2,
        left=0.08,
        right=0.99,
        bottom=0.10,
        top=0.90,
    )

    ### plot the regression
    std_f *= 3 ### plot 3-sigma region
    foo.fill_between(x_tst, f_tst-std_f, f_tst+std_f, color='grey', alpha=0.5)
    foo.plot(x_tst, f_tst, color='k')

#    res.fill_between(x_tst, -std_f/f_tst, +std_f/f_tst, color='grey', alpha=0.5)
    res.fill_between(x_tst, -std_f, +std_f, color='grey', alpha=0.5)
    res.plot(x_tst, zeros, color='k')

    ylim = foo.get_ylim(), 2*np.array(res.get_ylim())

    # iterate and overlay data
    for x, f in zip(x_obs, f_obs):
        color = foo.plot(x, f, '.-', alpha=0.5)[0].get_color()
        truth = (xmin<=x)*(x<=xmax)
        x = x[truth]
        f = f[truth]
        f_int = np.interp(x, x_tst, f_tst)
#        res.plot(x, (f_int-f)/f_int, '.-', color=color, alpha=0.5) ### plot the residual at the observed points
        res.plot(x, f_int-f, '.-', color=color, alpha=0.5) ### plot the residual at the observed points

    foo.set_ylim(ylim[0])
    res.set_ylim(ylim[1])

    # decorate regression
    plt.setp(foo.get_xticklabels(), visible=False)

    for ax in [foo, res]:
      ax.set_xlim(xmin=xmin, xmax=xmax)
      ax.grid(True, which='both')

    res.set_xlabel('$x$')
#    res.set_ylabel(r'$(f_\ast-f)/f_\ast$')
    res.set_ylabel(r'$f_\ast-f$')
    foo.set_ylabel('$f$')

    ### plot the regression of dfdx
    std_dfdx *= 3 ### plot 3-sigma region
    doo.fill_between(x_tst, dfdx_tst+std_dfdx, dfdx_tst-std_dfdx, color='grey', alpha=0.5)
    doo.plot(x_tst, dfdx_tst, color='k')

#    des.fill_between(x_tst, -std_dfdx/dfdx_tst, +std_dfdx/dfdx_tst, color='grey', alpha=0.5)
    des.fill_between(x_tst, -std_dfdx, +std_dfdx, color='grey', alpha=0.5)
    des.plot(x_tst, zeros, color='k')

    ylim = doo.get_ylim(), 2*np.array(des.get_ylim())

    # iterate and overlay data
    for x, dfdx in zip(x_obs, dfdx_obs):
        color = doo.plot(x, dfdx, '.-', alpha=0.5)[0].get_color()
        truth = (xmin<=x)*(x<=xmax)
        x = x[truth]
        dfdx = dfdx[truth]
        f_int = np.interp(x, x_tst, dfdx_tst)
#        des.plot(x, (f_int-dfdx)/f_int, '.-', color=color, alpha=0.5) ### plot the residual at the observed points
        des.plot(x, f_int-dfdx, '.-', color=color, alpha=0.5) ### plot the residual at the observed points

    doo.set_ylim(ylim[0])
    des.set_ylim(ylim[1])

    # decorate regression of dfdx
    plt.setp(doo.get_xticklabels(), visible=False)

    for ax in [doo, des]:
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.grid(True, which='both')

    des.set_xlabel('$x$')
#    des.set_ylabel(r'$([df/dx]_\ast - [df/dx])/[df/dx]_\ast$')
    des.set_ylabel(r'$[df/dx]_\ast - [df/dx]$')
    doo.set_ylabel('df/dx')

    ### plot the regression for phi
    std_phi *= 3 ### plot 3-sigma region
    phi.fill_between(x_tst, phi_tst+std_phi, phi_tst-std_phi, color='grey', alpha=0.5)
    phi.plot(x_tst, phi_tst, color='k')

#    pes.fill_between(x_tst, +std_phi/mean_phi, -std_phi/mean_phi, color='grey', alpha=0.5)
    pes.fill_between(x_tst, +std_phi, -std_phi, color='grey', alpha=0.5)
    pes.plot(x_tst, zeros, color='k')

    ylim = phi.get_ylim(), 2*np.array(pes.get_ylim())

    # iterate over data
    for x, p in zip(x_obs, phi_obs):
        color = phi.plot(x, p, '.-', alpha=0.5)[0].get_color()
        truth = (xmin<=x)*(x<=xmax)
        x = x[truth]
        p = p[truth]
        f_int = np.interp(x, x_tst, phi_tst)
#        pes.plot(x, (f_int-p)/f_int, '.-', color=color, alpha=0.5) ### plot the residual at the observed points
        pes.plot(x, (f_int-p), '.-', color=color, alpha=0.5) ### plot the residual at the observed points

    phi.set_ylim(ylim[0])
    pes.set_ylim(ylim[1])

    # decorate regression of phi
    plt.setp(phi.get_xticklabels(), visible=False)

    for ax in [phi, pes]:
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.grid(True, which='both')

    pes.set_xlabel('$x$')
#    pes.set_ylabel(r'$(\phi_\ast - \phi)/\phi_\ast$')
    pes.set_ylabel(r'$(\phi_\ast - \phi)$')
    phi.set_ylabel('$\phi$')

    ### return
    return fig, (foo, res), (doo, des), (phi, pes)
