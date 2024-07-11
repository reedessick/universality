"""a module that houses plotting routines to visualize gaussian processes
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import erfinv

### non-standard libraries
from . import utils as plt
from universality.gaussianprocess import gaussianprocess as gp

#-------------------------------------------------

def cov(model, colormap=plt.DEFAULT_COLORMAP, figwidth=plt.DEFAULT_COV_FIGWIDTH, figheight=plt.DEFAULT_COV_FIGHEIGHT, tanh=False):
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
    plt.plt.sca(eax)
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
    plt.plt.sca(vax)
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
        markersizes=None,
        xlabel=None,
        ylabel=None,
        figwidth=plt.DEFAULT_FIGWIDTH,
        figheight=plt.DEFAULT_FIGHEIGHT,
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
            ax = fig.add_axes(plt.MAIN_AXES_POSITION)
            rs = fig.add_axes(plt.RESIDUAL_AXES_POSITION)

        else:
            ax = fig.add_axes(plt.AXES_POSITION)
    else:
        if subax:
            fig, ax, rs = figtup
        else:
            fig, ax = figtup

    N = len(curves)
    if colors is None:
        colors = [plt.DEFAULT_COLOR2 for _ in range(N)]

    if alphas is None:
        alphas = [plt.DEFAULT_ALPHA for _ in range(N)]

    if linestyles is None:
        linestyles = [plt.DEFAULT_LINESTYLE for _ in range(N)]

    if markers is None:
        markers = [plt.DEFAULT_MARKER for _ in range(N)]

    if markersizes is None:
        markersizes = [plt.DEFAULT_MARKERSIZE for _ in range(N)]

    xmin = np.min([np.min(x) for x, _, _ in curves if len(x)])
    xmax = np.max([np.max(x) for x, _, _ in curves if len(x)])
    ymin = np.min([np.min(f) for _, f, _ in curves if len(f)])
    ymax = np.max([np.max(f) for _, f, _ in curves if len(f)])

    # plot the observed data
    for (x, f, label), c, l, a, m, s in zip(curves, colors, linestyles, alphas, markers, markersizes):
        ax.plot(x, f, linestyle=l, color=c, marker=m, label=label, alpha=a, markersize=s)

    # plot residuals, etc
    if subax:
        if reference_curve is None:
            x_ref, f_ref, _ = curves[0]
        else:
            x_ref, f_ref = reference_curve

        # plot the observed data
        for (x, f, _), c, l, a, m, s in zip(curves, colors, linestyles, alphas, markers, markersizes):
            f = np.interp(x_ref, x, f)

            if fractions:
                rs.plot(x_ref, (f-f_ref)/f_ref, linestyle=l, color=c, alpha=a, marker=m, markersize=s)

            elif residuals:
                rs.plot(x_ref, f-f_ref, linestyle=l, color=c, alpha=a, marker=m, markersize=s)

            elif ratios:
                rs.plot(x_ref, f/f_ref, linestyle=l, color=c, alpha=a, marker=m, markersize=s)

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
            rs.set_ylabel('$%s/%s_{\mathrm{ref}}$'%(ylabel.strip('$'), ylabel.strip('$')))

        return fig, ax, rs

    else:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        return fig, ax

def overlay_model(
        model,
        color=plt.DEFAULT_COLOR1,
        alpha=plt.DEFAULT_ALPHA,
        color_minimum=0.1,
        linestyle=plt.DEFAULT_LINESTYLE,
        marker=plt.DEFAULT_MARKER,
        levels=plt.DEFAULT_LEVELS, ### the confidence levels to include in shading
        figwidth=plt.DEFAULT_FIGWIDTH,
        figheight=plt.DEFAULT_FIGHEIGHT,
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
            ax = fig.add_axes(plt.MAIN_AXES_POSITION)
            rs = fig.add_axes(plt.RESIDUAL_AXES_POSITION)

        else:
            ax = fig.add_axes(plt.AXES_POSITION)
    else:
        if subax:
            fig, ax, rs = figtup
        else:
            fig, ax = figtup

    ### plot the confidence regions
    sigmas = [2**0.5*erfinv(level) for level in levels] ### base sigmas on Guassian cumulative distribution and the desired confidence levels

    weights = [m['weight'] for m in model]
    colors = plt.weights2color(weights, color, prefact=alpha/max(2., len(sigmas)*2.), minimum=color_minimum)

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
