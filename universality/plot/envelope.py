"""a module housing logic for envelop process plots (1D quantiles)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from . import utils as plt

#-------------------------------------------------

def envelope(
        x,
        quantiles,
        medians,
        names,
        colors,
        xcolumn_label,
        ycolumn_label,
        xcolumn_range,
        legend=False,
        neff_nkde=None,
        logxcolumn=False,
        logycolumn=False,
        grid=False,
        ymin=None,
        ymax=None,
        res_ymin=None,
        res_ymax=None,
        xsignposts=[],
        ysignposts=[],
        signpost_color=plt.DEFAULT_TRUTH_COLOR,
        y_reference=None,
        reference=[],
        reference_colors=None,
        residuals=False,
        ratios=False,
        filled=False,
        alpha=plt.DEFAULT_ALPHA,
        fig=None,
        figwidth=plt.DEFAULT_FIGWIDTH,
        figheight=plt.DEFAULT_FIGHEIGHT,
    ):

    # instantiate figure object or unpack it as needed
    if fig is None:
        fig = plt.figure(figsize=(figwidth, figheight)) ### FIXME: set figsize based on Ncol?
        if residuals or ratios:
            ax = fig.add_axes(plt.MAIN_AXES_POSITION)
            ax_res = fig.add_axes(plt.RESIDUAL_AXES_POSITION)
        else:
            ax = fig.add_axes(plt.AXES_POSITION)

    elif residuals or ratios:
        fig, ax, ax_res = fig

    else:
        fig, ax = fig

    Nn, Nq, Nx = np.shape(quantiles)
    assert Nn == len(names), 'bad shape for quantiles!'

    # iterate through data and plot
    for ind, label in enumerate(names):
        color = colors[label]
        # add quantiles
        for i in range(Nq/2): ### fill between pairs of quantiles
            if filled: ### fill in inter-quantile regions
                ax.fill_between(x, quantiles[ind,2*i,:], quantiles[ind,2*i+1,:], alpha=alpha, color=color)
                if residuals:
                    ax_res.fill_between(x, quantiles[ind,2*i,:]-y_reference, quantiles[ind,2*i+1,:]-y_reference, alpha=alpha, color=color)
                elif ratios:
                    ax_res.fill_between(x, quantiles[ind,2*i,:]/y_reference, quantiles[ind,2*i+1,:]/y_reference, alpha=alpha, color=color)

            ### plot quantile boundaries
            ax.plot(x, quantiles[ind,2*i,:], alpha=alpha, color=color)
            ax.plot(x, quantiles[ind,2*i+1,:], alpha=alpha, color=color)
            if residuals:
                ax_res.plot(x, quantiles[ind,2*i,:]-y_reference, alpha=alpha, color=color)
                ax_res.plot(x, quantiles[ind,2*i+1,:]-y_reference, alpha=alpha, color=color)
            elif ratios:
                ax_res.plot(x_test, quantiles[ind,2*i,:]/y_reference, alpha=alpha, color=color)
                ax_res.plot(x_test, quantiles[ind,2*i+1,:]/y_reference, alpha=alpha, color=color)

        # add median
        ax.plot(x, medians[ind,:], color=color, alpha=1.0) ### plot the median
        if residuals:
            ax_res.plot(x, medians[ind,:]-y_reference, color=color, alpha=1.0)
        elif ratios:
            ax_res.plot(test, medians[ind,:]/y_reference, color=color, alpha=1.0)

    # add reference curves
    for ref_label, curve in reference:
        X = curve[:,0]
        Y = curve[:,1]
        color = reference_colors[ref_label]
        ax.plot(X, Y, color=color, alpha=0.5)

        if residuals:
            ax_res.plot(x_test, np.interp(x, X, Y)-y_reference, color=color, alpha=0.5)
        elif ratios:
            ax_res.plot(x_test, np.interp(x, X, Y)/y_reference, color=color, alpha=0.5)

    ### decorate
    if residuals or ratios:
        fig = fig, ax, ax_res
    else:
        fig = fig, ax

    fig = annotate_envelope(
        fig,
        names,
        colors,
        xcolumn_label,
        ycolumn_label,
        xcolumn_range,
        legend=legend,
        neff_nkde=neff_nkde,
        logxcolumn=logxcolumn,
        logycolumn=logycolumn,
        grid=grid,
        ymin=ymin,
        ymax=ymax,
        res_ymin=res_ymin,
        res_ymax=res_ymax,
        xsignposts=xsignposts,
        ysignposts=ysignposts,
        signpost_color=signpost_color,
        y_reference=y_reference,
        residuals=residuals,
        ratios=ratios,
    )

    return fig

#------------------------

def annotate_envelope(
        fig,
        names,
        colors,
        xcolumn_label,
        ycolumn_label,
        xcolumn_range,
        legend=False,
        neff_nkde=None,
        logxcolumn=False,
        logycolumn=False,
        grid=False,
        ymin=None,
        ymax=None,
        res_ymin=None,
        res_ymax=None,
        xsignposts=[],
        ysignposts=[],
        signpost_color=plt.DEFAULT_TRUTH_COLOR,
        y_reference=None,
        residuals=False,
        ratios=False,
    ):

    xmin, xmax = xcolumn_range

    include_neff =  neff_nkde is not None
    if include_neff:
        neff, nkde = neff_nkde

    if residuals or ratios:
        fig, ax, ax_res = fig
    else:
        fig, ax = fig

    # add legend
    if legend:
        center = plt.MAIN_AXES_POSITION[0] + 0.5*plt.MAIN_AXES_POSITION[2]
        if len(names)==1:
            placement = [center]
        else:
            placement = np.linspace(plt.MAIN_AXES_POSITION[0], plt.MAIN_AXES_POSITION[0]+plt.MAIN_AXES_POSITION[2], len(names))

        for i, (placement, label) in enumerate(zip(placement, names)):
            legend = label
            if include_neff:
                legend = legend+": $N_\mathrm{eff} = %.1f,\ N_\mathrm{kde} = %.1f$"%(neff[i], nkde[i])
            if placement < center:
                ha='left'
            elif placement==center:
                ha='center'
            else:
                ha='right'
            fig.text(placement, 0.5*(1 + plt.MAIN_AXES_POSITION[1] + plt.MAIN_AXES_POSITION[3]), legend, color=colors[label], ha=ha, va='center')

    # scaling, etc
    if logxcolumn:
        ax.set_xscale('log')

    if logycolumn:
        ax.set_yscale('log')
        if ratios:
            ax_res.set_yscale('log')

    ax.grid(grid, which='both')
    if residuals or ratios:
        ax_res.grid(grid, which='both')

    # set limits
    ax.set_xlim(xmin=xmin, xmax=xmax)
    if residuals or ratios:
        ax_res.set_xscale(ax.get_xscale())
        ax_res.set_xlim(ax.get_xlim())

    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)

    if residuals or ratios:
        if res_ymin is not None:
            ax_res.set_ylim(ymin=res_ymin)
        if res_ymax is not None:
            ax_res.set_ylim(ymax=res_ymax)

    ### add signposts
    ylim = ax.get_ylim()
    for value in xsignposts:
        ax.plot([value]*2, ylim, color=signpost_color)
    ax.set_ylim(ylim)

    xlim = ax.get_xlim()
    for value in ysignposts:
        ax.plot(xlim, [value]*2, color=signpost_color)
    ax.set_xlim(xlim)

    if residuals or ratios:
        ylim = ax_res.get_ylim()
        for value in xsignposts:
            ax_res.plot([value]*2, ylim, color=signpost_color)
        ax_res.set_ylim(ylim)

        xlim = ax_res.get_xlim()
        for value in ysignposts:
            X = np.linspace(xlim[0], xlim[1], len(y_reference))
            if residuals:
                Y = value-np.ones_like(y_reference)
            else:
                Y = value/np.ones_like(y_reference)
            ax_res.plot(X, Y, color=signpost_color)
        ax_res.set_xlim(xlim)

    # set labels
    ax.set_ylabel(ycolumn_label)
    if residuals or ratios:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax_res.set_xlabel(xcolumn_label)
        if residuals:
            ax_res.set_ylabel('%s - %s'%(xcolumn_label, y_reference_label))
        if ratios:
            ax_res.set_ylabel('%s/%s'%(xcolumn_label, y_reference_label))
    else:
        ax.set_xlabel(xcolumn_label)

    # return
    if ratios or residuals:
        return fig, ax, ax_res

    else:
        return fig, ax
