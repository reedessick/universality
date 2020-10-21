"""a module that houses routines to solve for sequences of stellar models
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from .ode import (standard, logenthalpy)

#-------------------------------------------------

DEFAULT_MIN_NUM_MODELS = 10

DEFAULT_INTERPOLATOR_RTOL = 1e-2 ### used to determine accuracy of interpolator for macroscopic properties
DEFAULT_MIN_DPRESSUREC2_RTOL = 1e-2 ### used put a limit on how closely we space central pressures

DEFAULT_INTEGRATION_RTOL = 1e-6

KNOWN_FORMALISMS = ['logenthalpy', 'standard']
DEFAULT_FORMALISM = KNOWN_FORMALISMS[0]

#-------------------------------------------------

def stellar_sequence(
        min_central_pressurec2,
        max_central_pressurec2,
        eos,
        min_num_models=DEFAULT_MIN_NUM_MODELS,
        interpolator_rtol=DEFAULT_INTERPOLATOR_RTOL,
        min_dpressurec2_rtol=DEFAULT_MIN_DPRESSUREC2_RTOL,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        formalism=DEFAULT_FORMALISM,
        verbose=False,
        **kwargs
    ):
    """solve for a sequence of stellar models such that the resulting interpolator has relative error less than "interpolator_rtol"
    expect eos = (pressurec2, energy_densityc2)
    """
    if formalism == 'logenthalpy':
        integrate = logenthalpy.integrate
        macro_cols = logenthalpy.MACRO_COLS

    elif formalism == 'standard':
        integrate = standard.integrate
        macro_cols = standard.MACRO_COLS

    else:
        raise ValueError('formalism=%s not understood!'%formalism)

    R_ind = macro_cols.index('R') ### the column corresponding to the radius

    ### determine the initial grid of central pressures
    pressurec2 = eos[0]
    central_pressurec2 = list(np.logspace(np.log10(min_central_pressurec2), np.log10(max_central_pressurec2), min_num_models))

    ### recursively call integrator until interpolation is accurate enough
    central_pc2 = [central_pressurec2[0]]
    if verbose:
        print('computing stellar model with central pressure/c2 = %.6e'%central_pc2[-1])

    macro = [integrate(central_pc2[-1], eos, rtol=integration_rtol, **kwargs)]

    ### perform recursive search to get a good integrator
    for max_pc2 in central_pressurec2[1:]:
        new_central_pc2, new_macro = bisection_stellar_sequence(
            central_pc2[-1],
            max_pc2,
            eos,
            integrate,
            R_ind,
            min_pc2_macro=macro[-1],
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            integration_rtol=integration_rtol,
            verbose=verbose,
        )

        ### add the stellar models to the cumulative list
        central_pc2 += new_central_pc2[1:]
        macro += new_macro[1:]

    ### return the results
    return central_pc2, macro, macro_cols

def bisection_stellar_sequence(
        min_pc2,
        max_pc2,
        eos,
        foo,
        R_ind,
        min_pc2_macro=None,
        max_pc2_macro=None,
        interpolator_rtol=DEFAULT_INTERPOLATOR_RTOL,
        min_dpressurec2_rtol=DEFAULT_MIN_DPRESSUREC2_RTOL,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        verbose=False,
        **kwargs
    ):
    '''recursively compute estimates of the interpolator error until it is below rtol
    '''
    if min_pc2_macro is None:
        if verbose:
            print('computing stellar model with central pressure/c2 = %.6e'%min_pc2)
        min_pc2_macro = foo(min_pc2, eos, rtol=integration_rtol, **kwargs)
    min_pc2_macro = np.array(min_pc2_macro)

    if max_pc2_macro is None:
        if verbose:
            print('computing stellar model with central pressure/c2 = %.6e'%max_pc2)
        max_dr = 0.1*min_pc2_macro[R_ind] * 1e5 ### scale max step size with what we expect for the radius
                                                ### convert from km -> cm
        max_pc2_macro = foo(max_pc2, eos, rtol=integration_rtol, max_dr=max_dr, **kwargs)
    max_pc2_macro = np.array(max_pc2_macro)

    ### check to see whether central pressures are close enough
    if 2*(max_pc2 - min_pc2) < min_dpressurec2_rtol * (max_pc2 + min_pc2):
        return [min_pc2, max_pc2], [min_pc2_macro, max_pc2_macro]

    ### integrate at the mid point
    mid_pc2 = (min_pc2*max_pc2)**0.5
    if verbose:
        print('computing stellar model with central pressure/c2 = %.6e'%mid_pc2)
    max_dr = 0.1*min(min_pc2_macro[R_ind], max_pc2_macro[R_ind]) * 1e5 ### scale max step size with what we expect for the radius
                                                                       ### convert from km -> cm
    mid_pc2_macro = np.array(foo(mid_pc2, eos, rtol=integration_rtol, max_dr=max_dr, **kwargs))

    ### condition on whether we are accurate enough to determine recursive termination condition
    # compute errors based on a linear interpolation
    errors = mid_pc2_macro - (min_pc2_macro + (max_pc2_macro - min_pc2_macro) * (mid_pc2 - min_pc2) / (max_pc2 - min_pc2))

    if np.all(np.abs(errors) < integration_rtol*mid_pc2_macro): ### interpolation is "good enough"
        return [min_pc2, mid_pc2, max_pc2], [min_pc2_macro, mid_pc2_macro, max_pc2_macro]

    else: # interpolation is not good enough, so we recurse to compute mid-points of sub-intervals
        left_pc2, left_macro = bisection_stellar_sequence(
            min_pc2,
            mid_pc2,
            eos,
            foo,
            min_pc2_macro=min_pc2_macro,
            max_pc2_macro=mid_pc2_macro,
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            R_ind=R_ind,
            integration_rtol=integration_rtol,
            verbose=verbose,
            **kwargs
        )

        right_pc2, right_macro = bisection_stellar_sequence(
            mid_pc2,
            max_pc2,
            eos,
            foo,
            min_pc2_macro=mid_pc2_macro,
            max_pc2_macro=max_pc2_macro,
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            R_ind=R_ind,
            integration_rtol=integration_rtol,
            verbose=verbose,
            **kwargs
        )

        return left_pc2 + right_pc2[1:], left_macro + right_macro[1:] ### avoid returning repeated models
