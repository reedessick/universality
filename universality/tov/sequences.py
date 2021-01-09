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

DEFAULT_INTEGRATION_RTOL = 1e-4

KNOWN_FORMALISMS = [
    'standard',
    'standard_MR',
    'standard_MRLambda',
    'logenthalpy',
    'logenthalpy_MR',
    'logenthalpy_MRLambda',
]
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
    expect eos = (pressurec2, energy_densityc2, baryon_density, cs2c2)
    """
    if 'logenthalpy' in formalism: ### logenthalpy is the integration coordinate
        if formalism == 'logenthalpy':
            integrate = logenthalpy.integrate
            macro_cols = logenthalpy.MACRO_COLS

        elif formalism == 'logenthalpy_MR':
            integrate = logenthalpy.integrate_MR
            macro_cols = logenthalpy.MACRO_COLS_MR

        elif formalism == 'logenthalpy_MRLambda':
            integrate = logenthalpy.integrate_MRLambda
            macro_cols = logenthalpy.MACRO_COLS_MRLambda

        else:
            raise ValueError('logenthalpy-based formalism=%s not understood! Must be one of: %s'%(formalism, ', '.join(KNOWN_FORMALISMS)))

        R_ind = None ### don't pass max_dr to integrate

        ### compute the log(enthalpy per rest mass). Do this here so we only have to do it once
        pc2, ec2, rho, cs2c2 = eos
        logh = logenthalpy.eos2logh(pc2, ec2)
        eos = (logh, pc2, ec2, rho, cs2c2)

    elif 'standard' in formalism: ### radius is the integration coordinate
        if formalism == 'standard':
            integrate = standard.integrate
            macro_cols = standard.MACRO_COLS

        elif formalism == 'standard_MR':
            integrate = standard.integrate_MR
            macro_cols = standard.MACRO_COLS_MR

        elif formalism == 'standard_MRLambda':
            integrate = standard.integrate_MRLambda
            macro_cols = standard.MACRO_COLS_MRLambda

        else:
            raise ValueError('standard formalism=%s not understood! Must be one of: %s'%(formalism, ', '.join(KNOWN_FORMALISMS)))

        R_ind = macro_cols.index('R')

    else: ### formalism not understood
        raise ValueError('formalism=%s not understood! Must be one of: %s'%(formalism, ', '.join(KNOWN_FORMALISMS)))

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
            min_pc2_macro=macro[-1],
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            integration_rtol=integration_rtol,
            R_ind=R_ind,
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
        min_pc2_macro=None,
        max_pc2_macro=None,
        interpolator_rtol=DEFAULT_INTERPOLATOR_RTOL,
        min_dpressurec2_rtol=DEFAULT_MIN_DPRESSUREC2_RTOL,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        R_ind=None,
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
        if R_ind is not None:
            ### scale max step size with what we expect for the radius
            ### we need this to be pretty conservative, as this loop is is primarily entered when
            ### we are just starting a segment and there could be wild changes in the radius
            kwargs['max_dr'] = 0.001*min_pc2_macro[R_ind] * 1e5 ### convert from km -> cm

        max_pc2_macro = foo(max_pc2, eos, rtol=integration_rtol, **kwargs)
    max_pc2_macro = np.array(max_pc2_macro)

    ### check to see whether central pressures are close enough
    if 2*(max_pc2 - min_pc2) < min_dpressurec2_rtol * (max_pc2 + min_pc2):
        return [min_pc2, max_pc2], [min_pc2_macro, max_pc2_macro]

    ### integrate at the mid point
    mid_pc2 = (min_pc2*max_pc2)**0.5
    if verbose:
        print('computing stellar model with central pressure/c2 = %.6e'%mid_pc2)
    if R_ind is not None:
        ### here we can be less stringent with max_dr since we're interpolating between models and have a better idea of the behavior
        kwargs['max_dr'] = 0.1*min(min_pc2_macro[R_ind], max_pc2_macro[R_ind]) * 1e5 ### convert from km -> cm

    mid_pc2_macro = np.array(foo(mid_pc2, eos, rtol=integration_rtol, **kwargs))

    ### condition on whether we are accurate enough to determine recursive termination condition
    # compute errors based on a linear interpolation
    errors = mid_pc2_macro - (min_pc2_macro + (max_pc2_macro - min_pc2_macro) * (mid_pc2 - min_pc2) / (max_pc2 - min_pc2))

    if np.all(np.abs(errors) < interpolator_rtol*mid_pc2_macro): ### interpolation is "good enough"
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
