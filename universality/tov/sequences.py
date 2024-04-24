"""a module that houses routines to solve for sequences of stellar models
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import sys

import numpy as np

from .ode import (standard, logenthalpy)
from universality.utils import (io, utils)
from universality.properties.branches import (initial_stability, final_collapse)

#-------------------------------------------------

DEFAULT_MIN_NUM_MODELS = 2

DEFAULT_INTERPOLATOR_RTOL = 1e-2 ### used to determine accuracy of interpolator for macroscopic properties
DEFAULT_MIN_DPRESSUREC2_RTOL = 1e-2 ### used put a limit on how closely we space central pressures

DEFAULT_INTEGRATION_RTOL = 1e-4

KNOWN_FORMALISMS = [
    'logenthalpy',
    'logenthalpy_MR',
    'logenthalpy_MRLambda',
    'standard',
    'standard_MR',
    'standard_MRLambda',
]
DEFAULT_FORMALISM = KNOWN_FORMALISMS[0]

KNOWN_REGULAR_GRIDDINGS = [
    'logarithmic',
    'linear',
]
DEFAULT_REGULAR_GRIDDING = KNOWN_REGULAR_GRIDDINGS[0]

KNOWN_GRIDDINGS = ['bisection'] + KNOWN_REGULAR_GRIDDINGS
DEFAULT_GRIDDING = KNOWN_GRIDDINGS[0]

DEFAULT_CENTRAL_COLUMN_TEMPLATE = 'central_%s'

DEFAULT_PRESSUREC2_COLUMN = 'pressurec2'
DEFAULT_ENERGY_DENSITYC2_COLUMN = 'energy_densityc2'
DEFAULT_BARYON_DENSITY_COLUMN = 'baryon_density'

DEFAULT_EXTEND_DOWN_FACTOR = 1.1 # the factor by which we divide central_pc2 when extending range to lower pressures within stellar sequences
DEFAULT_EXTEND_UP_FACTOR = 2.0 # the factor by which we multiply central_pc2 when extending range to higher pressures within stellar sequences

#-------------------------------------------------

def process2sequences(
        eos,
        eostmp,
        mactmp,
        min_central_pressurec2,
        max_central_pressurec2,
        central_pressurec2=[],
        central_baryon_density_range=None,
        central_energy_densityc2_range=None,
        mod=1000,
        pressurec2_column=DEFAULT_PRESSUREC2_COLUMN,
        energy_densityc2_column=DEFAULT_ENERGY_DENSITYC2_COLUMN,
        baryon_density_column=DEFAULT_BARYON_DENSITY_COLUMN,
        cs2c2_column=None,
        central_eos_column=[],
        central_column_template=DEFAULT_CENTRAL_COLUMN_TEMPLATE,
        formalism=DEFAULT_FORMALISM,
        gridding=DEFAULT_GRIDDING,
        extend_up=False,
        extend_down=False,
        verbose=False,
        Verbose=False,
        **kwargs
    ):
    '''integrate stellar models for a whole set of EoS in a process
    '''
    verbose |= Verbose

    for draw in eos:
        tmp = {'draw':draw, 'moddraw':draw//mod}
        eospath = eostmp%tmp

        if verbose:
            print('loading EoS data from: '+eospath)
        cols = [pressurec2_column, energy_densityc2_column, baryon_density_column]
        if cs2c2_column is not None:
            cols.append(cs2c2_column)

        data, cols = io.load(eospath, cols+central_eos_column) ### NOTE: this will not produce duplicated columns

        pressurec2 = data[:,cols.index(pressurec2_column)]
        energy_densityc2 = data[:,cols.index(energy_densityc2_column)]
        baryon_density = data[:, cols.index(baryon_density_column)]

        if cs2c2_column is not None:
            cs2c2 = data[:,cols.index(cs2c2_column)]
        else:
            cs2c2 = utils.num_dfdx(energy_densityc2, pressurec2)

        ### get local copy of bounds for just this EoS
        max_central_pc2 = max_central_pressurec2
        min_central_pc2 = min_central_pressurec2

        ### sanity check that our integration range is compatible with the EoS data available
        max_pressurec2 = np.max(pressurec2)
        if max_central_pc2 > max_pressurec2:
            if verbose:
                print('limitting central_pressurec2 <= %.6e based on EoS data\'s range'%max_pressurec2)
            max_central_pc2 = max_pressurec2

        min_pressurec2 = np.min(pressurec2)
        if min_central_pc2 < min_pressurec2:
            if verbose:
                print('limitting central_pressurec2 >= %.6e based on EoS data\'s range'%min_pressurec2)
            min_central_pc2 = min_pressurec2

        ### additionally check whether we're obeying the requested bounds on central baryon and energy densities
        if central_baryon_density_range is not None:
            min_baryon_density, max_baryon_density = central_baryon_density_range

            # check minimum
            min_pc2 = np.interp(min_baryon_density, baryon_density, pressurec2)
            if min_pc2 > min_central_pc2:
                if verbose:
                    print('limitting central_pressurec2 >= %.6e based on min_baryon_density = %.6e'%(min_pc2, min_baryon_density))
                min_central_pc2 = min_pc2

            # check maximum
            max_pc2 = np.interp(max_baryon_density, baryon_density, pressurec2)
            if max_pc2 < max_central_pc2:
                if verbose:
                    print('limitting central_pressurec2 <= %.6e based on max_baryon_density = %.6e'%(max_pc2, max_baryon_density))
                max_central_pc2 = max_pc2

        if central_energy_densityc2_range is not None:
            min_energy_densityc2, max_energy_densityc2 = central_energy_densityc2_range

            # check minimum
            min_pc2 = np.interp(min_energy_densityc2, energy_densityc2, pressurec2)
            if min_pc2 > min_central_pc2:
                if verbose:
                    print('limitting central_pressurec2 >= %.6e based on min_energy_densityc2 = %.6e'%(min_pc2, min_energy_densityc2))
                min_central_p2 = min_pc2

            # check maximum
            max_pc2 = np.interp(max_baryon_density, energy_densityc2, pressurec2)
            if max_pc2 < max_central_pc2:
                if verbose:
                    print('limitting central_pressurec2 <= %.6e based on max_energy_densityc2 = %.6e'%(max_pc2, max_energy_densityc2))
                max_central_pc2 = max_pc2

        ### check to make sure the pressure bounds are sane, futz them if they are not
        if max_central_pc2 < min_central_pc2:
            if verbose:
                print('''WARNING: central pressure bounds are out of order! Reverting to original bounds!
    min_central_pressurec2 = %.6e
    max_central_pressurec2 = %.6e'''%(min_central_pressurec2, max_central_pressurec2))
            min_central_pc2, max_central_pc2 = min_central_pressurec2, max_central_pressurec2

        if verbose:
            print('''proceeding with central pressure bounds:
    min_central_pressurec2 = %.6e
    max_central_pressurec2 = %.6e'''%(min_central_pc2, max_central_pc2))

        ### now compute the stellar sequence
        if verbose:
            print('solving for sequence of stellar models with formalism=%s and gridding=%s'%(formalism, gridding))

        central_pc2, macro, macro_cols = stellar_sequence(
            min_central_pc2,
            max_central_pc2,
            (pressurec2, energy_densityc2, baryon_density, cs2c2),
            central_pressurec2=central_pressurec2,
            verbose=Verbose,
            formalism=formalism,
            gridding=gridding,
            extend_up=extend_up,
            extend_down=extend_down,
            **kwargs
        )

        if verbose:
            print('    evaluated %d stellar models'%len(central_pc2))

        sequence, columns = append_central_values(
            central_pc2,
            pressurec2,
            data,
            cols,
            macro,
            macro_cols,
            central_eos_column=central_eos_column,
            central_column_template=central_column_template,
            verbose=verbose,
        )

        ### write the output
        macpath = mactmp%tmp
        if verbose:
            print('writing stellar sequence to: '+macpath)
        io.write(macpath, sequence, columns)

#-------------------------------------------------

def append_central_values(
        central_pressurec2,
        pressurec2,
        eosdata,
        eoscols,
        macdata,
        maccols,
        central_eos_column=[],
        central_column_template=DEFAULT_CENTRAL_COLUMN_TEMPLATE,
        verbose=False,
    ):

    ### figure out the central values of all the EoS columns
    if verbose:
        print('extracting central values of all EoS parameters')
    Neos = len(central_eos_column)
    Nmac = len(maccols)

    sequence = np.empty((len(central_pressurec2), Neos+Nmac), dtype=float)
    columns = []

    # extract the central EoS parameters 
    for i, col in enumerate(central_eos_column):
        sequence[:,i] = np.interp(central_pressurec2, pressurec2, eosdata[:,eoscols.index(col)])
        columns.append(central_column_template%col)

    # add in the macro properties
    sequence[:,Neos:] = macdata
    columns += maccols

    return sequence, columns

#-------------------------------------------------

def stellar_sequence(
        min_central_pressurec2,
        max_central_pressurec2,
        eos,
        central_pressurec2=[],
        min_num_models=DEFAULT_MIN_NUM_MODELS,
        interpolator_rtol=DEFAULT_INTERPOLATOR_RTOL,
        min_dpressurec2_rtol=DEFAULT_MIN_DPRESSUREC2_RTOL,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        formalism=DEFAULT_FORMALISM,
        gridding=DEFAULT_GRIDDING,
        extend_up=False,
        extend_down=False,
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

        min_eos_pc2 = np.min(pc2)
        max_eos_pc2 = np.max(pc2)

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

        min_eos_pc2 = np.min(eos[0])
        max_eos_pc2 = np.max(eos[0])

    else: ### formalism not understood
        raise ValueError('formalism=%s not understood! Must be one of: %s'%(formalism, ', '.join(KNOWN_FORMALISMS)))

    ### determine the initial grid of central pressures
    # include any that were specifically requested through command-line argument
    for pc2 in central_pressurec2:
        assert (min_central_pressurec2 <= pc2), \
            'requested central_pressurec2=%.6e < min central_pressurec2=%.6e'%(pc2, min_central_pressurec2)
        assert (pc2 <= max_central_pressurec2), \
            'requested central_pressurec2=%.6e > max central_pressurec2=%.6e'%(pc2, max_central_pressurec2)

    #--------------------

    # figure out search algorithms based on gridding

    if gridding == 'bisection':
        central_pc2, macro = bisection_stellar_grid(
            min_central_pressurec2,
            max_central_pressurec2,
            integrate,
            eos,
            min_eos_pc2,
            max_eos_pc2,
            macro_cols,
            R_ind,
            min_num_models=min_num_models,
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            integration_rtol=integration_rtol,
            central_pressurec2=central_pressurec2,
            extend_up=extend_up,
            extend_down=extend_down,
            verbose=verbose,
            **kwargs
        )

    elif gridding in ['linear', 'logarithmic']:
        central_pc2, macro = regular_stellar_grid(
            min_central_pressurec2,
            max_central_pressurec2,
            integrate,
            eos,
            min_eos_pc2,
            max_eos_pc2,
            macro_cols,
            R_ind,
            gridding=gridding,
            min_num_models=min_num_models,
            integration_rtol=integration_rtol,
            central_pressurec2=central_pressurec2,
            extend_up=extend_up,
            extend_down=extend_down,
            verbose=verbose,
            **kwargs
        )

    else:
        raise ValueError('gridding=%s not understood; must be one of: %s'%(gridding, ', '.join(KNOWN_GRIDDINGS)))

    central_pc2 = np.array(central_pc2)
    macro = np.array(macro)

    #--------------------

    ### exponentiate logLambda -> Lambda
    if 'logLambda' in macro_cols:
        ind = macro_cols.index('logLambda')
        macro[:,ind] = np.exp(macro[:,ind])
        macro_cols = [col for col in macro_cols] # make a copy so we don't overwrite the default list
        macro_cols[ind] = 'Lambda'

    ### return the results
    return central_pc2, macro, macro_cols

#-------------------------------------------------

def bisection_stellar_grid(
        min_central_pressurec2,
        max_central_pressurec2,
        integrate,
        eos,
        min_eos_pc2,
        max_eos_pc2,
        macro_cols,
        R_ind,
        central_pressurec2=[],
        min_num_models=DEFAULT_MIN_NUM_MODELS,
        interpolator_rtol=DEFAULT_INTERPOLATOR_RTOL,
        min_dpressurec2_rtol=DEFAULT_MIN_DPRESSUREC2_RTOL,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        extend_up=False,
        extend_down=False,
        verbose=False,
        **kwargs
    ):

    central_pressurec2 = sorted(central_pressurec2 + \
        list(np.logspace(np.log10(min_central_pressurec2), np.log10(max_central_pressurec2), min_num_models)))

    #--------------------

    ### recursively call integrator until interpolation is accurate enough
    central_pc2 = [central_pressurec2[0]]
    if verbose:
        sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%central_pc2[-1])
        sys.stdout.flush()

    macro = [integrate(central_pc2[-1], eos, rtol=integration_rtol, **kwargs)]

    ### perform recursive search to get a good interpolator
    for max_pc2 in central_pressurec2[1:]:
        new_central_pc2, new_macro = _bisection_stellar_sequence(
            central_pc2[-1],
            max_pc2,
            integrate,
            eos,
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

    #---

    if extend_down: # extend the set of central_pc2 to lower values if needed
        if verbose:
            sys.stdout.write('\nextending search to lower central pressures if stopping criteria are not met')

        ind_M = macro_cols.index('M')
        M = [] # only look for the "new_macro"
               # NOTE: this could create annoying behavior if one is not careful with the initial guess
               #       for the minimum central pressurec2

        stable = initial_stability(M)
        if verbose and (stable is None): # get printing to look pretty both when we do and do not integrate more models
            sys.stdout.write('\n')

        while (stable is None) and (central_pc2[0] > min_eos_pc2):
            min_pc2 = max(min_eos_pc2, central_pc2[0] / DEFAULT_EXTEND_DOWN_FACTOR) # increment the minimum value

            new_central_pc2, new_macro = _bisection_stellar_sequence(
                min_pc2,
                central_pc2[0],
                integrate,
                eos,
                max_pc2_macro=macro[0],
                interpolator_rtol=interpolator_rtol,
                min_dpressurec2_rtol=min_dpressurec2_rtol,
                integration_rtol=integration_rtol,
                R_ind=R_ind,
                verbose=verbose,
            )

            ### add the stellar models to the cumulative list
            central_pc2 = new_central_pc2[:-1] + central_pc2
            macro = new_macro[:-1] + macro

            M = [_[ind_M] for _ in new_macro[:-1]] + M

            stable = initial_stability(M)

        if stable is not None:
            sys.stdout.write('\ninitial stopping criteria reached at pressurec2=%.6e (M=%.6f)' % \
                (central_pc2[stable], M[stable]))

        else:
            sys.stdout.write('\nWARNING! stopping criteria *not* reached before the minimum pressure within the EoS (%.6e)' % \
                central_pc2[0])

    #---

    if extend_up: # extend the set of central_pc2 to higher values if needed
        if verbose:
            sys.stdout.write('\nextending search to higher central pressures if stopping criteria are not met')

        ind_R = macro_cols.index('R') # look these up once
        ind_M = macro_cols.index('M')

        M = [_[ind_M] for _ in macro] # used to compute stopping criteria
        R = [_[ind_R] for _ in macro]

        collapsed = final_collapse(M,R)
        if verbose and (collapsed is None): # get printing to look pretty both when we do and do not integrate more models
            sys.stdout.write('\n')

        while (collapsed is None) and (central_pc2[-1] < max_eos_pc2):
            max_pc2 = min(max_eos_pc2, central_pc2[-1] * DEFAULT_EXTEND_UP_FACTOR) # increment the maximum pc2 geometrically

            new_central_pc2, new_macro = _bisection_stellar_sequence(
                central_pc2[-1],
                max_pc2,
                integrate,
                eos,
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

            M += [_[ind_M] for _ in new_macro[1:]]
            R += [_[ind_R] for _ in new_macro[1:]]

            collapsed = final_collapse(M, R)

        if collapsed is not None:
            sys.stdout.write('\nfinal stopping criteria reached at pressurec2=%.6e (M=%.6f R=%.6f)' % \
                (central_pc2[collapsed], M[collapsed], R[collapsed]))

        else:
            sys.stdout.write('\nWARNING! final stopping criteria *not* reached before the maximum pressure within the EoS (%.6e)' % \
                central_pc2[-1])

    #---

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return central_pc2, macro

#------------------------

def _bisection_stellar_sequence(
        min_pc2,
        max_pc2,
        foo,
        eos,
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
            sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%min_pc2)
            sys.stdout.flush()

        min_pc2_macro = foo(min_pc2, eos, rtol=integration_rtol, **kwargs)
    min_pc2_macro = np.array(min_pc2_macro)

    if max_pc2_macro is None:
        if verbose:
            sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%max_pc2)
            sys.stdout.flush()

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
        sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%mid_pc2)
        sys.stdout.flush()

    if R_ind is not None:
        ### here we can be less stringent with max_dr since we're interpolating between models and have a better idea of the behavior
        kwargs['max_dr'] = 0.1*min(min_pc2_macro[R_ind], max_pc2_macro[R_ind]) * 1e5 ### convert from km -> cm

    mid_pc2_macro = np.array(foo(mid_pc2, eos, rtol=integration_rtol, **kwargs))

    ### condition on whether we are accurate enough to determine recursive termination condition
    # compute errors based on a linear interpolation
    errors = mid_pc2_macro - (min_pc2_macro + (max_pc2_macro - min_pc2_macro) * (mid_pc2 - min_pc2) / (max_pc2 - min_pc2))

    if np.all(np.abs(errors) <= interpolator_rtol*mid_pc2_macro): ### interpolation is "good enough"
        return [min_pc2, mid_pc2, max_pc2], [min_pc2_macro, mid_pc2_macro, max_pc2_macro]

    else: # interpolation is not good enough, so we recurse to compute mid-points of sub-intervals
        left_pc2, left_macro = _bisection_stellar_sequence(
            min_pc2,
            mid_pc2,
            foo,
            eos,
            min_pc2_macro=min_pc2_macro,
            max_pc2_macro=mid_pc2_macro,
            interpolator_rtol=interpolator_rtol,
            min_dpressurec2_rtol=min_dpressurec2_rtol,
            R_ind=R_ind,
            integration_rtol=integration_rtol,
            verbose=verbose,
            **kwargs
        )

        right_pc2, right_macro = _bisection_stellar_sequence(
            mid_pc2,
            max_pc2,
            foo,
            eos,
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

#------------------------

def regular_stellar_grid(
        min_central_pressurec2,
        max_central_pressurec2,
        foo,
        eos,
        min_eos_pc2,
        max_eos_pc2,
        macro_cols,
        R_ind,
        gridding=DEFAULT_REGULAR_GRIDDING,
        central_pressurec2=[],
        min_num_models=DEFAULT_MIN_NUM_MODELS,
        integration_rtol=DEFAULT_INTEGRATION_RTOL,
        extend_up=False,
        extend_down=False,
        verbose=False,
        **kwargs
    ):

    if gridding == 'linear':
        central_pc2 = np.linspace(min_central_pressurec2, max_central_pressurec2, min_num_models)
        dpc2 = central_pc2[1] - central_pc2[0]

    elif gridding == 'logarithmic':
        central_pc2 = np.logspace(np.log10(min_central_pressurec2), np.log10(max_central_pressurec2), min_num_models)
        rpc2 = central_pc2[1] / central_pc2[0]

    else:
        raise ValueError('regular gridding=%s not understood; must be one of: %s'%(gridding, ', '.join(KNOWN_REGULAR_GRIDDINGS)))

    central_pc2 = sorted(central_pressurec2 + list(central_pc2))

    #--------------------

    ### iterate and compute models for each central_pressurec2
    macro = []
    for pc2 in central_pc2:
        if verbose:
            sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%pc2)
            sys.stdout.flush()
        macro.append( np.array(foo(pc2, eos, rtol=integration_rtol, **kwargs)) )

    #---

    if extend_down: # extend the set of central_pc2 to lower values if needed
        if verbose:
            sys.stdout.write('\nextending search to lower central pressures if stopping criteria are not met')

        ind_M = macro_cols.index('M')
        M = [] # only look for the "new_macro"
               # NOTE: this could create annoying behavior if one is not careful with the initial guess
               #       for the minimum central pressurec2

        stable = initial_stability(M)
        if verbose and (stable is None): # get printing to look pretty both when we do and do not integrate more models
            sys.stdout.write('\n')

        while (stable is None) and (central_pc2[0] > min_eos_pc2):
            if gridding == 'linear':
                pc2 = central_pc2[0] - dpc2
            elif gridding == 'logarithmic':
                pc2 = central_pc2[0] / rpc2
            else:
                raise ValueError('regular gridding=%s not understood'%gridding)

            # compute new stellar sequence
            pc2 = max(min_eos_pc2, pc2) # increment the minimum value

            if verbose:
                sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%pc2)
                sys.stdout.flush()

            new_macro = np.array(foo(pc2, eos, rtol=integration_rtol, **kwargs))

            ### add the stellar models to the cumulative list
            central_pc2.insert(0, pc2)
            macro.insert(0, new_macro)

            M.insert(0, new_macro[ind_M])

            stable = initial_stability(M)

        if stable is not None:
            sys.stdout.write('\ninitial stopping criteria reached at pressurec2=%.6e (M=%.6f)' % \
                (central_pc2[stable], M[stable]))

        else:
            sys.stdout.write('\nWARNING! stopping criteria *not* reached before the minimum pressure within the EoS (%.6e)' % \
                central_pc2[0])

    #---

    if extend_up: # extend the set of central_pc2 to higher values if needed
        if verbose:
            sys.stdout.write('\nextending search to higher central pressures if stopping criteria are not met')

        ind_R = macro_cols.index('R') # look these up once
        ind_M = macro_cols.index('M')

        M = [_[ind_M] for _ in macro] # used to compute stopping criteria
        R = [_[ind_R] for _ in macro]

        collapsed = final_collapse(M,R)
        if verbose and (collapsed is None): # get printing to look pretty both when we do and do not integrate more models
            sys.stdout.write('\n')

        while (collapsed is None) and (central_pc2[-1] < max_eos_pc2):
            if gridding == 'linear':
                pc2 = central_pc2[-1] + dpc2
            elif gridding == 'logarithmic':
                pc2 = central_pc2[-1] * rpc2
            else:
                raise ValueError('regular gridding=%s not understood'%gridding)

            pc2 = min(max_eos_pc2, pc2) # increment the maximum pc2 geometrically

            if verbose:
                sys.stdout.write('\r    computing stellar model with central pressure/c2 = %.6e'%pc2)
                sys.stdout.flush()

            new_macro = np.array(foo(pc2, eos, rtol=integration_rtol, **kwargs))

            ### add the stellar models to the cumulative list
            central_pc2.append(pc2)
            macro.append(new_macro)

            M.append(new_macro[ind_M])
            R.append(new_macro[ind_R])

            collapsed = final_collapse(M, R)

        if collapsed is not None:
            sys.stdout.write('\nfinal stopping criteria reached at pressurec2=%.6e (M=%.6f R=%.6f)' % \
                (central_pc2[collapsed], M[collapsed], R[collapsed]))

        else:
            sys.stdout.write('\nWARNING! final stopping criteria *not* reached before the maximum pressure within the EoS (%.6e)' % \
                central_pc2[-1])

    #---

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return central_pc2, macro
