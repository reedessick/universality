"""a module housing logic to identify (highly engineered) features from EoS
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from argparse import ArgumentParser

### non-standard libraries
from universality import utils
from universality import gaussianprocess as gp

#-------------------------------------------------

DEFAULT_EOS_COLUMNS = ['baryon_density', 'pressurec2', 'energy_densityc2', 'cs2c2']
DEFAULT_MACRO_COLUMNS = ['M', 'R', 'I', 'Lambda']

#-------------------------------------------------
# basic utilities for finding features before/after a particular point
#-------------------------------------------------

def find_preceeding(x_ref, x, y):
    '''assumes x is ordered from decreasing to increasing'''
    found = x <= x_ref
    if np.sum(found):
        return y[found][-1]
    else:
        raise RuntimeError('no preceeding value found!')

def find_following(x_ref, x, y):
    '''assumes x is ordered from decreasing to increasing'''
    return find_preceeding(-x_ref, -x[::-1], y[::-1])

#------------------------

def find_running_maxima(x):
    """find the indecies of the largest thing that's been seen up until the current index"""
    return find_running_minima(-np.array(x))

def find_running_minima(x):
    """find the indecies of the smallest thing that's been seen up until the current index"""
    mins = []
    best = +np.infty
    for i, X in enumerate(x):
        if X < best:
            mins.append(i)
            best = X
    return mins

#------------------------

def find_maxima(x):
    """find indecies of strict local maxima"""
    return find_minima(-np.array(x))

def find_minima(x):
    """find indecies of strict local minima"""
    mins = []
    old = +np.infty
    for i in range(len(x)-1):
        new = x[i]
        if (old > new) and (new < x[i+1]):
            mins.append(i)
        old = new
    if x[i+1] > x[i]:
        mins.append(i+1)
    return mins

#------------------------

def find_inclusive_maxima(x):
    """find indecies of local maxima, including cases when the value does not change from one sample to the next"""
    return find_inclusive_minima(-np.array(x))

def find_inclusive_minima(x):
    """find indecies of local minima, including cases when the value does not change from one sample to the next"""
    mins = []
    old = +np.infty
    for i in range(len(x)-1):
        new = x[i]
        if (old >= new) and (new <= x[i+1]):
            mins.append(i)
        old = new
    if x[i+1] > x[i]:
        mins.append(i+1)
    return mins

#-------------------------------------------------
# Features extracted from moment of inertia relations
#-------------------------------------------------

# templates for column names relevant to MoI features
MAX_CS2C2_TEMPLATE = 'max_cs2c2_%s'
#MAX_K_TEMPLATE = 'max_k_%s'
MIN_CS2C2_TEMPLATE = 'min_cs2c2_%s'
MIN_ARCTAN_DLNI_DLNM_TEMPLATE = 'min_arctan_dlnI_dlnM_%s'

DEFAULT_FLATTEN_THR = 0.0
DEFAULT_SMOOTHING_WIDTH = None
DEFAULT_DIFF_THR = 0.0
DEFAULT_CS2C2_COFACTOR = 3.0

#------------------------

### utility functions for identifying MoI features (including basic feature transformation)

def arctan_transform(rhoc, M, I, flatten_thr=DEFAULT_FLATTEN_THR, smoothing_width=DEFAULT_SMOOTHING_WIDTH):
    """compute the special arctan function we use to feed into our feature identification function
    """
    dlnI_drhoc = gp.num_dfdx(rhoc, I) / I
    dlnM_drhoc = gp.num_dfdx(rhoc, M) / M

    if smoothing_width:
        dlnI_drhoc = smooth(np.log(rhoc), dlnI_drhoc, smoothing_width)
        dlnM_drhoc = smooth(np.log(rhoc), dlnM_drhoc, smoothing_width)

    ### regularlize the derivatives so we don't get oscillatory behavior due to numerical precision when there is no change with respect to rhoc
    ### only filter placed where we've touched both dM and dI to account for numerical error
    spurious = (np.abs(dlnI_drhoc*rhoc)<flatten_thr) *(np.abs(dlnM_drhoc*rhoc)<flatten_thr)

    dlnI_drhoc[spurious] = 0
    dlnM_drhoc[spurious] = 0

    ans = np.arctan2(dlnI_drhoc, dlnM_drhoc)
    ans[ans>np.pi/2] -= 2*np.pi ### wrap this around so that all the region where dlnM_drhoc < 0 is "contiguous"

    return ans, (spurious, dlnM_drhoc, dlnI_drhoc)

def smooth(x, y, width):
    v = (width)**2
    ans = np.empty_like(y)
    inds = np.arange(len(x))
    for i, S in enumerate(x):
        weights = np.exp(-0.5*(S-x)**2/v) ### the basic Gaussian kernel
        weights /= np.sum(weights)
        ans[i] = np.sum(weights*y)
    return ans

#------------------------

### functions to extract parameters

def data2moi_features(
        rhoc,
        M,
        I,
        baryon_density,
        cs2c2,
        macro_data,
        macro_cols,
        eos_data,
        eos_cols,
        flatten_thr=DEFAULT_FLATTEN_THR,
        smoothing_width=DEFAULT_SMOOTHING_WIDTH,
        diff_thr=DEFAULT_DIFF_THR,
        cs2c2_cofactor=DEFAULT_CS2C2_COFACTOR,
        verbose=False,
    ):
    """
    looks for specific behavior in arctan(dlnI/dlnM) that seems to correspond to phase transitions.
    returns the corresponding values from data along with names based on cols
    """
    names = ['transition']
    for tmp in [
            MIN_CS2C2_TEMPLATE,
            MAX_CS2C2_TEMPLATE,
            MIN_ARCTAN_DLNI_DLNM_TEMPLATE,
        ]:
        names += [tmp%col for col in macro_cols]
        names += [tmp%col for col in eos_cols]
    params = []

    ### compute the absolute value of the curvature, which we use as an indicator variable
    arctan_dlnI_dlnM, (spurious, dlnM_drhoc, dlnI_drhoc) = arctan_transform(rhoc, M, I, flatten_thr=flatten_thr, smoothing_width=smoothing_width)

    ### find the possible end points as local minima of arctan_dlnI_dlnM
    ends = list(find_inclusive_minima(arctan_dlnI_dlnM)[::-1]) ### reverse so the ones with largest rhoc are first

    ### discard any local minima that are before the first stable branch
    while len(ends):
        end = ends[-1]
        if np.any(dlnM_drhoc[:end] > 0): ### something is stable before this
             break
        ends = ends[:-1] ### truncate this guy

    ### discard any local minima that are in the final unstable branch
    while len(ends):
        end = ends.pop(0)
        if np.any(dlnM_drhoc[end:] > 0): ### not part of the 'final unstable branch'
            ends.insert(0, end)
            break

    if ends: ### we have something to do

        ### if the lowest rhoc available is a minimum, we discard it since we can't tell if it is a local minimum or not
        if ends[-1] == 0:
            ends = ends[:-1]

        if ends[0] == len(rhoc)-1: ### same thing with the end point
            ends = ends[1:]

        ### local minima in sound speed
        min_cs2c2 = find_inclusive_minima(cs2c2)
        min_cs2c2 = np.array(min_cs2c2)
        min_cs2c2_baryon_density = baryon_density[min_cs2c2]

        ### global maxima in sound speed up to the current point
        max_cs2c2 = find_running_maxima(cs2c2)

        if not max_cs2c2: ### no qualifying maxima
            ends = [] ### this will make us skip all the ends because we can't match them to starting points
        else:
            max_cs2c2 = np.array(max_cs2c2)
            max_cs2c2_baryon_density = baryon_density[max_cs2c2]

        # iterate through and grab the following associated with each "end"
        Neos = len(eos_cols)
        Nmac = len(macro_cols)
        last_r = +np.infty ### logic to avoid overlapping phase transitions
        group = []
        for end in ends:
            r = rhoc[end]
            datum = []

            if verbose:
                print('    processing end=%d/%d at rhoc=%.6e'%(end, len(rhoc), r))

            ### min sound speed preceeding "end"
            try:
                ind = find_preceeding(r, min_cs2c2_baryon_density, min_cs2c2)
            except RuntimeError:
                if verbose:
                    print('    WARNING! coult not find preceeding minimum in cs2c2')
                continue

            min_r = baryon_density[ind]
            min_cs2c2_arctan = np.interp(min_r, rhoc, arctan_dlnI_dlnM)

            datum += [np.interp(baryon_density[ind], rhoc, macro_data[:,i]) for i in range(Nmac)]
            datum += list(eos_data[ind])

            ### max sound speed preceeding "end"
            try:
                ind = find_preceeding(min_r, max_cs2c2_baryon_density, max_cs2c2) ### look for local max before the local min
            except RuntimeError:
                if verbose:
                    print('    WARNING! could not find preceeding maximum in cs2c2')
                continue

            max_r = baryon_density[ind] ### expect max(cs2c2) to be the smallest density (required if we're to keep this possible transition)
            max_cs2c2_arctan = np.interp(max_r, rhoc, arctan_dlnI_dlnM)

            if (dlnM_drhoc[end] > 0) and (np.max(arctan_dlnI_dlnM[(max_r<=rhoc)*(rhoc<=r)]) - arctan_dlnI_dlnM[end] < diff_thr): ### does not pass our basic selection cut for being "big enough". Note that we add an exception if we're on an unstable branch (that's gotta be a strong phase transition...)
                if verbose:
                    print('    WARNING! difference in arctan_dlnI_dlnM is smaller than diff_thr; skipping this possible transition')
                continue

            if max(max_cs2c2_arctan, min_cs2c2_arctan) < arctan_dlnI_dlnM[end]: ### both are smaller, so we're kinda on an "upward sweep" that typically doesn't correspond to the behavior we want
                if verbose:
                    print('    WARNING! arctan(dlnI/dlnM) at max_cs2c2 and min_cs2c2 is less than at the local minimum; skipping this possible transition')
                continue

            if np.interp(r, baryon_density, cs2c2) > cs2c2_cofactor*cs2c2[ind]: ### recovery occurs at a larger sound speed than the onset...
                if verbose:
                    print('    WARNING! sound-speed at local minimum is larger than onset sound speed; skipping this possible transition')
                continue

            datum += [np.interp(max_r, rhoc, macro_data[:,i]) for i in range(Nmac)]
            datum += list(eos_data[ind])

            ### parameters at "end"
            ind = end
            datum += list(macro_data[end])
            datum += [np.interp(r, baryon_density, eos_data[:,i]) for i in range(Neos)]

            ### figure out if there is any overlap, keep the one with the minimum dlnI_drhoc[end]
            if len(group) and (r < last_r): ### we already have a group and this would *not* overlap with something we've already declared a phase transition, so figure out which is best and add it
                group.sort(key=lambda x:x[0]) ### sort so the smallest dlnI_drhoc is first
                params.append(group[0][1]) ### append the datum
                group = [] ### start a new group

            group.append((rhoc[end], datum)) ### FIXME: this whole "group" logic may be wasteful of memory if we know we're going to order by rhoc and we already iterate based on samples ordered by rhoc...

            last_r = max_r ### update this so we remember the extent of the current phase transition (avoid overlaps with the next one)

        if group: ### add the last identified transition
            group.sort(key=lambda x: -x[0]) ### bigger rhoc first
            params.append(group[0][1])

    params = [[ind]+thing for ind, thing in enumerate(params)] ### include a transition number for reference

    if not len(params):
        params = np.empty((0,len(names)), dtype=float)

    else:
        params = np.array(params, dtype=float)

    return params, names

#------------------------

def process2moi_features(
        data,
        eos_template,
        eos_num_per_dir,
        mac_template,
        macro_num_per_dir,
        summary_template,
        eos_rho,
        eos_cs2c2,
        macro_rhoc,
        macro_mass,
        macro_moi,
        output_eos_columns=DEFAULT_EOS_COLUMNS,
        output_macro_columns=DEFAULT_MACRO_COLUMNS,
        flatten_thr=DEFAULT_FLATTEN_THR,
        smoothing_width=DEFAULT_SMOOTHING_WIDTH,
        diff_thr=DEFAULT_DIFF_THR,
        cs2c2_cofactor=DEFAULT_CS2C2_COFACTOR,
        verbose=False,
    ):
    """extract the branches for each EoS specified
    """
    for eos in data:

        ### construct paths
        # where we're going to read in data
        eos_path = eos_template%{'moddraw':eos//eos_num_per_dir, 'draw':eos}

        tmp = {'moddraw':eos//macro_num_per_dir, 'draw':eos}
        mac_path = mac_template%tmp

        # where we're going to write data
        sum_path = summary_template%tmp

        if verbose:
            print('    loading macro: %s'%mac_path)
        mac_data, mac_cols = io.load(mac_path, [macro_rhoc, macro_mass, macro_moi]+output_macro_columns) ### NOTE: we load all columns because we're going to re-write them all into subdir as separate branches

        if args.verbose:
            print('    loading eos: %s'%eos_path)
        eos_data, eos_cols = io.load(eos_path, [eos_rho, eos_cs2c2]+output_eos_columns) ### NOTE: this guarantees that eos_rho is the first column!
        baryon_density = eos_data[:,eos_cols.index(eos_rho)] ### separate this for convenience
        cs2c2 = eos_data[:eos_cols.index(eos_cs2c2)]

        # use macro data to identify separate stable branches
        # NOTE: we expect this to be ordered montonically in rhoc
        M = mac_data[:,mac_cols.index(macro_mass)]
        I = mac_data[:,mac_cols.index(macro_moi)]
        rhoc = mac_data[:,mac_cols.index(macro_rhoc)]

        params, names = data2moi_features(
            rhoc,
            M,
            I,
            baryon_density,
            cs2c2,
            macro_data,
            macro_cols,
            eos_data,
            eos_cols,
            flatten_thr=flatten_thr,
            smoothing_width=smoothing_width,
            diff_thr=diff_thr,
            cs2c2_cofactor=cs2c2_cofactor,
            verbose=verbose,
        )

        if verbose:
            print('    writing summary into: %s'%sum_path)
        np.savetxt(sum_path, summary, comments='', header=sum_header, delimiter=',')
