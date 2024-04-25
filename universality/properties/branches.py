"""a module housing logic to identify separate stable branches from sequences of solutions to the TOV equations
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import os

import numpy as np

from universality.utils import io

#-------------------------------------------------

DEFAULT_EOS_COLUMNS = ['baryon_density', 'pressurec2', 'energy_densityc2']
DEFAULT_MACRO_COLUMNS = ['M', 'R', 'I', 'Lambda']

#-------------------------------------------------

def initial_stability(M):
    """assumes mass is ordered by increasing central pressure and looks for an initial "regaining" of stability
    """
    # look for an initial local minimum in M
    N = len(M)
    for ind in range(len(M)-2):
        if (M[ind] > M[ind+1]) and (M[ind+1] < M[ind+2]): # a local minimum in M, implying we have regained stability from WD branch
            return ind+1

    return None # criterion has not been found yet

#------------------------

def final_collapse(M, R):
    """stopping criteria for stellar sequence search. This looks for a local minimum of M for which dR/dpc > 0
    If such a point is found, we return True. Otherwise, we return False
    assumes M, R are ordered in terms of increasing central pressure
    """
    N = len(M)
    N1 = N-1
    for ind in range(N-2): # NOTE: we search from the end to the start of the lists
        if (M[N1-ind] > M[N1-(ind+1)]) and (M[N1-(ind+1)] < M[N1-(ind+2)]): # local minimum at M[ind+1]
            if (R[N1-(ind)] > R[N1-(ind+1)]) and (R[N1-(ind+1)] > R[N1-(ind+2)]): # consistently the case that dR/dpc > 0
                return N1-(ind+1) # this corresponds to the n=1 radial mode becoming unstable

    return None # this criterion has not been found yet

#------------------------

def Mrhoc2branches(M, rhoc, R=None):
    """take the M-rhoc curve and separate it into separate stable branches.
    Note! assumes models are ordered by increasing rhoc
    returns a list of boolean arrays denoting where each branch starts and ends
    """
    assert np.all(np.diff(rhoc)>=0), 'rhoc must be ordered from smallest to largest! (drhoc >= 0)'

    N = len(M)
    N1 = N - 1

    branches = []

    ### assume stellar models are ordered by increasing rhoc
    ### we just check the differences between M
    start = 0
    end = 0
    while end < N1:
        if M[end+1] > M[end]:
            end += 1
        else:
            if start!=end:
                branches.append(_bounds2bool(start, end, N))
            end += 1
            start = end

    if start!=end:
        branches.append(_bounds2bool(start, end, N))

    if R is not None:
        branches = [branch for branch in branches if (R[branch][1] < R[branch][0])]

    return branches

def _bounds2bool(start, end, N):
    ans = np.zeros(N, dtype=bool)
    ans[start:end+1] = True ### NOTE, this is inclusive
    return ans

#def Mrhoc2branches(M, rhoc):
#    """take the M-rhoc curve and separate it into separate stable branches.
#    Note! assumes models are ordered by increasing rhoc
#    returns a list of boolean arrays denoting where each branch starts and ends
#    """
#    # iterate over all data points, determining stability by numeric derivatives of dM/drhoc
#    stable = False
#    for i, dM_drhoc in enumerate(np.gradient(M, rhoc)):
#        if dM_drhoc > 0: ### stable branch
#            if stable:
#                branch.append(i)
#            else:
#                branch = [i]
#                stable = True
#        elif stable: ### was on a stable branch, and it just ended
#            stable = False
#            branches.append(_branch2bool(branch, N)) ### append
#
#    if stable:
#        branches.append(_branch2bool(branch, N)) ### append to pick up what was left when we existed the loop
#
#    return branches
#
#def _branch2bool(branch, N):
#    ans = np.zeros(N, dtype=bool)
#    ans[branch] = True
#    return ans

#------------------------

START_TEMPLATE = 'start_%s'
END_TEMPLATE = 'end_%s'

def data2branch_properties(rhoc, M, baryon_density, mac_data, mac_cols, eos_data, eos_cols, branch_template=None, verbose=False):

    inds = np.arange(len(rhoc)) ### used to look up indecies from boolean arrays later

    ### split maro data into stable branches
    branches = Mrhoc2branches(M, rhoc)
    if verbose:
        print('        identified %d branches'%len(branches))

    # iterate over stable branches to extract micro- and macroscopic parameters of these stellar configurations
    summary = [] # summary statistics for central values of EOS parameters at the start, end of each branch
    names = ['branch']+[START_TEMPLATE%col for col in eos_cols+mac_cols]+[END_TEMPLATE%col for col in eos_cols+mac_cols]

    Neos_column = len(eos_cols)
    Nmac_column = len(mac_cols)

    if branch_template is not None:
        subdir = os.path.dirname(branch_template)
        if not os.path.exists(subdir):
            try:
                os.makedirs(subdir)
            except OSError:
                pass # cateches a race condition where this directory already exists

    for ind, truth in enumerate(branches):

        if branch_template is not None: ### write out macros for individual branches
            # define the path for this branch
            branch_path = branch_template%{'branch':ind}
            if verbose:
                print('        writing branch %d into: %s'%(ind, branch_path))
            io.write(branch_path, mac_data[truth], mac_cols)

        # identify eos values at start, end of the branch
        branch = [ind] ### holder for central values, starting with the branch number

        for i in [inds[truth][0], inds[truth][-1]]: # add values for start of the branch and then the end of the branch
            rho = rhoc[i]
            branch += [np.interp(rho, baryon_density, eos_data[:,j]) for j in range(Neos_column)] # add the rest of the EoS columns
            branch += [mac_data[i,j] for j in range(Nmac_column)] # add the macro values

        # add to summary of all branches
        summary.append(branch)

    return np.array(summary), names

def process2branch_properties(
        data,
        eos_template,
        eos_num_per_dir,
        mac_template,
        macro_num_per_dir,
        summary_template,
        eos_rho,
        macro_rhoc,
        macro_mass,
        output_eos_columns=DEFAULT_EOS_COLUMNS,
        output_macro_columns=DEFAULT_MACRO_COLUMNS,
        branch_template=None,
        verbose=False,
    ):
    """extract the branches for each EoS specified
    """
    tmp = dict()
    if branch_template is not None: # do this backflip to make sure we can build string correctly
        branch_template = branch_template.replace('%(branch)06d', '%(branch_string)s')
        tmp['branch_string'] = '%(branch)06d'

    N = len(data)
    for ind, eos in enumerate(data):

        ### construct paths
        # where we're going to read in data
        eos_path = eos_template%{'moddraw':eos//eos_num_per_dir, 'draw':eos}

        tmp.update({'moddraw':eos//macro_num_per_dir, 'draw':eos})
        mac_path = mac_template%tmp

        # where we're going to write data
        sum_path = summary_template%tmp

        if verbose:
            print('    %d/%d'%(ind+1, N))
            print('    loading macro: %s'%mac_path)
        mac_data, mac_cols = io.load(mac_path, [macro_rhoc, macro_mass]+output_macro_columns) ### NOTE: we load all columns because we're going to re-write them all into subdir as separate branches

        if verbose:
            print('    loading eos: %s'%eos_path)
        eos_data, eos_cols = io.load(eos_path, [eos_rho]+output_eos_columns) ### NOTE: this guarantees that eos_rho is the first column!
        baryon_density = eos_data[:,0] ### separate this for convenience

        # use macro data to identify separate stable branches
        # NOTE: we expect this to be ordered montonically in rhoc
        M = mac_data[:,mac_cols.index(macro_mass)]
        rhoc = mac_data[:,mac_cols.index(macro_rhoc)]

        if branch_template is not None:
            branch_tmp = branch_template%tmp
        else:
            branch_tmp = None 

        params, names = data2branch_properties(
            rhoc,
            M,
            baryon_density,
            mac_data,
            mac_cols,
            eos_data,
            eos_cols,
            branch_template=branch_tmp,
            verbose=verbose,
        )
        if verbose:
            print('    writing summary into: %s'%sum_path)

        newdir = os.path.dirname(sum_path)
        if not os.path.exists(newdir):
            try:
                os.makedirs(newdir)
            except OSError:
                pass ### directory already exists

        io.write(sum_path, params, names)
