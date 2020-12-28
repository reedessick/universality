"""a module housing logic to identify separate stable branches from sequences of solutions to the TOV equations
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from universality.utils import io

#-------------------------------------------------

DEFAULT_EOS_COLUMNS = ['baryon_density', 'pressurec2', 'energy_densityc2']
DEFAULT_MACRO_COLUMNS = ['M', 'R', 'I', 'Lambda']

#-------------------------------------------------

def Mrhoc2branches(M, rhoc):
    """take the M-rhoc curve and separate it into separate stable branches.
    Note! assumes models are ordered by increasing rhoc
    returns a list of boolean arrays denoting where each branch starts and ends
    """
    assert np.all(np.diff(rhoc)>0), 'rhoc must be ordered from smallest to largest!'

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
    mac_header = ','.join(mac_cols) # header for the macro files representing each branch separately

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
            np.savetxt(branch_path, mac_data[truth], comments='', header=mac_header, delimiter=',')

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
        suppress_individual_branches=True,
        verbose=False,
    ):
    """extract the branches for each EoS specified
    """
    N = len(data)
    for ind, eos in enumerate(data):

        ### construct paths
        # where we're going to read in data
        eos_path = eos_template%{'moddraw':eos//eos_num_per_dir, 'draw':eos}

        tmp = {'moddraw':eos//macro_num_per_dir, 'draw':eos}
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

        if not suppress_individual_branches:
            branch_template = os.path(sum_path[:-4], os.path.basename(mac_path)[:-4]+'-%(branch)06d.csv')
        else:
            branch_template = None

        params, names = data2branch_properties(
            rhoc,
            M,
            baryon_density,
            mac_data,
            mac_cols,
            eos_data,
            eos_cols,
            branch_template=branch_template,
            verbose=verbose,
        )
        if verbose:
            print('    writing summary into: %s'%sum_path)
        np.savetxt(sum_path, params, comments='', header=','.join(names), delimiter=',')
