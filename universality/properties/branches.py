__doc__ = "a module housing logic to identify separate stable branches from sequences of solutions to the TOV equations"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

def data2branches(M, rhoc):
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

def process2branches(*args, **kwargs):
    raise NotImplementedError
