"""a module that houses utilities that compute statistics based on pre-computed KDEs
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

### bring in a few functions defined elsewhere as they depend on pre-computed KDEs
from universality.stats.information import (logkde2entropy, logkde2information, kldiv, sym_kldiv)

#-------------------------------------------------

def vects2vol(vects):
    """
    find approximate volume element
    """
    return np.prod([vect[1]-vect[0] for vect in vects])

def logkde2levels(logkde, levels):
    logkde = logkde.flatten()
    kde = np.exp(logkde-np.max(logkde))

    order = kde.argsort()[::-1] ### largest to smallest
    ckde = np.cumsum(kde[order]) ### cumulative distribution
    ckde /= np.sum(kde)

    ans = []
    for level in levels: ### iterate through levels, returning the kde value associated with that confidence
                         ### assume kde spacing is close enough that interpolation isn't worth while...
        ind = order[ckde<=level]
        if len(ind):
            ans.append(logkde[ind[-1]])
        else: ### nothing is smaller than the first level, so we just add the first element
              ### this issue should go away if we increase the number of samples in the kde...
            ans.append(logkde[order[0]])

    return ans

def logkde2median(vect, logkde):
    assert vect.ndim==1 and logkde.ndim==1, 'logkde2crbounds only works with 1-dimensional vectors!'
    assert np.all(np.diff(vect) > 0), 'vect must be strictly increasing!'
    return np.interp(0.5, logkde2cdf(logkde), vect) ### find the median via interpolation

def logkde2mean(vect, logkde):
    assert vect.ndim==1 and logkde.ndim==1, 'logkde2crbounds only works with 1-dimensional vectors!'
    assert np.all(np.diff(vect) > 0), 'vect must be strictly increasing!'
    weights = np.exp(logkde-np.max(logkde))
    weights /= np.sum(weights)
    return samples2mean(vect, weights=weights)

def logkde2cdf(logkde):
    cweights = np.cumsum(np.exp(logkde-np.max(logkde)))
    cweights /= cweights[-1]
    return cweights

def logkde2crbounds(vect, logkde, levels):
    """
    only works with 1D kde
    returns the bounds on the smallest contiguous region that contain the specified confidence
    """
    assert vect.ndim==1 and logkde.ndim==1, 'logkde2crbounds only works with 1-dimensional vectors!'
    assert np.all(np.diff(vect) > 0), 'vect must be strictly increasing!'
    return cdf2crbounds(vect, logkde2cdf(logkde), levels)

def logkde2cr(vect, logkde, levels):
    """
    only works with 1D kde
    returns the smallest credible region associated with each level. These may not be contiguous, so a list (with an even number of elements) is returned for each level
    """
    assert vect.ndim==1 and logkde.ndim==1, 'logkde2cr only works with 1-dimensional vectors!'
    dvect = vect[1]-vect[0] ### assume equal spacing

    bounds = []
    inds = np.arange(len(vect))
    order = logkde.argsort()[::-1] ### from biggets to smallest
    cordered = np.cumsum(logkde[order])
    cordered /= cordered[-1] ### normalize the sum

    for level in levels:
        these = vects[order[cordered <= level]] ### extract the vector elements included up to this cumulative level
        these.sort() ### put these in order
        bound_list = []

        if len(these):
            start = these[0]-0.5*dvect
            end = these[0]
            for e in these[1:]:
                if e == end+dvect: ### same segment
                    end = e
                else: # new segment
                    bound_list.append( (start, end+0.5*dvect) )
                    start = e-0.5*dvect
                    end = e
            bound_list.append( (start, end+0.5*dvect) )

        bounds.append(bound_list)

    return bounds

def logkde2crsize(vects, logkde, levels):
    """
    compute the volumes of confidence regions associated with levels
    """
    ### find approximate volume element
    vol = vects2vol(vects)

    ### finde confidence levels and associated sizes
    return [vol*np.sum(logkde>=thr) for thr in logkde2levels(logkde, levels)]

'''
def dlogkde(point, vects, logkde):
    """
    returns the difference in loglikelihood between a point and the argmax of kde
    """
    m = np.max(logkde)
    return _interpn(point, vects, logkde) - np.max(logkde)

def _interpn(point, vects, logkde):
    """
    use a weighted average of nearest neighbors within the grid to perform interpolation
    weights for each neighbor are determined by their relative distance
        wi \propto 1./|x-xi|
    this is equivalent to linear interpolation in the 1D case and we do this using the nearest neighbors only

    NOTE:
        I do not guarantee that this interpolation proceedure will be immune to weird corner cases!
        It should be well behaved in 1D and at least reasonable in higher dimensions as long as the grid size 
        is much smaller than the typical length scale in the underlying function (we resolve the function reasonably well)
    """
    # find indecies of all nodes to include in harmonic sum for each dimension separately
    indecies = []
    for x, vect in zip(point, vects):
        if x<=vect[0]: ### before beginning of array
            indecies.append((0,))

        elif x >=vect[-1]: ### after end of array
            indecies.append((len(vect)-1,))

        else: ### is is contained in vects
            N = len(vect)
            i = 1
            while i<N:
                if vect[i]>x:
                    indecies.append((i-1,i))
                    break
                else:
                    i += 1
            else:
                raise RuntimeError('location of index is did not complete successfully')

    # compute all look-up list from the identified indecies
    lookups = [()]
    for inds in indecies:
        for i in xrange(len(lookups)):
            lookup = lookups[i]
            for ind in inds:
                lookups.append(lookup+(ind,))

    # extract all grid points, computing weights
    N = len(lookups)
    logkdes = np.empty(N, dtype=float)
    distances = np.empty(N, dtype=float)
    for l, lookup in enumerate(lookups):
        xi = np.array([vect[i] for vect, i in zip(vects, lookup)])
        logkdes[l] = logkde[lookup]
        distances[l] = np.sum((point-xi)**2)**0.5 ### euclidean distance

    # return the weighted sum
    if np.any(distances==0):
        return logkde[distance==0] ### handle this as a special case so I don't divide by zero
    else:
        np.sum(logkde/distance)/np.sum(1./distance)
'''

def logkde2argmax(vects, logkde):
    """
    returns the argmax of the kde
    """
    grid = np.meshgrid(*vects, indexing='ij')
    arg = logkde.flatten().argmax()
    return np.array([g.flatten()[arg] for g in grid])
