__doc__ = "a module that houses utilities that compute statistics based on KDEs"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

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

def vects2vol(vects):
    """
    find approximate volume element
    """
    return np.prod([vect[1]-vect[0] for vect in vects])

def logkde2cr(vects, logkde, levels):
    """
    compute the volumes of confidence regions associated with levels
    """
    ### find approximate volume element
    vol = vects2vol(vects)

    ### finde confidence levels and associated sizes
    kde = np.exp(logkde-np.max(logkde))
    return [vol*np.sum(kde>=thr) for thr in kde2levels(kde, levels)]

def logkde2entropy(vects, logkde):
    """
    computes the entropy of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    vol = vects2vol(vects)
    truth = logkde > -np.infty
    return -vects2vol(vects)*np.sum(np.exp(logkde[truth])*logkde[truth])

def logkde2information(vects, logkde):
    """
    computes the information of the kde
    incorporates vects so that kde is properly normalized (transforms into a truly discrete distribution)
    """
    vol = vects2vol(vects)
    return np.log(len(logkde.flatten())*vol) - logkde2entropy(vects, logkde)

def kldiv(vects, logkde1, logkde2):
    """
    computes the KL divergence from kde1 to kde2
        Dkl(k1||k2) = sum(k1*log(k1/k2))
    """
    truth = logkde1 > -np.infty
    return vects2vol(vects)*np.sum(np.exp(logkde1[truth]*(logkde1[truth] - logkde2[truth])))

def sym_kldiv(vects, logkde1, logkde2):
    """
    Dkl(k1||k2) + Dkl(k2||k1)
    """
    return kldiv(vects, logkde1, logkde2) + kldiv(vects, logkde2, logkde1)

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

def argmax(vects, logkde):
    """
    returns the argmax of the kde
    """
    grid = np.meshgrid(*vects, indexing='ij')
    arg = logkde.flatten().argmax()
    return np.array([g.flatten()[arg] for g in grid])
