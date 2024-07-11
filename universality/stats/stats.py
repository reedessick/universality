"""a module that houses utilities that compute statistics based on KDEs
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------
# basic statistical quantities about convergence of monte-carlo integrals
#-------------------------------------------------

def nkde(weights):
    """the number of samples that determine the scaling of the variance of our KDE estimates"""
    weights /= np.sum(weights)
    return 1./np.sum(weights**2)

def neff(weights):
    """the effective number of samples based on a set of weights"""
    return np.exp(entropy(weights, base=np.exp(1)))

def entropy(weights, base=2.):
    """compute the entropy of the distribution"""
    weights = np.array(weights, dtype=float)
    truth = weights > 0
    weights /= np.sum(weights)
    return -np.sum(weights[truth]*np.log(weights[truth])) / np.log(base)

def information(weights, base=2.):
    """compute the information in the distribution"""
    return np.log(len(weights))/np.log(base) - entropy(weights, base=base)

#-------------------------------------------------
# basic statistical quantities to be derived from samples
#-------------------------------------------------

def quantile(x, quantiles, weights=None):
    if weights is None:
        return np.percentile(x, np.array(quantiles)*100)

    else:
        order = x.argsort()
        x = x[order]
        csum = np.cumsum(weights[order])
        csum /= csum[-1]

        return np.interp(quantiles, csum, x)
#        return np.array([x[[csum<=q]][-1] for q in quantiles])

def samples2cdf(data, weights=None):
    """estimate a CDF (integrating from small values to large values in data) based on weighted samples
    returns data, cweights (data is sorted from smallest to largest values)
    """
    if weights is None:
        N = len(data)
        weights = np.ones(N, dtype=float)/N

    order = data.argsort()
    data = data[order]
    weights = weights[order]
    cweights = np.cumsum(weights)/np.sum(weights)

    return data, cweights

def samples2range(data, pad=0.1):
    m = np.min(data)
    M = np.max(data)
    delta = (M-m)*pad
    return (m-delta, M+delta)

def samples2median(data, weights=None):
    data, cweights = samples2cdf(data, weights=weights)
    return np.interp(0.5, cweights, data) ### find the median via interpolation

def samples2mean(data, weights=None):
    if weights is None:
        weights = np.ones_like(data, dtype=float)
    return np.sum(weights*data)/np.sum(weights)

def samples2crbounds(data, levels, weights=None):
    """
    expects 1D data and returns the smallest contiguous confidence region that contains a certain amount of the cumulative weight
    returns a contiguous confidence region for level
    does this by trying all possible regions (defined by data's sampling) that have at least as much cumulative weight as each level and selecting the smallest
    """
    N = len(data)
    stop = N-1
    if weights is None:
        weights = np.ones(N, dtype=float)/N

    data, cweights = samples2cdf(data, weights=weights)

    return cdf2crbounds(data, cweights, levels)

def cdf2crbounds(data, cweights, levels):
    N = len(data)
    stop = N-1
    bounds = []
    for level in levels:
        i = 0
        j = 0
        best = None
        best_size = np.infty
        while (i < stop):
            while (j < stop) and (cweights[j]-cweights[i] < level): ### make the region big enough to get the confidence we want
                j += 1

            if cweights[j] - cweights[i] < level: ### too small!
                break

            size = data[j]-data[i]
            if size < best_size:
                best_size = size
                best = (data[i], data[j])
            i += 1 # increment starting point and then repeat

        bounds.append( best )

    return bounds

#-------------------------------------------------
# statistical interpretations of kdes
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
        for i in range(len(lookups)):
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
