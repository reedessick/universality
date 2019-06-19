"""a module that houses logic to perform rates and populations inferences based on weighted samples contained. \
We approximate integrals over individual events' observables via weighted monte-carlo sums and dynamically adjust the weights to reflect the prior odds associated with the population parameters
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import utils

#-------------------------------------------------

class Population(object):
    """an object representing a population. We can use this to evaluate the prior suitable for each event as well as the exposure. \
This object should be extended to implement different population models (eg, power law vs Gaussian in component masses).
    """

    def __init__(self, columns, **params):
        self.columns = columns
        self.params = params

    def logprior(self, data):
        raise NotImplementedError('need to evaluate the logprior associated with this set of params')

    def vt(self, exposure):
        raise NotImplementedError('need to evaluate the VT given the exposure and this model')

#-------------------------------------------------

class Observation(object):
    """an object representing the observations for a single system
    """

    def __init__(self, inpath, columns, weight_columns, logcolumns=[], max_num_samples=np.infty, logweightcolumns=[]):

        ### load and store data
        self.inpath = inpath
        data, columns = utils.load(inpath, columns, logcolumns=logcolumns, max_num_samples=max_num_samples)
        self.data = np.array(data, dtype=[(col, 'float') for col in columns]) ### store a structured array

        self.log_weights = utils.load_logweights(inpath, weight_columns, logweightcolumns=logweightcolumns, max_num_samples=max_num_samples)
        self.log_weights /= utils.sum_log(self.log_weights) ### normalize these
                                                            ### OK because we only care about ratios and not the absolue value of the marginal likelihood

        ### discard any data assigned zero weight
        truth = self.log_weights > -np.infty
        self.data = self.data[truth]
        self.log_weights = self.log_weights[truth]

    def marginalize(self, population):
        """compute the marginal integral with respect to prior given the weighted samples
        """
        logprior = population.logprior(self.data) ### compute the prior weight for each sample
        return utils.sum_log(self.log_weights + logprior) ### return the log of the monte-carlo marginalization

class ObservationSet(object):
    """an object representing an ensemble of observations, which provides access to inference over the entire population.
    """

    def __init__(self, *observations):
        self.observation = observations

    def marginalize(self, population):
        """returns the product of all individual event marginalizations.
        """
        return np.sum(obs.marginalize(population) for obs in self.observations) ### returns the log of the monte-carlo marginalization

#-------------------------------------------------


