"""a module that houses logic to perform rates and populations inferences based on weighted samples contained. \
We approximate integrals over individual events' observables via weighted monte-carlo sums and dynamically adjust the weights to reflect the prior odds associated with the population parameters
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import utils

#-------------------------------------------------

class Exposure(object):
    """an object representing the exposure of an observation set. Should be able to return everything needed to compute VT et al given a population
    """

    def __init__(self, inpath, **kwargs):

        ### store data
        self.inpath = inpath
        raise NotImplementedError('need to figure out how we want to represent this')

    @staticmethod
    def _vt(params, population):
        raise NotImplementedError

    def vt(self, population):
        """compute the VT given the population
        """
        return self._vt(population.params, population)

    @staticmethod
    def _count(params, population):
        raise NotImplementedError

    def count(self, population):
        """compute the expected number of events given the population
        """
        return self._count(population.params, params)

#-------------------------------------------------

class Population(object):
    """an object representing a population. We can use this to evaluate the prior suitable for each event as well as the exposure. \
This object should be extended to implement different population models (eg, power law vs Gaussian in component masses).
    """

    def __init__(self, columns, *params):
        self.columns = columns
        self._params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new):
        assert len(new)==len(self._params), 'new parameters must have the same dimension as the current parameters'
        self._params = new

    def vt(self, exposure):
        """compute the VT given the exposure
        """
        return exposure._vt(self.params, self)

    def count(self, exposure):
        """compute the expected number of events given the exposure
        """
        return exposure._count(self.params, self)

    @staticmethod
    def _logprior(params, data):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('need to evaluate the logprior associated with this set of params')

    def logprior(self, data):
        """compute the appropriate log(prior) for single-event parameters given the population
        """
        return self._logprior(self.params, data)

    @staticmethod
    def _loghyperprior(params):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('return the hyperprior on population parameters evaluated at the current parameters')

    def loghyperprior(self):
        """return the log(hyperprior) for the population parameters
        """
        return self._loghyperprior(self.params)

#-------------------------------------------------

class Observation(object):
    """an object representing the observations for a single system
    """

    def __init__(self, inpath, columns, weight_columns, logcolumns=[], max_num_samples=np.infty, logweightcolumns=[], label=None):

        ### load and store data
        self.label = label if label is not None else inpath

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

    @staticmethod
    def _marginalize(params, population, data, log_weights)
        logprior = population._logprior(params, data)
        return utils.sum_log(log_weights + logprior) ### return the log of the monte-carlo marginalization

    def marginalize(self, population):
        """compute the marginal integral with respect to prior given the weighted samples
        """
        return self._marginalize(population.params, population, self.data, self.log_weights)

class ObservationSet(object):
    """an object representing an ensemble of observations, which provides access to inference over the entire population.
    """

    def __init__(self, exposure, *observations):
        self.exposure = exposure
        self.observation = observations

    @staticmethod
    def _marginalize(params, population, observations):
        return np.sum(obs._marginalize(params, population, obs.data, obs.log_weights) for obs in observations)

    def marginalize(self, population):
        """returns the product of all individual event marginalizations.
        """
        return self._marginalize(population.params, population, self.observations) ### returns the log of the monte-carlo marginalization

    @staticmethod
    def _loglike(params, population, exposure, observations):
        return self._marginalize(params, population, observations) + population._count(params, exposure)

    def loglike(self, population):
        """return loglike of observation set given the population
        """
        return self._loglike(population.params, population, self.exposure, self.observations)

#-------------------------------------------------

def logpost(params, population, observationset):
    return observationset._loglike(params, population, observationset.exposure, observationset.observations) + population._loghyperprior(params)
