"""a module that houses logic to perform rates and populations inferences based on weighted samples contained. \
We approximate integrals over individual events' observables via weighted monte-carlo sums and dynamically adjust the weights to reflect the prior odds associated with the population parameters
"""
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

### non-standard libraries
from . import utils

#-------------------------------------------------

DEFAULT_SAMPLE_SIZE = 1000

#------------------------

DEFAULT_M_NAME = 'm'
DEFAULT_R_NAME = 'R'

DEFAULT_M1_NAME = 'm1'
DEFAULT_M2_NAME = 'm2'
DEFAULT_L1_NAME = 'lambda1'
DEFAULT_L2_NAME = 'lambda2'

#-------------------------------------------------

def m1m2_to_mc(m1, m2):
    return (m1*m2)**0.6/(m1+m2)**0.2

#-------------------------------------------------

class Exposure(object):
    """an object representing the exposure of an observation set. Should be able to return everything needed to compute VT et al given a population
    """
    def __init__(self, population):
        self.population = population

    @staticmethod
    def _vt(params, population, size=DEFAULT_SAMPLE_SIZE):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('need to compute VT via a monte-carlo integral over samples from the population')

    def vt(self, size=DEFAULT_SAMPLE_SIZE):
        """compute the VT given the population via a monte-carlo integral over the population
        """
        return self._vt(self.population.params, self.population, size=size)

    @staticmethod
    def _count(rate, params, population, size=DEFAULT_SAMPLE_SIZE):
        return rate*self._vt(params, population, size=size)

    def count(self, size=DEFAULT_SAMPLE_SIZE):
        """compute the expected number of events given the population
        """
        return self._count(self.population.rate, self.population.params, self.population)

#------------------------

class GravitationalWaveExposure(Exposure):
    """an object representing the exposure of a network of Gravitational-Wave detectors. Does some basic type checking
    """

    def __init__(self, *args, **kwargs):
        Exposure.__init__(self, *args, **kwargs)
        assert isinstance(self.population, GravitationalWavePopulation), 'population must be an instance of GravitationalWavePopulation!'
        print('WARNING: Exposure currently implements an unrealistic toy model! Please take all results with a healthy amount of skepticism')

    @staticmethod
    def _vt(params, population, size=DEFAULT_SAMPLE_SIZE):
        """A very simple model of VT, assuming that the VT for masses simply scales as (Mchirp)**3
        """
        samples = population._sample_prior(params, population, size=size, m1=DEFAULT_M1_NAME, m2=DEFAULT_M2_NAME)
        return m1m2_to_mc(samples[DEFAULT_M1_NAME], samples[DEFAULT_M2_NAME])**3

class NicerExposure(Exposure):
    """an object representing the exposure of Nicer observations. Does some basic type checking
    """

    def __init__(self, *args, **kwargs):
        Exposure.__init__(self, *args, **kwargs)
        assert isinstance(self.population, NicerPopulation), 'population must be an instance of NicerPopulation!'

    @staticmethod
    def _vt(params, population, size=DEFAULT_SAMPLE_SIZE):
        return 0. ### we assume selection effects do not affect this because we do not marginalize over the "rate" of Nicer measurements

class MassExposure(Exposure):
    """an object representing the exposure of mass observations. Does some basic type checking
    """

    def __init__(self, *args, **kwargs):
        Exposure.__init__(self, *args, **kwargs)
        assert isinstance(self.population, MassPopulation), 'population must be an instance of MassPopulation!'

    @staticmethod
    def _vt(params, population, size=DEFAULT_SAMPLE_SIZE):
        return 0. ### we assume selection effects do not affect this because we do not marginalize over the "rate" of mass measurements

#-------------------------------------------------

class Population(object):
    """an object representing a population. We can use this to evaluate the prior suitable for each event as well as the exposure. \
This object should be extended to implement different population models (eg, power law vs Gaussian in component masses).
    """

    def __init__(self, rate, *params):
        self.rate = rate
        self._params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new):
        assert len(new)==len(self._params), 'new parameters must have the same dimension as the current parameters'
        self._params = new

    def vt(self, exposure, **kwargs):
        """compute the VT given the exposure
        """
        return exposure._vt(self.params, self, **kwargs)

    def count(self, exposure, **kwargs):
        """compute the expected number of events given the exposure
        """
        return exposure._count(self.rate, self.params, self, **kwargs)

    @staticmethod
    def _logprior(rate, params, data, **names):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('need to evaluate the logprior associated with this set of params')

    def logprior(self, data, **names):
        """compute the appropriate log(prior) for single-event parameters given the population with column-names in data given by names
        """
        return self._logprior(self.rate, self.params, data, **names)

    @staticmethod
    def _sample_prior(params, size=DEFAULT_SAMPLE_SIZE, **names):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('return samples of parameters drawn from the prior induced by this population')

    def sample_prior(self, size=DEFAULT_SAMPLE_SIZE, **names):
        """return samples for single-event parameters drawn from the prior described by these hyperparameters as a structured array with column-names given by names
        """
        return self._sample_prior(self.params, size=size, **names)

    @staticmethod
    def _loghyperprior(rate, params):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('return the hyperprior on population parameters evaluated at the current parameters')

    def loghyperprior(self):
        """return the log(hyperprior) for the population parameters
        """
        return self._loghyperprior(rate, self.params)

    @staticmethod
    def _sample_hyperprior(rate, params, size=DEFAULT_SAMPLE_SIZE):
        """NOTE: child classes should overwrite this!
        """
        raise NotImplementedError('return hyperparameter samples drawn from the hyperprior')

    def sample_hyperprior(self, size=DEFAULT_SAMPLE_SIZE):
        """return samples from the hyperprior
        """
        return self._sample_hyperprior(self.rate, self.params, size=size)

#------------------------

class GravitationalWavePopulation(Population):
    """an object representing a population of possible Gravitational-Wave sources.
This is essentially a distribution of component masses and distances

NOTE: this implementes a simple prior on component masses

    p(m1, m2) ~ m1**alpha * m2**alpha | mmin <= m2 <= m1 < mmax

and a flat hyperprior

    p(rate, alpha, mmin, mmax) ~ constant | mmin, mmax > 0 AND mmax > mmin
    """

    @staticmethod
    def _logprior(rate, params, data, m1=DEFAULT_M1_NAME, m2=DEFAULT_M2_NAME):
        """evaluate the prior induced by params at all the samples in data with names in data given by m1, m2
        """
        ### extract samples from data
        m1 = data[m1]
        m2 = data[m2]

        ### extract hyperparameters
        alpha, mmin, mmax = params

        ### compute prior at these samples and return
        ans = np.log(rate) + alpha*np.log(m1) + alpha*np.log(m2)

        ans[m2<mmin] = -np.infty ### out of bounds
        ans[m1<mmin] = -np.infty
        ans[m2>=mmax] = -np.infty
        ans[m2>=mmax] = -np.infty
        ans[m2>m1] = -np.infty

        return ans

    @staticmethod
    def _sample_prior(params, size=DEFAULT_SAMPLE_SIZE, m1=DEFAULT_M1_NAME, m2=DEFAULT_M2_NAME):
        """sample from the prior induced by params and return a structured array with names given by m1, m2
        """
        alpha, mmin, mmax = params
        if alpha==-1:
            raise NotImplementedError

        else:
            raise NotImplementedError

    @staticmethod
    def _loghyperprior(rate, params):
        ans, mmin, mmax = params
        if (mmin < 0) or (mmax < 0) or (mmax < mmin):
            return -np.infty
        else:
            return 0.

#    @staticmethod
#    def _sample_hyperprior(rate, params, size=DEFAULT_SAMPLE_SIZE):
#        raise NotImplementedError('return hyperparameter samples drawn from the hyperprior')

class NicerPopulation(Population):
    """an object representing a population of possible Nicer sources.
This is essentially a distribution over mass.

NOTE: this implements a simple prior on mass

    p(m) ~ m**alpha | mmin <= m < mmax

and a flat hyperprior

    p(rate, alpha, mmin, mmax) ~ constant | mmin, mmax > 0 AND mmax > mmin
    """

    @staticmethod
    def _logprior(rate, params, data, m=DEFAULT_M_NAME):
        """evaluate the prior induced by params at all the samples within data with the column-name in data given by m
        """
        ### extract samples from data
        m = data[m]

        ### extract parameters
        alpha, mmin, mmax = params

        ### compute prior at these samples
        ans = np.log(rate) + alpha*np.log(m)

        ans[m<mmin] = -np.infty  ### add the prior bounds
        ans[m>=mmax] = -np.infty

        return ans

    @staticmethod
    def _sample_prior(params, size=DEFAULT_SAMPLE_SIZE, m=DEFAULT_M_NAME):
        """sample from the prior induced by params and return a structured array with column-name given by m
        """
        alpha, mmin, mmax = params
        if alpha==-1:
            return np.array(
                mmin * np.exp(np.log(mmax/mmin)*np.random.random(size)),
                dtype=[(m, float)],
            )

        else:
            alpha_plus_one = alpha+1
            return np.array(
                ( (mmax**alpha_plus_one - mmin**alpha_plus_1)*np.random.random(size) + mmin**alpha_plus_one )**(1./alpha_plus_one),
                dtype=[(m, float)],
            )

    @staticmethod
    def _loghyperprior(rate, params):
        return GravitationalWavePopulation._loghyperprior(rate, params)

#    @staticmethod
#    def _sample_hyperprior(rate, params, size=DEFAULT_SAMPLE_SIZE):
#        raise NotImplementedError('return hyperparameter samples drawn from the hyperprior')

class MassPopulation(Population):
    """an object representing a population of possible mass-measurement sources
This is essentially a distribution over mass.

NOTE: this implements a simple prior on mass

    p(m) ~ m**alpha  | mmin <= m < mmax

and a flat hyperprior

    p(rate, alpha, mmin, mmax) ~ constant | mmin, mmax > 0 AND mmax > mmin

ALSO NOTE: we current delegate to NicerPopulation for pretty much everything because the underlying model is the same. I'm keeping this object around in the expectation that we will eventually change the model and this could save some boilerplate typing later on.
    """

    @staticmethod
    def _logprior(rate, params, data, m=DEFAULT_M_NAME):
        """evaluate the prior induced by params at all samples within data with the column-name in data given by m
        """
        return NicerPopulation._logprior(rate, params, data, m=m)

    @staticmethod
    def _sample_prior(params, size=DEFAULT_SAMPLE_SIZE, m=DEFAULT_M_NAME):
        return NicerPopulation._logprior(params, size=size, m=m)

    @staticmethod
    def _loghyperprior(rate, params):
        return NicerPopulation._loghyperprior(rate, params)

#    @staticmethod
#    def _sample_hyperprior(params, size=DEFAULT_SAMPLE_SIZE):
#        raise NotImplementedError('return hyperparameter samples drawn from the hyperprior')

#-------------------------------------------------

class Observation(object):
    """an object representing the observations for a single system
    """

    def __init__(self, inpath, columns, weight_columns, logcolumns=[], max_num_samples=np.infty, logweightcolumns=[], label=None):

        self.names = dict((col, col) for col in columns) ### this is overwritten by child classes and is supported here for syntactic completion

        ### load and store data
        self.label = label if label is not None else inpath

        self.inpath = inpath
        data, self.columns = utils.load(inpath, columns, logcolumns=logcolumns, max_num_samples=max_num_samples)
        self.data = np.array(data, dtype=[(col, 'float') for col in columns]) ### store a structured array

        self.log_weights = utils.load_logweights(inpath, weight_columns, logweightcolumns=logweightcolumns, max_num_samples=max_num_samples)
        self.log_weights /= utils.sum_log(self.log_weights) ### normalize these
                                                            ### OK because we only care about ratios and not the absolue value of the marginal likelihood

        ### discard any data assigned zero weight
        truth = self.log_weights > -np.infty
        self.data = self.data[truth]
        self.log_weights = self.log_weights[truth]

    @staticmethod
    def _marginalize(params, population, data, log_weights, **names):
        logprior = population._logprior(params, data, **names)
        return utils.sum_log(log_weights + logprior) ### return the log of the monte-carlo marginalization

    def marginalize(self, population, **names):
        """compute the marginal integral with respect to prior given the weighted samples
        """
        return self._marginalize(population.params, population, self.data, self.log_weights, **self.names)

#------------------------

class GravitationalWaveObservation(Observation):
    """an object representing a Gravitational-Wave observation
    """

    def __init__(self,
            inpath,
            weight_columns,
            max_num_samples=np.infty,
            label=None,
            m1=DEFAULT_M1_NAME, ### column names relevant for this type of observation
            m2=DEFAULT_M2_NAME,
            lambda1=DEFAULT_L1_NAME,
            lambda2=DEFAULT_L2_NAME,
        ):
        columns = [m1, m2, lambda1, lambda2]
        Observation.__init__(self, inpath, columns, weight_columns, max_num_samples=max_num_samples, logweightcolumns=logweightcolumns, label=label)

        ### overwrite what was set
        self.names = {
            'm1':m1,
            'm2':m2,
            'lambda1':lambda1,
            'lambda2':lambda2,
        }

class NicerObservation(Observation):
    """an object representing a Nicer observation
    """

    def __init__(self,
            inpath,
            weight_columns,
            max_num_samples=np.infty,
            label=None,
            m=DEFAULT_M_NAME, ### column names relevant for this type of observation
            r=DEFAULT_R_NAME,
        ):
        columns = [m, r]
        Observation.__init__(self, inpath, columns, weight_columns, max_num_samples=max_num_samples, logweightcolumns=logweightcolumns, label=label)

        ### overwrite what was set
        self.names = {'m':m, 'r':r}

class MassObservation(Observation):
    """an object representing a Mass observation
    """

    def __init__(self,
            inpath,
            weight_columns,
            max_num_samples=np.infty,
            label=None,
            m=DEFAULT_M_NAME, ### column names relevant for this type of observation
        ):
        columns = [m]
        Observation.__init__(self, inpath, columns, weight_columns, max_num_samples=max_num_samples, logweightcolumns=logweightcolumns, label=label)

        ### overwrite what was set
        self.names = {'m':m}

#-------------------------------------------------

class ObservationSet(object):
    """an object representing an ensemble of observations, which provides access to inference over the entire population.
    """

    def __init__(self, exposure, *observations):
        self.exposure = exposure
        self.observation = observations

    @staticmethod
    def _marginalize(params, population, observations):
        return np.sum(obs._marginalize(params, population, obs.data, obs.log_weights, **obs.names) for obs in observations)

    def marginalize(self, population):
        """returns the product of all individual event marginalizations.
        """
        return self._marginalize(population.params, population, self.observations) ### returns the log of the monte-carlo marginalization

    @staticmethod
    def _loglike(rate, params, population, exposure, observations, size=DEFAULT_SAMPLE_SIZE):
        return self._marginalize(rate, params, population, observations) + exposure._count(rate, params, population, size=size)

    def loglike(self, size=DEFAULT_SAMPLE_SIZE):
        """return loglike of observation set given the population
        """
        population = self.exposure.population
        return self._loglike(population.rate, population.params, population, self.exposure, self.observations, size=size)

#------------------------

class GravitationalWaveObservationSet(ObservationSet):
    """an object representing a set of Gravitational-Wave observations. Does some basic type checking
    """

    def __init__(self, *args, **kwargs):
        ObservationSet.__init__(self, *args, **kwargs)
        assert isinstance(self.exposure, GravitationalWaveExposure), \
            'exposure must be an instance of GravitationalWaveExposure'
        assert all(isinstance(obs, GravitationalWaveObservation) for obs in self.observations), \
            'each observation must be an instance of GravitationalWaveObservation!'

class NicerObservationSet(ObservationSet):
    """an object representing a set of Nicer observations. Does some basic type checking
    """

    def __init__(self, *args, **kwargs):
        ObservationSet.__init__(self, *args, **kwargs)
        assert isinstance(self.exposure, NicerExposure), \
            'exposure must be an instance of NicerExposure'
        assert all(isinstance(obs, NicerObservation) for obs in self.observations), \
            'each observation must be an instance of NicerObservation!'

class MassObservation(ObservationSet):
    """an object representing a set of mass observations. Does some basid type checking
    """

    def __init__(self, *args, **kwargs):
        ObservationSet.__init__(self, *args, **kwargs)
        assert isinstance(self.exposure, MassExposure), \
            'exposure must be an instance of MassExposure'
        assert all(isinstance(obs, MassObservation) for obs in self.observations), \
            'each observation must be an instance of MassObservation!'

#-------------------------------------------------

def logpost(rate_params, observationset):
    """assumes the first element of rate_params is the overall rate and the rest describe the population
    """
    ### dig out all the references we need
    exposure = observationset.exposure
    population = exposure.population
    observations = observationset.observations

    rate = rate_params[0]
    params = rate_params[1:]

    ### delegate
    return observationset._loglike(rate, params, population, exposure, observations) + population._loghyperprior(rate, params)
