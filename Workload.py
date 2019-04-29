from scipy.stats import truncnorm
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import pareto
import logging
import numpy as np


class Distribution(object):
    ''' Base class for all distribution classes that can be used
    to create a workload of jobs '''

    def get_low_bound(self):
        ''' Method for getting the distribution lower bound -
        has to be over 0 '''
        return 0

    def get_user_friendly_name(self):
        ''' Method returns the distribution name that is used
        to return to users possible options '''
        return 'noname'


class ConstantDistr(Distribution):
    ''' Constant class - generating execution times of contant value '''

    def __init__(self, value):
        assert (value > 0),\
            "Constant value must be greater than 0"

        self.__value = value

    def PDF_func(self, t):
        return 1

    def CDF_func(self, t):
        return 1

    def random_sample(self, count):
        return [self.__value] * count

    def get_up_bound(self):
        return self.__value

    def get_user_friendly_name(self):
        return 'constant'


class BetaDistr(Distribution):
    ''' Beta Distribution - generating execution times between 0 and 1 '''

    def __init__(self, alpha, beta):
        assert (alpha > 0),\
            "Alpha must be greater than 0 in the beta distribution"
        assert (beta > 0),\
            "Beta must be greater than 0 in the beta distribution"

        self.__alpha = alpha
        self.__beta = beta

    def PDF_func(self, t):
        return beta.pdf(t, self.__alpha, self.__beta)

    def CDF_func(self, t):
        return beta.cdf(t, self.__alpha, self.__beta)

    def random_sample(self, count):
        return beta.rvs(self.__alpha, self.__beta, size=count)

    def get_up_bound(self):
        return 1

    def get_user_friendly_name(self):
        return 'beta'


class ExponentialDistr(Distribution):
    ''' Exponential Distribution -
    generating execution times between 0 and 20 '''

    def __init__(self, lambda_exp):
        assert (lambda_exp >= 0),\
            "Lambda must be > 0 in the exponential distribution"
        self.__scale = 1. / lambda_exp

    def PDF_func(self, t):
        return expon.pdf(t, scale=self.__scale)

    def CDF_func(self, t):
        return expon.cdf(t, scale=self.__scale)

    def random_sample(self, count):
        return expon.rvs(scale=self.__scale, size=count)

    # need to be changed to some upper limit that makes more sense
    def get_up_bound(self):
        return 15

    def get_user_friendly_name(self):
        return 'exponential'


class TruncNormalDistr(Distribution):
    ''' Truncated Normal Distribution - generating execution times between
    the provided lower and upper bound '''

    def __init__(self, lower_bound, upper_bound, mu, sigma):
        assert (lower_bound >= 0),\
            "Lower bound must be >= 0 in the truncted normal distribution"
        assert (upper_bound > lower_bound),\
            """Upper bound must be > lower bound in the truncted normal
            distribution"""
        assert (sigma >= 1),\
            "Sigma must be >= 1 in the truncated normal distribution"

        self.__up_bound = (upper_bound - mu) / sigma
        self.__low_bound = (lower_bound - mu) / sigma
        self.mu = mu
        self.sigma = sigma

    def PDF_func(self, t):
        return truncnorm.pdf(
            t,
            self.__low_bound,
            self.__up_bound,
            loc=self.mu,
            scale=self.sigma)

    def CDF_func(self, t):
        return truncnorm.cdf(
            t,
            self.__low_bound,
            self.__up_bound,
            loc=self.mu,
            scale=self.sigma)

    def random_sample(self, count):
        return truncnorm.rvs(self.__low_bound, self.__up_bound,
                             loc=self.mu, scale=self.sigma, size=count)

    def get_low_bound(self):
        return self.__low_bound * self.sigma + self.mu

    def get_up_bound(self):
        return self.__up_bound * self.sigma + self.mu

    def get_user_friendly_name(self):
        return 'truncnormal'


class ParetoDistr(Distribution):
    ''' Bounded Pareto Distribution - generating execution times between
    the provided lower and upper bound '''

    def __init__(self, lower_bound, upper_bound, alpha):
        assert (lower_bound >= 1),\
            "Lower bound must be >= 1 in the pareto distribution"
        assert (upper_bound > lower_bound),\
            "Upper bound must be > lower bound in the pareto distribution"

        self.__up_bound = upper_bound
        self.__low_bound = lower_bound
        self.__alpha = alpha

    def PDF_func(self, t):
        return (self.__alpha * pow(self.__low_bound, self.__alpha) * pow(t,
                -self.__alpha - 1)) / (1 - pow(self.__low_bound /
                                       self.__up_bound, self.__alpha))

    def CDF_func(self, t):
        if t > 0:
            return (1 - pow(self.__low_bound, self.__alpha) *
                    pow(t, -self.__alpha)) / (1 - pow(self.__low_bound /
                                              self.__up_bound, self.__alpha))
        return 0

    def random_sample(self, count):
        sequence = []
        i = 1
        while len(sequence) < count:
            sequence = np.array(pareto.rvs(
                self.__alpha, size=i * count)) + (self.__low_bound - 1)
            sequence = sequence[sequence < self.__up_bound]
            i += 1
        return sequence[:count]

    def get_up_bound(self):
        return self.__up_bound

    def get_low_bound(self):
        return self.__low_bound

    def get_user_friendly_name(self):
        return 'pareto'
