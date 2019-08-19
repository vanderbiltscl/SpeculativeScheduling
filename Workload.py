from scipy.stats import truncnorm
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import pareto
import logging
import math
import numpy as np
from SpeculativeBackfill import StochasticApplication


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


class RequestSequence():

    def __init__(self, distribution, discret_samples=100):

        self._a = distribution.get_low_bound()
        self._b = distribution.get_up_bound()
        self._n = discret_samples
        self._delta = float((self._b - self._a) / self._n)

        self.CDF_func = distribution.CDF_func

        self._E = {}
        self._request_sequence = []

    def compute_F(self, i):
        vi = (self._a + self._delta * i)
        fi = self.CDF_func(vi)
        if i > 0:
            fi -= self.CDF_func(vi-self._delta)
        return fi

    def compute_FV(self, i):
        vi = (self._a + self._delta * i)
        fi = self.compute_F(i)
        return vi * fi

    def compute_sum_F(self):
        sumF = (self._n + 1) * [0]
        for k in range(self._n - 1, -1, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
        return sumF

    def get_optimal(self):
        return -1

    def compute_request_sequence(self):
        return self._request_sequence


class ATOptimalSequence(RequestSequence):
    '''Optimal sequence of request values when a job runs together with
    a continuous stream of small jobs that can be used for backfill. '''

    def __init__(self, backfill_rate, distribution, discret_samples=100):
        super(ATOptimalSequence, self).__init__(distribution,
                                                   discret_samples)
        self.__backfill_rate = backfill_rate
        self.__sumF = self.__compute_sum_F()
        self.__sumFV = self.__compute_sum_FV()
        E_val = self.compute_E_value(1, 0, 0)
        self.__t1 = self._a + E_val[1] * self._delta
        self.__makespan = E_val[0]

    def __compute_sum_FV(self):
        sumFV = []
        sumFV.append(self.compute_FV(0))
        for k in range(1, self._n + 1):
            sumFV.append(self.compute_FV(k) +
                         sumFV[k - 1])
        return sumFV

    def __compute_sum_F(self):
        sumF = []
        sumF.append(self.compute_F(0))
        for k in range(1, self._n + 1):
            sumF.append(self.compute_F(k) +
                        sumF[k - 1])
        return sumF

    def get_optimal(self):
        return self.__makespan

    def __compute_sum_bound(self, i, m, k, j):
        spec = math.floor(j - self.__backfill_rate *
                          (j + k + (m + 1) * self._a / self._delta))
        return max(i - 1, int(spec))

    def compute_sum(self, sfrom, sto, sfun):
        if sto < sfrom:
            return 0
        return sfun[sto] - sfun[sfrom - 1]

    def __compute_E_table(self, i, m, k):
        if i == self._n + 1:
            return (0, self._n)

        min_makespan = -1
        min_request = 0
        for j in range(i, self._n + 1):
            bound = self.__compute_sum_bound(i, m, k, j)
            makespan = float(self.compute_sum(i, bound, self.__sumF) *
                             (self._a + self._delta * j))
            makespan += (self.compute_sum(bound + 1, j, self.__sumF) *
                         (m * self._a + k * self._delta) * 1. *
                         self.__backfill_rate / (1 - self.__backfill_rate))
            makespan += (self.compute_sum(bound + 1, j, self.__sumFV) /
                         float(1 - self.__backfill_rate))
            makespan += (self.compute_sum(j + 1, self._n, self.__sumF) *
                         (self._a + self._delta * j))

            if (j + 1, m + 1, k + j) in self._E:
                makespan += self._E[(j + 1, m + 1, k + j)][0]
            else:
                E_val = self.__compute_E_table(j + 1, m + 1, k + j)
                makespan += E_val[0]
                self._E[(j + 1, m + 1, k + j)] = E_val

            if min_makespan > makespan or min_makespan == -1:
                min_makespan = makespan
                min_request = j

        return (min_makespan, min_request)

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        E_val = (0, 0)
        j = 1
        m = 0
        k = 0
        while E_val[1] < self._n:
            E_val = self.compute_E_value(j, m, k)
            self._request_sequence.append(self._a + E_val[1] * self._delta)
            j = E_val[1] + 1
            m += 1
            k += E_val[1]
        return self._request_sequence

    def compute_E_value(self, i, m, k):
        if (i, m, k) in self._E:
            return self._E[(i, m, k)]
        E_val = self.__compute_E_table(i, m, k)
        self._E[(i, m, k)] = E_val
        return E_val


class UOptimalSequence(RequestSequence):
    ''' Sequence that optimizes the job utilization '''

    def __init__(self, distribution, discret_samples=100):
        super(UOptimalSequence, self).__init__(distribution, discret_samples)
        E_val = self.compute_E_value(1, 0, 0)
        self.__t1 = self._a + E_val[1] * self._delta
        self.__utilization = E_val[0]

    def get_optimal(self):
        return self.__utilization

    def __compute_E_table(self, i, m, k):
        if i == self._n + 1:
            return (0, self._n)

        max_utilization = 0
        max_request = 0
        FV = 0
        for j in range(i, self._n + 1):
            FV += self.compute_FV(j)
            utilization = float(FV / ((m + 1) * self._a +
                                      (k + j) * self._delta))
            if (j + 1, m + 1, k + j) in self._E:
                utilization += self._E[(j + 1, m + 1, k + j)][0]
            else:
                E_val = self.__compute_E_table(j + 1, m + 1, k + j)
                utilization += E_val[0]
                self._E[(j + 1, m + 1, k + j)] = E_val

            if max_utilization < utilization:
                max_utilization = utilization
                max_request = j

        return (max_utilization, max_request)

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        E_val = (0, 0)
        j = 1
        m = 0
        k = 0
        while E_val[1] < self._n:
            E_val = self.compute_E_value(j, m, k)
            self._request_sequence.append(self._a + E_val[1] * self._delta)
            j = E_val[1] + 1
            m += 1
            k += E_val[1]
        return self._request_sequence

    def compute_E_value(self, i, m, k):
        if (i, m, k) in self._E:
            return self._E[(i, m, k)]
        E_val = self.__compute_E_table(i, m, k)
        self._E[(i, m, k)] = E_val
        return E_val


class TOptimalSequence(RequestSequence):
    ''' Sequence that optimizes the total makespan of a job. Defined in the
    Aupy et al paper published in IPDPS 2019. '''

    def __init__(self, distribution, discret_samples=100):
        super(TOptimalSequence, self).__init__(distribution, discret_samples)
        self.__sumF = self.compute_sum_F()
        E_val = self.compute_E_value(1)
        self.__t1 = self._a + E_val[1] * self._delta
        self.__makespan = E_val[0]

    def get_optimal(self):
        return self.__makespan

    def __compute_E_table(self, i):
        if i == self._n + 1:
            return (0, self._n)

        min_makespan = -1
        min_request = -1
        for j in range(i, self._n + 1):
            makespan = float(self.__sumF[i] * (self._a + self._delta * j))
            if j + 1 in self._E:
                makespan += self._E[j + 1][0]
            else:
                E_val = self.__compute_E_table(j + 1)
                makespan += E_val[0]
                self._E[j + 1] = E_val

            if min_request == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_request = j
        return (min_makespan, min_request)

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        E_val = (0, 0)
        j = 1
        while E_val[1] < self._n:
            E_val = self.compute_E_value(j)
            self._request_sequence.append(self._a + E_val[1] * self._delta)
            j = E_val[1] + 1
        return self._request_sequence

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        E_val = self.__compute_E_table(i)
        self._E[i] = E_val
        return E_val


class CheckpointSequence(RequestSequence):

    def __init__(self, distribution, discret_samples=100):
        super(TOptimalSequence, self).__init__(distribution, discret_samples)
        self.C = 0.15
        self.R = 0.15
        self.__sumF = self.compute_sum_F()
        E_val = self.compute_E_value(0, 0)

    def __compute_makespan(self, ic, il, j, R, delta):
        makespan = 0
        start = 0
        if ic == 0:
            start = self._a
        new_ic = (1 - delta) * ic + delta * j
        if (new_ic, j) in self._E:Your location
            makespan += self._E[(new_ic, j)][0]
        else:
            E_val = self.__compute_E_table(new_ic, j)
            makespan += E_val[0]
            self._E[(new_ic, j)] = E_val
        makespan += ((R + delta * C + self._delta * (j - ic) + start) * self._sumF[il])
        return makespan

    def __compute_E_table(self, ic, il):
        if il == self.n:
            return (0, self._n, 0)
        R = self.R
        if ic == 0:
            R = 0
        min_makespan = -1
        min_j = -1
        min_delta = -1

        for j in range(il + 1, self._n + 1):
            makespan_wo = __compute_makespan(self, ic, il, j, R, 0)
            delta = 1
            makespan = __compute_makespan(self, ic, il, j, R, 1)
            if makespan_wo < makespan:
                makespan = makespan_wo
                delta = 0

        if min_makespan == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_j = j
                min_delta = delta
        return (min_makespan, min_j, min_delta)

    def compute_request_sequence(self):
        return self._request_sequence

    def compute_E_value(self, ic, il):
        if (ic, il) in self._E:
            return self._E[(ic, il)]
        E_val = self.__compute_E_table(ic, il)
        self._E[(ic, il)] = E_val
        return E_val


class Workload(object):
    ''' Class for generating the list of jobs used by the simulator '''

    def __init__(self, distribution, count):
        ''' Constructor method that takes a distribution object and
        the number of jobs to be generated using the given distribution '''

        self.logger = logging.getLogger(__name__)

        exec_times = distribution.random_sample(count)
        # eliminate the rare cases when the distribution returns
        # a walltime of 0
        for i in range(len(exec_times)):
            if int(exec_times[i] * 3600) == 0:
                exec_times[i] = 0.01

        self.walltimes = [int(i * 3600) for i in exec_times]
        self.upper_bound = distribution.get_up_bound()
        self.__request = [i for i in self.walltimes]
        self.__procs = [1 for i in range(count)]
        self.__submission = [0 for i in range(count)]
        self.__sequence = []
        self.__distribution = distribution

    def set_processing_units(self, procs_function=None, distribution=None):
        assert (procs_function is not None or
                distribution is not None),\
            'No valid method provided for generating processing units'
        assert (procs_function is None or distribution is None),\
            'Need to provide only one method for generating processing'

        if procs_function is not None:
            self.__procs = procs_function(self.walltimes)

        if distribution is not None:
            self.__procs = distribution.random_sample(len(self.walltimes))
            self.__procs = [int(p) for p in self.__procs]

        assert (len(self.__procs) == len(self.walltimes)),\
            'Processor list has different lenght than the walltimes list'

    def set_request_time(self, request_function=None, request_sequence=None):
        assert (request_function is not None or
                request_sequence is not None), \
            'No valid method for generating request times provided'

        self.__sequence = []
        self.__request = []
        if request_function is not None:
            self.__request = request_function(self.walltimes)

        if request_sequence is not None:
            sequence_start = 0
            if len(self.__request) == 0:
                self.__request = [int(request_sequence[0] * 3600)
                                  for i in range(len(self.walltimes))]
                sequence_start += 1
            self.__sequence = [int(i * 3600) for i in
                               request_sequence[sequence_start:]]

        assert (len(self.__request) <= len(self.walltimes)),\
            'Request list has more entries than the walltimes list'

    def set_submission_time(self, submission_function):
        self.__submission = submission_function(self.walltimes)
        assert (len(self.__submission) == len(self.walltimes)),\
            'Submission list has more entries than the walltimes list'

    def generate_workload(self):
        ''' Method for generating the list of Jobs '''

        prev_instance = len(self.walltimes) - len(self.__request)
        jobs = [StochasticApplication(
            self.__procs[i + prev_instance],
            self.__submission[i + prev_instance],
            self.walltimes[i + prev_instance],
            [self.__request[i]] + self.__sequence,
            resubmit_factor=1.5) for i in range(len(self.__request))]
        return jobs
