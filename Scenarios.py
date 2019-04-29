import Workload
import math


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
        sumF[self._n] = self.compute_F(self._n)
        for k in range(self._n - 1, 0, -1):
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
