import numpy as np

class RequestSequence():

    def __init__(self, lower_bound, upper_bound, CDF_func,
                 discret_samples=100):

        self._a = lower_bound
        self._b = upper_bound
        self._n = discret_samples
        self._delta = float((self._b - self._a) / self._n)

        self.CDF_func = CDF_func
        self._E = {}
        self._request_sequence = []

    def compute_F(self, i):
        vi = (self._a + self._delta * i)
        fi = self.CDF_func(vi)
        if np.isnan(fi):
            fi=0
        if i > 0:
            fi -= self.CDF_func(vi-self._delta)
        if np.isnan(fi):
            fi=0
        return fi / self.CDF_func(self._b)

    def compute_sum_F(self):
        sumF = (self._n + 1) * [0]
        for k in range(self._n - 1, 0, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
        return sumF

    def get_optimal(self):
        return -1

    def compute_request_sequence(self):
        return self._request_sequence

class TOptimalSequence(RequestSequence):
    ''' Sequence that optimizes the total makespan of a job. Defined in the
    Aupy et al paper published in IPDPS 2019. '''

    def __init__(self, lower_bound, upper_bound, CDF_func,
                 discret_samples=100):
        super(TOptimalSequence, self).__init__(lower_bound, upper_bound,
                                               CDF_func, discret_samples)
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

            if min_request == -1 or min_makespan >= makespan:
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


class TODiscretSequence(RequestSequence):
    ''' Sequence that optimizes the total makespan of a job for discret
    values (instead of a continuous space) '''

    def __init__(self, upper_bound, discret_values, probability_values):
        self.discret_values = discret_values
        self.__prob = probability_values
        self.upper_limit = upper_bound
        self._E = {}
        self._request_sequence = []
        
        self.__sumF = self.get_discret_sum_F()
        E_val = self.compute_E_value(0)
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def get_discret_sum_F(self):
        sumF = (len(self.discret_values) + 1) * [0]
        for k in range(len(self.discret_values) - 1, -1, -1):
            sumF[k] = self.__prob[k] + sumF[k + 1]
        return sumF

    def __compute_E_table(self, i):
        if i == len(self.discret_values):
            return (0, len(self.discret_values))

        min_makespan = -1
        min_request = -1
        for j in range(i, len(self.discret_values)):
            makespan = float(self.__sumF[i] * self.discret_values[j])
            if j + 1 in self._E:
                makespan += self._E[j + 1][0]
            else:
                E_val = self.__compute_E_table(j + 1)
                makespan += E_val[0]
                self._E[j + 1] = E_val

            if min_request == -1 or min_makespan >= makespan:
                min_makespan = makespan
                min_request = j
        return (min_makespan, min_request)

    def __compute_E_table_iter(self, first):
        self._E[len(self.discret_values)] = (0, len(self.discret_values))
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            if i in self._E:
                continue
            min_makespan = -1
            min_request = -1
            for j in range(i, len(self.discret_values)):
                makespan = float(self.__sumF[i] * self.discret_values[j])
                makespan += self._E[j + 1][0]

                if min_request == -1 or min_makespan >= makespan:
                    min_makespan = makespan
                    min_request = j
            self._E[i] = (min_makespan, min_request)
        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        j = 0
        E_val = self.compute_E_value(j)
        while E_val[1] < len(self.discret_values):
            self._request_sequence.append(self.discret_values[E_val[1]])
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)
        if self._request_sequence[-1] != self.upper_limit:
            self._request_sequence.append(self.upper_limit)
        return self._request_sequence

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        if len(self.discret_values)<600:
            E_val = self.__compute_E_table(i)
        else:
            E_val = self.__compute_E_table_iter(i)
        self._E[i] = E_val
        return E_val
