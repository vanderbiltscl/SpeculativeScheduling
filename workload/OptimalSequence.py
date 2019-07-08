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