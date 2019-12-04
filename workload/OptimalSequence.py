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
        sumF = (self._n + 2) * [0]
        for k in range(self._n, -1, -1):
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
            self._request_sequence.append((self._a + E_val[1] * self._delta, ))
            j = E_val[1] + 1

        if self._request_sequence[-1][0] != self._b:
            self._request_sequence.append((self._b, ))

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

    def compute_F(self, vi):
        fi = self.__prob[vi]
        if vi > 0:
            fi -= self.__prob[vi-1]
        return fi / self.__prob[-1]

    def get_discret_sum_F(self):
        sumF = (len(self.discret_values) + 1) * [0]
        for k in range(len(self.discret_values) - 1, -1, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
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

            if min_request == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_request = j
        return (min_makespan, min_request)

    def __compute_E_table_iter(self, first):
        self._E[len(self.discret_values)] = (0, len(self.discret_values))
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            if i in self._E:
                continue
            min_makespan = 0
            min_request = len(self.discret_values)
            for j in range(i, len(self.discret_values)):
                makespan = float(self.__sumF[i] * self.discret_values[j])
                makespan += self._E[j + 1][0]

                if min_makespan == 0 or min_makespan >= makespan:
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
            self._request_sequence.append((self.discret_values[E_val[1]], ))
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)

        if self._request_sequence[-1][0] != self.upper_limit:
            self._request_sequence.append((self.upper_limit, ))
        
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


class CheckpointSequence(RequestSequence):

    def __init__(self, lower_bound, upper_bound, CDF_func,
                 discret_samples=100, always_checkpoint=False,
                 C=-1, R=-1):
        super(CheckpointSequence, self).__init__(lower_bound, upper_bound,
                                               CDF_func, discret_samples)
        
        # if no C/R values are provided, the cost is equal to 10% of average
        self._C = C
        self.R = R
        if C== -1:
            self._C = (upper_bound - lower_bound) / 10
        if R == -1:
            self._R = (upper_bound - lower_bound) / 10

        self.always_checkpoint = always_checkpoint
        self._sumF = self.compute_sum_F()
        E_val = self.compute_E_value(0, 0)

    def __compute_makespan(self, ic, il, j, R, delta):
        makespan = 0
        start = 0
        if ic == 0:
            start = self._a
        new_ic = (1 - delta) * ic + delta * j
        if (new_ic, j) in self._E:
            makespan += self._E[(new_ic, j)][0]
        else:
            if self.always_checkpoint:
                E_val = self.__compute_E_table_checkpoint(new_ic, j) 
            else:
                E_val = self.__compute_E_table(new_ic, j)
            makespan += E_val[0]
            self._E[(new_ic, j)] = E_val
        makespan += ((R + delta * self._C + self._delta * (j - ic) + start) * self._sumF[il])
        return makespan

    def __compute_E_table(self, ic, il):
        if il == self._n:
            return (0, self._n, 0)
        R = self.R
        if ic == 0:
            R = 0
        min_makespan = -1
        min_j = -1
        min_delta = -1

        for j in range(il + 1, self._n + 1):
            makespan_wo = self.__compute_makespan(ic, il, j, R, 0)
            delta = 1
            makespan = self.__compute_makespan(ic, il, j, R, 1)
            if makespan_wo < makespan:
                makespan = makespan_wo
                delta = 0

            if min_makespan == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_j = j
                min_delta = delta
        return (min_makespan, min_j, min_delta)

    def __compute_E_table_checkpoint(self, ic, il):
        if il == self._n:
            return (0, self._n, 0)
        R = self.R
        if ic == 0:
            R = 0
        min_makespan = -1
        min_j = -1

        for j in range(il + 1, self._n + 1):
            makespan = self.__compute_makespan(ic, il, j, R, 1)
            if min_makespan == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_j = j
        return (min_makespan, min_j, 1)

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        E_val = (0, 0)
        ic = 0
        il = 0
        total_walltime = 0
        while E_val[1] < self._n:
            E_val = self.compute_E_value(ic, il)
            start = 0
            if ic == 0:
                start = self._a
            self._request_sequence.append(
                (start + (E_val[1] - ic) * self._delta, E_val[2]))
            # if the reservation is checkpointed add the walltime
            if E_val[2] == 1:
                total_walltime += (start + (E_val[1] - ic) * self._delta)
            ic = (1 - E_val[2]) * ic + E_val[1] * E_val[2]
            il = E_val[1]

        time_left = self._b - (self._request_sequence[-1][0] + total_walltime)
        if time_left != 0:
            self._request_sequence.append((time_left, 0))

        return self._request_sequence

    def compute_E_value(self, ic, il):
        if (ic, il) in self._E:
            return self._E[(ic, il)]
        if self.always_checkpoint:
            E_val = self.__compute_E_table_checkpoint(ic, il) 
        else:
            E_val = self.__compute_E_table(ic, il)
        self._E[(ic, il)] = E_val
        return E_val
