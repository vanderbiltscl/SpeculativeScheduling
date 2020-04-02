import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

class Workload():
    def __init__(self, data, interpolation_model=None,
                 verbose=False):
        self.verbose = verbose
        self.best_fit = None
        self.fit_model = None
        self.discrete_data = None
        self.discrete_cdf = None
        self.default_interpolation = True
        
        assert (len(data) > 0), "Invalid data provided"
        self.__set_workload(data)
        if interpolation_model is not None:
            self.__set_interpolation_model(interpolation_model)
            self.default_interpolation = False
        elif len(data) < 100:
            self.__set_interpolation_model(DistInterpolation(data))

    def __set_workload(self, data):
        self.data = data
        self.__compute_discrete_cdf()
        self.best_fit = None

    def __set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        if len(self.fit_model)==0:
            self.fit_model = None
            return -1
        best_fit = self.__compute_best_cdf_fit()
        return best_fit
    
    # Function that returns the best fit (for debug or printing purposes)
    def get_best_fit(self):
        if self.best_fit is None:
            self.__compute_best_cdf_fit()
        return self.best_fit

    def __compute_discrete_cdf(self):
        assert (self.data is not None),\
            'Data needs to be set to compute the discrete CDF'

        discret_data = sorted(self.data)
        cdf = [1 for _ in self.data]
        todel = []
        for i in range(len(self.data) - 1):
            if discret_data[i] == discret_data[i + 1]:
                todel.append(i)
                cdf[i + 1] += cdf[i]
        todel.sort(reverse=True)
        for i in todel:
            del discret_data[i]
            del cdf[i]
        cdf = [i * 1. / len(cdf) for i in cdf]
        for i in range(1, len(cdf)):
            cdf[i] += cdf[i-1]
        # normalize the cdf
        for i in range(len(cdf)):
            cdf[i] /= cdf[-1]

        self.discrete_data = discret_data
        self.discrete_cdf = cdf
        self.cdf = cdf
        return discret_data, cdf

    def __compute_best_cdf_fit(self):
        if self.fit_model is None:
            return -1
        
        # set dicrete data and cdf to the original ones
        self.__compute_discrete_cdf()

        best_fit = self.fit_model[0].get_empty_fit()
        best_i = -1
        for i in range(len(self.fit_model)):
            fit = self.fit_model[i].get_best_fit(
                self.discrete_data, self.discrete_cdf)
            if fit[2] < best_fit[2]:
                best_fit = fit
                best_i = i
        self.best_fit = best_fit
        self.best_fit_index = best_i
        return best_i

    def get_interpolation_cdf(self, all_data):
        if self.best_fit is None:
            self.__compute_best_cdf_fit()
        self.discrete_data, self.cdf = self.fit_model[
            self.best_fit_index].get_discrete_cdf(all_data, self.best_fit)
       
        return self.discrete_data, self.cdf
    
    def compute_cdf(self, data=None):
        if data is None:
            data = self.data
        if self.fit_model is not None:
            self.get_interpolation_cdf(data)
        else:
            self.__compute_discrete_cdf()
        return self.discrete_data, self.cdf

    def compute_request_sequence(self, max_request=-1,
                                 alpha=1, beta=0, gamma=0):
        self.compute_cdf()
        if max_request == -1:
            max_request = max(self.discrete_data)
        handler = RequestSequence(max_request, self.discrete_data,
                                  self.cdf, alpha=alpha, beta=beta,
                                  gamma=gamma)
        return handler.compute_request_sequence()

    def compute_sequence_cost(self, sequence, data):
        handler = LogDataCost(sequence)
        return handler.compute_cost(data)

#-------------
# Classes for defining how the interpolation will be done
#-------------

class InterpolationModel():
    # define the format of the return values for the get_best_fit functions
    def get_empty_fit(self):
        return (-1, -1, np.inf)
    
    def discretize_data(self, data, discrete_steps):
        step = (max(data) - min(data)) / discrete_steps
        return np.unique(
            [min(data) + i * step for i in range(discrete_steps)] \
            + [max(data)])


class FunctionInterpolation(InterpolationModel):
    # function could be any function that takes one parameter (e.g. log, sqrt)
    def __init__(self, function, order=1, discretization=500):
        self.fct = function
        self.order = order
        self.discrete_steps =  discretization - 1

    def get_discrete_cdf(self, data, best_fit):
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [max(0, min(1, np.polyval(best_fit, self.fct(d)))) for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1,len(all_cdf)):
            if all_cdf[i] < all_cdf[i-1]:
                all_cdf[i] = all_cdf[i-1]
        return all_data, all_cdf

    # fitting the function a + b * fct
    def get_best_fit(self, x, y):
        try:
            params = np.polyfit(self.fct(x), y, self.order)
        except:
            return self.get_empty_fit()
        err = np.sum((np.polyval(params, self.fct(x)) - y)**2)
        return (self.order, params, err)
    
    def get_cdf(self, x, params):
        return np.polyval(params, self.fct(x))


class PolyInterpolation(InterpolationModel):

    def __init__(self, max_order=10, discretization=500):
        self.max_order = max_order
        self.discrete_steps =  discretization - 1

    def get_discrete_cdf(self, data, best_fit):
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [max(0, min(1, np.polyval(best_fit, d))) for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1,len(all_cdf)):
            if all_cdf[i] < all_cdf[i-1]:
                all_cdf[i] = all_cdf[i-1]
        return all_data, all_cdf

    def get_best_fit(self, x, y):
        empty = self.get_empty_fit()
        best_err = empty[2]
        best_params = empty[1]
        best_order = empty[0]
        for order in range(1, self.max_order):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    params = np.polyfit(x, y, order)
                except:
                    break

                err = np.sum((np.polyval(params, x) - y)**2)
                if err < best_err:
                    best_order = order
                    best_params = params
                    best_err = err
        
        return (best_order, best_params, best_err)


class DistInterpolation(InterpolationModel):
    def __init__(self, data, list_of_distr=[], discretization=500):
        self.distr = list_of_distr
        self.data = data
        self.discrete_steps = discretization - 1
    
    def get_discrete_cdf(self, data, best_fit):
        arg = best_fit[1][:-2]
        loc = best_fit[1][-2]
        scale = best_fit[1][-1]
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [best_fit[0].cdf(d, loc=loc, scale=scale, *arg) for d in all_data]
        return all_data, all_cdf

    def get_best_fit(self, x, y): 
        dist_list = self.distr
        if len(dist_list)==0:
            # list of distributions to check
            dist_list = [        
                st.alpha,st.beta,st.cosine,st.dgamma,st.dweibull,st.exponnorm,
                st.exponweib,st.exponpow,st.genpareto,st.gamma,st.halfnorm,
                st.invgauss,st.invweibull,st.laplace,st.loggamma,st.lognorm,
                st.lomax,st.maxwell,st.norm,st.pareto,#st.pearson3,st.rice,
                st.truncexpon,st.truncnorm,st.uniform,st.weibull_min,
                st.weibull_max]

        # Best holders
        best_distribution = -1
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # estimate distribution parameters from data
        for distribution in dist_list:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(self.data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
            except Exception:
                pass

        return (best_distribution, best_params, best_sse)

#-------------
# Classes for computing the sequence of requests
#-------------

class RequestSequence():
    ''' Sequence that optimizes the total makespan of a job for discret
    values (instead of a continuous space) '''

    def __init__(self, max_value, discrete_values, cdf_values,
                 alpha=1, beta=1, gamma=0):
        # default pay what you reserve (AWS model) (alpha 1 beta 0 gamma 0)
        # pay what you use (HPC model) would be alpha 1 beta 1 gamma 0
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

        assert (len(discrete_values) > 0), "Invalid input"
        assert (len(discrete_values) == len(cdf_values)), "Invalid cdf"
        assert (max_value >= max(discrete_values)), "Invalid max value"
        
        self.discret_values = discrete_values
        self.__cdf = cdf_values
        self.upper_limit = max_value
        self._E = {}
        self._request_sequence = []
        
        self.__sumF = self.get_discrete_sum_F()
        self.__sumFV = self.compute_FV()
        E_val = self.compute_E_value(-1)
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def compute_F(self, vi):
        fi = self.__cdf[vi]
        if vi > 0:
            fi -= self.__cdf[vi-1]
        return fi / self.__cdf[-1]

    def compute_FV(self):
        FV = 0
        for i in range(len(self.discret_values)):
            FV += (self.discret_values[i] * self.compute_F(i))
        return FV

    # Compute sumF[i] as sum_k=i,n f[k]
    def get_discrete_sum_F(self):
        sumF = (len(self.discret_values) + 1) * [0]
        for k in range(len(self.discret_values) - 1, -1, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
        return sumF

    def makespan_init_value(self, i, j):
        init = float(self.__alpha * self.discret_values[j] + self.__gamma) \
               * self.__sumF[i + 1]
        init += self.__beta * self.discret_values[j] * self.__sumF[j + 1]
        return init

    def compute_E_table(self, first):
        self._E[len(self.discret_values) - 1] = (self.__beta * self.__sumFV,
                                                 len(self.discret_values) - 1)
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            if i in self._E:
                continue
            min_makespan = -1
            min_request = -1
            for j in range(i + 1, len(self.discret_values)):
                makespan = self.makespan_init_value(i, j)
                makespan += self._E[j][0]

                if min_makespan == -1 or min_makespan >= makespan:
                    min_makespan = makespan
                    min_request = j
            self._E[i] = (min_makespan, min_request)
        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        j = 0
        E_val = self.compute_E_value(j)
        while E_val[1] < len(self.discret_values) - 1:
            self._request_sequence.append((self.discret_values[E_val[1]], ))
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)

        self._request_sequence.append((self.discret_values[E_val[1]], ))
        if self._request_sequence[-1][0] != self.upper_limit:
            self._request_sequence.append((self.upper_limit, ))
        
        return self._request_sequence

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        E_val = self.compute_E_table(i)
        self._E[i] = E_val
        return E_val

#-------------
# Classes for defining how the cost is computed
#-------------

class SequenceCost():
    def compute_cost(self, data):
        return -1

class LogDataCost(SequenceCost):

    def __init__(self, sequence):
        # if entries in the sequence use a multi information format
        # extract only the execution time
        if not isinstance(sequence[0], tuple):
            self.sequence = sequence
        else:
            self.sequence = [i[0] for i in sequence]

    def compute_cost(self, data):
        cost = 0
        for instance in data:
            # get the sum of all the values in the sequences <= walltime
            cost += sum([i for i in self.sequence if i < instance])
            # add the first reservation that is >= current walltime
            idx = 0
            if len(self.sequence) > 1:
                idx_list = [i for i in range(1,len(self.sequence)) if
                            self.sequence[i-1] < instance and
                            self.sequence[i] >= instance]
                if len(idx_list) > 0:
                    idx = idx_list[0]
            # if the reservation is fixed the cost += self.sequence[idx]
            cost += instance
        cost = cost / len(data)
        return cost
