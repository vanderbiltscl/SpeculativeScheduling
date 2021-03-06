import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys
from enum import IntEnum

class CRStrategy(IntEnum):
    ''' Enumeration class to hold the types of Checkpoint/Restart
        strategies available to the application '''

    NeverCheckpoint = 0
    AlwaysCheckpoint = 1
    AdaptiveCheckpoint = 2


class ClusterCosts():
    ''' Class for storing the costs of running on the cluster
        For a job of actual length t, a reservation of lenth t1
        will cost alpha * t + beta * min(t, t1) + gamma '''

    def __init__(self, reservation_cost=1, utilization_cost=1, deploy_cost=0,
                 checkpoint_cost=1, restart_cost=1):
        # default pay what you reserve (AWS model) (alpha 1 beta 0 gamma 0)
        # pay what you use (HPC model) would be alpha 1 beta 1 gamma 0
        self.alpha = reservation_cost
        self.beta = utilization_cost
        self.gamma = deploy_cost
        self.C = checkpoint_cost
        self.R = restart_cost


class ResourceEstimator():
    ''' Class used to generate the sequence of resource requests
        needed to be used for application submissions '''

    def __init__(self, past_runs, interpolation_model=None,
                 CR_strategy=CRStrategy.NeverCheckpoint, verbose=False):
        self.verbose = verbose
        self.fit_model = None
        self.discrete_data = None
        self.default_interpolation = True
        self.checkpoint_strategy = CR_strategy

        assert (len(past_runs) > 0), "Invalid log provided"
        self.__set_workload(past_runs)
        if interpolation_model is not None:
            self.set_interpolation_model(interpolation_model)
            self.default_interpolation = False
        elif len(past_runs) < 100:
            self.set_interpolation_model(DistInterpolation())

    ''' Private functions '''
    def __set_workload(self, past_runs):
        self.data = past_runs
        self.best_fit = None

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
        self.cdf = cdf
        return discret_data, cdf

    def __compute_best_fit(self):
        if self.fit_model is None:
            return -1
        
        # set dicrete data and cdf to the original ones
        ddata, dcdf = self.__compute_discrete_cdf()

        best_fit = self.fit_model[0].get_empty_fit()
        best_i = -1
        for i in range(len(self.fit_model)):
            fit = self.fit_model[i].get_best_fit(
                ddata, dcdf)
            if fit[2] < best_fit[2]:
                best_fit = fit
                best_i = i
        self.best_fit = best_fit
        self.best_fit_index = best_i
        return best_i

    def __get_interpolation_cdf(self, all_data):
        if self.best_fit is None:
            self.__compute_best_fit()
        self.discrete_data, self.cdf = self.fit_model[
            self.best_fit_index].get_discrete_cdf(all_data, self.best_fit)
       
        return self.discrete_data, self.cdf

    def __get_sequence_type(self):
        if self.checkpoint_strategy == CRStrategy.AdaptiveCheckpoint:
            return CheckpointSequence
        # by default return request times when checkpoint is not availabe
        return RequestSequence

    ''' Functions used for debuging or printing purposes '''
    # Function that returns the best fit 
    def _get_best_fit(self):
        if self.best_fit is None:
            self.__compute_best_fit()
        return self.best_fit

    # Function that computes the cdf
    def _compute_cdf(self):
        # if all runs have the same execution time
        if all(elem == self.data[0] for elem in self.data):
            self.discrete_data = [self.data[0]]
            self.cdf = [1]
            return

        if self.fit_model is not None:
            self.__get_interpolation_cdf(self.data)
            valid = self._check_cdf_validity(self.cdf)
            if valid:
                return
        
        self.__compute_discrete_cdf()

    def _get_cdf(self):
        self._compute_cdf()
        return self.discrete_data, self.cdf

    # Function to check if the cdf is [0,1] and strictly increasing
    def _check_cdf_validity(self, cdf):
        test = all(elem >= 0 and elem <= 1 for elem in cdf)
        if not test:
            return False
        return all(cdf[i - 1] < cdf[i] for i in range(1, len(cdf)))

    ''' Public functions '''
    def set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        self.best_fit = None
        if len(self.fit_model)==0:
            self.fit_model = None
            return -1

    def set_CR_strategy(self, CR_strategy):
        self.checkpoint_strategy = CR_strategy
    
    def compute_request_sequence(self, cluster_cost=None):
        if cluster_cost == None:
            cluster_cost = ClusterCosts()
        self._compute_cdf()
        sequence_type = self.__get_sequence_type()
        handler = sequence_type(self.discrete_data, self.cdf, cluster_cost)
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
        all_cdf = [max(0, min(1, np.polyval(best_fit[1], d))) for d in all_data]
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
    def __init__(self, list_of_distr=[], discretization=500):
        self.distr = list_of_distr
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
                    params = distribution.fit(x)

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

class DefaultRequests():
    ''' Default class for generating the sequence of requests given 
    an application behavior and system properties '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost):
        self._alpha = cluster_cost.alpha
        self._beta = cluster_cost.beta
        self._gamma = cluster_cost.gamma

        assert (len(discrete_values) > 0), "Invalid input"
        assert (len(discrete_values) == len(cdf_values)), "Invalid cdf"

        self.discret_values = discrete_values
        self._cdf = cdf_values
        self.upper_limit = max(self.discret_values)
        self._E = {}
        self._request_sequence = []
        
        self._sumF = self.get_discrete_sum_F()
        self._sumFV = self.compute_FV()

    def compute_F(self, vi):
        fi = self._cdf[vi]
        if vi > 0:
            fi -= self._cdf[vi-1]
        return fi / self._cdf[-1]

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

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        E_val = self.compute_E_table(i)
        self._E[i] = E_val
        return E_val


class RequestSequence(DefaultRequests):
    ''' Sequence that optimizes the total makespan of a job for discret
    values (instead of a continuous space) '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost):

        super().__init__(discrete_values, cdf_values, cluster_cost)
        E_val = self.compute_E_value(0)
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def makespan_init_value(self, i, j):
        init = float(self._alpha * self.discret_values[j] + self._gamma) \
               * self._sumF[i]
        init += self._beta * self.discret_values[j] * self._sumF[j + 1]
        return init

    def compute_E_table(self, first):
        self._E[len(self.discret_values)] = (self._beta * self._sumFV,
                                             len(self.discret_values) - 1)
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            min_makespan = -1
            min_request = -1
            for j in range(i, len(self.discret_values)):
                makespan = self.makespan_init_value(i, j)
                makespan += self._E[j + 1][0]

                if min_makespan == -1 or min_makespan >= makespan:
                    min_makespan = makespan
                    min_request = j
            self._E[i] = (min_makespan, min_request, 0)
        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        j = 0
        E_val = self.compute_E_value(j)
        while E_val[1] < len(self.discret_values) - 1:
            self._request_sequence.append((self.discret_values[E_val[1]], E_val[2]))
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)

        self._request_sequence.append((self.discret_values[E_val[1]], E_val[2]))
        if self._request_sequence[-1][0] != self.upper_limit:
            self._request_sequence.append((self.upper_limit, 0))

        return self._request_sequence


class CheckpointSequence(DefaultRequests):
    ''' Sequence that optimizes the total makespan of a job when the
    application or system is capable of taking checkpoints '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost):

        self._C = cluster_cost.C
        self._R = cluster_cost.R
        super().__init__(discrete_values, cdf_values, cluster_cost)
        E_val = self.compute_E_value((0, 0))
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def makespan_init_value(self, ic, il, j, delta, R):
        vic = self.discret_values[ic]
        if R == 0:
            vic = 0

        init = (self._alpha * (R + self.discret_values[j] - vic + \
                delta * self._C) + self._beta * R + self._gamma) \
                * self._sumF[il + 1]
        init += self._beta * ((1 - delta) * (self.discret_values[j] - vic) \
                              + delta * self._C) * self._sumF[j + 1]
        return init

    def compute_E(self, ic, il, R):
        min_makespan = -1
        min_request = -1
        for j in range(il, len(self.discret_values) - 1):
            # makespan with checkpointing the last sequence (delta = 1)
            makespan = self.makespan_init_value(ic, il, j, 1, R)
            makespan += self._E[(j + 1, j + 1)][0]
            if min_makespan == -1 or min_makespan >= makespan:
                min_makespan = makespan
                min_request = j
                min_delta = 1

            # makespan without checkpointing the last sequence (delta = 0)
            makespan = self.makespan_init_value(ic, il, j, 0, R)
            makespan += self._E[(ic, j + 1)][0]
            if min_makespan == -1 or min_makespan >= makespan:
                min_makespan = makespan
                min_request = j
                min_delta = 0

        self._E[(ic, il)] = (min_makespan, min_request, min_delta)

    def compute_E_table(self, first):
        for ic in range(len(self.discret_values) - 1, -1, -1):
            self._E[(ic, len(self.discret_values) - 1)] = (
                self._beta * self._sumFV, len(self.discret_values) - 1, 0)

        for il in range(len(self.discret_values) - 2, -1, -1):
            for ic in range(len(self.discret_values) - 1, 0, -1):
                if (ic, il) in self._E:
                    continue
                self.compute_E(ic, il, self._R)
            self.compute_E(0, il, 0)

        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        ic = 0
        il = 0
        E_val = self.compute_E_value((ic, il))
        already_compute = 0
        while E_val[1] < len(self.discret_values) - 1:
            self._request_sequence.append(
                (self.discret_values[E_val[1]] - already_compute, E_val[2]))
            ic = (1 - E_val[2]) * ic + (E_val[1] + 1) * E_val[2]
            il = E_val[1] + 1
            if E_val[2] == 1:
                already_compute = self.discret_values[E_val[1]]
            E_val = self.compute_E_value((ic, il))

        self._request_sequence.append(
            (self.discret_values[E_val[1]] - already_compute, E_val[2]))
        return self._request_sequence

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
