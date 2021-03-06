import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

import OptimalSequence

class WorkloadFit():
    def __init__(self, data, cost_model=None, interpolation_model=None,
                 bins=100, verbose=False):
        self.verbose = verbose
        self.best_fit = None
        self.fit_model = None 
        if len(data) > 0:
            self.set_workload(data, bins=bins)
            # values will be overwritten by the cost model if provided
            self.lower_limit = min(data)
            self.upper_limit = max(data)
        if cost_model is not None:
            self.set_cost_model(cost_model)
        if interpolation_model is not None:
            self.set_interpolation_model(interpolation_model)
        self.sequence_model = OptimalSequence.TOptimalSequence
        # default argument for computing the sequence: no split intervals
        self.sequece_args = [500]

    def set_workload(self, data, bins=100):
        self.data = data
        self.y, self.x = np.histogram(data, bins=bins, density=True)
        self.x = (self.x + np.roll(self.x, -1))[:-1] / 2.0
        self.best_fit = None
        
    def set_cost_model(self, cost_model):
        self.cost_model = cost_model
        limits = self.cost_model.get_limits()
        self.lower_limit = limits[0]
        self.upper_limit = limits[1]

    def set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        self.best_fit = None
    
    def change_default_sequence_model(self, sequence_model, args=[]):
        self.sequence_model = sequence_model
        self.sequece_args = [self.sequece_args[0]] + args

    def add_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model.append(interpolation_model)
        else:
            self.fit_model += interpolation_model
        self.best_fit = None

    # best_fit has the format returned by the best_fit functions in the
    # interpolation model: [distr, params] or [order, params]
    def set_best_fit(self, best_fit):
        self.best_fit = best_fit

    def __compute_discrete_cdf(self):
        # sort data and merge entries with the same value
        discret_data = sorted(self.data)
        self.cdf = [1 for _ in self.data]
        todel = []
        for i in range(len(self.data)-1):
            if discret_data[i] == discret_data[i + 1]:
                todel.append(i)
                self.cdf[i + 1] += self.cdf[i]
        todel.sort(reverse=True)
        for i in todel:
            del discret_data[i]
            del self.cdf[i]
        self.cdf = [i*1./len(self.cdf) for i in self.cdf]
        return discret_data

    def compute_discrete_sequence(self):
        discret_data = self.__compute_discrete_cdf()
        handler = OptimalSequence.TODiscretSequence(
            self.upper_limit, discret_data, self.cdf)
        sequence = handler.compute_request_sequence()
        if self.verbose:
            print(sequence)
        return sequence

    def __interpolation_sequence(self, cdf, limits=-1):
        if limits == -1:
            limits = [self.lower_limit, self.upper_limit]
        handler = self.sequence_model(
            limits[0], limits[1], cdf, *self.sequece_args)
        sequence = handler.compute_request_sequence()
        if self.verbose:
            print(sequence)
        return sequence

    def compute_discrete_cost(self):
        sequence = self.compute_discrete_sequence()
        cost = self.cost_model.compute_sequence_cost(sequence)
        return cost

    def compute_best_fit(self):
        best_fit = (-1, -1, np.inf)
        best_i = -1
        for i in range(len(self.fit_model)):
            fit = self.fit_model[i].get_best_fit(
                self.data, self.x, self.y)
            if fit[2] < best_fit[2]:
                best_fit = fit
                best_i = i
        self.best_fit = best_fit
        return best_i

    def compute_interpolation_sequence(self):
        assert (self.fit_model is not None or self.best_fit is not None),\
            "No interpolation model provided"

        if self.best_fit is None:
            best_idx = self.compute_best_fit() 
        if self.best_fit[0] == -1:
            print("Data cannot be fitted with the given interpolation model")
            return -1

        cdf = lambda val: self.fit_model[best_idx].get_cdf(
            self.lower_limit, self.upper_limit, val, self.best_fit)
        sequence = self.__interpolation_sequence(cdf)
        return sequence

    def compute_interpolation_cost(self):
        sequence = self.compute_interpolation_sequence()
        cost = self.cost_model.compute_sequence_cost(sequence)
        return cost
    
    def compute_cdf_cost(self, cdf):
        sequence = self.__interpolation_sequence(cdf)
        cost = self.cost_model.compute_sequence_cost(sequence)
        return cost

    def get_best_fit(self):
        if self.best_fit is None:
            self.compute_best_fit()
        return self.best_fit

#-------------
# Classes for defining how the interpolation will be done
#-------------

class InterpolationModel():
    def get_cdf(self, start, end, x, params):
        return 0

    def get_best_fit(self, data, x, y):
        return (-1, -1)


class DistInterpolation(InterpolationModel):

    def __init__(self, list_of_distr=[]):
        self.distr = list_of_distr

    def get_cdf(self, start, end, x, all_params):
        distribution = all_params[0]
        params = all_params[1]
        arg = params[:-2]
        if x >= end:
            return distribution.cdf(end, loc=params[-2],
                                    scale=params[-1], *arg)
        if x <= start:
            return distribution.cdf(start, loc=params[-2],
                                    scale=params[-1], *arg)
        return distribution.cdf(x, loc=params[-2], 
                                scale=params[-1], *arg)


    def get_best_fit(self, data, x, y): 
        dist_list = self.distr
        if len(dist_list)==0:
            # list of distributions to check
            dist_list = [        
                st.alpha,st.beta,st.cosine,st.dgamma,st.dweibull,st.exponnorm,
                st.exponweib,st.exponpow,st.genpareto,st.gamma,st.halfnorm,
                st.invgauss,st.invweibull,st.laplace,st.loggamma,st.lognorm,
                st.lomax,st.maxwell,st.norm,st.pareto,st.pearson3,st.rayleigh,
                st.rice,st.truncexpon,st.truncnorm,st.uniform,st.weibull_min,
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
                    params = distribution.fit(data)

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


class PolyInterpolation(InterpolationModel):

    def __init__(self, max_order=10):
        self.max_order = max_order

    def get_cdf(self, start, end, x, all_params):
        params = all_params[1]
        if x >= end:
            return integrate.quad(np.poly1d(params), start,
                                  end, epsrel=1.49e-05)[0]
        if x <= start:
            return 0
        return integrate.quad(np.poly1d(params), start,
                              x, epsrel=1.49e-05)[0]

    def get_best_fit(self, data, x, y):
        best_err = np.inf
        best_z = -1
        best_order = -1
        for order in range(1, self.max_order):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    z = np.polyfit(x, y, order)
                except:
                    break

                err = np.sum((np.polyval(z, x) - y)**2)
                if err < best_err:
                    best_order = order
                    best_z = z
                    best_err = err
        
        return (best_order, best_z, best_err)


#-------------
# Classes for defining how the cost is computed
#-------------

class SequenceCost():
    def compute_sequence_cost(self, sequence):
        return -1

class LogDataCost(SequenceCost):

    def __init__(self, testing_data):
        self.testing = testing_data

    def compute_sequence_cost(self, sequence):
        cost = 0
        for instance in self.testing:
            # get the sum of all the values in the sequences <= walltime
            cost += sum([i[0] for i in sequence if i[0] < instance])
            # add the first reservation that is >= current walltime
            idx = 0
            if len(sequence) > 1:
                idx_list = [i for i in range(1,len(sequence)) if
                            sequence[i-1][0] < instance and
                            sequence[i][0] >= instance]
                if len(idx_list) > 0:
                    idx = idx_list[0]
            cost += sequence[idx][0]
        cost = cost / len(self.testing)
        return cost
    
    def get_limits(self):
        return [min(self.testing), max(self.testing)]


class SyntheticDataCost(SequenceCost):

    def __init__(self, cdf_function, limits):
        self.cdf = cdf_function
        self.limits = limits

    def compute_sequence_cost(self, sequence):
        # for all sequences < lower_limit consider the cdf = 0
        cost = sum([sequence[i][0] for i in range(len(sequence)-1)
                    if sequence[i][0] <= self.limits[0]])
        # the cost is computed based on the original distribution
        # normalized so that cdf(upper_limit) is 1
        scale = self.cdf(self.limits[1])
        cost = sum([sequence[i+1][0]*(1-self.cdf(sequence[i])/scale)
                    for i in range(len(sequence)-1)
                    if sequence[i][0] > self.limits[0]])
        cost += sequence[0][0]
        return cost
    
    def get_limits(self):
        return self.limits
