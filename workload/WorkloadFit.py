import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

import OptimalSequence

class WorkloadFit():
    def __init__(self, data, cost_model, interpolation_model,
                 bins=100, verbose=False):
        self.verbose = verbose
        self.update_workload(data)
        self.cost_model = cost_model
        self.fit_model = interpolation_model

    def update_workload(self, data, bins=100):
        self.data = data
        self.y, self.x = np.histogram(data, bins=bins, density=True)
        self.x = (self.x + np.roll(self.x, -1))[:-1] / 2.0
        self.lower_limit = min(data)
        self.upper_limit = max(data)

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

    # TODO self.upperlimit aici e max(testing) nu max(training) pentru LogDataCost
    def compute_discrete_sequence(self):
        discret_data = self.__compute_discrete_cdf()
        handler = OptimalSequence.TODiscretSequence(
            self.upper_limit, discret_data, self.cdf)
        sequence = handler.compute_request_sequence()
        if self.verbose:
            print(sequence)
        return sequence

    # TODO limits in cazul asta e lower si upper limit al distributiei pentru Syntetic
    def compute_interpolation_sequence(self, cdf, limits=-1):
        if limits == -1:
            limits = [self.lower_limit, self.upper_limit]
        handler = OptimalSequence.TOptimalSequence(
            limits[0], limits[1], cdf, discret_samples=500)
        sequence = handler.compute_request_sequence()
        if sequence[-1]!=upper_limit:
            sequence[-1] = upper_limit
        if self.verbose:
            print(sequence)
        return sequence

    def compute_discrete_cost(self):
        sequence = self.compute_discrete_sequence()
        cost = self.cost_model.compute_sequence_cost(sequence)
        return cost

    def compute_interpolation_cost(self):
        params = self.fit_model.get_best_fit(self.data, self.x, self.y)
        if params[0] == -1:
            print("Data cannot be fitted with the given interpolation model")
            return -1

        print(params)
        cdf = lambda val: get_cdf(self.x[0], self.x[-1], val, params[1])
        sequence = self.compute_interpolation_sequence(cdf)
        cost = self.cost_model.compute_sequence_cost(sequence)
        return cost

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

    def get_cdf(self, start, end, x, params):
        if x >= end:
            return 1
        if x <= start:
            return 0
        scale = distribution.cdf(end, loc=params[-2], scale=params[-1], *arg)
        arg = params[:-2]
        return distribution.cdf(val, loc=params[-2], 
                                scale=params[-1], *arg)/scale


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

        if self.verbose:
            progressbar_width = len(dist_list)
            sys.stdout.write("[%s]" % ("." * progressbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (progressbar_width + 1))

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

            if self.verbose:
                sys.stdout.write("=")
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write("]\n")
        return (best_distribution, best_params, best_sse)


class PolyInterpolation(InterpolationModel):

    def __init__(self, max_order=10):
        self.max_order = max_order

    def get_cdf(self, start, end, x, params):
        if x >= end:
            return 1
        if x <= start:
            return 0
        return integrate.quad(np.poly1d(params), start, x, epsrel=1.49e-05)[0]

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
        
        return (best_order, best_z)


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
            cost += sum([i for i in sequence if i < instance])
            # add the first reservation that is >= current walltime
            idx = 0
            if len(sequence) > 1:
                idx_list = [i for i in range(1,len(sequence)) if
                            sequence[i-1] < instance and
                            sequence[i] >= instance]
                if len(idx_list) > 0:
                    idx = idx_list[0]
            cost += sequence[idx]
        cost = cost / len(self.testing)
        return cost


# TODO - need ro make sure cdf is not <0 or > 1
class SyntheticDataCost(SequenceCost):

    def __init__(self, data, distribution, params):
        self.distribution = distribution
        self.arg = params[:-2]
        self.loc = params[-2]
        self.mu = params[-1]

    def compute_sequence_cost(sequence):
        arg = [self.lower_bound, self.upper_bound]
        # Compute the cost based on the original distribution
        cost = sum([sequence[i+1]*(1-self.distribution.cdf(sequence[i],
                                                           loc=self.mu,
                                                           scale=self.sigma,
                                                           *self.arg))
                    for i in range(len(sequence)-1)])
        cost += sequence[0]
        return cost