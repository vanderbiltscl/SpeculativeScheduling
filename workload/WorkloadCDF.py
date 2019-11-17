import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

class WorkloadCDF():
    def __init__(self, data, cost_model=None, interpolation_model=None,
                 verbose=False):
        self.verbose = verbose
        self.best_fit = None
        self.fit_model = None
        self.discrete_data = None
        self.discrete_cdf = None
        if len(data) > 0:
            self.set_workload(data)
        if interpolation_model is not None:
            self.set_interpolation_model(interpolation_model)

    def set_workload(self, data):
        self.data = data
        self.discrete_data, self.discrete_cdf = self.compute_discrete_cdf()
        self.lower_limit = min(data)
        self.upper_limit = max(data)
        self.best_fit = None

    def set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        self.best_fit = None
    
    def add_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model.append(interpolation_model)
        else:
            self.fit_model += interpolation_model
        self.best_fit = None

    # best_fit has the format returned by the best_fit functions in the
    # interpolation model: [distr, params] or [order, params], ['log', params]
    def set_best_fit(self, best_fit):
        self.best_fit = best_fit

    def get_best_fit(self):
        if self.best_fit is None:
            self.compute_best_fit()
        return self.best_fit

    def compute_discrete_cdf(self):
        assert (self.data is not None),\
            'Data needs to be set to compute the discrete CDF'

        if self.discrete_cdf is not None and self.discrete_data is not None:
            return self.discrete_data, self.discrete_cdf

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
        cdf = [i * 1 / len(cdf) for i in cdf]
        for i in range(1, len(cdf)):
            cdf[i] += cdf[i-1]
        # normalize the cdf
        for i in range(len(cdf)):
            cdf[i] /= cdf[-1]

        return discret_data, cdf

    def compute_best_cdf_fit(self):
        assert (len(self.fit_model)>0), "No fit models available"

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

    def get_interpolation_cdf(self, all_data, best_fit):
        if self.best_fit is None:
            self.compute_best_fit()
        return self.fit_model[self.best_fit_index].get_discrete_cdf(
            all_data, best_fit)

#-------------
# Classes for defining how the interpolation will be done
#-------------

class InterpolationModel():
    # define the format of the return values for the get_best_fit functions
    def get_empty_fit(self):
        return (-1, -1, np.inf)


class FunctionInterpolation(InterpolationModel):
    # function could be any function that takes one parameter (e.g. log, sqrt)
    def __init__(self, function, order=1):
        self.fct = function
        self.order = order

    def get_interpolation_value(self, params, x):
        # for now just return the function including the decreasing areas
        return np.polyval(params, self.fct(x))

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

    def __init__(self, max_order=10):
        self.max_order = max_order

    def get_discrete_cdf(self, data):
        all_data = np.unique(data)
        all_cdf = [min(1,np.polyval(best_fit_poly[1], d)) for d in all_data]
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
    def __init__(self, data, list_of_distr=[]):
        self.distr = list_of_distr
        self.data = data

    def get_best_fit(self, x, y): 
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