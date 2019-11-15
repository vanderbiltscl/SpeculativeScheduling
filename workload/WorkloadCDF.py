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