import numpy as np
import pandas as pd
import warnings
import scipy.integrate as integrate
from scipy.stats import truncnorm
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

import OptimalSequence

sample_count = list(range(10, 101, 10)) + list(range(110, 511, 50))
bins = 100
current_distribution = st.expon
lower_limit = 0
upper_limit = 9
mu = 1
sigma = 1.5
upper_bound = 9
lower_bound = 0

def compute_cost_discret(data):
    # sort data and merge entries with the same value
    discret_data = sorted(data)
    cdf = [1 for _ in data]
    todel = []
    for i in range(len(data)-1):
        if discret_data[i] == discret_data[i + 1]:
            todel.append(i)
            cdf[i + 1] += cdf[i]
    todel.sort(reverse=True)
    for i in todel:
        del discret_data[i]
        del cdf[i]
    cdf = [i*1./len(cdf) for i in cdf]
    # compute cost
    handler = OptimalSequence.TODiscretSequence(upper_limit, discret_data, cdf)
    sequence = handler.compute_request_sequence()
    if verbose:
        print(sequence)
    
    # Compute the expected makespan (MS)
    MS = sum([sequence[i+1]*(1-(current_distribution.cdf(sequence[i], loc=mu, scale=sigma)/current_distribution.cdf(upper_limit, loc=mu, scale=sigma)))
              for i in range(len(sequence)-1)])
    MS += sequence[0]
    return MS

def compute_cost(cdf, limits = [lower_limit, upper_limit]):
    handler = OptimalSequence.TOptimalSequence(limits[0], limits[1], cdf, discret_samples=500)
    sequence = handler.compute_request_sequence()
    if sequence[-1]!=upper_limit:
        sequence[-1] = upper_limit
    if verbose:
        print(sequence)
    
    # Compute the expected makespan (MS)
    MS = sum([sequence[i+1]*(1-(current_distribution.cdf(sequence[i], loc=mu, scale=sigma)/current_distribution.cdf(upper_limit, loc=mu, scale=sigma)))
              for i in range(len(sequence)-1)])
    MS += sequence[0]
    # MS = sum([sequence[i+1]*cdf(sequence[i]) for i in range(len(sequence)-1)])
    return MS

def get_cdf(start, x, params):
    return integrate.quad(np.poly1d(params), start, x, epsrel=1.49e-05)[0]

def best_fit_distribution(data, x, y, bins, distr=[]): 
    dist_list = distr
    if len(dist_list)==0:
        # Distributions to check
        dist_list = [        
            st.alpha,st.beta,st.cosine,st.dgamma,st.dweibull,st.exponnorm,st.exponweib,
            st.exponpow,st.genpareto,st.gamma,st.halfnorm,st.invgauss,st.invweibull,
            st.laplace,st.loggamma,st.lognorm,st.lomax,st.maxwell,st.norm,st.pareto,
            st.pearson3,st.rayleigh,st.rice,st.truncexpon,st.truncnorm,st.uniform,
            st.weibull_min,st.weibull_max
        ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    progressbar_width = len(dist_list)
    if verbose:
        sys.stdout.write("[%s]" % ("." * progressbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progressbar_width + 1))
    # Estimate distribution parameters from data
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

        if verbose:
            sys.stdout.write("=")
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("]\n")
    return (best_distribution, best_params, best_sse)

def generate_workload(distribution, samples_count):
    all_data = distribution.rvs(loc=mu, scale=sigma, size=samples_count)
    return [i for i in all_data if i>lower_bound and i<upper_bound]

if __name__ == '__main__':
    verbose = False
    if len(sys.argv)>1:
        verbose = True

    df = pd.DataFrame(columns=["Function", "Parameters", "Cost", "Sample", "EML"])

    for count in sample_count:
        data = generate_workload(current_distribution, count)
        print("Number samples: %d %d" %(count, len(data)))
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        if verbose:
            print("-----------")
            print("Optimal FIT")
        distribution = current_distribution
        cdf = lambda val: distribution.cdf(val, loc=mu, scale=sigma)
        optimal_cost = compute_cost(cdf)
        df.loc[len(df)] = ["Optimal", distribution.name, optimal_cost, count, 0]

        if verbose:
            print("-----------")
            print("Discreet FIT")
        cost_discreet = compute_cost_discret(data)
        df.loc[len(df)] = ["Discreet", "", cost_discreet, count, (cost_discreet-optimal_cost)*1./optimal_cost]

        if verbose:
            print("-----------")
            print("Continuous FIT")
        
        best_order = 0
        best_cost = np.inf

        if verbose:
            print("-- Polynomial fit")
        best_err = np.inf
        best_z = -1
        for order in range(1,6):
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
        try:
            cdf = lambda val: get_cdf(x[0], val, best_z)
            cost = compute_cost(cdf, limits=[x[0], x[-1]])
            if cost < best_cost:
                best_cost = cost
                best_order = "Polynomial "+str(best_order)
        except:
            pass

        if verbose:
            print("-- Distribution fit")
        distribution, params, err = best_fit_distribution(data, x, y, bins)
        arg = params[:-2]
        # if the cdf of the first element is almost 1, this is not a good fit
        if not np.isclose(distribution.cdf(0, loc=params[-2], scale=params[-1], *arg), 1, rtol=1e-03):            
            cdf = lambda val: distribution.cdf(val, loc=params[-2], scale=params[-1], *arg)
            cost = compute_cost(cdf)
            if cost < best_cost:
                best_cost = cost
                best_order = distribution.name

        df.loc[len(df)] = ["Continuous", best_order, best_cost, count, (best_cost-optimal_cost)*1./optimal_cost]

        if verbose:
            print("-----------")
            print("Semi-clairvoyant FIT")
        distribution, params, err = best_fit_distribution(data, x, y, bins, distr=[current_distribution])
        if err!=np.inf:
            arg = params[:-2]
            cdf = lambda val: distribution.cdf(val, loc=params[-2], scale=params[-1], *arg)
            cost = compute_cost(cdf)
            df.loc[len(df)] = ["Semi-clairvoyant", distribution.name, cost, count, (cost-optimal_cost)*1./optimal_cost]

        if verbose:
            print("-----------")
            print("Clairvoyant FIT")
        #distribution = current_distribution
        mu_guess = np.mean(data)
        sigma_guess = np.std(data)
        #print(mu_guess, sigma_guess, arg)
        cdf = lambda val: current_distribution.cdf(val, loc=mu_guess, scale=sigma_guess)
        cost = compute_cost(cdf)
        df.loc[len(df)] = ["Clairvoyant", current_distribution.name, cost, count, (cost-optimal_cost)*1./optimal_cost]
        
    with open("./"+current_distribution.name+"_expon.csv", 'a') as f:
        df.to_csv(f, header=True)
        