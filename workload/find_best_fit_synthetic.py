import numpy as np
import pandas as pd
import warnings
import scipy.integrate as integrate
from scipy.stats import truncnorm
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

import OptimalSequence

sample_count = list(range(50, 501, 50))
bins = 100
current_distribution = st.truncnorm
lower_bound = 0
upper_bound = 20
mu = 8
sigma = 2
upper_bound = (upper_bound - mu) / sigma
lower_bound = (lower_bound - mu) / sigma

def func_exp(x, a, b, c):
    return a * np.exp(b * x) + c

def discreet_cdf(x, histogram, histogram_counts):
    # find the position in the histogram closest to x
    cum_histogram = np.cumsum(histogram_counts)
    min_dist = np.abs(x - histogram[0])
    count = cum_histogram[0]
    for bins in range(len(histogram)):
        dist = np.abs(x - histogram[bins])
        if dist < min_dist:
            min_dist = dist
            count = cum_histogram[bins]
    return count

def compute_cost(cdf):
    handler = OptimalSequence.TOptimalSequence(min(walltimes), max(walltimes), cdf, discret_samples=500)
    sequence = handler.compute_request_sequence()
    # Compute the expected makespan (MS)
    MS = sum([sequence[i+1]*cdf(sequence[i] for i in range(len(sequence)-1))])
    return MS

def get_cdf(pdf, start, x):
    return integrate.quad(pdf, start, x, epsrel=1.49e-05)[0]

def best_fit_distribution(x, y, bins, distr=[]): 
    distr_list = distr
    if len(distr_list)==0:
        # Distributions to check
        dist_list = [        
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    progressbar_width = len(dist_list)
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

        sys.stdout.write("=")
        sys.stdout.flush()
        
    sys.stdout.write("]\n")
    return (best_distribution, best_params, best_sse)

def get_pdf(data, params, bins):
    f = np.poly1d(params)
    # compute the integral from xp[0] to xp[-1]
    area = integrate.quad(f, data[0], data[-1], epsrel=1.49e-05)
    if np.isclose(area[0], 1, rtol=1e-03):
        return f
    params = [i/area[0] for i in params]
    return np.poly1d(params)

def generate_workload(distribution, samples_count):
    all_data = distribution.rvs(lower_bound, upper_bound,
                             loc=mu, scale=sigma, size=samples_count)
    return pd.Series(all_data)

if __name__ == '__main__':
    
    df = pd.DataFrame(columns=["Function", "Parameters", "Cost", "Sample"])

    for count in sample_count:
        data = generate_workload(current_distribution, count)
        print("Number samples: %d" %(count))
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        y = [i/sum(y) for i in y]
        print("-----------")
        print("Optimal FIT")
        distribution = current_distribution
        cdf = lambda val: distribution.cdf(val, loc=mu, scale=sigma, *arg)
        cost = compute_cost(cdf)
        df.loc[len(df)] = ["Optimal", distribution.name, cost, count]

        print("-----------")
        print("FIT Dataset")
        print("Using the discreet distribution ...")
        cdf_just_training = lambda val: discreet_cdf(val, x, y)
        cost_discreet = compute_cost(cdf_just_training)
        df.loc[len(df)] = ["Discreet", "", cost_discreet, count]

        print("Using the continuous distribution ...")
        print("-- Polynomial fit")
        best_order = 0
        best_cost = -1
        for order in range(1,5):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    z = np.polyfit(x, y, order)
                except:
                    break
                err = np.sum((np.polyval(z, x) - y)**2)
                try:
                    pdf = get_pdf(x,z,bins)
                    cdf = lambda val: get_cdf(pdf, x[0], val)
                    print("---- Polynomial Order", order)
                    cost = compute_cost(cdf)
                    if best_cost == -1 or cost < best_cost:
                        best_cost = cost
                        best_order = "Polynomial "+str(order)
                except:
                    continue

        print("-- Distribution fit")
        distribution, params, err = best_fit_distribution(x, y, bins)
        arg = params[:-2]
        cdf = lambda val: distribution.cdf(val, loc=params[-2], scale=params[-1], *arg)
        cost = compute_cost(cdf)
        if cost < best_cost:
            best_cost = cost
            best_order = distribution.name

        print("-- Exponantial fit")
        popt, pcov = curve_fit(func_exp, x, y)
        pdf = lambda val: func_exp(val, *popt)
        cdf = lambda val: get_cdf(pdf, x[0], val)
        cost = compute_cost(cdf)
        if cost < best_cost:
            best_cost = cost
            best_order = "Exponential"
        df.loc[len(df)] = ["Continuous", best_order, best_cost, count]
        
        print("-----------")
        print("Semi-clairvoyant FIT")
        distribution, params, err = best_fit_distribution(x, y, bins, distr=[current_distribution])
        if err==np.inf:
            df.loc[len(df)] = ["Semi-clairvoyant", distribution.name, -1, count]
        else:
            arg = params[:-2]
            cdf = lambda val: distribution.cdf(val, loc=params[-2], scale=params[-1], *arg)
            cost = compute_cost(cdf)
            df.loc[len(df)] = ["Semi-clairvoyant", distribution.name, cost, count]

        print("-----------")
        print("Clairvoyant FIT")
        distribution = current_distribution
        mu_guess = np.mean(data)
        sigma_guess = np.std(data)
        cdf = lambda val: distribution.cdf(val, loc=mu_guess, scale=sigma_guess, *arg)
        cost = compute_cost(cdf)
        df.loc[len(df)] = ["Clairvoyant", distribution.name, cost, count]
    
    print(df.head())
    with open("ACCRE/"+dataset+"_synthetic.csv", 'a') as f:
        df.to_csv(f, header=True)
        