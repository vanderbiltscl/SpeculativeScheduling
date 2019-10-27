import numpy as np
import pandas as pd
import warnings
import scipy.integrate as integrate
from scipy.stats import truncnorm
import scipy.stats as st
from scipy.optimize import curve_fit
import sys
import random

import OptimalSequence

optimal_sequence = []

def func_exp(x, a, b, c):
    return a * np.exp(b * x) + c

def compute_cost_discret(data, testing, optimal=False):
    global optimal_sequence
    sequence = optimal_sequence
    if not optimal or len(optimal_sequence) == 0:
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
        handler = OptimalSequence.TODiscretSequence(max(testing), discret_data, cdf)
        sequence = handler.compute_request_sequence()
        if optimal:
            optimal_sequence = sequence[:]
            print("Optimal sequence:", optimal_sequence)
    print(sequence)
    cost = 0
    for instance in testing:
        # get the sum of all the values in the sequences <= current walltime
        cost += sum([i for i in sequence if i < instance])
        # add the first reservation that is >= current walltime
        idx = 0
        if len(sequence) > 1:
            idx_list = [i for i in range(1,len(sequence)) if
                        sequence[i-1] < instance and sequence[i] >= instance]
            if len(idx_list) > 0:
                idx = idx_list[0]
        cost += sequence[idx]
    cost = cost / len(testing)
    return cost
    

def compute_cost(cdf, walltimes):
    handler = OptimalSequence.TOptimalSequence(min(walltimes), max(walltimes), cdf, discret_samples=500)
    sequence = handler.compute_request_sequence()
    print(sequence)
    cost = 0
    for instance in walltimes:
        # get the sum of all the values in the sequences <= current walltime
        cost += sum([i for i in sequence if i < instance])
        # add the first reservation that is >= current walltime
        idx = 0
        if len(sequence) > 1:
            idx_list = [i for i in range(1,len(sequence)) if
                        sequence[i-1] < instance and sequence[i] >= instance]
            if len(idx_list) > 0:
                idx = idx_list[0]
        cost += sequence[idx]
    cost = cost / len(walltimes)
    return cost

def get_cdf(start, x, params, end):
    if x >= end:
        return 1
    if x <= start:
        return 0
    return integrate.quad(np.poly1d(params), start, x, epsrel=1.49e-05)[0]

def best_fit_distribution(data, x, y, bins, distr=[], verbose=True): 
    dist_list = distr
    if len(dist_list)==0:
        # Distributions to check
        dist_list = [        
            st.alpha,st.beta,st.cosine,st.dgamma,st.dweibull,st.exponnorm,st.exponweib,
            st.exponpow,st.genpareto,st.gamma,st.halfnorm,st.invgauss,st.invweibull,
            st.laplace,st.loggamma,st.lognorm,st.lomax,st.maxwell,st.norm,st.pareto,
            #st.pearson3,
            st.rayleigh,#st.rice,
            st.truncexpon,st.truncnorm,st.uniform,
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

def load_workload():
    all_data = np.loadtxt("ACCRE/"+dataset+".out", delimiter=' ')
    print("Load %d entries from %s" %(len(all_data), "ACCRE/"+dataset+".out"))
    return pd.Series(all_data)

if __name__ == '__main__':
    train_perc = list(range(10, 511, 50))

    if len(sys.argv) < 2:
        print("Usage: %s dataset_file [number_of_bins]" %(sys.argv[0]))
        print("Example: %s ACCRE/Multi_Atlas.out 100" %(sys.argv[0]))
        exit()

    dataset = sys.argv[1].split("/")[1]
    dataset = dataset[:-4]

    all_data = load_workload()
    if len(all_data)<600:
        exit()
    df = pd.DataFrame(columns=["Function", "Parameters", "Cost", "Trainset", "EML"])
    bins = 100
    random_start = np.random.randint(0, len(all_data)-600)
    #random.shuffle(all_data) 

    for perc in train_perc:
        #print(perc, bins)
        test_cnt = perc #int(len(all_data)*perc/100)
        testing = all_data[:] #[test_cnt:]
        data =  all_data[random_start:random_start+test_cnt]
        #data = data.append(pd.Series(max(all_data)))
        #data = data.append(pd.Series(min(all_data)))
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        yall, xall = np.histogram(all_data, bins=bins, density=True)
        xall = (xall + np.roll(xall, -1))[:-1] / 2.0

        optimal_cost = compute_cost_discret(all_data, testing, optimal=True)
        df.loc[len(df)] = ["Optimal", bins, optimal_cost, perc, 0]
        
        print("----------")
        print("TRAIN SET ",perc)
        print("----------")
        print("Using the discreet distribution ...")
        cost_discreet = compute_cost_discret(data, testing)
        df.loc[len(df)] = ["Discreet", "", cost_discreet, perc, (cost_discreet-optimal_cost)*1./optimal_cost]

        print("Using the continuous distribution ...")
        best_order = 0
        best_cost = np.inf

        print("-- Polynomial fit")
        best_err = np.inf
        best_z = -1
        for order in range(1,10):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    z = np.polyfit(x, y, order)
                except:
                    break
                err = np.sum((np.polyval(z, x) - y)**2)
                if err < best_err:
                    cdf = lambda val: get_cdf(x[0], val, z, x[-1])
                    cost = compute_cost(cdf, testing)
                    #print("order",order, x[0], x[-1], cdf(min(testing)), cdf(max(testing)), cost)
                    if cost < best_cost:
                        best_order = order
                        best_z = z
                        best_err = err
        try:
            cdf = lambda val: get_cdf(x[0], val, best_z, x[-1])
            cost = compute_cost(cdf, testing)
            if cost < best_cost:
                best_cost = cost
                best_order = "Polynomial "+str(best_order)
        except:
            pass

        print("-- Distribution fit")
        distribution, params, err = best_fit_distribution(data, x, y, bins)
        arg = params[:-2]
        # if the cdf of the first element is almost 1, this is not a good fit
        if not np.isclose(distribution.cdf(0, loc=params[-2], scale=params[-1], *arg), 1, rtol=1e-03):            
            cdf = lambda val: distribution.cdf(val, loc=params[-2], scale=params[-1], *arg)
            cost = compute_cost(cdf, testing)
            if cost < best_cost:
                best_cost = cost
                best_order = distribution.name
        '''
        print("-- Exponantial fit")
        popt, pcov = curve_fit(func_exp, x, y)
        pdf = lambda val: func_exp(val, *popt)
        cdf = lambda val: integrate.quad(pdf, x[0], val, epsrel=1.49e-05)[0]
        cost = compute_cost(cdf, testing)
        if cost < best_cost:
                best_cost = cost
                best_order = "Exponential"
        '''
        df.loc[len(df)] = ["Continuous", best_order, best_cost, perc, (best_cost-optimal_cost)*1./optimal_cost]

    print(df)
    with open("ACCRE/"+dataset+"_eml.csv", 'a') as f:
        df.to_csv(f, header=True)
        