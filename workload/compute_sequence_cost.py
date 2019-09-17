import numpy as np
import pandas as pd
import scipy.stats as st
import sys
import random

import WorkloadFit

class LogDataRun():
    def load_workload(self, dataset):
        all_data = np.loadtxt("ACCRE/"+dataset+".out", delimiter=' ')
        print("Load %d runs from %s" %(len(all_data), "ACCRE/"+dataset+".out"))
        random.shuffle(all_data)
        self.all_data = all_data
        return all_data

    def get_training_set(self, test_cnt):
        random_start = np.random.randint(0, len(self.all_data)-test_cnt)
        self.testing = self.all_data[:]
        training = self.all_data[random_start:random_start+test_cnt]
        return training
    
    def get_cost_model(self):
        assert (self.testing is not None),\
            'get_train_test_sets needs to be called before get_cost_model'
        return WorkloadFit.LogDataCost(self.testing)
    
    def get_optimal_cost(self, wf):
        return wf.compute_discrete_cost()


class DistributionRuns():
    def set_cdf(self):
        self.cdf_function = lambda val: np.sum(
            [self.distribution.cdf(val, loc=self.mu[i],
                                   scale=self.sigma[i],
                                   *self.args[i])
             for i in range(len(self.mu))]) / len(self.mu)

    def get_training_set(self, samples_count):
        self.all_data = list([])
        for i in range(len(self.mu)):
            self.all_data += list(self.distribution.rvs(
                loc=self.mu[i], scale=self.sigma[i],
                size=samples_count, *self.args[i])) 
        return self.all_data

    def get_cost_model(self):
        last = len(self.mu) - 1
        return WorkloadFit.SyntheticDataCost(self.cdf_function,
                                             [self.lower_limit[0],
                                              self.upper_limit[last]])

    def get_optimal_cost(self, wf):
        assert (self.all_data is not None),\
            "Data needs to be set for optimal cost"
        assert (self.cdf_function is not None),\
            "CDF function needs to be set for optimal cost"

        wf.set_workload(self.all_data)
        return wf.compute_cdf_cost(self.cdf_function)


class DoubleNormRun(DistributionRuns):
    def load_workload(self):
        self.distribution = st.truncnorm
        self.lower_limit = [0.5, 6]
        self.upper_limit = [4, 20]
        self.mu = [4, 10]
        self.sigma = [2, 8]
        i = len(self.mu) - 1
        self.upper_bound = [
            (self.upper_limit[i] - self.mu[i]) / self.sigma[i]
            for i in range(len(self.mu))]
        self.lower_bound = [
            (self.lower_limit[i] - self.mu[i]) / self.sigma[i]
            for i in range(len(self.mu))]
        self.args = [(self.lower_bound[i], self.upper_bound[i])
                     for i in range(len(self.mu))]
        self.set_cdf()
        return []


class ExponentialRun(DistributionRuns):
    def load_workload(self):
        self.distribution = st.expon
        self.lower_limit = [0]
        self.upper_limit = [9]
        self.mu = [1]
        self.sigma = [1.5]
        self.upper_bound = [9]
        self.lower_bound = [0]
        self.args = [[]]
        self.set_cdf()
        return []


class TruncNormRun(DistributionRuns):
    def load_workload(self):
        self.distribution = st.truncnorm
        self.lower_limit = [0]
        self.upper_limit = [20]
        self.mu = [8]
        self.sigma = [2]
        self.upper_bound = [
            (self.upper_limit[i] - self.mu[i]) / self.sigma[i]
            for i in range(len(self.mu))]
        self.lower_bound = [
            (self.lower_limit[i] - self.mu[i]) / self.sigma[i]
            for i in range(len(self.mu))]
        self.args = [(self.lower_bound[i], self.upper_bound[i])
                     for i in range(len(self.mu))]
        self.set_cdf()
        return []


if __name__ == '__main__':
    train_perc = list(range(10, 511, 50))

    verbose = False
    if len(sys.argv) < 2:
        print("Usage: %s {dataset_file or distribution} [verbose]" %(sys.argv[0]))
        print("Accepted distributions: [truncnorm, expon, doublenorm]")
        print("Example dataset: %s ACCRE/Multi_Atlas.out" %(sys.argv[0]))
        exit()

    if len(sys.argv) > 2:
        verbose = True

    all_data = []
    if sys.argv[1]=="truncnorm":
        dataset = "truncnorm"
        scenario = TruncNormRun()
        all_data = scenario.load_workload()
    elif sys.argv[1]=="expon":
        dataset = "expon"
        scenario = ExponentialRun()
        all_data = scenario.load_workload()
    elif sys.argv[1]=="doublenorm":
        dataset = "doublenorm"
        scenario = DoubleNormRun()
        all_data = scenario.load_workload()

    else:
        dataset = sys.argv[1].split("/")[1]
        dataset = dataset[:-4]

        scenario = LogDataRun()
        all_data = scenario.load_workload(dataset)
        if len(all_data)<600:
            exit()

    df = pd.DataFrame(columns=["Function", "Fit", "Parameters",
                               "Cost", "Trainset", "EML"])
    bins = 100
    for perc in train_perc:
        print("Training size %d" %(perc))
        test_cnt = perc 
        training = scenario.get_training_set(perc)
        cost_model = scenario.get_cost_model()

        wf = WorkloadFit.WorkloadFit(all_data, cost_model, bins=bins)
        optimal_cost = scenario.get_optimal_cost(wf)
        df.loc[len(df)] = ["Optimal", "", "", optimal_cost, perc, 0]
        if verbose:
            print("Optimal cost %f" %(optimal_cost))

        wf.set_workload(training)
        cost = wf.compute_discrete_cost()
        df.loc[len(df)] = ["Discrete", "", "", cost, perc,
                           (cost-optimal_cost)*1./optimal_cost]
        if verbose:
            print("Discrete cost %f" %(cost))

        wf.set_interpolation_model(
            WorkloadFit.PolyInterpolation(max_order=20))
        best_cost = wf.compute_interpolation_cost()
        best_params = wf.get_best_fit()
        if verbose:
            print("Polynomial cost %f" %(best_cost))

        wf.set_interpolation_model(WorkloadFit.DistInterpolation())
        cost = wf.compute_interpolation_cost()
        if cost < best_cost:
            best_cost = cost
            best_params = list(wf.get_best_fit())
            best_params[0] = best_params[0].name
        if verbose:
            print("Distribution cost %f" %(cost))

        df.loc[len(df)] = ["Continuous", best_params[0],
                           ' '.join([str(i) for i in best_params[1]]),
                           best_cost, perc,
                           (best_cost-optimal_cost)*1./optimal_cost]

    with open("ACCRE/"+dataset+"_cost.csv", 'a') as f:
        df.to_csv(f, header=True)
        