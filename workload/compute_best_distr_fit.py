import numpy as np
import pandas as pd
import scipy.stats as st
import sys
import random

import WorkloadFit

class LogDataRun():
    def load_workload(self, dataset):
        all_data = np.loadtxt("ACCRE/"+dataset+".out", delimiter=' ')
        #print("Load %d runs from %s" %(len(all_data), "ACCRE/"+dataset+".out"))
        self.testing = all_data
        return all_data

    def get_cost_model(self):
        assert (self.testing is not None),\
            'get_train_test_sets needs to be called before get_cost_model'
        return WorkloadFit.LogDataCost(self.testing)
    
    def get_optimal_cost(self, wf):
        return wf.compute_discrete_cost()



if __name__ == '__main__':
    train_perc = list(range(10, 511, 50))

    verbose = False
    if len(sys.argv) < 2:
        print("Usage: %s dataset_file [verbose]" %(sys.argv[0]))
        print("Example dataset: %s ACCRE/Multi_Atlas.out" %(sys.argv[0]))
        exit()

    if len(sys.argv) > 2:
        verbose = True

    all_data = []
    dataset = sys.argv[1].split("/")[1]
    dataset = dataset[:-4]

    scenario = LogDataRun()
    all_data = scenario.load_workload(dataset)

    bins = 100
    cost_model = scenario.get_cost_model()

    wf = WorkloadFit.WorkloadFit(all_data, cost_model, bins=bins)
    wf.set_interpolation_model(WorkloadFit.DistInterpolation())
    
    wf.compute_best_fit()
    best_params = list(wf.get_best_fit())
    best_params[0] = best_params[0].name
    print(dataset, best_params[0], min(all_data), max(all_data), best_params[1])
