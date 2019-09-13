import numpy as np
import pandas as pd
import scipy.stats as st
import sys
import random

import WorkloadFit

def load_workload(dataset):
    all_data = np.loadtxt("ACCRE/"+dataset+".out", delimiter=' ')
    print("Load %d runs from %s" %(len(all_data), "ACCRE/"+dataset+".out"))
    return pd.Series(all_data)

if __name__ == '__main__':
    train_perc = list(range(10, 511, 50))

    if len(sys.argv) < 2:
        print("Usage: %s dataset_file" %(sys.argv[0]))
        print("Example: %s ACCRE/Multi_Atlas.out" %(sys.argv[0]))
        exit()

    dataset = sys.argv[1].split("/")[1]
    dataset = dataset[:-4]

    all_data = load_workload(dataset)
    if len(all_data)<600:
        exit()
    df = pd.DataFrame(columns=["Function", "Fit", "Parameters",
                               "Cost", "Trainset", "EML"])
    bins = 100
    random_start = np.random.randint(0, len(all_data)-600)
    random.shuffle(all_data) 

    for perc in train_perc:
        test_cnt = perc #int(len(all_data)*perc/100)
        testing = all_data[:] #[test_cnt:]
        training =  all_data[random_start:random_start+test_cnt]
        
        cost_model = WorkloadFit.LogDataCost(testing)
        wf = WorkloadFit.WorkloadFit(all_data, cost_model, bins=bins)
        optimal_cost = wf.compute_discrete_cost()
        df.loc[len(df)] = ["Optimal", "", "", optimal_cost, perc, 0]
        
        wf.set_workload(training)
        cost = wf.compute_discrete_cost()
        df.loc[len(df)] = ["Discrete", "", "", cost, perc,
                           (cost-optimal_cost)*1./optimal_cost]
        
        wf.set_interpolation_model(
            WorkloadFit.PolyInterpolation(max_order=20))
        best_cost = wf.compute_interpolation_cost()
        best_params = wf.get_best_fit()
        wf.set_interpolation_model(WorkloadFit.DistInterpolation())
        cost = wf.compute_interpolation_cost()
        if cost < best_cost:
            best_cost = cost
            best_params = list(wf.get_best_fit())
            best_params[0] = best_params[0].name

        df.loc[len(df)] = ["Continuous", best_params[0],
                           ' '.join([str(i) for i in best_params[1]]),
                           best_cost, perc,
                           (best_cost-optimal_cost)*1./optimal_cost]

    with open("ACCRE/"+dataset+"_cost.csv", 'a') as f:
        df.to_csv(f, header=True)
        