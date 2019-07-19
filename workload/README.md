## Find best fit functions

Computes the cost of using discreet or continuous fits for a given dataset. 
The dataset containing execution times is used to create a histogram with the walltime distribution.
This distribution is either used by itself (the discreet case)
or different functions are being used to fit it (the continuous case) in order to create PDF and CDF functions for it.

The CDF is used to compute the sequence of request time for the given dataset (using the `OptimalSequence` class). 
The sequence is afterwards used to compute the wasted time used by an application by requesting less time than needed (the cost).

### Input dataset format

A list of execution times for a given application either separated by space or one per line. 

### Output format

A CSV file containing one line per experiment: `Function Parameters Cost`

 * **Function**: represents the function used for fitting the data (either `Discreet` or for the continuous case: `Polynomial`, `Exponential` or `Distribution`)
     * The function can also be `Optimal` when the discreet algorithm is using all the data for providind the CDF (and not just the training data)
 * **Parameters**: represents details about the parameters used by the fit (either the degree of the polynomial, the distribution used for the Distribution fit)
 * **Cost**: represents the amount of time wasted with requests smaller than the required execution time
     * Computed on the testing data as 
     ```python
     foreach walltime in testing_data:
       cost += sum([i for i in sequence if i < walltime])
       cost += first_request_time_over_walltime(sequence)
     ```

### Functionalities

**1. Default**

`find_best_fit.py` takes an input dataset, creates the histogram using 50 bins using 1% of the total dataset. Based on the histogram it computes the cost of using discreet or continuous fit for the rest of 99% of the data and writes the results in the csv file. For the `Optimal` case, it creates the histogram using 100% of the dataset and tests it on the last 99%.

*Output* Results for different datasets are appended in the cost.csv file

**2. Differnt number of bins**

**3. Differnt training percentaces**

**4. Adapting the sequence**
