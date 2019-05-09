import sys
sys.path.append("./ScheduleFlow_v1.0")
import ScheduleFlow
import SpeculativeBackfill
import Workload
import Scenarios
import numpy as np
import sys

num_jobs_large = 70
num_jobs_small = 30


def req_time_large(walltimes):
    return [i+np.random.randint(3600, 7200) for i in walltimes]


def req_time_small(walltimes):
    return [i+np.random.randint(1800, 5400) for i in walltimes]


def req_procs(walltimes):
    distr = Workload.BetaDistr(2, 2)
    sequence = distr.random_sample(len(walltimes))
    return [max(1, int(i*10)) for i in sequence]


def generate_workload(sequence, procs, wd_large, wd_small,
                      distr_large, distr_small):

    if sequence == "TOptimal":
        sw = Workload.TOptimalSequence(distr_large, discret_samples=127)
        wd_large.set_request_time(
            request_sequence=sw.compute_request_sequence())
    elif sequence == "MASI":
        wd_large.set_request_time(request_function=req_time_large)
    elif sequence == "HPC":
        request_sequence = Workload.ConstantDistr(
            wd_large.upper_bound).random_sample(1)
        wd_large.set_request_time(request_sequence=request_sequence)
    apl = wd_large.generate_workload()

    if sequence == "TOptimal":
        sw = Workload.TOptimalSequence(distr_small, discret_samples=127)
        wd_small.set_request_time(
            request_sequence=sw.compute_request_sequence())
    elif sequence == "MASI":
        wd_small.set_request_time(request_function=req_time_small)
    elif sequence == "HPC":
        request_sequence = Workload.ConstantDistr(
            wd_small.upper_bound).random_sample(1)
        wd_small.set_request_time(request_sequence=request_sequence)

    apl_small = wd_small.generate_workload()
    for job in apl_small:
        job.job_id = job.job_id + num_jobs_large
        job.submission_time = 10

    return apl + apl_small


if __name__ == '__main__':
    print("Generating the workload...")
    scenario_list = ["TOptimal", "HPC"]
    procs_list = ["beta", "full"]

    distr_large = Workload.TruncNormalDistr(1, 8, 4, 2)
    wd_large = Workload.Workload(distr_large, num_jobs_large)
    distr_small = Workload.TruncNormalDistr(0.1, 6, 1, 1)
    wd_small = Workload.Workload(distr_small, num_jobs_small)

    for procs in procs_list:
        if procs == "full":
            wd_large.set_processing_units(
                distribution=Workload.ConstantDistr(10))
            wd_small.set_processing_units(
                distribution=Workload.ConstantDistr(10))
        elif procs == "beta":
            wd_large.set_processing_units(procs_function=req_procs)
            wd_small.set_processing_units(procs_function=req_procs)

        outf = sys.stdout
        for sequence in scenario_list:
            #outf = open("backfill_%s_%s_%d" % (sequence.lower(),
            #                                   procs,
            #                                   num_jobs_small), "a")
            simulation = ScheduleFlow.Simulator(check_correctness=True,
                                                output_file_handler=outf)
            apl = generate_workload(sequence, procs, wd_large,
                                    wd_small, distr_large, distr_small)
            print(sequence, procs)
            print("Running the new backfilling scheme...")

            sch = SpeculativeBackfill.SpeculativeBatchScheduler(
                    ScheduleFlow.System(10))
            simulation.create_scenario("speculative", sch)
            simulation.add_applications(apl)
            ret = simulation.run()

            print("Running the classic HPC backfilling scheme...")
            sch = ScheduleFlow.BatchScheduler(ScheduleFlow.System(10))
            simulation.create_scenario("classic", sch)
            simulation.add_applications(apl)
            ret = simulation.run()

            #outf.close()
