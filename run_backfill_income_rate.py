import argparse
import logging
import inspect
import os
import sys
sys.path.append("./ScheduleFlow_v1.1")
import ScheduleFlow
import Workload
import SpeculativeSubmission
import numpy as np


def prepare_small_jobs_hpc(small_jobs, scale, distr):
    upper_limit = int(3600 * distr.get_up_bound() * scale)
    for job in small_jobs:
        job.request_walltime = upper_limit
        job.request_sequence = []


def prepare_small_jobs_zeta0(small_jobs, scale, distr, param):
    try:
        with open("request_sequence/toptimal_%s_%s" % (
            distr.get_user_friendly_name(),
            "_".join([str(i) for i in param])), "r") as fp:
            line = fp.readline().split(" ")
            sequence = [float(i) for i in line]
    except IOError:
        sw = Workload.TOptimalSequence(distr, discret_samples=127)
        sequence = sw.compute_request_sequence()
    for job in small_jobs:
        job.request_walltime = int(sequence[0] * 3600 * scale)
        job.request_sequence = [int(3600 * i * scale) for i in sequence[1:]]


def scale_jobs(job_list, new_work):
    work = sum([job.walltime for job in job_list])
    job_rate = int(new_work/len(job_list))
    scale = float(new_work / work)
    i = 0 
    for job in job_list:
        job.job_id = job.job_id + 100
        job.submission_time = 1 + i * job_rate
        job.walltime = int(job.walltime * scale)
        job.request_walltime = job.walltime # int(job.request_walltime * scale)
        for i in range(len(job.request_sequence)):
            job.request_sequence[i] = int(job.request_sequence[i] *
                                          scale)
        i += 1
    return (job_list, scale)

def generate_small_jobs(makespan, zeta, distr, param):
    small_job_work = int(makespan * zeta)
    total_small_jobs = int(400 * zeta) 
    scenario = SpeculativeSubmission.ATOptimalScenario(0, distr,
                                              param, 0)
    wd = Workload.Workload(distr, total_small_jobs)
    scenario.set_procs_request_method(wd, arg_list['procs'])
    scenario.set_time_request_method(wd)
    # scale execution/request times so that sum of work equals total work
    job_list = scale_jobs(wd.generate_workload(), small_job_work)
    print("Mean jobs: %.2f; Total jobs: %d" % (
        int(np.mean([job.walltime for job in job_list[0]])),
        len(job_list[0])))
    return job_list
 

def run_simulation(simulation, scenario, procs, wd, small_jobs):
    print("\nRunning the %s scenario ..." % (scenario.scenario_name))
    apl = []
    scenario.set_time_request_method(wd)

    apl = wd.generate_workload()
    apl += small_jobs

    sch = ScheduleFlow.BatchScheduler(ScheduleFlow.System(procs))

    simulation.create_scenario(scenario.scenario_name, sch)
    simulation.add_applications(apl)
    simulation.run()
    ret = simulation.get_execution_log()
    return max([max([i[1] for i in ret[job]]) for job in ret])


def create_workload(distribution, param, total_jobs, outf):
    distr = None
    for DistrType in Workload.Distribution.__subclasses__():
        distr_name = DistrType.get_user_friendly_name(DistrType)
        if distr_name == "noname":
            distr_name = str(DistrType).split("'")[1][len('Workload') + 1:]
        if distribution == distr_name:
            distr = DistrType(*param)
    if distr is None:
        return None
    return distr


def get_class_param_info():
    param_len = {}
    for DistrType in Workload.Distribution.__subclasses__():
        sig = inspect.signature(DistrType.__init__)
        distr_name = DistrType.get_user_friendly_name(DistrType)
        if distr_name == "noname":
            distr_name = str(DistrType).split("'")[1][len('Workload') + 1:]
        param_len[distr_name] = []
        for param in sig.parameters:
            if param == "self":
                continue
            param_len[distr_name].append(param)
    return param_len


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run the simulator on a given distribution')
    parser.add_argument('procs', metavar="#cores", type=int,
                        help='number of processing units')
    parser.add_argument('jobs', metavar="#jobs", type=int,
                        help='total number of jobs')
    parser.add_argument(
        'distribution',
        metavar='distr_type',
        choices=[
            distr for distr in distr_param],
        help='type of distribution for walltime generation')
    parser.add_argument(
        'param',
        metavar='distr_param',
        type=float,
        nargs='+',
        help=r'distribution parameters %s' % (distr_param))
    parser.add_argument(
        '--save_results',
        metavar="file_name",
        help="""save the results by providing the filename
             where to append the results of the current experiment
             (default do not save)""")
    parser.add_argument(
        '--create_gif',
        action='store_true',
        help="""create a gif animation with the schedule for each
        simulated scenario (saved in the draw folder)""")
    parser.add_argument(
        '--loops',
        metavar="iter",
        type=int,
        default=1,
        help="""Number of simulation loops (default one loop)""")

    arg_list = vars(parser.parse_args())
    if len(arg_list['param']) != len(distr_param[arg_list['distribution']]):
        logger.error("""Invalid number of parameters.
                        %s distribution requires %d: %s %d provided""" % (
                        arg_list['distribution'],
                        len(distr_param[arg_list['distribution']]),
                        distr_param[arg_list['distribution']],
                        len(arg_list['param'])))
        exit()

    return arg_list


if __name__ == '__main__':
    os.environ["SF_DRAW_PATH"] = "./ScheduleFlow_v1.1/draw"
    distr_param = get_class_param_info()
    logger = logging.getLogger(__name__)

    arg_list = parse_arguments()

    outf = sys.stdout
    if arg_list['save_results'] is not None:
        outf = open(arg_list['save_results'], "a")
        outf.write("Distribution : %s : %s\n" %
                   (arg_list['distribution'], arg_list['param']))

    distr = create_workload(arg_list['distribution'],
                            arg_list['param'],
                            arg_list['jobs'], outf)
    if distr is None:
        if arg_list['save_results'] is not None:
            outf.close()
        logger.error(
            "Distribution %s was not found in the implemented classes" %
            (arg_list['distribution']))
        exit()

    simulation_z0 = ScheduleFlow.Simulator(loops=1)
    simulation = ScheduleFlow.Simulator(generate_gif=arg_list['create_gif'],
                                        loops=1,
                                        check_correctness=False,
                                        output_file_handler=outf)
    # check correctness set to false because we change the request
    # time and sequence of job without using the Application
    # interface (so the internal logs are not updated)

    for loop in range(arg_list['loops']):
        wd = Workload.Workload(distr, arg_list['jobs'])
        scenario_z0 = SpeculativeSubmission.ATOptimalScenario(0, distr,
                                                     arg_list['param'], 0)
        scenario_z0.set_procs_request_method(wd, arg_list['procs'])
        scenario_hpc = SpeculativeSubmission.HPCScenario(0)
        makespan = run_simulation(simulation_z0, scenario_z0,
                                  arg_list['procs'], wd, [])
        for i in range(1, 10):
            zeta = float(i / 10)
            print("\n\nZeta: %.2f" %(zeta))
            scenario = SpeculativeSubmission.ATOptimalScenario(0, distr,
                                                      arg_list['param'],
                                                      zeta)
            ret = generate_small_jobs(makespan, zeta,
                                      distr, arg_list['param'])
            small_jobs = ret[0]
            scale = ret[1]
            #print("Initial small jobs: ", len(small_jobs), small_jobs[0])
            run_simulation(simulation, scenario, arg_list['procs'],
                           wd, small_jobs)
            #prepare_small_jobs_zeta0(small_jobs, scale, distr, arg_list['param'])
            #print("ZETA 0 small jobs: ", len(small_jobs), small_jobs[0])
            run_simulation(simulation, scenario_z0, arg_list['procs'],
                           wd, small_jobs)
            #prepare_small_jobs_hpc(small_jobs, scale, distr)
            #print("HPC small jobs: ", len(small_jobs), small_jobs[0])
            run_simulation(simulation, scenario_hpc, arg_list['procs'],
                           wd, small_jobs)

    if arg_list['save_results'] is not sys.stdout:
        outf.close()
