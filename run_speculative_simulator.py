import argparse
import logging
import inspect
import sys
sys.path.append("./ScheduleFlow_v1.0")
import ScheduleFlow
import Workload
import Scenarios


def run_simulation(simulation, scenario, procs, wd):
    print("\nRunning the %s scenario ..." % (scenario.scenario_name))
    apl = []
    scenario.set_time_request_method(wd)

    apl = wd.generate_workload()
    apl = apl[scenario.get_remove_entries_count():]
    sch = ScheduleFlow.BatchScheduler(ScheduleFlow.System(procs))

    simulation.create_scenario(scenario.scenario_name, sch)
    simulation.add_applications(apl)
    ret = simulation.run()
    return ret


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
        '--run_neuro',
        metavar="prev_entries",
        type=int,
        const=5,
        nargs='?',
        help="""add the neuroscience scenario to the simulations
             (if prev_entries is not provided using by default the
             last 5 previous runs for estimation)""")
    parser.add_argument(
        '--create_gif',
        action='store_true',
        help="""create a gif animation with the schedule for each
        simulated scenario (saved in the draw folder)""")
    parser.add_argument(
        '--loops_runtime',
        metavar="iter",
        type=int,
        default=1,
        help="""Number of testing iterations for running the runtime
        on the same application set (default one loop)""")
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
    distr_param = get_class_param_info()
    logger = logging.getLogger(__name__)

    arg_list = parse_arguments()

    prev_instance = 0
    if arg_list['run_neuro'] is not None:
        prev_instance = arg_list['run_neuro']

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

    scenario = []
    scenario.append(Scenarios.HPCScenario(prev_instance))
    scenario.append(Scenarios.MaxScenario(prev_instance))
    scenario.append(Scenarios.TOptimalScenario(prev_instance, distr,
                                               arg_list['param']))
    scenario.append(Scenarios.UOptimalScenario(prev_instance, distr,
                                               arg_list['param']))
    if arg_list['run_neuro'] is not None:
        scenario.append(Scenarios.NeuroScenario(prev_instance))

    simulation = ScheduleFlow.Simulator(generate_gif=arg_list['create_gif'],
                                        loops=arg_list['loops_runtime'],
                                        check_correctness=True,
                                        output_file_handler=outf)
    for loop in range(arg_list['loops']):
        wd = Workload.Workload(distr, arg_list['jobs'])
        scenario[0].set_procs_request_method(wd, arg_list['procs'])
        for temp in scenario:
            run_simulation(simulation, temp, arg_list['procs'], wd)

    if arg_list['save_results'] is not None:
        outf.close()
