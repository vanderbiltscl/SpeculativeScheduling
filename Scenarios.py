import Workload


class ScenarioInfo():
    def __init__(self, scenario_name, prev_entries):
        self.scenario_name = scenario_name
        self.prev_instances = prev_entries

    def set_procs_request_method(self, wd, procs):
        wd.set_processing_units(distribution=Workload.ConstantDistr(procs))

    def set_procs_request_method_truncnormal(self, wd, procs):
        wd.set_processing_units(distribution=Workload.TruncNormalDistr(
            1, procs, int(procs/2), 2))

    def set_procs_request_method_half(self, wd, procs):
        wd.set_processing_units(distribution=Workload.ConstantDistr(
            int(procs/2)))

    def req_procs(self, walltimes):
        distr = Workload.BetaDistr(2, 2)
        sequence = distr.random_sample(len(walltimes))
        return [max(1, int(i*10)) for i in sequence]

    def set_procs_request_method_beta(self, wd, procs):
        wd.set_processing_units(procs_function=self.req_procs)

    def get_remove_entries_count(self):
        return self.prev_instances


class HPCScenario(ScenarioInfo):
    def __init__(self, prev_entries):
        super(HPCScenario, self).__init__('HPC', prev_entries)

    def set_time_request_method(self, wd):
        request_sequence = Workload.ConstantDistr(
            wd.upper_bound).random_sample(1)
        wd.set_request_time(request_sequence=request_sequence)


class MaxScenario(ScenarioInfo):
    def __init__(self, prev_entries):
        super(MaxScenario, self).__init__('Max', prev_entries)

    def req_time(self, walltimes):
        return [max(walltimes) for i in walltimes]

    def set_time_request_method(self, wd):
        wd.set_request_time(request_function=self.req_time)


class NeuroScenario(ScenarioInfo):
    def __init__(self, prev_entries):
        super(NeuroScenario, self).__init__('Neuro', prev_entries)

    def req_time_neuroscience(self, walltimes):
        return [max(walltimes[i - self.prev_instances:i])
                for i in range(self.prev_instances, len(walltimes))]

    def set_time_request_method(self, wd):
        wd.set_request_time(request_function=self.req_time_neuroscience)

    def get_remove_entries_count(self):
        return 0


class TOptimalScenario(ScenarioInfo):
    def __init__(self, prev_entries, distr, distr_param):
        super(TOptimalScenario, self).__init__('TOptimal', prev_entries)
        self.__distribution = distr
        self.__param = distr_param
        try:
            with open("request_sequence/toptimal_%s_%s" % (
                      self.__distribution.get_user_friendly_name(),
                      "_".join([str(i) for i in self.__param])), "r") as fp:
                line = fp.readline().split(" ")
                self.sequence_toptimal = [float(i) for i in line]
        except IOError:
            sw = Workload.TOptimalSequence(distr, discret_samples=127)
            self.sequence_toptimal = sw.compute_request_sequence()

    def set_time_request_method(self, wd):
        wd.set_request_time(request_sequence=self.sequence_toptimal)


class ATOptimalScenario(ScenarioInfo):
    def __init__(self, prev_entries, distr, distr_param, zeta):
        super(ATOptimalScenario, self).__init__('WBackfill_'+str(zeta),
                                                   prev_entries)
        self.__distribution = distr
        self.__param = distr_param
        try:
            with open("request_sequence/atoptimal_%.2f_%s_%s" % (zeta,
                      self.__distribution.get_user_friendly_name(),
                      "_".join([str(i) for i in self.__param])), "r") as fp:
                line = fp.readline().split(" ")
                self.sequence = [float(i) for i in line]
        except IOError:
            sw = Workload.ATOptimalSequence(zeta, distr, discret_samples=30)
            self.sequence = sw.compute_request_sequence()

    def set_time_request_method(self, wd):
        wd.set_request_time(request_sequence=self.sequence)


class UOptimalScenario(ScenarioInfo):
    def __init__(self, prev_entries, distr, distr_param):
        super(UOptimalScenario, self).__init__('UOptimal', prev_entries)
        self.__distribution = distr
        self.__param = distr_param
        try:
            with open("request_sequence/uoptimal_%s_%s" % (
                      self.__distribution.get_user_friendly_name(),
                      "_".join([str(i) for i in self.__param])), "r") as fp:
                line = fp.readline().split(" ")
                self.sequence_uoptimal = [float(i) for i in line]
        except IOError:
            sw = Workload.UOptimalSequence(distr, discret_samples=23)
            self.sequence_uoptimal = sw.compute_request_sequence()

    def set_time_request_method(self, wd):
        wd.set_request_time(request_sequence=self.sequence_uoptimal)
