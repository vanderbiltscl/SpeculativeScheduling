import unittest
import SpeculativeSubmission
import sys
sys.path.append("./ScheduleFlow_v1.1")
import ScheduleFlow
import Workload


class TestWorkload(unittest.TestCase):

    def __req_procs(self, walltime):
        return [8 for i in walltime]

    def __req_time(self, walltime):
        return [i+100 for i in walltime]

    def __set_submission(self, walltime):
        return [0]+[100 for i in range(len(walltime)-1)]

    def test_workload_procs(self):
        distr = Workload.ExponentialDistr(1)
        wd = Workload.Workload(distr, 10)
        wd.set_processing_units(distribution=Workload.ConstantDistr(5))
        apl = wd.generate_workload()
        for job in apl:
            self.assertEqual(job.nodes, 5)
        wd.set_processing_units(procs_function=self.__req_procs)
        apl = wd.generate_workload()
        for job in apl:
            self.assertEqual(job.nodes, 8)

    def test_workload_walltimes(self):
        param_list = {'truncnormal': [0, 20, 8, 2], 'exponential': [1],
                      'beta': [2, 2], 'pareto': [1, 20, 2.1], 'constant': [3]}
        for DistrType in Workload.Distribution.__subclasses__():
            distr_name = DistrType.get_user_friendly_name(DistrType)
            param = param_list[distr_name]
            distr = DistrType(*param)
            wd = Workload.Workload(distr, 10)
            apl = wd.generate_workload()
            for job in apl:
                self.assertGreater(job.walltime / 3600, distr.get_low_bound())
                self.assertLessEqual(job.walltime / 3600, distr.get_up_bound())

    def test_workload_submission(self):
        wd = Workload.Workload(Workload.ExponentialDistr(1), 10)
        wd.set_submission_time(self.__set_submission)
        apl = wd.generate_workload()
        submissions = [job.submission_time for job in apl]
        submissions.sort()
        self.assertEqual(submissions[0], 0)
        for i in submissions[1:]:
            self.assertEqual(i, 100)

    def test_workload_request(self):
        distr = Workload.ConstantDistr(1)
        wd = Workload.Workload(distr, 5)
        wd.set_processing_units(distribution=Workload.ConstantDistr(10))
        sequence = [1.5, 3]
        wd.set_request_time(request_sequence=sequence)
        apl = wd.generate_workload()
        for job in apl:
            self.assertEqual(job.walltime, 3600)
            self.assertEqual(job.request_walltime, 5400)
        wd.set_request_time(request_function=self.__req_time)
        apl = wd.generate_workload()
        for job in apl:
            self.assertEqual(job.request_walltime, job.walltime+100)

    def test_metrics_full(self):
        distr = Workload.ConstantDistr(1)
        wd = Workload.Workload(distr, 5)
        wd.set_processing_units(distribution=Workload.ConstantDistr(10))
        sequence = [1.5, 3]
        wd.set_request_time(request_sequence=sequence)
        apl = wd.generate_workload()
        sch = ScheduleFlow.BatchScheduler(ScheduleFlow.System(10))
        simulator = ScheduleFlow.Simulator(output_file_handler=None)
        simulator.create_scenario("test_batch", sch, job_list=apl)
        simulator.run()
        stats = simulator.stats
        self.assertEqual(stats.total_makespan(), 7*3600)
        self.assertAlmostEqual(stats.system_utilization(), 5/7, places=3)
        self.assertAlmostEqual(stats.average_job_utilization(), 10/15,
                               places=3)
        self.assertEqual(stats.average_job_wait_time(), 3*3600)
        self.assertEqual(stats.average_job_response_time(), 4*3600)
        self.assertAlmostEqual(stats.average_job_stretch(), 4,
                               places=3)

    def __req_incremental_time(self, walltimes):
        request = [i+1800 for i in walltimes]
        request[0] = 3000
        return request

    def test_metrics_failure(self):
        distr = Workload.ConstantDistr(1)
        wd = Workload.Workload(distr, 5)
        wd.set_processing_units(distribution=Workload.ConstantDistr(10))
        wd.set_request_time(request_function=self.__req_incremental_time)
        apl = wd.generate_workload()
        sch = ScheduleFlow.BatchScheduler(ScheduleFlow.System(10))
        simulator = ScheduleFlow.Simulator(output_file_handler=None)
        simulator.create_scenario("test_batch", sch, job_list=apl)
        simulator.run()
        stats = simulator.stats
        self.assertAlmostEqual(stats.total_makespan(), 28200, places=3)
        self.assertAlmostEqual(stats.system_utilization(), 0.6383, places=3)
        self.assertAlmostEqual(stats.average_job_utilization(), 0.6293,
                               places=3)
        self.assertEqual(stats.average_job_wait_time(), 9000)
        self.assertAlmostEqual(stats.average_job_response_time(), 15000,
                               places=3)
        self.assertAlmostEqual(stats.average_job_stretch(), 4.1666,
                               places=3)

    def test_sequence_values(self):
        distr = Workload.TruncNormalDistr(0, 20, 8, 2)
        sw = Workload.TOptimalSequence(distr, discret_samples=100)
        sequence = sw.compute_request_sequence()
        self.assertSequenceEqual(sequence,
                                 [10.8, 13.4, 15.4, 17.2, 18.8, 20.0])
        sw = Workload.UOptimalSequence(distr, discret_samples=10)
        sequence = sw.compute_request_sequence()
        self.assertSequenceEqual(sequence,
                                 [10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    def test_invalid_distr(self):
        with self.assertRaises(AssertionError):
            Workload.TruncNormalDistr(-1, 20, 8, 2)
        with self.assertRaises(AssertionError):
            Workload.TruncNormalDistr(2, 1, 8, 2)
        with self.assertRaises(AssertionError):
            Workload.ParetoDistr(0, 20, 2.1)
        with self.assertRaises(AssertionError):
            Workload.ParetoDistr(10, 2, 2.1)
        with self.assertRaises(AssertionError):
            Workload.BetaDistr(-2, 2)
        with self.assertRaises(AssertionError):
            Workload.BetaDistr(2, -2)
        with self.assertRaises(AssertionError):
            Workload.ExponentialDistr(-2)
        with self.assertRaises(AssertionError):
            Workload.ConstantDistr(-2)
