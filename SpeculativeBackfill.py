import sys
sys.path.append("./ScheduleFlow_v1.1")
import ScheduleFlow


class StochasticApplication(ScheduleFlow.Application):
    ''' Application together with its distribution for past walltimes '''

    def __init__(self, nodes, submission_time, walltime,
                 requested_walltimes, resubmit_factor=-1,
                 distribution=None):
        ''' Optionaly, the constructor can receive a distribution
        object in addition to regular parameters for Applications '''
        
        super(StochasticApplication, self).__init__(
            nodes, submission_time, walltime, requested_walltimes,
            resubmit_factor)
        self.distribution = distribution

    def get_last_used_request(self):
        '''Method for returning the request time used for the previous
        failed execution '''
        change_list = [i for i in self.__execution_log if
                       i[0] == JobChangeType.RequestChange]
        num_submissions = len(change_list)
        if num_submissions == 0:
            return int(self.distribution.get_low_bound() * 3600)
        return change_list[num_submissions - 1][1]


class SpeculativeBatchScheduler(ScheduleFlow.BatchScheduler):
    ''' Reservation based scheduler with speculative backfilling '''

    def stochastic_fit_in_schedule(self, job, reserved_jobs, ts, time):
        if len(reserved_jobs) == 0:
            return False
        gap_list = self.create_gaps_list(reserved_jobs, ts)
        if len(gap_list) == 0:
            return False

        # check every gap that starts with ts
        for gap in gap_list:
            if gap[0] != ts:
                continue
            if job.nodes <= gap[2] and time <= (gap[1] - ts):
                return True
        return False

    def __get_max_gap(self, job, reserved_jobs, min_ts):
        if len(reserved_jobs) == 0:
            return (-1, 0)
        gap_list = super(BatchScheduler, self).create_gaps_list(
            reserved_jobs, min_ts)
        if len(gap_list) == 0:
            return (-1, 0)
        max_gap = 0
        chosen_ts = -1
        # return the largest gap that fits the jobs processing requirements
        for gap in gap_list:
            if gap[1] <= job.submission_time:
                continue
            ts = max(gap[0], job.submission_time)
            if job.nodes <= gap[2]:
                if (gap[1] - ts) > max_gap:
                    max_gap = (gap[1] - ts)
                    chosen_ts = ts
        return (chosen_ts, max_gap)

    def __compute_equation_member(self, distr, x):
        return (x - distr.mu) / (distr.sigma * scipy.sqrt(2))

    def __compute_benefit_score(self, job, upper_limit):
        lower_limit = job.get_last_used_request()
        if upper_limit <= lower_limit:
            return 0
        distr = job.distribution
        upper_limit = upper_limit / 3600
        lower_limit = lower_limit / 3600

        if distr.get_user_friendly_name() == "truncnormal":
            xlow = self.__compute_equation_member(distr, lower_limit)
            xup = self.__compute_equation_member(distr, upper_limit)
            score = distr.mu * (scipy.special.erf(xup) -
                                scipy.special.erf(xlow))
            score += (distr.sigma * scipy.sqrt(2 / scipy.pi) *
                      (scipy.exp(-(xlow * xlow)) - scipy.exp(-(xup * xup))))
            xup_distr = self.__compute_equation_member(distr,
                                                       distr.get_up_bound())
            xlow_distr = self.__compute_equation_member(distr,
                                                        distr.get_low_bound())
            score /= (scipy.special.erf(xup_distr) -
                      scipy.special.erf(xlow_distr))
        if score < 0.18:
            return 0
        return score * job.nodes

    def backfill_request(self, stop_job, reservation, min_ts):
        ''' Overwrite the backfill function to allow jobs that
        would normally not fit in the gap to be scheduled based
        on the CDF of its walltime distribution'''

        batch_jobs = self.get_batch_jobs()
        selected_jobs = super(SpeculativeBatchScheduler,
                              self).backfill_request(stop_job, reservation,
                                                     min_ts)

        reserved_jobs = reservation.copy()
        for job in selected_jobs:
            reserved_jobs[job[1]] = job[0]
        # there are no more job in the batch list that fit in the gaps
        potential_jobs = []
        for job in batch_jobs:
            # skip over jobs that were already selected
            if job in reserved_jobs:
                continue
            # skip over jobs that are not stochastic
            if job.distribution is None:
                continue
            max_gap = self.__get_max_gap(job, reserved_jobs, min_ts)
            if max_gap[0] == -1:
                continue
            # compute the benefit score of scheduling job in the given
            # max_gap request time
            score = self.__compute_benefit_score(job, max_gap[1])
            if score == 0:
                continue
            potential_jobs.append((score, max_gap[0], max_gap[1], job))

        potential_jobs.sort(reverse=True)
        # schedule in the order of their CDF as many jobs as possible
        for entry in potential_jobs:
            job = entry[3]
            tm = entry[1]
            new_request = entry[2]
            ret = self.stochastic_fit_in_schedule(job, reserved_jobs,
                                                  tm, new_request)
            if ret:
                reserved_jobs[job] = tm
                selected_jobs.append((tm, job))
                self.wait_queue.remove(job)
                job.speculative_run(new_request)
        return selected_jobs
