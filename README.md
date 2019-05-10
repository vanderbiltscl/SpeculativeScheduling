# SpeculativeScheduling
Extension of the ScheduleFlow Simulator (v1.0) to allow speculative request times at submission and during backfill

![Simulator workflow](docs/simulator_diagram.png)

The code represents an extension of the classic reservation-based HPC schedulers by using speculation to determine the resource requirements of stochastic applications based on their past behavior. Specifically, we augment the existing HPC model by speculatively overwriting the request times (including the initial one and subsequent ones in case of failures) of an application during submission.

The backfilling algorithm is also extended to include speculation by allowing stochastic jobs to be
scheduled into smaller backfilling spaces than their requests (temporary overwrite of their requirements).
