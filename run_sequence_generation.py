import Workload
import sys
import time

start = 1
end = 101
scenario = "atoptimal"
zeta_steps = 0
check = False
sequence_options = ['atoptimal', 'uoptimal', 'toptimal', 'checkpoint']

def print_usage():
        print("usage: python %s start end sequence_type [options]" %(sys.argv[0]))
        print("sequence_type: %s" %(sequence_options))
        print("options: for checkpoint 1 for taking checkpoints for every reservation (by default checkpoints are not forced)")

if len(sys.argv) > 2:
    start = int(sys.argv[1])
    end = int(sys.argv[2])

if len(sys.argv) < 2:
    print_usage()
    exit()

if len(sys.argv) > 3:
    scenario = sys.argv[3]
    if scenario not in sequence_options:
        print_usage()
        exit()
    if scenario == "checkpoint" and len(sys.argv) > 4:
        check = True

if len(sys.argv) > 4:
    zeta_steps = int(sys.argv[4])

for n in range(start,end):
    start = time.time()
    distr = Workload.TruncNormalDistr(0, 20, 8, 2)

    if scenario == "uoptimal":
        sw = Workload.UOptimalSequence(distr, discret_samples=n)
        val = sw.compute_E_value(1, 0, 0)
    elif scenario == "toptimal":
        sw = Workload.TOptimalSequence(distr, discret_samples=n)
        val = sw.compute_E_value(1)
    elif scenario=="atoptimal":
        sw = Workload.ATOptimalSequence(0, distr, discret_samples=n)
        val = sw.compute_E_value(1, 0, 0)
    else:
        sw = Workload.CheckpointSequence(distr, discret_samples=n, always_checkpoint=check)
        val = sw.compute_E_value(0, 0)

    sequence = sw.compute_request_sequence()
    end = time.time()
    print("%s, %d, %.6f, %.2f, %s" % (scenario, n, val[0], end-start, sequence))

    if scenario != "atoptimal":
        continue

    print("atoptimal: zeta, sequence")
    for i in range(zeta_steps):
        zeta = float(i / zeta_steps)
        sw = Workload.ATOptimalSequence(zeta, distr, discret_samples = n)
        val = sw.compute_E_value(1, 0, 0)
        sequence = sw. compute_request_sequence()
        end = time.time()
        print("%.2f, %s" % (zeta, sequence))
