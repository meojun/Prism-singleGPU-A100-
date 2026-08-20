# v6 sweep -- 2026-08-20T08:31:34+00:00

Ran: arms [paper-faithful-v4] x workloads [bursty steady] x rates [8] x seeds [1 2 3]
Completed 6 of 6; 0 failed.
tau=0.00035 (the v4 sweep's, not this box's derived 0.15992 -- that one
admits under 1% of decisions and would leave nothing to migrate).

```
==================================================================================
1. Did the mechanism engage?   (v4 is the control -- it must read zero)
==================================================================================
workload   rate arm                     n  stash  inject   reqs    KV MiB  capfail  paths
bursty        8 paper-faithful-v4       3      0       0      0       0.0        0  -
steady        8 paper-faithful-v4       3      0       0      0       0.0        0  -


==================================================================================
2. Did it change anything?   (from summary.csv -- the project's own definitions)
==================================================================================
workload   rate arm                     n           goodput         joint SLO  per-seed goodput
bursty        8 paper-faithful-v4       3    4.940 +- 0.846    0.615 +- 0.110   5.53, 5.54, 3.74
steady        8 paper-faithful-v4       3    3.579 +- 0.221    0.443 +- 0.028   3.80, 3.28, 3.66

Verdict per condition -- a gap inside the combined spread is not a finding:
```
