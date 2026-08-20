# TP=2 validation: FAIL

- [FAIL] server_started
- [FAIL] tp_size_2_configured
- [FAIL] both_gpus_in_placement
- [PASS] ranks_observed_on_distinct_gpus
- [PASS] nccl_mentioned_in_logs
- [FAIL] inference_succeeded
- [FAIL] no_runtime_errors
- [FAIL] load_phase_exit_zero

startup: 355.0s
TP rank -> GPU: {'0': [0, 1, 2, 3], '1': [1], '2': [2], '3': [3]}
