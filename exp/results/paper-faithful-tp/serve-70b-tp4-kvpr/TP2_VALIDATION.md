# TP=2 validation: PARTIAL

- [PASS] server_started
- [FAIL] tp_size_2_configured
- [FAIL] both_gpus_in_placement
- [PASS] ranks_observed_on_distinct_gpus
- [PASS] nccl_mentioned_in_logs
- [PASS] inference_succeeded
- [PASS] no_runtime_errors
- [PASS] load_phase_exit_zero

startup: 368.0s
TP rank -> GPU: {'0': [0, 1, 2, 3], '1': [1], '2': [2], '3': [3]}
