# Log index

No large raw log was duplicated. The canonical evidence remains under `raw/`:

- Algorithm 2 same-GPU ordering: `../raw/alg2-ordering/`
  - `pipeline.log`, `pipeline.rc`, `monitor/status.json`,
    `monitor/heartbeat.jsonl`, `scheduler_proof.txt`, result JSON
- Algorithm 2 steady8 sanity: `../raw/alg2-sanity-D1/`
  - `pipeline.log`, `pipeline.rc`, `monitor/status.json`,
    `monitor/heartbeat.jsonl`, `scheduler_proof.txt`, result JSON,
    `alg2_runtime_events.csv`, `final_validation.json`
- D2 instrumentation-only failed launch (preserved):
  `../raw/migration-D2/run/`
- D2 diagnostic attempt with four migration decisions:
  `../raw/migration-D2/run-attempt2/`
  - `pipeline.log`, `pipeline.rc`, `monitor/status.json`,
    `monitor/heartbeat.jsonl`, `server-logs/gpu_timeline.txt`, split server
    logs, `weight_transfers.jsonl`, `kv_transfers.jsonl`, and
    `migration_summary.json`
- D2 workload provenance JSON files:
  `../raw/migration-D2/workload/paired_requests_r20_s1.json` and
  `phases_r20_s1.json`. Their hashes match the preregistered inputs recorded in
  `../META.txt`.

Top-level normalized evidence files are `alg2_runtime_events.csv`,
`alg2_sanity_summary.csv`, and `migration_timeline.csv`.

The repository's existing `.gitignore` excludes generated `*.pkl` workload
objects. Those binary workload files remain on the diagnostic server and are
not required to inspect the D2 failure: all D2 pipeline, watchdog, GPU,
controller, scheduler, model-service, stdout, transfer, and summary evidence is
committed. Server-only generated files are:

- `../raw/alg2-ordering/abc_same_gpu.pkl`
- `../raw/migration-D2/workload/bursty_r20_s1.pkl`
- `../raw/migration-D2/workload/steady_r20_s1.pkl`
