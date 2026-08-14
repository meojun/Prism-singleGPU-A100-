#!/usr/bin/env python3
"""Apply the Paper-Faithful Prism implementation onto the pinned prism-research clone.

    python patches/paper_faithful/apply_patches.py [--repo /workspace/prism-exp/prism-research]

Idempotent: re-running is a no-op.  Every edit is additive and gated behind a
CLI flag, so with no flags the released prototype's code path is unchanged.

Files ADDED to prism-research:
    sglang/multi_model/scheduling/gpu/moore_hodgson.py      paper Algorithm 2
    sglang/multi_model/scheduling/gpu/request_queue_mh.py   Alg-2 admission control
    sglang/multi_model/scheduling/policy/kvpr_global.py     paper Algorithm 1

Files EDITED (marked with  # PAPER-FAITHFUL):
    scheduling/gpu/request_queue.py     mixin + dispatch on a flag
    scheduling/gpu/gpu_scheduler.py     wire the flag/config through
    scheduling/controller_global.py     register the kvpr-global policy
    multi_model_server_args.py          new flags
"""
import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
MARK = "# PAPER-FAITHFUL"

DIRECT_TRACE_BODY = '''        # PAPER-FAITHFUL: direct trace -- the pickle already carries everything.
        if getattr(self, "_direct", False):
            out, n = [], req_count
            for r in self.requests:
                for k in range(config.replication):
                    ttft = r.slo_ttft * config.ttft_slo_scale
                    tpot = r.slo_tpot * config.tpot_slo_scale
                    # keep the pickle's own request_id: the paired bursty and
                    # steady traces carry the SAME id for the same payload, and
                    # a sequential renumber (which is by arrival order) would
                    # destroy exactly the pairing the study depends on.
                    rid = getattr(r, "req_id", None) or str(n)
                    rid = rid if config.replication == 1 else f"{rid}r{k}"
                    out.append(Request(
                        req_id=rid, prompt=r.prompt, prompt_len=r.prompt_len,
                        output_len=r.output_len,
                        arrival_time=r.arrival_time * config.time_scale,
                        model=r.model, slo=ttft, slo_ttft=ttft, slo_tpot=tpot))
                    n += 1
            out.sort(key=lambda x: x.arrival_time)
            span = (out[-1].arrival_time - out[0].arrival_time) if out else 0.0
            print("[PRISM_DIRECT] %d requests, span %.1fs" % (len(out), span))
            return out
'''



def edit(path: pathlib.Path, anchor: str, addition: str, where="after", count=1):
    """Insert `addition` relative to the first `count` occurrence(s) of `anchor`."""
    src = path.read_text()
    if addition.strip().splitlines()[0] in src:
        return False
    if anchor not in src:
        sys.exit(f"FATAL: anchor not found in {path}:\n{anchor!r}")
    if where == "after":
        src = src.replace(anchor, anchor + addition, count)
    else:
        src = src.replace(anchor, addition + anchor, count)
    path.write_text(src)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/workspace/prism-exp/prism-research")
    a = ap.parse_args()
    mm = pathlib.Path(a.repo) / "python/sglang/multi_model"
    if not mm.is_dir():
        sys.exit(f"FATAL: {mm} is not a directory")

    # ---------------------------------------------------------------- new files
    for src, dst in [
        ("moore_hodgson.py", mm / "scheduling/gpu/moore_hodgson.py"),
        ("request_queue_mh.py", mm / "scheduling/gpu/request_queue_mh.py"),
        ("kvpr_global.py", mm / "scheduling/policy/kvpr_global.py"),
    ]:
        shutil.copyfile(HERE / src, dst)
        print(f"  copied {dst.relative_to(a.repo)}")

    # ------------------------------------------------------- request_queue.py
    rq = mm / "scheduling/gpu/request_queue.py"
    edit(rq, "logger = logging.getLogger(__name__)",
         f"\n\n{MARK}: paper Algorithm 2 lives in request_queue_mh.py and is inert\n"
         "# unless configure_moore_hodgson(enabled=True, ...) has been called.\n"
         "from sglang.multi_model.scheduling.gpu.request_queue_mh import MooreHodgsonMixin")
    src = rq.read_text()
    if "class RequestQueue(MooreHodgsonMixin):" not in src:
        rq.write_text(src.replace("class RequestQueue:",
                                  "class RequestQueue(MooreHodgsonMixin):", 1))
    edit(rq, "        self.last_log_time = 0\n",
         f"        self._mh_enabled = False   {MARK}\n")
    edit(rq, '        admitted = defaultdict(list)\n\n'
             '        # Calculate total resources consumed by activating state requests\n',
         f'        {MARK}: route to paper Algorithm 2 when enabled.\n'
         '        if getattr(self, "_mh_enabled", False):\n'
         '            return self.admission_control_mh(\n'
         '                available_resources, model_backend_queue_lens,\n'
         '                model_states, allow_sending_when_activating)\n',
         where="before")
    print("  edited scheduling/gpu/request_queue.py")

    # ------------------------------------------------------- gpu_scheduler.py
    gs = mm / "scheduling/gpu/gpu_scheduler.py"
    edit(gs, "        self.queue = RequestQueue(model_name_to_cell_size)\n",
         f"""        {MARK}: paper Algorithm 2 configuration.
        if getattr(multi_model_server_args, "enable_moore_hodgson", False):
            import json as _json, os as _os
            _speed = {{}}
            _f = getattr(multi_model_server_args, "prefill_speed_file", None)
            if _f and _os.path.exists(_f):
                _raw = _json.load(open(_f))
                _raw = _raw.get("model_prefill_speed", _raw)
                _p2n = {{v: k for k, v in model_names_to_model_paths.items()}}
                for _k, _v in _raw.items():
                    _v = _v.get("c_i") if isinstance(_v, dict) else _v
                    if _k in model_names_to_model_paths:
                        _speed[_k] = float(_v)
                    elif _k in _p2n:
                        _speed[_p2n[_k]] = float(_v)
            _log = getattr(multi_model_server_args, "log_file", None)
            _log = f"{{_log}}.alg2_gpu{{gpu_id}}.jsonl" if _log else None
            self.queue.configure_moore_hodgson(True, _speed, _log)
""")
    print("  edited scheduling/gpu/gpu_scheduler.py")

    # --------------------------------------------------- controller_global.py
    cg = mm / "scheduling/controller_global.py"
    edit(cg, "from sglang.multi_model.scheduling.policy.simple_global import SimpleGlobalPolicy",
         f"\n{MARK}\nfrom sglang.multi_model.scheduling.policy.kvpr_global import KVPRGlobalPolicy")
    edit(cg, '        else:\n            raise ValueError(f"Unknown policy: {self.server_args.policy}")',
         f'''        elif self.server_args.policy == "kvpr-global":
            {MARK}: paper Algorithm 1.
            import json as _json, os as _os
            _tpot = {{}}
            _f = getattr(self.server_args, "slo_base_file", None)
            if _f and _os.path.exists(_f):
                _raw = _json.load(open(_f))
                _scale = float(getattr(self.server_args, "kvpr_tpot_slo_scale", 1.0) or 1.0)
                _p2n = {{v: k for k, v in self.model_names_to_model_paths.items()}}
                for _k, _v in _raw.items():
                    if isinstance(_v, dict):
                        _v = _v.get("tpot") or _v.get("tpot_slo") or _v.get("tpot_p95_ms")
                    if _v is None:
                        continue
                    _v = float(_v)
                    _v = _v / 1000.0 if _v > 1.0 else _v          # ms -> s
                    _name = _k if _k in self.model_names_to_model_paths else _p2n.get(_k)
                    if _name:
                        _tpot[_name] = _v * _scale
            logger.info(f"[PAPER-ALG1] tpot_slo_s={{_tpot}}")
            self.policy = KVPRGlobalPolicy(
                num_gpus=len(self.gpu_ids),
                gpu_mem=gpu_mem,
                model_weights_info=self.model_weights_info_after_renamed,
                workers_per_gpu=self.server_args.workers_per_gpu,
                tau=float(getattr(self.server_args, "kvpr_tau", 0.35)),
                rate_window=float(getattr(self.server_args, "kvpr_rate_window", 30.0)),
                migration_cooldown=float(getattr(self.server_args, "kvpr_migration_cooldown", 30.0)),
                tpot_slo_s=_tpot,
            )
''', where="before")
    print("  edited scheduling/controller_global.py")

    # ------------------------------------------------- multi_model_server_args
    ar = mm / "multi_model_server_args.py"
    edit(ar, '    policy: str = "simple-global"\n',
         f"""    {MARK}
    enable_moore_hodgson: bool = False
    prefill_speed_file: str = None
    slo_base_file: str = None
    kvpr_tau: float = 0.35
    kvpr_rate_window: float = 30.0
    kvpr_migration_cooldown: float = 30.0
    kvpr_tpot_slo_scale: float = 1.0
""")
    edit(ar, '            choices=[\n                "simple-global",\n            ],',
         '            choices=[\n                "simple-global",\n                "kvpr-global",   ' + MARK + '\n            ],')
    edit(ar, '        parser.add_argument(\n            "--queue-id",',
         f'''        {MARK}: paper Algorithm 1 / Algorithm 2 knobs.
        parser.add_argument("--enable-moore-hodgson", action="store_true",
            help="Paper Algorithm 2 (Moore-Hodgson) GPU-local request scheduling.")
        parser.add_argument("--prefill-speed-file", type=str, default=None,
            help="JSON of per-model chunked-prefill speed c_i in tokens/s (Algorithm 2).")
        parser.add_argument("--slo-base-file", type=str, default=None,
            help="JSON of per-model SLO baselines; supplies s_j to Algorithm 1.")
        parser.add_argument("--kvpr-tau", type=float, default=MultiModelServerArgs.kvpr_tau,
            help="Algorithm 1 migration threshold tau (relative peak-KVPR improvement).")
        parser.add_argument("--kvpr-rate-window", type=float,
            default=MultiModelServerArgs.kvpr_rate_window,
            help="Sliding window in seconds for the Algorithm 1 token-rate estimate.")
        parser.add_argument("--kvpr-migration-cooldown", type=float,
            default=MultiModelServerArgs.kvpr_migration_cooldown,
            help="Minimum seconds between Algorithm 1 migrations.")
        parser.add_argument("--kvpr-tpot-slo-scale", type=float,
            default=MultiModelServerArgs.kvpr_tpot_slo_scale,
            help="Scale applied to the TPOT SLO baseline used as s_j in Algorithm 1.")
''', where="before")
    print("  edited multi_model_server_args.py")

    # ------------------------------------------------------------- trace.py
    # Harness plumbing, NOT an algorithm change: it applies identically to
    # every arm.  generate_e2e_benchmark_reqs() maps a request's adapter rank
    # onto one of eight fixed slots and looks its SLO up in a table keyed by
    # slot, which makes any model set outside those eight slots impossible to
    # express.  A "direct" trace carries model, arrival_time and per-model SLO
    # baselines in the pickle itself and is passed straight through; the
    # ttft/tpot scale flags still apply, so both arms keep sharing one knob.
    tr = pathlib.Path(a.repo) / "benchmark/multi-model/trace.py"
    edit(tr, "            adapter_dirs, self.requests = obj[0], obj[1]\n",
         '            self._direct = bool(adapter_dirs) and adapter_dirs[0] == "__PRISM_DIRECT__"   '
         + MARK + "\n")
    edit(tr, '        """Generate e2e benchmark requests from trace_1.py"""\n',
         DIRECT_TRACE_BODY)
    print("  edited benchmark/multi-model/trace.py")

    # ------------------------------------------------------- srt/server_args.py
    # ServerArgs is constructed from MultiModelServerArgs by forwarding **vars()
    # minus an explicit drop-list, so any field added to MultiModelServerArgs
    # that ServerArgs does not know about crashes engine launch with
    # "unexpected keyword argument".  Our flags are controller/scheduler-side
    # only, so they join the drop-list next to --policy and --enable-controller.
    sa = pathlib.Path(a.repo) / "python/sglang/srt/server_args.py"
    edit(sa, '            "num_model_service_workers",\n',
         '            "enable_moore_hodgson",   ' + MARK + '\n'
         '            "prefill_speed_file",\n'
         '            "slo_base_file",\n'
         '            "kvpr_tau",\n'
         '            "kvpr_rate_window",\n'
         '            "kvpr_migration_cooldown",\n'
         '            "kvpr_tpot_slo_scale",\n')
    print("  edited python/sglang/srt/server_args.py")
    print("PAPER-FAITHFUL patches applied.")


if __name__ == "__main__":
    main()
