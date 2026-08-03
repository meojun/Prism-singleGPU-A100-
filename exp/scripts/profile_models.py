"""Add models to Prism's profiled model_info.json.

Prism's GPU scheduler (--enable-gpu-scheduler) refuses to start for any model
that is not present in
  python/sglang/multi_model/utils/model_info.json
because it needs the KV cell size and weight size to do admission control and
KVPR-based placement. The shipped file only covers the models used in the paper.

Usage (inside the prism venv):
    python profile_models.py Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct

Each model is briefly loaded onto GPU 0 to measure its real weight footprint,
then freed. Existing entries are kept unless --force is given.
"""

import argparse
import json
import os
import sys
import time

import torch
from vllm.config import CacheConfig, DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.model_executor.model_loader import get_model

from sglang.multi_model.utils.profile_model_info import (
    clean_up,
    get_cell_size,
    init_torch_distributed,
)
from sglang.srt.utils import (
    get_available_gpu_memory,
    monkey_patch_vllm_dummy_weight_loader,
)


def load_model(model_path):
    """Same as the upstream profiler, but passes a real CacheConfig.

    Upstream passes cache_config=None, which works for Llama but crashes in
    vLLM's Qwen2 model (it reads cache_config.sliding_window at build time).
    """
    monkey_patch_vllm_dummy_weight_loader()
    load_config = LoadConfig(load_format="auto")
    vllm_model_config = VllmModelConfig(
        model=model_path,
        quantization=None,
        tokenizer=None,
        tokenizer_mode=None,
        trust_remote_code=True,
        dtype="auto",
        seed=42,
        skip_tokenizer_init=True,
    )
    dtype = vllm_model_config.dtype
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )

    tic = time.time()
    mem_before = get_available_gpu_memory(device="cuda", gpu_id=0)
    model = get_model(
        model_config=vllm_model_config,
        load_config=load_config,
        device_config=DeviceConfig(device="cuda"),
        parallel_config=None,
        scheduler_config=None,
        lora_config=None,
        cache_config=cache_config,
    )
    mem_after = get_available_gpu_memory(device="cuda", gpu_id=0)
    model_size = mem_before - mem_after
    print(
        f"Load {model_path} end. Time cost: {time.time() - tic:.4f}s. "
        f"Model size: {model_size:.2f} GB"
    )
    del model
    clean_up()
    return model_size, dtype

MODEL_INFO_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(sys.modules["sglang.multi_model.utils.profile_model_info"].__file__)
    ),
    "model_info.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_paths", nargs="+")
    ap.add_argument("--force", action="store_true", help="re-profile existing entries")
    ap.add_argument("--path", default=MODEL_INFO_PATH)
    args = ap.parse_args()

    with open(args.path) as f:
        model_info = json.load(f)
    print(f"model_info.json: {args.path} ({len(model_info)} entries)")

    todo = [m for m in args.model_paths if args.force or m not in model_info]
    if not todo:
        print("nothing to do; all models already profiled")
        return

    init_torch_distributed()
    for model_path in todo:
        clean_up()
        model_size, dtype = load_model(model_path)
        cell_size = get_cell_size(model_path, dtype)
        model_info[model_path] = {"model_size": model_size, "cell_size": cell_size}
        print(f"  + {model_path}: size={model_size:.2f} GB cell={cell_size} B")

    with open(args.path, "w") as f:
        json.dump(model_info, f, indent=2, sort_keys=True)
    print(f"saved {len(model_info)} entries to {args.path}")


if __name__ == "__main__":
    torch.cuda.init()
    main()
