# This box re-measured — transport mechanisms hold, and P2P is faster here

Run before the gated meta-llama weights were available, so this uses the three
**ungated Qwen models only** (22.8 GiB total). That is a real limit and the
numbers are labelled for it: the v4 report's headline figures came from a
6-model set including Llama-3.1-8B. What a Qwen-only run *can* settle is
whether page-locking still pays and whether the P2P path is live, because both
are properties of the transport rather than of the model.

`peer access: {'0->1': True, '1->0': True}` — the §4.3 check the handover flags
as the one thing a 2-GPU box might fail. This box passes it.

## Loading (3 models onto GPU 0, 3 reps)

| Arm | this box | v4 box (6 models) |
| --- | ---: | ---: |
| sequential | 11.05 GB/s | 9.01 |
| v3-parallel-activation | 14.62 | 11.39 |
| **v4-parallel-loading** | **24.86** | **25.51** |
| v4-pipelined-helper | 20.52 | 12.67 |

Page-locking still carries the win: v4 is 1.70x over v3 here (2.24x there).

The pipelined-helper ablation is the interesting change. It was 2.0x *slower*
than the v4 default on the v4 box and is only 1.21x slower here — the per
sub-chunk event cost that sank it is much less dominant on this box. It still
loses, so it stays an ablation and not the production path, but the margin is
now small enough that it is worth re-running against the full 6-model set
before the question is called settled.

## Migration 0 -> 1 (3 reps)

| Model | Arm | latency | downtime | GB/s | path | NVLink Rx |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Qwen2.5-1.5B (2.88 GiB) | prototype-source-first | 0.267 s | 0.227 s | 11.6 | host + helper p2p | 1.44 GiB |
| | v3-target-first | 0.197 | 0.000 | 16.0 | host + helper p2p | 1.44 |
| | **v4-p2p-target-first** | **0.121** | 0.000 | **25.5** | gpu-to-gpu-p2p | **2.88** |
| Qwen2.5-3B (5.75 GiB) | prototype-source-first | 0.495 | 0.439 | 12.5 | host + helper p2p | 2.87 |
| | v3-target-first | 0.339 | 0.000 | 18.3 | host + helper p2p | 2.87 |
| | **v4-p2p-target-first** | **0.185** | 0.000 | **34.1** | gpu-to-gpu-p2p | **5.75** |
| Qwen2.5-7B (14.19 GiB) | prototype-source-first | 0.906 | 0.838 | 16.8 | host + helper p2p | 7.09 |
| | v3-target-first | 0.683 | 0.000 | 22.3 | host + helper p2p | 7.09 |
| | **v4-p2p-target-first** | **0.145** | 0.000 | **105.3** | gpu-to-gpu-p2p | **14.19** |

The path is measured, not assumed: on the broker path exactly half the model
crosses the link, on the P2P path the whole model does, and 105 GB/s is far
above any PCIe ceiling. Same signature as the v4 box, so the mechanism
reproduces.

The largest model gains most, which is the shape the v4 box showed too — but
more so: 105.3 GB/s here against 72.9 GB/s there for a comparable model.
**Do not carry the v4 box's migration latencies over.** Both boxes are NV12
all-pairs, but this one reaches the link's bandwidth more completely, and the
absolute numbers differ by 1.4x on the mechanism v4 exists to optimise.

## What is still owed

The 6-model set including Llama-3.1-8B, once HF_TOKEN is available, so these
are comparable to the v4 report model-for-model rather than only in shape.
