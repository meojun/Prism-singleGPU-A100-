# Llama 70B serving: **PASS**

Prototype 과의 성능 비교가 아니라 capability/stability 검증이다. 프로토타입은 worker-pool 경로에서 TP>1 을 아예 못 돌리므로 비교 대상이 없다.


## 판정

| 항목 | 결과 |
| --- | --- |
| startup: TP ranks created | PASS |
| startup: ranks on distinct GPUs (anti-affinity held) | PASS |
| basic inference succeeded | PASS |
| sustained window completed | PASS |
| no failed / abandoned requests in sustained | PASS |
| no CUDA/NCCL/OOM fatal in logs | PASS |
| no memory drift > 8 GiB per GPU | PASS |

## TP rank 배치

| tp_rank | GPU |
| ---: | --- |
| 0 | 0 |
| 1 | 1 |

한 rank 가 여러 GPU 로 보이면 그 디렉터리에서 여러 번 실행된 로그의 합집합이다. 한 번의 실행 안에서는 1:1 이다.

## Stage 2 — basic inference

```
{
  "requests": 4,
  "successful": 4,
  "failed": 0,
  "errors": [],
  "latency_s": {
    "mean": 1.7757946252822876,
    "all": [
      3.1917994022369385,
      1.2820706367492676,
      1.2957141399383545,
      1.3335943222045898
    ]
  }
}
```

## Stage 3 — sustained

* 지속 시간: 30.1 분
* 발송 1828 / 완료 1828 / 실패 0 / 미반환 0
* 처리량 1.011 req/s, 출력 토큰 105.6 tok/s
* TTFT p50 5.021 / p95 12.985 / p99 13.312 / max 14.913 (n=1828)
* TPOT p50 0.000 / p95 0.000 / p99 0.000 / max 0.000 (n=1828)
* E2E p50 5.021 / p95 12.985 / p99 13.312 / max 14.913 (n=1828)

## GPU 메모리 추이 (누수 징후)

| GPU | 시작(MiB) | 끝(MiB) | 변화 |
| ---: | ---: | ---: | ---: |
| 0 | 73283 | 73921 | +638 |
| 1 | 73309 | 73921 | +612 |
| 2 | 0 | 0 | +0 |
| 3 | 0 | 0 | +0 |

KV 캐시가 부하에 따라 늘고 주는 것은 정상이다. 여기서 보는 것은 창 전체에 걸친 단조 증가다. 8 GiB 를 넘으면 FAIL 로 잡는다.

## 로그의 치명적 패턴

| 패턴 | 건수 |
| --- | ---: |
| `CUDA error` | 0 |
| `NCCL WARN` | 0 |
| `NCCL error` | 0 |
| `out of memory` | 0 |
| `CUDA out of memory` | 0 |
| `Segmentation fault` | 0 |

## 환경

```
generated: 2026-08-21T05:44:29+00:00
model: meta-llama/Llama-3.1-70B
tp_size: 2
gpus: 0,1 of 4 A100-SXM4-80GB
calibration: reused; user confirmed same hardware (not used by simple-global)
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-435b8647-c13a-0cc8-d748-41679ddb08d1)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-bc692286-b1c5-2909-5328-a5a23087f4aa)
GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-915ac7c9-50de-d59c-e310-15a580b86c91)
GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-24a09821-2da2-18fe-7b33-87f43f38025f)
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	NV12	NV12	0-31,64-95	0		N/A
GPU1	NV12	 X 	NV12	NV12	32-63,96-127	1		N/A
GPU2	NV12	NV12	 X 	NV12	32-63,96-127	1		N/A
GPU3	NV12	NV12	NV12	 X 	32-63,96-127	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
