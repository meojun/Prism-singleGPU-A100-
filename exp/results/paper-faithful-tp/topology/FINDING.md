# 이 박스의 토폴로지와 TP 전제 조건

TP 는 all-reduce 가 매 디코드 스텝에 있으므로, 인터커넥트가 PCIe-only 면 TP 수치
자체가 무의미하다. 그래서 구현 전에 먼저 재고 기록한다.

## 결과: 통과

4 x NVIDIA A100-SXM4-80GB. 모든 GPU 쌍이 **NV12** (12개 NVLink 본딩, 링크당
25 GB/s) 로 연결되어 있고 NVSwitch 를 거친다. `topology.txt` 가 원본이다.

```
     GPU0  GPU1  GPU2  GPU3
GPU0  X    NV12  NV12  NV12
GPU1 NV12   X    NV12  NV12
GPU2 NV12  NV12   X    NV12
GPU3 NV12  NV12  NV12   X
```

peer access 는 12개 순서쌍 전부 `true` 이며 `all_pairs_peer: true` 다
(`peer_access.json`). GPU 2장짜리 결과를 4장 토폴로지로 확대 해석하지 않기 위해
**4장 전부에 대해** 측정했다.

## 이것이 무엇을 허용하는가

* TP=2 를 서로 다른 **6가지** GPU 쌍 중 어디에 놓아도 인터커넥트가 동등하다.
  {0,1} {0,2} {0,3} {1,2} {1,3} {2,3} 중 어느 것을 골라도 NV12 다.
  → anti-affinity 검증에서 배치가 달라져도 **인터커넥트 품질이 교란변수가 되지
  않는다.** 2장짜리 박스에서는 선택지가 하나뿐이라 이 통제가 불가능했다.
* TP=4 는 전 GPU 를 쓰므로 배치 선택지가 하나이고, 제약이 자동 만족된다.
  돌아간다는 증거로만 1회 남긴다.

## 유의

`nvidia-smi topo -m` 의 NUMA affinity 는 GPU0 만 노드 0 이고 GPU1-3 은 노드 1 이다.
호스트 메모리를 경유하는 경로(가중치 로딩의 CPU 측)는 GPU0 과 나머지가 비대칭일 수
있다. GPU 간 경로는 NVSwitch 라 대칭이다.
