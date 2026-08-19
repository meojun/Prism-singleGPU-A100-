#!/usr/bin/env python3
"""Read the V5_2 instrumentation back out and say where the two costs go."""
import argparse, glob, json, math, os, re, statistics as st
from collections import defaultdict
from pathlib import Path


def logs(run_dir, pat):
    out = []
    for p in glob.glob(os.path.join(run_dir, "server-logs", "*.log")):
        try:
            for line in open(p, errors="replace"):
                i = line.find(pat)
                if i >= 0:
                    try: out.append(json.loads(line[i + len(pat):]))
                    except json.JSONDecodeError: pass
        except OSError: pass
    return out


def ms(v, d=3):
    v = [x for x in v if x is not None and not math.isnan(x)]
    if not v: return "—"
    return f"{st.fmean(v):.{d}f}" + (f" ± {st.stdev(v):.{d}f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True)
    base = Path(ap.parse_args().base)

    print("# V5_2 — where the two unexplained costs actually go\n")
    print("계측만 넣었고 동작은 바꾸지 않았다. 두 질문에 답한다.\n")

    # ---------------- (a) scheduler loop
    print("## (가) Algorithm 2 를 켜면 왜 goodput 이 사라지는가\n")
    print("released-prototype 과 v3-alg2only 는 GPU 스케줄러 루프 안의 함수 하나만 다르다.")
    print("그 루프를 구간별로 잰다. 값은 iteration 당 밀리초.\n")
    print("| Arm | seed | iters | 루프 주기 | 메모리읽기 | Redis | admission | 디스패치 | 최대 iter |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    per_arm = defaultdict(lambda: defaultdict(list))
    for done in sorted(glob.glob(str(base / "raw/*/steady/rate_8/seed_*/DONE"))):
        d = os.path.dirname(done); parts = Path(d).parts
        arm, seed = parts[-4], parts[-1].split("_")[1]
        recs = logs(d, "[V5-LOOP] ")
        if not recs: continue
        it = sum(r["iters"] for r in recs)
        if not it: continue
        f = lambda k: 1000 * sum(r[k] for r in recs) / it
        row = dict(iters=it, iter=f("t_iter"), mem=f("t_mem"), redis=f("t_redis"),
                   admit=f("t_admit"), send=f("t_send"),
                   imax=1000 * max(r["iter_max"] for r in recs))
        for k, v in row.items(): per_arm[arm][k].append(v)
        print(f"| {arm} | {seed} | {it} | {row['iter']:.3f} | {row['mem']:.3f} | "
              f"{row['redis']:.3f} | {row['admit']:.3f} | {row['send']:.3f} | {row['imax']:.1f} |")
    if len(per_arm) >= 2:
        print("\n| Arm | 루프 주기 (ms) | admission (ms) | Redis (ms) |")
        print("| --- | ---: | ---: | ---: |")
        for arm, d in per_arm.items():
            print(f"| {arm} | {ms(d['iter'])} | {ms(d['admit'])} | {ms(d['redis'])} |")
        arms = list(per_arm)
        proto = next((a for a in arms if "prototype" in a), None)
        mh = next((a for a in arms if "alg2only" in a), None)
        if proto and mh:
            di = st.fmean(per_arm[mh]['iter']) - st.fmean(per_arm[proto]['iter'])
            da = st.fmean(per_arm[mh]['admit']) - st.fmean(per_arm[proto]['admit'])
            print(f"\nMoore-Hodgson 을 켜면 루프 주기가 **{di:+.3f} ms**, 그 중 admission 이 "
                  f"**{da:+.3f} ms** 다.")
            print("루프가 길어지면 새로 도착한 요청이 디스패치까지 더 기다린다. 8 req/s 에서")
            print(f"평균 추가 대기는 루프 주기 증가의 절반, 약 **{di/2:+.3f} ms** 이다.")
            print("관측된 TTFT 차이와 비교하면 이 경로가 설명하는 몫을 알 수 있다.")

    # ---------------- (b) deactivation hops
    print("\n## (나) deactivation 의 82% 는 어디에 있는가\n")
    print("[V5-HOP] 은 제어 요청을 보낸 시점과 엔진의 응답이 돌아온 시점을 잰다.")
    print("엔진 자체 teardown 은 v4 측정에서 평균 0.96 초였다.\n")
    print("| Arm | seed | action | n | 등록·전송 (s) | 엔진 대기 (s) | 총 (s) |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    agg = defaultdict(list)
    for done in sorted(glob.glob(str(base / "raw/*/*/rate_*/seed_*/DONE"))):
        d = os.path.dirname(done); parts = Path(d).parts
        arm, seed = parts[-4], parts[-1].split("_")[1]
        recs = logs(d, "[V5-HOP] ")
        by = defaultdict(list)
        for r in recs: by[r["action"]].append(r)
        for act, rs in sorted(by.items()):
            print(f"| {arm} | {seed} | {act} | {len(rs)} | "
                  f"{ms([r['register_s'] for r in rs])} | "
                  f"{ms([r['wait_for_engine_s'] for r in rs])} | "
                  f"{ms([r['total_s'] for r in rs])} |")
            if act == "DeactivateReqInput":
                agg["wait"].extend(r["wait_for_engine_s"] for r in rs)
    if agg["wait"]:
        w = st.fmean(agg["wait"])
        print(f"\nDeactivate 의 엔진 대기 평균 **{w:.2f} 초** 대 엔진 자체 teardown 0.96 초.")
        print(f"차이 **{w - 0.96:.2f} 초**가 요청이 스케줄러에 도달해 엔진이 실제로 teardown 을")
        print("시작하기까지의 구간이다 — 스케줄러 루프가 그 요청을 집어들 때까지의 대기가")
        print("여기에 포함된다. (가)의 루프 주기와 같은 원인일 수 있다.")
    print()


if __name__ == "__main__":
    main()
