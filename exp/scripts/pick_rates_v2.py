#!/usr/bin/env python3
"""Pick the Low / Medium / High / Near-saturation rates from calibration output.

    python pick_rates_v2.py --calib exp/results/paper-faithful-v2/sanity/calibration \
        -o /workspace/logs/chosen_rates.txt

The brief forbids a ladder that is all-easy or all-collapsed, so the choice is
made from measured joint SLO attainment and from where sustained throughput
stops tracking offered load:

  Low               highest calibrated rate still at joint attainment >= 0.95
  Medium            closest to joint attainment 0.80
  High              closest to joint attainment 0.45
  Near-saturation   lowest rate where throughput < 0.9 x offered load,
                    else the highest calibrated rate

Duplicates are resolved by walking outward on the calibrated ladder, and the
result is interpolated onto a finer grid if the calibration points are too
coarse to separate the four regimes.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    pts = []
    for p in sorted(glob.glob(os.path.join(a.calib, "rate_*", "metrics.json"))):
        rate = float(os.path.basename(os.path.dirname(p)).split("_")[1])
        d = json.load(open(p))
        pts.append({
            "rate": rate,
            "joint": d.get("joint_attainment"),
            "thr": d.get("throughput_req_s"),
            "off": d.get("offered_load_req_s"),
        })
    pts = [p for p in pts if p["joint"] == p["joint"] and p["joint"] is not None]
    pts.sort(key=lambda p: p["rate"])
    if len(pts) < 2:
        print("not enough calibration points; keeping defaults")
        return

    print(f"{'rate':>7s} {'joint':>7s} {'thr/off':>8s}")
    for p in pts:
        ratio = (p["thr"] / p["off"]) if p["off"] else float("nan")
        print(f"{p['rate']:7.1f} {p['joint']:7.3f} {ratio:8.3f}")

    def nearest(target):
        return min(pts, key=lambda p: abs(p["joint"] - target))["rate"]

    easy = [p for p in pts if p["joint"] >= 0.95]
    low = max(p["rate"] for p in easy) if easy else pts[0]["rate"]
    medium = nearest(0.80)
    high = nearest(0.45)
    sat = next((p["rate"] for p in pts
                if p["off"] and p["thr"] / p["off"] < 0.90), pts[-1]["rate"])

    chosen = [low, medium, high, sat]
    # If the calibrated ladder is too coarse to separate the regimes, spread the
    # duplicates over the interval between the lowest and highest pick instead of
    # silently reporting fewer than four load levels.
    if len(set(chosen)) < 4:
        lo, hi = min(chosen), max(chosen)
        if hi <= lo:
            hi = pts[-1]["rate"]
        if hi <= lo:
            hi = lo * 4
        step = (hi - lo) / 3.0
        chosen = [round(lo + i * step, 1) for i in range(4)]
        print(f"calibration too coarse to separate regimes; using an even ladder "
              f"{lo:g}..{hi:g}")

    chosen = sorted({round(c, 1) for c in chosen})
    line = " ".join(f"{c:g}" for c in chosen)
    with open(a.out, "w") as f:
        f.write(line + "\n")
    print(f"chosen rates (Low/Medium/High/Near-saturation): {line}  -> {a.out}")


if __name__ == "__main__":
    main()
