#!/usr/bin/env python3
"""
Calibrate per-ticker thresholds from JSONL diagnostics logs.

Usage:
  python3 daytrade/calibrate.py --logs logs/daytrade-20250909 [logs/daytrade-20250910 ...] \
                                --out daytrade/thresholds.json

Logic (simple, transparent):
- Read JSONL diagnostics written by the CLI/UI (one line per poll).
- For each ticker, derive thresholds from empirical distributions with guardrails:
  - Only use bias samples where activity is non-trivial (V5 ≥ 1000 and V15 ≥ 2000) to avoid
    saturated ±1.0 bias from ultra-thin prints.
  - Bias thresholds use p60 of |bias| with clamping to [0.30, 0.70].
  - Volume floors use p70 of V with floors: V5 ≥ 2000, V15 ≥ 4000.
  - If there are too few eligible samples (n < 10), fall back to all samples but keep clamping.
- Persistence is fixed at 2 by default (tune later if desired).

You can rerun this over additional days to refine the thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np


def read_jsonl(paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return rows


def collect_from_dirs(log_dirs: List[str]) -> Dict[str, List[dict]]:
    per_tkr: Dict[str, List[dict]] = {}
    for d in log_dirs:
        if not os.path.isdir(d):
            continue
        for path in glob(os.path.join(d, "*.jsonl")):
            ticker = os.path.splitext(os.path.basename(path))[0].upper()
            per_tkr.setdefault(ticker, []).extend(read_jsonl([path]))
    return per_tkr


def pct(v: List[float], q: float) -> float:
    if not v:
        return 0.0
    return float(np.percentile(np.array(v, dtype=float), q))


def weighted_pct(values: List[float], weights: List[float], q: float) -> float:
    """Weighted percentile of values at quantile q in [0,100].
    Returns 0.0 if inputs are empty or weights sum to zero.
    """
    if not values or not weights:
        return 0.0
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 0:
        return float(np.percentile(x, q))
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    target = (q / 100.0) * tot
    idx = np.searchsorted(cw, target, side='left')
    idx = min(max(int(idx), 0), len(x_sorted) - 1)
    return float(x_sorted[idx])


def propose_thresholds(rows: List[dict], *,
                       bias_q: float = 60.0,
                       bias_lo: float = 0.30,
                       bias_hi: float = 0.70,
                       vol_q: float = 70.0,
                       minV5_elig: float = 1000.0,
                       minV15_elig: float = 2000.0,
                       min_mix_ratio: float = 0.0,
                       min_samples: int = 10,
                       floorV5: float = 2000.0,
                       floorV15: float = 4000.0) -> Tuple[float, float, float, float, int]:
    b5s: List[float] = []
    b15s: List[float] = []
    v5s: List[float] = []
    v15s: List[float] = []
    c5s: List[float] = []
    p5s: List[float] = []
    c15s: List[float] = []
    p15s: List[float] = []
    for r in rows:
        try:
            i = r.get("intervals", {})
            b5s.append(abs(float(i.get("5m",{}).get("bias", 0.0))))
            b15s.append(abs(float(i.get("15m",{}).get("bias", 0.0))))
            v5s.append(float(i.get("5m",{}).get("V", 0.0)))
            v15s.append(float(i.get("15m",{}).get("V", 0.0)))
            c5s.append(float(i.get("5m",{}).get("C", 0.0)))
            p5s.append(float(i.get("5m",{}).get("P", 0.0)))
            c15s.append(float(i.get("15m",{}).get("C", 0.0)))
            p15s.append(float(i.get("15m",{}).get("P", 0.0)))
        except Exception:
            continue

    # Volume-aware filtering to avoid saturated ±1.0 biases on tiny prints
    def _mix_ratio(c: float, p: float) -> float:
        den = abs(c) + abs(p)
        if den <= 0:
            return 0.0
        return min(abs(c), abs(p)) / den

    eligible_idx = []
    for i, (_v5, _v15, _c5, _p5, _c15, _p15) in enumerate(zip(v5s, v15s, c5s, p5s, c15s, p15s)):
        if _v5 >= minV5_elig and _v15 >= minV15_elig:
            if min_mix_ratio <= 0:
                eligible_idx.append(i)
            else:
                if _mix_ratio(_c5, _p5) >= min_mix_ratio and _mix_ratio(_c15, _p15) >= min_mix_ratio:
                    eligible_idx.append(i)
    def _eligible(vals: List[float]) -> List[float]:
        if eligible_idx and len(eligible_idx) >= int(min_samples):
            return [vals[i] for i in eligible_idx]
        return vals  # fall back to all

    b5_pool = _eligible(b5s)
    b15_pool = _eligible(b15s)
    v5_pool = _eligible(v5s)
    v15_pool = _eligible(v15s)

    # Bias percentiles with clamping (weighted by volume to reduce ±1 saturation impact)
    B_Q = float(bias_q)
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    b5_raw = weighted_pct(b5_pool, v5_pool, B_Q) if (b5_pool and v5_pool) else pct(b5_pool, B_Q)
    b15_raw = weighted_pct(b15_pool, v15_pool, B_Q) if (b15_pool and v15_pool) else pct(b15_pool, B_Q)
    b5 = _clamp(b5_raw, float(bias_lo), float(bias_hi))
    b15 = _clamp(b15_raw, max(0.20, float(bias_lo)*0.67), float(bias_hi))

    # Volume floors: use p70 with sensible minimums
    V5 = max(float(floorV5), pct(v5s, float(vol_q)))
    V15 = max(float(floorV15), pct(v15s, float(vol_q)))
    persist = 2
    return b5, b15, V5, V15, persist


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate per-ticker thresholds from JSONL logs")
    ap.add_argument("--logs", nargs="+", required=True, help="One or more log folders (e.g., logs/daytrade-YYYYMMDD)")
    ap.add_argument("--out", required=True, help="Output thresholds.json path")
    ap.add_argument("--bias_quantile", type=float, default=60.0, help="Quantile (0-100) for |bias| calibration (default: 60)")
    ap.add_argument("--bias_lo", type=float, default=0.30, help="Lower clamp for bias thresholds (default: 0.30)")
    ap.add_argument("--bias_hi", type=float, default=0.70, help="Upper clamp for bias thresholds (default: 0.70)")
    ap.add_argument("--vol_quantile", type=float, default=70.0, help="Quantile (0-100) for V calibration (default: 70)")
    ap.add_argument("--minV5_elig", type=float, default=1000.0, help="Eligibility gate for 5m volume (default: 1000)")
    ap.add_argument("--minV15_elig", type=float, default=2000.0, help="Eligibility gate for 15m volume (default: 2000)")
    ap.add_argument("--min_mix_ratio", type=float, default=0.0, help="Min min(|C|,|P|)/( |C|+|P| ) on 5m & 15m to be eligible (default: 0.0)")
    ap.add_argument("--min_samples", type=int, default=10, help="Min eligible rows before falling back to all (default: 10)")
    ap.add_argument("--floorV5", type=float, default=2000.0, help="Floor for minV5 (default: 2000)")
    ap.add_argument("--floorV15", type=float, default=4000.0, help="Floor for minV15 (default: 4000)")
    args = ap.parse_args()

    per_ticker_rows = collect_from_dirs(args.logs)
    out: Dict[str, dict] = {}
    for tkr, rows in per_ticker_rows.items():
        if not rows:
            continue
        b5, b15, V5, V15, persist = propose_thresholds(
            rows,
            bias_q=args.bias_quantile,
            bias_lo=args.bias_lo,
            bias_hi=args.bias_hi,
            vol_q=args.vol_quantile,
            minV5_elig=args.minV5_elig,
            minV15_elig=args.minV15_elig,
            min_mix_ratio=args.min_mix_ratio,
            min_samples=args.min_samples,
            floorV5=args.floorV5,
            floorV15=args.floorV15,
        )
        out[tkr] = {
            "thresh5": round(b5, 3),
            "thresh15": round(b15, 3),
            "minV5": float(f"{V5:.0f}"),
            "minV15": float(f"{V15:.0f}"),
            "persistence": persist,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote thresholds for {len(out)} tickers -> {args.out}")


if __name__ == "__main__":
    main()
