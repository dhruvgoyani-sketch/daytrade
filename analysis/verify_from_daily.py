#!/usr/bin/env python3
"""Recompute the paper's day-level results from the committed daily series.

Needs no market data. Reads analysis/paper_stats/daily_edge_15m.csv
(one mean signed 15-minute return per trading day, in basis points).

The bootstrap interval can differ from the paper in the last digit:
paper_stats.py draws it from a random generator shared with the placebo
and all three horizons, so a standalone re-draw lands slightly differently.

Exits with an error if any result differs from the committed paper outputs.

Run:  python analysis/verify_from_daily.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
x = pd.read_csv(os.path.join(HERE, "paper_stats", "daily_edge_15m.csv"))["edge_bps"].to_numpy()
D = len(x)
se = x.std(ddof=1) / np.sqrt(D)
t = x.mean() / se
p_sign = stats.binomtest(int((x > 0).sum()), D).pvalue

rng = np.random.default_rng(0)
boot = rng.choice(x, size=(10000, D), replace=True).mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
mde = (stats.t.ppf(0.975, D - 1) + stats.t.ppf(0.80, D - 1)) * se

print(f"Trading days used:              {D}")
print(f"Mean signed return (bps):       {x.mean():.4f}")
print(f"Day-clustered t:                {t:.3f}")
print(f"Days signal was right:          {(x > 0).mean():.1%}  (sign test p = {p_sign:.4f})")
print(f"Day bootstrap 95% CI (bps):     [{lo:.2f}, {hi:.2f}]")
print(f"Min detectable effect (bps):    {mde:.2f}  (80% power, 5% two-sided)")

# Check against the numbers the paper reports (written by paper_stats.py and revision_stats.py).
ref = pd.read_csv(os.path.join(HERE, "paper_stats", "directional_edge.csv")).set_index("horizon").loc[15]
rev = json.load(open(os.path.join(HERE, "paper_stats", "revision_stats.json")))
ref_mde = next(m["mde"] for m in rev["mde"] if m["H"] == 15)
checks = {
    "trading days": (D, ref["cl_D"]),
    "mean signed return": (x.mean(), ref["cl_mean_bps"]),
    "day-clustered t": (t, ref["cl_t"]),
    "sign test p": (p_sign, ref["sign_test_p"]),
    "min detectable effect": (mde, ref_mde),
}
bad = [k for k, (got, want) in checks.items() if not np.isclose(got, want, rtol=1e-9, atol=1e-12)]
if bad:
    print("MISMATCH with paper outputs:", ", ".join(bad))
    sys.exit(1)
print("All day-level numbers match analysis/paper_stats/.")
