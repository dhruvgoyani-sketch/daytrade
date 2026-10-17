#!/usr/bin/env python3
"""Simulate the paper's failure mode with no market data.

Price is a pure random walk, so the true predictability of any signal is zero.
The signal is a persistent random long/short state, polled every 15 seconds
like the engine. We test it against overlapping 15-minute forward returns two
ways and count how often each test (wrongly) finds an effect at the 5% level.

A valid test should reject about 5% of the time.

This illustrates the mechanism. It is not a replication of the paper's
numbers: the simulated signal's persistence is not tuned to the real one.

Run:  python analysis/simulate_overlap.py [--reps 200]
"""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--reps", type=int, default=200)
p.add_argument("--days", type=int, default=150)
p.add_argument("--polls", type=int, default=1536, help="polls per day (15 s cadence)")
p.add_argument("--horizon", type=int, default=60, help="forward window in polls (60 = 15 min)")
p.add_argument("--stay", type=float, default=0.98, help="prob. signal keeps its sign each poll")
a = p.parse_args()

rng = np.random.default_rng(20261008)
naive_rej = clustered_rej = 0
for _ in range(a.reps):
    day_means, all_e = [], []
    for _ in range(a.days):
        steps = rng.standard_normal(a.polls + a.horizon)
        price = np.concatenate([[0.0], np.cumsum(steps)])
        fwd = price[a.horizon:a.horizon + a.polls] - price[:a.polls]
        flips = rng.random(a.polls) > a.stay
        sign = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0) * rng.choice([-1.0, 1.0])
        e = sign * fwd
        all_e.append(e)
        day_means.append(e.mean())
    e = np.concatenate(all_e)
    t_naive = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
    d = np.array(day_means)
    t_clust = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    naive_rej += abs(t_naive) > 1.96
    clustered_rej += abs(t_clust) > 1.96

print(f"True predictability: 0. Replications: {a.reps}")
print(f"Naive test rejects:           {naive_rej / a.reps:.1%}")
print(f"Day-clustered test rejects:   {clustered_rej / a.reps:.1%}  (valid test: ~5%)")
