#!/usr/bin/env python3
"""sampling_rate_law.py -- post-acceptance extension. Not in the CSIT 2026 paper.

Question: how fast can a signal be sampled before the naive t-test starts
finding effects that are not there?

The paper evaluates the engine at its native ~15 s polling against 15-minute
forward returns. Here the same signal, on the same 150 trading days, is thinned
to one poll per interval (15 s up to 15 min). The 15-minute forward return is
defined exactly as in paper_stats.py. For each interval this reports:

  * the naive t and the day-clustered t of the real signal;
  * how often random placebo signals pass each test at the 5% level
    (placebo = the paper's within-day circular shift of the real signal);
  * the naive pass rate predicted from the committed autocorrelation alone.

Usage:
  python extensions/sampling_rate_law.py --predict   # no data needed; run in CI
  python extensions/sampling_rate_law.py --plot      # redraw figure from committed CSV
  python extensions/sampling_rate_law.py             # full run, needs raw logs

The full run reads the same logs as analysis/paper_stats.py (DT_LOGDIR) and
writes extensions/results/sampling_rate_law.csv and the figure.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(HERE, "results")
ACF = os.path.join(ROOT, "analysis", "paper_stats", "acf_signed_15m.csv")
H = 15                                   # forward-return horizon, minutes
INTERVALS = [15, 30, 60, 120, 300, 600, 900]   # seconds between kept polls
N_PLACEBO = 1000
TRUNC_LAG = 55                           # same truncation the paper uses at 15 min


def predicted(acf_path: str = ACF) -> pd.DataFrame:
    """Variance inflation and naive false-positive rate from the committed ACF.

    The ACF is indexed in directional polls (~15 s apart). Keeping every k-th
    poll keeps the autocorrelations at lags k, 2k, 3k, ...; the naive test's
    variance is understated by VIF = 1 + 2 * sum of those autocorrelations,
    so a zero-effect signal passes |t| > 1.96 with probability
    2 * (1 - Phi(1.96 / sqrt(VIF))).
    """
    a = pd.read_csv(acf_path)
    rho = dict(zip(a["lag_polls"], a["acf"]))
    rows = []
    for sec in INTERVALS:
        k = max(1, round(sec / 15))
        s = sum(rho.get(j * k, 0.0) for j in range(1, TRUNC_LAG // k + 1))
        vif = max(1.0 + 2.0 * s, 1.0)
        rows.append(dict(interval_s=sec, keep_every_k=k, vif_pred=vif,
                         t_inflation_pred=math.sqrt(vif),
                         naive_pass_pred=2 * (1 - norm.cdf(1.96 / math.sqrt(vif)))))
    return pd.DataFrame(rows)


def thin(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    """Keep the first poll in each `sec`-second bucket of each trading day."""
    if sec <= 15:
        return df
    t = df["ts"].astype("datetime64[ns, UTC]").astype("int64") // 10**9
    t0 = t.groupby(df["day"]).transform("min")
    bucket = (t - t0) // sec
    return df[~pd.DataFrame({"d": df["day"], "b": bucket}).duplicated()]


def tstats(days: np.ndarray, e: np.ndarray):
    ok = ~np.isnan(e)
    x, d = e[ok], days[ok]
    t_naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    dm = pd.Series(x).groupby(d).mean().to_numpy()
    t_day = dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm)))
    return len(x), len(dm), t_naive, t_day


def placebo_pass(g: pd.DataFrame, rng: np.random.Generator):
    """Share of within-day circularly shifted signals that pass each test."""
    days = []
    for day, gd in g.groupby("day"):
        sig = gd["signal"].to_numpy()
        d = np.where(sig == "long", 1.0, np.where(sig == "short", -1.0, np.nan))
        days.append((day, d, gd[f"r{H}"].to_numpy()))
    naive = clus = 0
    for _ in range(N_PLACEBO):
        es, ds = [], []
        for day, d, r in days:
            n = len(d)
            if n < 4:
                continue
            e = np.roll(d, int(rng.integers(1, n))) * r
            es.append(e)
            ds.append(np.full(n, day))
        _, _, tn, tc = tstats(np.concatenate(ds), np.concatenate(es))
        naive += abs(tn) > 1.96
        clus += abs(tc) > 1.96
    return naive / N_PLACEBO, clus / N_PLACEBO


def full_run() -> pd.DataFrame:
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import paper_stats as ps                      # same loaders and conventions
    df = ps.add_forward(ps.load_spx(), 0)
    rng = np.random.default_rng(20261027)
    rows = []
    for sec in INTERVALS:
        g = thin(df, sec)
        e = ps.signed(g, H)
        n, D, tn, tc = tstats(g["day"].to_numpy(), e)
        pn, pc = placebo_pass(g, rng)
        rows.append(dict(interval_s=sec, n_obs=n, days=D, naive_t=tn, day_clustered_t=tc,
                         placebo_naive_pass=pn, placebo_clustered_pass=pc))
        print(f"{sec:>4}s  n={n:>6}  naive t={tn:5.2f}  clustered t={tc:5.2f}  "
              f"placebo pass: naive {pn:5.1%}, clustered {pc:5.1%}", flush=True)
    return pd.DataFrame(rows).merge(predicted(), on="interval_s")


def plot(res: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = res["interval_s"].to_numpy()
    ticks = [15, 30, 60, 120, 300, 900]
    labels = ["15 s", "30 s", "1 min", "2 min", "5 min", "15 min"]
    accent, dark, grey = "#b2182b", "#222222", "#888888"
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))

    a.plot(x, 100 * res["placebo_naive_pass"], "o-", color=accent, label="naive test (measured)")
    a.plot(x, 100 * res["naive_pass_pred"], "--", color=grey, label="naive test (predicted from ACF)")
    a.plot(x, 100 * res["placebo_clustered_pass"], "s-", color=dark, label="day-clustered test")
    a.axhline(5, color=grey, lw=0.8, ls=":")
    a.text(16, 6.5, "5% = what a valid test should give", ha="left", va="bottom", color=grey, fontsize=8)
    a.set_ylabel("random signals that 'pass' (%)")
    a.set_title("How often a random signal looks significant", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")

    b.plot(x, res["naive_t"], "o-", color=accent, label="naive t")
    b.plot(x, res["day_clustered_t"], "s-", color=dark, label="day-clustered t")
    b.axhline(1.96, color=grey, lw=0.8, ls=":")
    b.text(900, 2.15, "|t| = 1.96", ha="right", va="bottom", color=grey, fontsize=8)
    b.set_ylabel("t-statistic of the real signal")
    b.set_title("The same signal, sampled at different speeds", fontsize=10)
    b.legend(frameon=False, fontsize=8)

    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("time between polls (15-minute forward return)")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true",
                    help="prediction from the committed ACF only; needs no market data")
    ap.add_argument("--plot", action="store_true",
                    help="redraw the figure from the committed results CSV; needs no market data")
    a = ap.parse_args()
    csv = os.path.join(RES, "sampling_rate_law.csv")
    if a.predict:
        p = predicted()
        for r in p.itertuples():
            print(f"every {r.interval_s:>4}s: naive t inflated x{r.t_inflation_pred:4.2f}, "
                  f"random signal passes naive test {r.naive_pass_pred:5.1%}")
        if os.path.exists(csv):            # the committed results must agree with the prediction code
            ref = pd.read_csv(csv).set_index("interval_s")
            got = p.set_index("interval_s")
            cols = ["vif_pred", "naive_pass_pred"]
            if not np.allclose(got[cols].to_numpy(), ref.loc[got.index, cols].to_numpy(), rtol=1e-9):
                print("MISMATCH with", csv)
                sys.exit(1)
            print("Prediction matches extensions/results/sampling_rate_law.csv.")
        return
    if a.plot:
        plot(pd.read_csv(csv), os.path.join(HERE, "fig_sampling_rate.png"))
        return
    os.makedirs(RES, exist_ok=True)
    out = full_run()
    out.to_csv(os.path.join(RES, "sampling_rate_law.csv"), index=False)
    print("wrote", os.path.join(RES, "sampling_rate_law.csv"))
    plot(out, os.path.join(HERE, "fig_sampling_rate.png"))


if __name__ == "__main__":
    main()
