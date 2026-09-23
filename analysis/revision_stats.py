#!/usr/bin/env python3
"""revision_stats.py -- additional statistics requested in the CSIT 2026 review.
Reuses the loaders and estimators in paper_stats.py so conventions are identical.
Writes analysis/paper_stats/revision_stats.json
"""
import json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_stats as ps

OUT = ps.OUT
res = {}
df = ps.load_spx()
df = ps.add_forward(df, 0)
days = df["day"].to_numpy()
cad = 14.994177103042603

# ---- (R2.5) day accounting ----
alld = sorted(df["day"].unique())
dirmask = df["signal"].isin(["long", "short"])
dird = set(df.loc[dirmask, "day"])
res["days_total"] = len(alld)
res["days_no_directional"] = [d for d in alld if d not in dird]
e15 = ps.signed(df, 15)
d_with_e15 = set(df.loc[~np.isnan(e15), "day"])
res["days_no_valid_edge15"] = [d for d in alld if d not in d_with_e15]
res["polls_on_excluded_days"] = {d: int((df.day == d).sum()) for d in res["days_no_valid_edge15"]}

# ---- (R2.1) overlap table ----
def all_poll_neighbors(h):
    tot = cnt = 0
    for _, gd in df.groupby("day"):
        t = gd["ts"].astype("int64").to_numpy() / 1e9
        lo = np.searchsorted(t, t - h * 60, side="left")
        hi = np.searchsorted(t, t + h * 60, side="right")
        tot += (hi - lo - 1).sum(); cnt += len(t)
    return tot / cnt

def dir_neighbors(h):
    tot = cnt = 0
    for _, gd in df.groupby("day"):
        e = ps.signed(gd, h)
        t = (gd["ts"].astype("int64").to_numpy() / 1e9)[~np.isnan(e)]
        if len(t) < 2: continue
        lo = np.searchsorted(t, t - h * 60, side="left")
        hi = np.searchsorted(t, t + h * 60, side="right")
        tot += (hi - lo - 1).sum(); cnt += len(t)
    return tot / cnt

edge = pd.read_csv(os.path.join(OUT, "directional_edge.csv"))
rows = []
for h in ps.H:
    acf = ps.acf_by_minutes(df, h, max_lag_polls=400)
    rho = acf["acf"].to_numpy()
    upto = acf[acf["lag_med_min"] <= h + 10].index.max()
    srho = float(np.clip(rho[: upto + 1], 0, None).sum())
    n = int(edge.loc[edge.horizon == h, "n_polls"].iloc[0])
    fac = 1 + 2 * srho
    rows.append(dict(H=h, polls_per_window=h * 60 / cad,
                     nbr_all=all_poll_neighbors(h), nbr_dir=dir_neighbors(h),
                     rho1=float(rho[0]), sum_rho=srho, trunc_lag=int(upto + 1),
                     factor=fac, sqrt_factor=math.sqrt(fac), n=n, n_eff=n / fac))
res["overlap_table"] = rows

# ---- (R2.2) decomposition of the t drop ----
dec = []
for h in ps.H:
    e = ps.signed(df, h)
    tmp = pd.DataFrame({"day": days, "x": e}).dropna()
    n = len(tmp); m_pw = tmp.x.mean()
    se_naive = tmp.x.std(ddof=1) / math.sqrt(n)
    # cluster-robust (CR1) SE of the poll-weighted mean, clusters = days
    g = tmp.groupby("day")["x"].agg(["sum", "count"])
    u = g["sum"] - g["count"] * m_pw
    G = len(g)
    se_cr = math.sqrt((G / (G - 1)) * (u ** 2).sum()) / n
    cl = ps.clustered_stats(days, e)
    dec.append(dict(H=h, mean_poll=m_pw, se_naive=se_naive, t_naive=m_pw / se_naive,
                    se_poll_cluster=se_cr, t_poll_cluster=m_pw / se_cr,
                    mean_day=cl["mean_daily"], se_day=cl["se"], t_day=cl["t"],
                    se_ratio_day_vs_naive=cl["se"] / se_naive,
                    mean_ratio=m_pw / cl["mean_daily"] if cl["mean_daily"] else None,
                    t_ratio=(m_pw / se_naive) / cl["t"]))
    # covariance identity: poll mean = day mean + cov(w_d, ebar_d), w_d = n_d/nbar
    ebar = tmp.groupby("day")["x"].mean(); nd = g["count"]
    w = nd / nd.mean()
    dec[-1]["cov_term"] = float(((w - 1) * ebar).mean())
res["decomposition"] = dec

# ---- (R2.4) minimum detectable effect, D days, 80% power, two-sided 5% ----
mde = []
for r in dec:
    D = 150
    tc = sps.t.ppf(0.975, D - 1); tp = sps.t.ppf(0.80, D - 1)
    mde.append(dict(H=r["H"], se_day=r["se_day"], mde=(tc + tp) * r["se_day"],
                    sd_daily=r["se_day"] * math.sqrt(D)))
res["mde"] = mde

# ---- (R2.6) naive |t| under the null: theory + placebo ----
fac15 = [r for r in rows if r["H"] == 15][0]["factor"]
k = math.sqrt(fac15)
res["theory_naive_reject_share_15"] = float(2 * (1 - sps.norm.cdf(1.96 / k)))
rng = np.random.default_rng(20260922)
dd = []
for _, gd in df.groupby("day"):
    d = np.where(gd.signal.to_numpy() == "long", 1.0, np.where(gd.signal.to_numpy() == "short", -1.0, np.nan))
    dd.append((d, gd["r15"].to_numpy()))
tn, tc_ = [], []
for rep in range(200):
    allx, dm = [], []
    for d, r in dd:
        if len(d) < 4: continue
        x = np.roll(d, int(rng.integers(1, len(d)))) * r
        x = x[~np.isnan(x)]
        if len(x): allx.append(x); dm.append(x.mean())
    a = np.concatenate(allx); dm = np.array(dm)
    tn.append(a.mean() / (a.std(ddof=1) / math.sqrt(len(a))))
    tc_.append(dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm))))
tn = np.abs(tn); tc_ = np.abs(tc_)
res["placebo_naive_15"] = dict(share_gt_196=float((tn > 1.96).mean()), mean_abs=float(tn.mean()),
                               p95=float(np.percentile(tn, 95)))
res["placebo_cluster_15_rerun"] = dict(share_gt_196=float((tc_ > 1.96).mean()), p95=float(np.percentile(tc_, 95)))

# ---- (minor) signal-state means: poll vs day weighting, short side ----
sr = []
for sig in ["long", "flat", "short"]:
    sub = df[df.signal == sig].dropna(subset=["r15"])
    g2 = sub.groupby("day")["r15"].agg(["mean", "count"])
    sr.append(dict(signal=sig, poll_mean=float(sub.r15.mean()), day_mean=float(g2["mean"].mean()),
                   corr_count_mean=float(np.corrcoef(g2["count"], g2["mean"])[0, 1]),
                   top_decile_share=float(g2["count"].sort_values(ascending=False).head(15).sum() / g2["count"].sum())))
res["state_weighting_15"] = sr

# ---- (minor) long vs short frequency ----
cnt = df[dirmask].groupby(["day", "signal"]).size().unstack(fill_value=0)
res["days_short_gt_long"] = int((cnt["short"] > cnt["long"]).sum())
res["days_with_dir"] = int(len(cnt))
res["median_daily_short_share"] = float((cnt["short"] / (cnt["short"] + cnt["long"])).median())
# raw gate direction before hysteresis/veto: all four gates + agree, sign of b5
raw_pass = df.b5p & df.v5p & df.b15p & df.v15p & df.agree
res["raw_gate_long"] = int((raw_pass & (df.b5 > 0)).sum())
res["raw_gate_short"] = int((raw_pass & (df.b5 < 0)).sum())
# day-level SPX return vs short share
dret = df.groupby("day")["spot"].agg(lambda s: 1e4 * (s.iloc[-1] / s.iloc[0] - 1))
ss = (cnt["short"] / (cnt["short"] + cnt["long"]))
res["corr_shortshare_dayret"] = float(np.corrcoef(ss, dret.loc[ss.index])[0, 1])

# ---- (R2.7) V15 floor change: post-change subsample ----
post = df[df.day >= "20251016"]
sub_rows = []
for h in ps.H:
    e = ps.signed(post, h)
    cl = ps.clustered_stats(post.day.to_numpy(), e)
    nv = ps.naive_stats(e)
    sub_rows.append(dict(H=h, D=cl["D"], mean=cl["mean_daily"], t=cl["t"], naive_t=nv["t"]))
res["post_20251016"] = sub_rows
res["pre_change_days"] = int((pd.Series(alld) < "20251016").sum())

# ---- moneyness band share ----
res["mny_band_002_share_pct"] = float(100 * (df.mny_band == 0.02).mean())

json.dump(res, open(os.path.join(OUT, "revision_stats.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
