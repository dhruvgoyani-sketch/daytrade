#!/usr/bin/env python3
"""
paper_stats.py -- single source of truth for every statistic in the paper
"Overlapping-Observation t-Inflation in Evaluating Algorithmic Intraday Trading
Signals: A Reproducible Null on S&P 500 Option Flow" (SPX only).

Reads  : logs/daytrade-*/SPX.jsonl   (the engine's own per-poll diagnostic logs)
Writes : analysis/paper_stats/*.csv  + stats.json (headline numbers cited in the paper)
         analysis/paper_stats/polls.parquet (cached per-poll table)

Conventions:
  * a poll is valid iff the JSON line parses and has ts_ny and spot>0;
  * signal = decision.signal (post-gate, post-wall-veto, post-hysteresis);
  * forward return r_{t,H} = 1e4*(S_{t+H}/S_t - 1), S_{t+H} = spot of the first
    poll at/after t+H *within the same day*, accepted iff the realised gap lies
    in [(H-1)*60, (H+10)*60] seconds;
  * signed directional edge e = d*r with d=+1 (long), -1 (short), flat excluded;
  * day-clustered inference: mean of within-day means across D days.

Run:  python3 analysis/paper_stats.py [--fresh]
"""
import glob
import json
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

try:
    import orjson as _json

    def _loads(line):
        return _json.loads(line)
except Exception:  # pragma: no cover
    def _loads(line):
        return json.loads(line)

from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.environ.get("DT_LOGDIR", os.path.join(HERE, "..", "logs"))
OUT = os.path.join(HERE, "paper_stats")
os.makedirs(OUT, exist_ok=True)
H = [5, 15, 30]
RNG = np.random.default_rng(20260702)  # fixed seed
N_BOOT = 10_000
N_PLACEBO = 200


# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
def load_spx() -> pd.DataFrame:
    cache = os.path.join(OUT, "polls.parquet")
    if os.path.exists(cache) and "--fresh" not in sys.argv:
        return pd.read_parquet(cache)
    rows, bad = [], 0
    files = sorted(glob.glob(os.path.join(LOGDIR, "daytrade-*", "SPX.jsonl")))
    for f in files:
        day = os.path.basename(os.path.dirname(f)).replace("daytrade-", "")
        with open(f, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = _loads(line)
                except Exception:
                    bad += 1
                    continue
                spot, ts = d.get("spot"), d.get("ts_ny")
                if spot is None or ts is None:
                    bad += 1
                    continue
                try:
                    spot = float(spot)
                except Exception:
                    bad += 1
                    continue
                if spot <= 0:
                    bad += 1
                    continue
                dec = d.get("decision") or {}
                base = d.get("decision_baseline")
                g = dec.get("gates") or {}
                iv5 = (d.get("intervals") or {}).get("5m") or {}
                strat = d.get("strategy") or {}
                cfg = d.get("config") or {}
                reasons = dec.get("reasons") or []
                rows.append(
                    (
                        day,
                        ts,
                        spot,
                        (dec.get("signal") or "flat"),
                        ((base or {}).get("signal") or dec.get("signal") or "flat"),
                        base is not None,
                        bool(d.get("wall_guard_applied")),
                        "wall_guard_applied" in d,
                        float(iv5.get("bias")) if iv5.get("bias") is not None else np.nan,
                        str(strat.get("active") or "baseline"),
                        float(cfg.get("mny_band")) if cfg.get("mny_band") is not None else np.nan,
                        float(cfg.get("poll_secs")) if cfg.get("poll_secs") is not None else np.nan,
                        bool(g.get("b5_pass")), bool(g.get("V5_pass")),
                        bool(g.get("b15_pass")), bool(g.get("V15_pass")),
                        bool(g.get("agree")),
                        ("late_session_block" in reasons),
                    )
                )
    df = pd.DataFrame(
        rows,
        columns=["day", "ts", "spot", "signal", "signal_base", "has_base", "veto",
                 "has_veto_field", "b5", "strategy", "mny_band", "poll_secs",
                 "b5p", "v5p", "b15p", "v15p", "agree", "late_block"],
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["day", "ts"]).reset_index(drop=True)
    df["bad_lines"] = bad  # persisted (attrs are lost in parquet round-trips)
    print(f"loaded {len(df):,} valid polls, {bad} rejected lines, "
          f"{df['day'].nunique()} days, {df['day'].min()}..{df['day'].max()}")
    df.to_parquet(cache)
    return df


# ----------------------------------------------------------------------
# 2. forward returns  (repo convention; optional skip-k entry)
# ----------------------------------------------------------------------
def add_forward(df: pd.DataFrame, lag_polls: int = 0) -> pd.DataFrame:
    for h in H:
        df[f"r{h}"] = np.nan
    for _, gd in df.groupby("day"):
        idx = gd.index.to_numpy()
        t = gd["ts"].astype("int64").to_numpy() / 1e9
        s = gd["spot"].to_numpy()
        n = len(t)
        ent = np.arange(n) + lag_polls
        ok = ent < n
        te = np.where(ok, t[np.clip(ent, 0, n - 1)], np.nan)
        se = np.where(ok, s[np.clip(ent, 0, n - 1)], np.nan)
        for h in H:
            target = te + h * 60
            pos = np.searchsorted(t, np.nan_to_num(target, nan=1e18), side="left")
            fok = ok & (pos < n)
            fp = np.clip(pos, 0, n - 1)
            gap = t[fp] - te
            good = fok & (gap <= (h + 10) * 60) & (gap >= (h - 1) * 60)
            r = np.full(n, np.nan)
            r[good] = (s[fp][good] / se[good] - 1.0) * 1e4
            df.loc[idx, f"r{h}"] = r
    return df


def signed(df: pd.DataFrame, h: int, sigcol: str = "signal") -> np.ndarray:
    d = np.where(df[sigcol].to_numpy() == "long", 1.0,
                 np.where(df[sigcol].to_numpy() == "short", -1.0, np.nan))
    return d * df[f"r{h}"].to_numpy()


# ----------------------------------------------------------------------
# 3. the two inference regimes
# ----------------------------------------------------------------------
def naive_stats(e: np.ndarray) -> dict:
    x = e[~np.isnan(e)]
    n = len(x)
    m = x.mean() if n else np.nan
    se = x.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    return dict(n=n, mean=m, se=se, t=(m / se if se and se > 0 else np.nan))


def clustered_stats(days: np.ndarray, e: np.ndarray) -> dict:
    tmp = pd.DataFrame({"day": days, "x": e}).dropna()
    daily = tmp.groupby("day")["x"].mean()
    D = daily.shape[0]
    m = daily.mean()
    sd = daily.std(ddof=1)
    se = sd / math.sqrt(D)
    kpos = int((daily > 0).sum())
    kneg = int((daily < 0).sum())
    sign_p = sps.binomtest(kpos, kpos + kneg, 0.5).pvalue if kpos + kneg else np.nan
    return dict(D=D, mean_daily=m, se=se, t=(m / se if se > 0 else np.nan),
                pct_pos=100.0 * kpos / D if D else np.nan,
                sign_test_p=sign_p, daily=daily)


def block_bootstrap(daily: pd.Series, n_boot: int = N_BOOT) -> dict:
    v = daily.to_numpy()
    D = len(v)
    idx = RNG.integers(0, D, size=(n_boot, D))
    bmeans = v[idx].mean(axis=1)
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    p = 2 * min((bmeans <= 0).mean(), (bmeans >= 0).mean())
    return dict(ci_lo=lo, ci_hi=hi, p=p)


def hac_t(days: np.ndarray, e: np.ndarray, L: int) -> dict:
    """Newey-West t for the mean of e over directional polls; autocovariances
    computed only across pairs within the same trading day (no overnight lags)."""
    tmp = pd.DataFrame({"day": days, "x": e}).dropna()
    n = len(tmp)
    m = tmp["x"].mean()
    xc = tmp["x"] - m
    S = float((xc**2).sum())
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        for _, g in xc.groupby(tmp["day"].values):
            a = g.to_numpy()
            if len(a) > l:
                S += 2.0 * w * float((a[:-l] * a[l:]).sum())
    var_m = S / n**2
    se = math.sqrt(max(var_m, 0.0))
    return dict(n=n, mean=m, se=se, t=(m / se if se > 0 else np.nan), L=L)


# ----------------------------------------------------------------------
# 4. overlap diagnostics
# ----------------------------------------------------------------------
def acf_by_minutes(df: pd.DataFrame, h: int, max_lag_polls: int = 240) -> pd.DataFrame:
    """Autocorrelation of the signed-edge series at poll lags, translated to a
    median wall-clock lag; pairs are taken within day, both polls directional."""
    recs = []
    e_all, t_all = [], []
    for _, gd in df.groupby("day"):
        e = signed(gd, h)
        mask = ~np.isnan(e)
        e_all.append(e[mask])
        t_all.append((gd["ts"].astype("int64").to_numpy() / 1e9)[mask])
    m = np.concatenate([x for x in e_all]).mean()
    g0 = np.concatenate([(x - m) ** 2 for x in e_all]).sum()
    for l in range(1, max_lag_polls + 1):
        num, dts = 0.0, []
        for e, t in zip(e_all, t_all):
            if len(e) > l:
                num += float(((e[:-l] - m) * (e[l:] - m)).sum())
                dts.append(t[l:] - t[:-l])
        if not dts:
            break
        dt_med = float(np.median(np.concatenate(dts)))
        recs.append(dict(lag_polls=l, lag_med_min=dt_med / 60.0, acf=num / g0))
    return pd.DataFrame(recs)


def overlap_share(df: pd.DataFrame, h: int) -> float:
    """Average number of other directional polls whose forward window overlaps
    a given directional poll's window (same day)."""
    tot, cnt = 0, 0
    for _, gd in df.groupby("day"):
        e = signed(gd, h)
        t = (gd["ts"].astype("int64").to_numpy() / 1e9)[~np.isnan(e)]
        if len(t) < 2:
            continue
        for i in range(len(t)):
            lo = np.searchsorted(t, t[i] - h * 60, side="left")
            hi = np.searchsorted(t, t[i] + h * 60, side="right")
            tot += hi - lo - 1
            cnt += 1
    return tot / cnt if cnt else np.nan


def non_overlapping(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Greedy within-day thinning: keep a directional poll only if >= h minutes
    after the last kept one. Returns kept subsample."""
    keep_idx = []
    for _, gd in df.groupby("day"):
        e = signed(gd, h)
        t = gd["ts"].astype("int64").to_numpy() / 1e9
        last = -np.inf
        for i, (ee, tt) in enumerate(zip(e, t)):
            if np.isnan(ee):
                continue
            if tt - last >= h * 60:
                keep_idx.append(gd.index[i])
                last = tt
    return df.loc[keep_idx]


# ----------------------------------------------------------------------
# 5. placebo: circular within-day rotation of the signal sequence
# ----------------------------------------------------------------------
def placebo_ts(df: pd.DataFrame, h: int, n_rep: int = N_PLACEBO) -> np.ndarray:
    days = []
    for _, gd in df.groupby("day"):
        d = np.where(gd["signal"].to_numpy() == "long", 1.0,
                     np.where(gd["signal"].to_numpy() == "short", -1.0, np.nan))
        r = gd[f"r{h}"].to_numpy()
        days.append((d, r))
    ts = np.empty(n_rep)
    for k in range(n_rep):
        dm = []
        for d, r in days:
            n = len(d)
            if n < 4:
                continue
            shift = int(RNG.integers(1, n))
            e = np.roll(d, shift) * r
            e = e[~np.isnan(e)]
            if len(e):
                dm.append(e.mean())
        dm = np.asarray(dm)
        ts[k] = dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm)))
    return ts


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    df = load_spx()
    stats = {}
    stats["n_polls"] = int(len(df))
    stats["n_days"] = int(df["day"].nunique())
    stats["day_min"], stats["day_max"] = df["day"].min(), df["day"].max()

    # cadence
    dt_all = []
    for _, gd in df.groupby("day"):
        t = gd["ts"].astype("int64").to_numpy() / 1e9
        dt_all.append(np.diff(t))
    dt_all = np.concatenate(dt_all)
    stats["cadence_med_s"] = float(np.median(dt_all))
    stats["cadence_p10_s"] = float(np.percentile(dt_all, 10))
    stats["cadence_p90_s"] = float(np.percentile(dt_all, 90))
    stats["polls_per_day_med"] = float(df.groupby("day").size().median())
    stats["bad_lines"] = int(df["bad_lines"].iloc[0])
    stats["spot_min"], stats["spot_max"] = float(df["spot"].min()), float(df["spot"].max())
    stats["mny_band_values"] = {str(k): int(v) for k, v in
                                df["mny_band"].value_counts(dropna=False).items()}
    stats["mny_band_001_days"] = int(df.loc[df.mny_band == 0.01, "day"].nunique())
    stats["mny_band_0015_days"] = int(df.loc[df.mny_band == 0.015, "day"].nunique())
    dmask0 = df["signal"].isin(["long", "short"])
    stats["dir_polls_narrow_band_pct"] = float(
        100.0 * (dmask0 & (df.mny_band < 0.02)).sum() / dmask0.sum())
    stats["strategy_active"] = {str(k): int(v) for k, v in
                                df["strategy"].value_counts().items()}
    # decision_baseline exists only in later log-schema versions
    stats["baseline_field_coverage_pct"] = float(100.0 * df["has_base"].mean())
    cov = df[df["has_base"]]
    stats["decision_vs_baseline_mismatch_pct_on_covered"] = float(
        100.0 * (cov["signal"] != cov["signal_base"]).mean())
    stats["veto_field_coverage_pct"] = float(100.0 * df["has_veto_field"].mean())
    stats["late_block_polls"] = int(df["late_block"].sum())
    # regular trading hours
    ny0 = df["ts"].dt.tz_convert("America/New_York")
    mins = ny0.dt.hour * 60 + ny0.dt.minute
    rth = (mins >= 570) & (mins < 960)
    stats["non_rth_polls"] = int((~rth).sum())
    stats["non_rth_directional"] = int(((~rth) & dmask0).sum())
    # why flat: gate-failure shares among flat polls (categories overlap)
    fl = df[df["signal"] == "flat"]
    stats["flat_fail_shares_pct"] = {
        "sign_disagree": float(100.0 * (~fl["agree"]).mean()),
        "bias_threshold": float(100.0 * (~(fl["b5p"] & fl["b15p"])).mean()),
        "activity_floor": float(100.0 * (~(fl["v5p"] & fl["v15p"])).mean()),
    }
    # concentration of directional polls across days
    dcnt = df.loc[dmask0].groupby("day").size().sort_values(ascending=False)
    stats["dir_polls_per_day_median"] = float(dcnt.median())
    ntop = int(np.ceil(0.10 * len(dcnt)))
    stats["dir_polls_top_decile_days_share_pct"] = float(
        100.0 * dcnt.head(ntop).sum() / dcnt.sum())

    # signal frequency
    fc = df["signal"].value_counts()
    stats["freq"] = {k: dict(n=int(fc.get(k, 0)), pct=float(100 * fc.get(k, 0) / len(df)))
                     for k in ["flat", "long", "short"]}
    stats["veto_n"] = int(df["veto"].sum())
    stats["veto_pct_of_polls"] = float(100 * df["veto"].mean())

    df = add_forward(df, 0)
    # forward-return match rate
    stats["fwd_match_pct"] = {str(h): float(100 * df[f"r{h}"].notna().mean()) for h in H}

    # ---------------- per-signal returns (Table 3 / Fig 2) ----------------
    rows = []
    for h in H:
        for sig in ["long", "flat", "short"]:
            sub = df[df["signal"] == sig]
            r = sub[f"r{h}"].dropna()
            cl = clustered_stats(sub["day"].to_numpy(), sub[f"r{h}"].to_numpy())
            rows.append(dict(horizon=h, signal=sig, n=len(r),
                             mean_bps=r.mean(),
                             naive_se=r.std(ddof=1) / math.sqrt(len(r)),
                             cl_mean_daily=cl["mean_daily"], cl_se=cl["se"],
                             cl_D=cl["D"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "signal_returns.csv"), index=False)

    # ---------------- directional edge: naive vs clustered (Table 4/Fig 3) -
    rows, daily_store = [], {}
    for h in H:
        e = signed(df, h)
        nv = naive_stats(e)
        cl = clustered_stats(df["day"].to_numpy(), e)
        bb = block_bootstrap(cl["daily"])
        # HAC lag: polls spanning the horizon at median cadence
        L = int(math.ceil(h * 60 / stats["cadence_med_s"]))
        hc = hac_t(df["day"].to_numpy(), e, min(L, 400))
        daily_store[h] = cl["daily"]
        rows.append(dict(horizon=h, n_polls=nv["n"],
                         naive_mean_bps=nv["mean"], naive_se=nv["se"], naive_t=nv["t"],
                         cl_D=cl["D"], cl_mean_bps=cl["mean_daily"], cl_se=cl["se"],
                         cl_t=cl["t"], pct_days_pos=cl["pct_pos"],
                         sign_test_p=cl["sign_test_p"],
                         boot_ci_lo=bb["ci_lo"], boot_ci_hi=bb["ci_hi"], boot_p=bb["p"],
                         hac_t=hc["t"], hac_L=hc["L"]))
    edge = pd.DataFrame(rows)
    edge.to_csv(os.path.join(OUT, "directional_edge.csv"), index=False)
    daily_store[15].rename("edge_bps").to_csv(os.path.join(OUT, "daily_edge_15m.csv"))
    stats["edge"] = edge.to_dict("records")

    # ---------------- skip-one-poll ----------------
    df1 = add_forward(df.drop(columns=[f"r{h}" for h in H]).copy(), 1)
    rows = []
    for h in H:
        e = signed(df1, h)
        nv, cl = naive_stats(e), clustered_stats(df1["day"].to_numpy(), e)
        rows.append(dict(horizon=h, naive_t=nv["t"], cl_t=cl["t"],
                         cl_mean_bps=cl["mean_daily"], pct_days_pos=cl["pct_pos"]))
    skip = pd.DataFrame(rows)
    skip.to_csv(os.path.join(OUT, "skip_one_poll.csv"), index=False)
    stats["skip1"] = skip.to_dict("records")

    # ---------------- non-overlapping subsample ----------------
    rows = []
    for h in H:
        sub = non_overlapping(df, h)
        e = signed(sub, h)
        nv, cl = naive_stats(e), clustered_stats(sub["day"].to_numpy(), e)
        rows.append(dict(horizon=h, n_kept=nv["n"], naive_t=nv["t"],
                         cl_t=cl["t"], cl_mean_bps=cl["mean_daily"]))
    novl = pd.DataFrame(rows)
    novl.to_csv(os.path.join(OUT, "non_overlapping.csv"), index=False)
    stats["non_overlap"] = novl.to_dict("records")

    # ---------------- long / short split ----------------
    rows = []
    for h in H:
        for side, sgn in [("long", 1.0), ("short", -1.0)]:
            sub = df[df["signal"] == side]
            e = sgn * sub[f"r{h}"].to_numpy()
            cl = clustered_stats(sub["day"].to_numpy(), e)
            rows.append(dict(horizon=h, side=side, cl_mean_bps=cl["mean_daily"],
                             cl_t=cl["t"], D=cl["D"]))
    ls = pd.DataFrame(rows)
    ls.to_csv(os.path.join(OUT, "long_short_split.csv"), index=False)
    stats["long_short"] = ls.to_dict("records")

    # ---------------- time-of-day (regular trading hours only) ----------------
    ny = df["ts"].dt.tz_convert("America/New_York")
    mins2 = ny.dt.hour * 60 + ny.dt.minute
    rth2 = ((mins2 >= 570) & (mins2 < 960)).to_numpy()
    tod = np.where(~rth2, "non-RTH",
                   np.where(ny.dt.hour < 11, "open(9:30-11)",
                            np.where(ny.dt.hour < 14, "midday(11-14)", "close(14-16)")))
    rows = []
    for h in H:
        for b in ["open(9:30-11)", "midday(11-14)", "close(14-16)"]:
            sub = df[tod == b]
            e = signed(sub, h)
            cl = clustered_stats(sub["day"].to_numpy(), e)
            rows.append(dict(horizon=h, bucket=b, cl_mean_bps=cl["mean_daily"],
                             cl_t=cl["t"], D=cl["D"]))
    todf = pd.DataFrame(rows)
    todf.to_csv(os.path.join(OUT, "time_of_day.csv"), index=False)
    stats["tod_15m"] = todf[todf.horizon == 15].to_dict("records")

    # ---------------- volatility split (median split of realised day vol) --
    dayvol = {}
    for day, gd in df.groupby("day"):
        r1 = np.diff(np.log(gd["spot"].to_numpy())) * 1e4
        dayvol[day] = float(np.std(r1)) if len(r1) > 10 else np.nan
    dv = pd.Series(dayvol)
    med = dv.median()
    rows = []
    for h in H:
        for lab, sel in [("low-vol", dv[dv <= med].index), ("high-vol", dv[dv > med].index)]:
            sub = df[df["day"].isin(sel)]
            e = signed(sub, h)
            cl = clustered_stats(sub["day"].to_numpy(), e)
            rows.append(dict(horizon=h, bucket=lab, cl_mean_bps=cl["mean_daily"],
                             cl_t=cl["t"], D=cl["D"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "vol_split.csv"), index=False)

    # ---------------- signal-strength split (|b5| median among directional) -
    dmask = df["signal"].isin(["long", "short"])
    b5med = df.loc[dmask, "b5"].abs().median()
    rows = []
    for h in H:
        for lab, sel in [("weak", df["b5"].abs() <= b5med), ("strong", df["b5"].abs() > b5med)]:
            sub = df[dmask & sel]
            e = signed(sub, h)
            cl = clustered_stats(sub["day"].to_numpy(), e)
            rows.append(dict(horizon=h, bucket=lab, cl_mean_bps=cl["mean_daily"],
                             cl_t=cl["t"], D=cl["D"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "strength_split.csv"), index=False)

    # ---------------- OPEX exclusion (3rd Friday of each month) ------------
    def is_opex(daystr):
        d = date(int(daystr[:4]), int(daystr[4:6]), int(daystr[6:]))
        return d.weekday() == 4 and 15 <= d.day <= 21
    opex_days = {d for d in df["day"].unique() if is_opex(d)}
    sub = df[~df["day"].isin(opex_days)]
    rows = []
    for h in H:
        e = signed(sub, h)
        cl = clustered_stats(sub["day"].to_numpy(), e)
        rows.append(dict(horizon=h, cl_mean_bps=cl["mean_daily"], cl_t=cl["t"],
                         D=cl["D"], n_opex_excluded=len(opex_days)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ex_opex.csv"), index=False)

    # ---------------- ACF / overlap diagnostics ----------------
    acf15 = acf_by_minutes(df, 15)
    acf15.to_csv(os.path.join(OUT, "acf_signed_15m.csv"), index=False)
    rho = acf15["acf"].to_numpy()
    upto = acf15[acf15["lag_med_min"] <= 25.0].index.max() or len(rho) - 1
    ess_factor = 1.0 + 2.0 * float(np.clip(rho[: upto + 1], 0, None).sum())
    n_dir15 = int(edge.loc[edge.horizon == 15, "n_polls"].iloc[0])
    stats["ess_15m"] = dict(factor=ess_factor, n=n_dir15,
                            n_eff=int(n_dir15 / ess_factor))
    for h in H:
        stats[f"overlap_neighbors_{h}m"] = float(overlap_share(df, h))

    # ---------------- placebo ----------------
    pts = placebo_ts(df, 15)
    np.save(os.path.join(OUT, "placebo_t15.npy"), pts)
    actual_t15 = float(edge.loc[edge.horizon == 15, "cl_t"].iloc[0])
    stats["placebo_15m"] = dict(
        n_rep=len(pts), mean_abs_t=float(np.abs(pts).mean()),
        p95_abs_t=float(np.percentile(np.abs(pts), 95)),
        share_abs_ge_actual=float((np.abs(pts) >= abs(actual_t15)).mean()))

    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str)[:4000])
    print("\nDONE ->", OUT)


if __name__ == "__main__":
    main()
