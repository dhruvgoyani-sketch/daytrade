#!/usr/bin/env python3
"""
Day-trading flow bot (intraday) using Convex options data.

Core idea
- Use multi-interval net aggressor flow (volmbs_5m, 15m, 30m, 60m, and since-open volm_bs)
  to form a flow-centric directional view for the underlying.
- Focus on 0DTE (today's expiration) by default; optionally include the next few expirations.
- Weight flows by moneyness (and optionally delta) to emphasize near-spot strikes.
- Generate signals when strong 5m flow is confirmed by the 15m window.

Notes
- This script does not require long-term storage. It fetches live chains and computes signals on the fly.
- You may run it once or in a polling loop via --poll.
- Credentials are taken from env vars by default: CONVEX_EMAIL, CONVEX_PASSWORD, CONVEX_ENV.
  If not present, you can pass them as CLI flags.

Example
  python3 daytrade/dt.py --tickers SPX --mode 0dte --spot_auto --weighting mny_delta \
      --mny_band 0.02 --thresh5 0.30 --thresh15 0.20 --minV5 2e4 --minV15 4e4
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import json
import csv
import sqlite3
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from datetime import datetime, date as _DT_DATE, time as _DT_TIME
from zoneinfo import ZoneInfo

try:
    from convexlib.api import ConvexApi
except Exception as e:  # pragma: no cover
    raise SystemExit("convexlib.api not found. Please install convexlib and retry.")
try:
    from daytrade.store import insert_intraday_ctx as _insert_ctx_cli
    from daytrade.store import upsert_iv_ctx as _upsert_iv_ctx
except Exception:
    _insert_ctx_cli = None
    _upsert_iv_ctx = None

from daytrade.walls import WallDetector, WallGuard

NY_TZ = ZoneInfo("America/New_York")
EPS = 1e-9

_WALL_DETECTORS: Dict[str, WallDetector] = {}
_WALL_GUARD = WallGuard()
_BIAS_FAIL_STATS: Dict[str, Dict[str, int]] = {}
_SIGN_DISAGREE_STATS: Dict[str, Dict[str, int]] = {}
_BIAS_FAIL_WINDOW: Dict[str, Dict[str, deque[Tuple[float, int]]]] = {}
_BIAS_FAIL_WINDOW_SUM: Dict[str, Dict[str, int]] = {}

_ALT_VIX_ASSUMED = 16.0
_ALT_EXPIRY_WEIGHT_DESC = "1/sqrt(DTE+1)"

ZERO_GAMMA_CFG = {
    "gamma_mode": "BS",          # "BS" or "DOLLAR"
    "strike_window_pts": 150,
    "grid_halfwidth": 80,
    "grid_step": 5,
    "ema_alpha": 0.5,
}

LEDGER_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "pg_database_spx.db")
)

# ---------------------- Normalization ----------------------
#
# Notes (references & rationale for intraday SPX logic in this file)
# - 0DTE vs 1DTE microstructure:
#   Dealers’ same‑day hedging intensity grows into the close as T→0. Today’s (0DTE)
#   flow exerts the largest immediate gamma/theta impact intraday; tomorrow (1DTE)
#   is relevant early (rolls/new risk), dips midday (0DTE dominates), and rises again
#   late as books are re‑hedged for overnight. We encode this via time‑of‑day weights.
#   Practitioners: common intraday weighting practice. Academic priors: Gatheral’s
#   volatility surface (gamma/vanna mechanics), demand‑based option pricing (Gârleanu,
#   Pedersen, Poteshman), and empirical microstructure of order‑flow → IV surface
#   (Bollen & Whaley).
#
# - Anchor (show‑only):
#   A simple EWM of 5m flow bias (expiry‑weighted) with ~30m half‑life provides a
#   “state” of intraday flow. We surface it for context; it does not affect decisions.
#   Similar to intraday state filters used by practitioners to stabilize flips.
#
# - Implementation scope:
#   The expiry weights and anchor are applied to SPX only (safe, explainable). Other
#   tickers retain the default behavior.

def norm_ticker(t: object, default: str = "SPX") -> str:
    try:
        s = str(t).strip().upper()
        return s if s else default
    except Exception:
        return default


def _update_zero_gamma_cache(ticker: str, df: pd.DataFrame, spot: float) -> None:
    key = norm_ticker(ticker, "SPX")
    try:
        keep_cols = [
            col for col in (
                "strike_price",
                "call_gamma",
                "put_gamma",
                "call_volm_bs",
                "put_volm_bs",
                "expiration_date",
                "expiration_dt",
            ) if col in df.columns
        ]
        subset = df[keep_cols].copy() if keep_cols else pd.DataFrame()
    except Exception:
        subset = pd.DataFrame()
    _ZERO_GAMMA_CACHE[key] = {
        "df": subset,
        "spot": float(spot) if _finite_float(spot) is not None else 0.0,
        "ts": time.time(),
    }


def get_zero_gamma_snapshot(ticker: str = "SPX") -> ZeroGammaSnapshot:
    key = norm_ticker(ticker, "SPX")
    cache = _ZERO_GAMMA_CACHE.get(key)
    if not cache:
        last = getattr(_ZERO_GAMMA_ENGINE, "_last_pub", None)
        return ZeroGammaSnapshot(
            ts=time.time(),
            spot=0.0,
            zero_gamma=round(last, 1) if isinstance(last, (float, int)) and math.isfinite(last) else None,
            has_root=False,
            n_opts_used=0,
        )

    df_cached = cache.get("df")
    spot_cached = _finite_float(cache.get("spot"))
    rows: List[OptionRow] = []
    today = now_ny().date()

    ledger_df = _load_spx_ledger_for_date(today)
    ledger_map: Dict[Tuple[float, str], Tuple[float, float]] = {}
    ledger_map_by_date: Dict[Tuple[float, str], Tuple[float, float]] = {}
    if isinstance(ledger_df, pd.DataFrame) and not ledger_df.empty:
        for row in ledger_df.itertuples():
            try:
                strike_val = float(row.strike_price)
            except Exception:
                continue
            call_open = _finite_float(getattr(row, "cust_call_open", None)) or 0.0
            put_open = _finite_float(getattr(row, "cust_put_open", None)) or 0.0
            expiration_str = str(getattr(row, "expiration", "")).strip()
            if expiration_str:
                ledger_map[(strike_val, expiration_str)] = (call_open, put_open)
                if "T" in expiration_str:
                    ledger_map_by_date[(strike_val, expiration_str.split("T", 1)[0])] = (call_open, put_open)
            else:
                ledger_map_by_date[(strike_val, today.isoformat())] = (call_open, put_open)

    if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
        df_work = df_cached.copy()
        exp_series = None
        if "expiration_date" in df_work.columns:
            try:
                exp_series = pd.to_datetime(df_work["expiration_date"], errors="coerce").dt.date
            except Exception:
                exp_series = None
        if exp_series is not None:
            df_work = df_work.assign(_exp_date=exp_series)
            df_work = df_work[df_work["_exp_date"] == today]
        else:
            df_work = df_work.iloc[0:0]

        for rec in df_work.itertuples():
            strike_val = _finite_float(getattr(rec, "strike_price", None))
            if strike_val is None:
                continue
            c_gamma = _finite_float(getattr(rec, "call_gamma", None))
            c_flow = _finite_float(getattr(rec, "call_volm_bs", None)) or 0.0
            exp_dt = getattr(rec, "expiration_dt", None)
            exp_iso = None
            if isinstance(exp_dt, pd.Timestamp):
                if pd.notna(exp_dt):
                    exp_iso = exp_dt.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S')
            elif isinstance(exp_dt, datetime):
                exp_iso = exp_dt.strftime('%Y-%m-%dT%H:%M:%S')
            elif hasattr(rec, "expiration"):
                exp_val = getattr(rec, "expiration", None)
                if isinstance(exp_val, str) and exp_val:
                    exp_iso = exp_val

            ledger_vals = None
            if exp_iso:
                ledger_vals = ledger_map.get((strike_val, exp_iso))
                if ledger_vals is None:
                    ledger_vals = ledger_map_by_date.get((strike_val, exp_iso.split('T', 1)[0]))
            if ledger_vals is None:
                ledger_vals = ledger_map_by_date.get((strike_val, today.isoformat()))
            call_base = ledger_vals[0] if ledger_vals else 0.0
            put_base = ledger_vals[1] if ledger_vals else 0.0

            if c_gamma is not None:
                rows.append(OptionRow(strike=strike_val, cp='C', gamma=c_gamma, qty_customer=call_base + c_flow))
            p_gamma = _finite_float(getattr(rec, "put_gamma", None))
            p_flow = _finite_float(getattr(rec, "put_volm_bs", None)) or 0.0
            if p_gamma is not None:
                rows.append(OptionRow(strike=strike_val, cp='P', gamma=p_gamma, qty_customer=put_base + p_flow))

        if (spot_cached is None or spot_cached <= 0) and rows:
            try:
                strike_vals = [row.strike for row in rows if math.isfinite(row.strike)]
                if strike_vals:
                    median_strike = float(np.median(strike_vals))
                    if math.isfinite(median_strike) and median_strike > 0:
                        spot_cached = median_strike
            except Exception:
                pass

    spot_val = spot_cached if (spot_cached is not None and math.isfinite(spot_cached)) else 0.0
    snapshot = _ZERO_GAMMA_ENGINE.compute_snapshot(float(spot_val), rows)
    return snapshot


# ---------------------- Config models ----------------------

WeightingMode = str  # 'raw' | 'mny' | 'mny_delta'


@dataclass
class FlowConfig:
    weighting: WeightingMode = "mny_delta"
    mny_band: float = 0.02
    exp_count: int = 7        # fetch next N expirations via exps=[0..N-1]
    use_spot_auto: bool = True


@dataclass
class SignalConfig:
    thresh5: float = 0.30      # 5m bias threshold
    thresh15: float = 0.20     # 15m bias threshold (confirmation)
    minV5: float = 2e4         # activity floor for 5m weighted volume
    minV15: float = 4e4        # activity floor for 15m weighted volume
    b5_fail_ratio_alert: float = 0.20
    b5_alert_min_polls: int = 12


@dataclass
class Credentials:
    email: str
    password: str
    env: str = "live"  # or 'pro'


@dataclass
class OptionRow:
    strike: float
    cp: str            # 'C' or 'P'
    gamma: float       # per-contract gamma (BS or Dollar)
    qty_customer: float  # net +buy / -sell since open


@dataclass
class ZeroGammaSnapshot:
    ts: float
    spot: float
    zero_gamma: Optional[float]
    has_root: bool
    n_opts_used: int


def _finite_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _trade_date_code(d: _DT_DATE) -> Optional[str]:
    try:
        return d.strftime("%y%m%d")
    except Exception:
        return None


def _load_spx_ledger_for_date(trade_date: _DT_DATE) -> pd.DataFrame:
    code = _trade_date_code(trade_date)
    if not code or not os.path.exists(LEDGER_DB_PATH):
        return pd.DataFrame()
    conn = None
    try:
        conn = sqlite3.connect(LEDGER_DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT strike_price,
                   expiration,
                   cust_call_open,
                   cust_put_open
            FROM spx_ledger
            WHERE trade_date = ?
            """,
            conn,
            params=(code,),
        )
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _dollar_gamma(spot: float, gamma_api: float, mode: str) -> float:
    if not math.isfinite(gamma_api):
        return 0.0
    if mode == "BS":
        return 0.0 if spot <= 0 else gamma_api * (spot ** 2) * 100.0
    if mode == "DOLLAR":
        return gamma_api
    raise ValueError(f"Unknown gamma mode: {mode}")


def _total_dealer_gex_at(spot: float, chain: List[OptionRow], mode: str) -> float:
    total = 0.0
    for row in chain:
        if not math.isfinite(row.qty_customer):
            continue
        dg = _dollar_gamma(spot, row.gamma, mode)
        if not math.isfinite(dg):
            continue
        sign = -1.0 if str(row.cp).upper() == "C" else 1.0
        total += sign * row.qty_customer * dg
    return total


def _solve_zero_gamma(
    spot_now: float,
    chain: List[OptionRow],
    mode: str,
    half_width: int,
    step: int,
) -> Tuple[bool, Optional[float]]:
    if spot_now <= 0 or not chain:
        return False, None
    grid = np.arange(spot_now - half_width, spot_now + half_width + step, step, dtype=float)
    totals: List[float] = []
    for s in grid:
        val = _total_dealer_gex_at(s, chain, mode)
        totals.append(val if math.isfinite(val) else float("nan"))

    for idx in range(len(grid) - 1):
        a, b = grid[idx], grid[idx + 1]
        fa, fb = totals[idx], totals[idx + 1]
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        if abs(fa) < 1e-6:
            return True, round(a, 1)
        if fa * fb < 0:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                fm = _total_dealer_gex_at(mid, chain, mode)
                if not math.isfinite(fm):
                    break
                if abs(fm) < 1e-3 or (hi - lo) <= 0.1:
                    return True, round(mid, 1)
                if flo * fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            return True, round(0.5 * (lo + hi), 1)
    return False, None


class ZeroGammaEngine:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = dict(cfg or ZERO_GAMMA_CFG)
        self._last_pub: Optional[float] = None

    def _ema(self, new: float) -> float:
        alpha = float(self.cfg.get("ema_alpha", 0.5))
        if self._last_pub is None:
            self._last_pub = new
        else:
            self._last_pub = alpha * new + (1.0 - alpha) * self._last_pub
        return self._last_pub

    def compute_snapshot(self, spot: float, rows: List[OptionRow]) -> ZeroGammaSnapshot:
        window = float(self.cfg.get("strike_window_pts", 150))
        mode = str(self.cfg.get("gamma_mode", "BS"))
        half_width = int(self.cfg.get("grid_halfwidth", 80))
        step = int(self.cfg.get("grid_step", 5))

        filtered = [
            row for row in rows
            if math.isfinite(row.strike) and abs(row.strike - spot) <= window
        ]

        has_root, root_val = _solve_zero_gamma(spot, filtered, mode, half_width, step)
        published: Optional[float]
        if has_root and root_val is not None:
            published = round(self._ema(root_val), 1)
        else:
            published = round(self._last_pub, 1) if self._last_pub is not None else None

        return ZeroGammaSnapshot(
            ts=time.time(),
            spot=float(spot),
            zero_gamma=published,
            has_root=has_root,
            n_opts_used=len(filtered),
        )


_ZERO_GAMMA_ENGINE = ZeroGammaEngine()
_ZERO_GAMMA_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------- Utilities ----------------------

def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def is_weekend(ts: datetime | None = None) -> bool:
    ref = ts or now_ny()
    return ref.weekday() >= 5


def today_iso() -> str:
    return now_ny().date().isoformat()


def day_fraction_ny() -> float:
    """Trading day fraction in [0,1] from 09:30 to 16:00 NY."""
    now = now_ny()
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now <= start:
        return 0.0
    if now >= end:
        return 1.0
    return (now - start).total_seconds() / max(1.0, (end - start).total_seconds())


def _market_closed_ny() -> bool:
    """Return True if current NY local time is at or after 16:00."""
    try:
        now_t = datetime.now(NY_TZ).time()
        return (now_t >= _DT_TIME(16, 0))
    except Exception:
        # If timezone fails, be permissive and do not block
        return False


def _spx_expiry_weight_for_date(exp_date: pd.Series) -> np.ndarray:
    """Per-row expiry weights for SPX based on DTE and time-of-day.

    Motivation (microstructure):
      - 0DTE (today) drives same‑day hedging; intensity grows into the close.
      - 1DTE (tomorrow) matters at the open (rolls/new positions), dips midday, and
        regains relevance in the last hour as dealers re‑hedge for overnight.
      - 2D+ carries little same‑day directional power; we give a small constant.

    Schedule (piecewise; simple, robust):
      f ∈ [0,1] is day fraction from 09:30→16:00 NY.
        w0(f) = 1.0 + 0.5·f          # 0DTE strictly increasing (max ≈1.5x by close)
        w1(f) = 0.35 (f≤0.35), 0.25 (mid), 0.40 (f≥0.75)  # U‑shape for 1DTE
        w2+   = 0.10                 # small, non‑zero floor

    Returns np.ndarray aligned to exp_date, used to scale moneyness/Δ weights.
    """
    try:
        today = now_ny().date()
        f = day_fraction_ny()
        dte = (pd.to_datetime(exp_date, errors='coerce').dt.date - today).apply(lambda d: 0 if pd.isna(d) else (d).days)
        d = dte.fillna(10).astype(int).to_numpy()
        # piecewise schedule
        w0 = 1.0 + 0.5 * f
        if f <= 0.35:
            w1 = 0.35
        elif f >= 0.75:
            w1 = 0.40
        else:
            w1 = 0.25
        w2p = 0.10
        w = np.where(d <= 0, w0, np.where(d == 1, w1, w2p))
        return w.astype(float)
    except Exception:
        return np.ones(len(exp_date), dtype=float)


_ANCHOR_STATE: Dict[str, Dict[str, object]] = {}


def update_anchor(ticker: str, bias_val: float, half_life_min: float = 30.0) -> float:
    """EWM anchor of (expiry‑weighted) 5m bias — context only, not used for gating.

    - Exponential weighted moving average updated by wall‑clock elapsed time, so it
      is insensitive to small poll jitter. Half‑life default ~30 minutes.
    - Intended to be displayed alongside dte0/dte1 shares to provide a holistic view
      of intraday flow state. Not used anywhere in the decision path.
    """
    now = now_ny()
    state = _ANCHOR_STATE.get(ticker, None)
    if state is None:
        anchor = float(bias_val)
        _ANCHOR_STATE[ticker] = {"ts": now, "val": anchor}
        return anchor
    prev_ts = state.get("ts", now)
    prev_val = float(state.get("val", 0.0))
    dt_sec = max(1.0, (now - prev_ts).total_seconds())
    hl_sec = max(60.0, half_life_min * 60.0)
    alpha = 1.0 - math.exp(-math.log(2.0) * dt_sec / hl_sec)
    anchor = (1 - alpha) * prev_val + alpha * float(bias_val)
    _ANCHOR_STATE[ticker] = {"ts": now, "val": anchor}
    return float(anchor)


def _is_zero_flow_metric(metric: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metric, dict):
        return True
    try:
        V_val = float(metric.get('V', 0.0) or 0.0)
    except (TypeError, ValueError):
        V_val = 0.0
    try:
        bias_val = float(metric.get('bias', 0.0) or 0.0)
    except (TypeError, ValueError):
        bias_val = 0.0
    return (abs(V_val) <= 1e-6) and (abs(bias_val) <= 1e-6)


def compute_dte_shares(df: pd.DataFrame, interval_param: str, spot: float, mode: WeightingMode, mny_band: float,
                       exp_weights: np.ndarray) -> Dict[str, float]:
    """Compute weighted V shares by DTE (0 and 1) for a given interval."""
    try:
        c_col = f"call_{interval_param}"
        p_col = f"put_{interval_param}"
        if (c_col not in df.columns) or (p_col not in df.columns):
            return {"d0": 0.0, "d1": 0.0}
        strike = df["strike_price"].astype(float)
        call_delta = df.get("call_delta", pd.Series(np.zeros(len(df))))
        put_delta = df.get("put_delta", pd.Series(np.zeros(len(df))))
        w_c, w_p = leg_weights(strike, spot, mode, mny_band, call_delta, put_delta)
        ew = np.asarray(exp_weights, dtype=float)
        if ew.shape[0] == len(df):
            w_c = w_c * ew
            w_p = w_p * ew
        c_arr = df[c_col].astype(float).to_numpy() * w_c
        p_arr = df[p_col].astype(float).to_numpy() * w_p
        V = np.abs(c_arr) + np.abs(p_arr)
        # DTE buckets
        today = now_ny().date()
        exp_dates = pd.to_datetime(df.get('expiration_date'), errors='coerce').dt.date
        dte = pd.Series([(ed - today).days if pd.notna(ed) else 99 for ed in exp_dates])
        d0 = float(np.nansum(V[(dte == 0).to_numpy()]))
        d1 = float(np.nansum(V[(dte == 1).to_numpy()]))
        tot = float(np.nansum(V))
        if tot <= 0:
            return {"d0": 0.0, "d1": 0.0}
        return {"d0": d0 / tot, "d1": d1 / tot}
    except Exception:
        return {"d0": 0.0, "d1": 0.0}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Intraday options flow bot (daytrade)")
    ap.add_argument("--ui", action="store_true", help="Launch the Daytrade UI (port 8052) instead of CLI")
    ap.add_argument("--tickers", nargs="+", default=["SPX"], help="Tickers to analyze")
    # Deprecated: --mode/--next_n_days kept for backward-compatibility; no longer used.
    ap.add_argument("--mode", choices=["0dte", "next_n"], default="0dte", help=argparse.SUPPRESS)
    ap.add_argument("--next_n_days", type=int, default=2, help=argparse.SUPPRESS)
    ap.add_argument("--exp_count", type=int, default=7, help="Number of expiries to fetch (exps=[0..N-1])")
    ap.add_argument("--weighting", choices=["raw", "mny", "mny_delta"], default="mny_delta")
    ap.add_argument("--mny_band", type=float, default=0.02, help="Moneyness band fraction (e.g., 0.02 = ±2% around spot)")
    ap.add_argument("--spot", type=float, default=None, help="Underlying spot override")
    # Default: auto spot ON; allow disabling with --no_spot_auto
    ap.add_argument("--spot_auto", dest="spot_auto", action="store_true", default=True,
                    help="Auto-fetch spot via convexlib get_und (default: on)")
    ap.add_argument("--no_spot_auto", dest="spot_auto", action="store_false",
                    help="Disable auto spot fetch (fallback to median strike if no --spot)")
    ap.add_argument("--thresh5", type=float, default=0.30)
    ap.add_argument("--thresh15", type=float, default=0.20)
    ap.add_argument("--minV5", type=float, default=2e4)
    ap.add_argument("--minV15", type=float, default=4e4)
    ap.add_argument("--poll", type=int, default=0, help="Poll every N seconds (0 = run once)")
    ap.add_argument("--log", action="store_true", help="Write diagnostics logs (JSONL per ticker + daily CSV) like the UI")
    ap.add_argument("--log_dir", default=None, help="Custom logs dir (default: logs/daytrade-YYYYMMDD)")
    ap.add_argument("--email", default=os.environ.get("CONVEX_EMAIL"))
    ap.add_argument("--password", default=os.environ.get("CONVEX_PASSWORD"))
    ap.add_argument("--convex_env", default=os.environ.get("CONVEX_ENV", "live"))
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(__file__), 'thresholds.json'), help="Path to thresholds.json (optional)")
    ap.add_argument("--after_hours", action="store_true", help="Allow running after 16:00 NY (override market close guard)")
    return ap.parse_args()


def build_params(intervals: List[str]) -> List[str]:
    """Return param names for Convex get_chain in the order we will parse.
    intervals: list like ['5m','15m','30m','60m','td'] where 'td' means since-open (volm_bs).
    """
    mapping = {
        "5m": "volmbs_5m",
        "15m": "volmbs_15m",
        "30m": "volmbs_30m",
        "60m": "volmbs_60m",
        "td": "volm_bs",
    }
    params = ["oi", "oi_ch"]
    for k in intervals:
        if k not in mapping:
            continue
        params.append(mapping[k])
    # always include these needed fields after flows
    params += ["gamma", "expiration_ts", "price", "delta", "volatility"]
    return params


_FAIL_WINDOW_SECONDS = 30 * 60


def _update_fail_window_ratio(ticker: str, key: str, failed: bool, ts: datetime) -> float:
    window = _BIAS_FAIL_WINDOW.setdefault(ticker, {}).setdefault(key, deque())
    sums = _BIAS_FAIL_WINDOW_SUM.setdefault(ticker, {})
    current_sum = sums.get(key, 0)
    now_epoch = float(ts.timestamp())
    flag = 1 if failed else 0
    window.append((now_epoch, flag))
    current_sum += flag
    cutoff = now_epoch - _FAIL_WINDOW_SECONDS
    while window and window[0][0] < cutoff:
        _, old_flag = window.popleft()
        current_sum -= int(old_flag)
    sums[key] = max(current_sum, 0)
    length = len(window)
    if length <= 0:
        return 0.0
    return float(sums[key]) / float(length)


def fetch_spot(api: ConvexApi, ticker: str) -> float | None:
    """Robust spot fetch for an underlying symbol.
    Handles both observed shapes:
      {'data': [['SPX', 4500.6, 5.24e9], ...]}
      {'data': [[[ 'SPX', 4500.6 ]], ...]}
    Returns None on failure.
    """
    sym = str(ticker or "").upper()
    try:
        print(f"[dt] get_und symbols={[sym]} params=['price']")
        und = api.get_und(symbols=[sym], params=["price"])  # type: ignore
        print(f"[dt] datas={und}")
        data = und.get("data") if isinstance(und, dict) else None
        if not data:
            return None
        first = data[0]
        # shape A: ['SPX', 4500.6, ...]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (str, bytes)):
            return float(first[1])
        # shape B: [[ 'SPX', 4500.6 ]] (extra nesting)
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            inner = first[0]
            if len(inner) >= 2:
                return float(inner[1])
        return None
    except Exception as e:
        try:
            import traceback
            print(f"[dt] get_und ERROR ticker={ticker}: {e!r}")
            print(traceback.format_exc())
        except Exception:
            print("[dt] get_und ERROR (unable to format traceback)")
        return None


# get_und volume-based metrics removed for now (insufficient data fidelity)

def _get_chain_with_debug(api: ConvexApi, ticker: str, params: List[str], exps: List[int], rng: int = 1):
    t = norm_ticker(ticker)
    print(f"[dt] get_chain ticker={t} exps={exps} rng={rng} params={params}")
    chain = api.get_chain(t, params=params, exps=exps, rng=rng)  # type: ignore
    # Print light shape info to help diagnose empties
    try:
        d = chain.get('data') if isinstance(chain, dict) else None
        n = len(d) if d is not None else 0
        print(f"[dt] get_chain → data_len={n}")
        if n:
            # peek first element structure
            print(f"[dt] get_chain → first_type={type(d[0]).__name__}")
    except Exception as e:
        print(f"[dt] get_chain debug failed: {e}")
    return chain


def parse_chain_to_df(chain: Dict, params: List[str]) -> pd.DataFrame:
    """Parse Convex chain payload into a flat DataFrame with per-leg columns.
    The order of 'calls'/'puts' arrays matches params.
    """
    rows = []
    # Build per-param index helper: symbol at pos 0; then params arranged in the same order
    for data_group in chain.get("data", []):
        for _, option_groups, _ in data_group.get("chain", []):
            for option_group in option_groups:
                strike_price = option_group[0]
                calls, puts = option_group[1], option_group[2]
                # calls: [c_symbol, values for params...]
                # puts:  [p_symbol, values for params...]
                def pack(side_arr, side_prefix: str) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    # side_arr[0] is symbol
                    for i, p in enumerate(params, start=1):
                        key = f"{side_prefix}_{p}"
                        try:
                            out[key] = float(side_arr[i]) if side_arr[i] is not None else 0.0
                        except Exception:
                            out[key] = 0.0
                    return out

                row = {
                    "strike_price": float(strike_price),
                }
                row.update(pack(calls, "call"))
                row.update(pack(puts, "put"))
                rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Parse expiration_ts from either call or put (identical)
    exp_col = "call_expiration_ts" if "call_expiration_ts" in df.columns else None
    if exp_col is None and "put_expiration_ts" in df.columns:
        exp_col = "put_expiration_ts"
    if exp_col:
        dt = pd.to_datetime(df[exp_col], unit="ms", errors="coerce")
        df["expiration_dt"] = dt.dt.tz_localize(None)
        df["expiration_date"] = df["expiration_dt"].dt.date
    return df


def get_chain_df_spot(api: ConvexApi, ticker: str, fc: FlowConfig,
                      spot_override: float | None = None) -> Tuple[pd.DataFrame, float, str]:
    """Fetch chain with multi-interval flows, parse, and resolve spot.
    Returns (df_all, spot, spot_src). Uses fc.exp_count to select expirations.
    """
    intervals = ["5m", "15m", "30m", "60m", "td"]
    params = build_params(intervals)
    # Always fetch closest 7 expiries at most (index 0..6)
    n = min(7, max(1, int(getattr(fc, 'exp_count', 7) or 7)))
    exps = list(range(0, n))
    t = norm_ticker(ticker)
    chain = _get_chain_with_debug(api, t, params, exps, rng=100)
    df = parse_chain_to_df(chain, params)
    if df.empty:
        print(f"[dt] parse_chain_to_df → empty frame for ticker={ticker}")
        return df, 0.0, "empty"

    # Spot
    if spot_override is not None and spot_override > 0:
        spot = float(spot_override)
        spot_src = "manual"
    elif fc.use_spot_auto:
        s = fetch_spot(api, t)
        spot = float(s) if (s is not None and s > 0) else float(np.median(df["strike_price"]))
        spot_src = "auto" if s is not None else "median_strike"
    else:
        spot = float(np.median(df["strike_price"]))
        spot_src = "median_strike"

    try:
        _update_zero_gamma_cache(t, df, spot)
    except Exception:
        pass

    return df, spot, spot_src


def get_chain_both(api: ConvexApi, ticker: str, fc: FlowConfig,
                   spot_override: float | None = None) -> Tuple[pd.DataFrame, float, str]:
    """Fetch chain once and return (df_all, spot, spot_src)."""
    intervals = ["5m", "15m", "30m", "60m", "td"]
    params = build_params(intervals)
    n = min(7, max(1, int(getattr(fc, 'exp_count', 7) or 7)))
    exps = list(range(0, n))
    t = norm_ticker(ticker)
    chain = _get_chain_with_debug(api, t, params, exps, rng=100)
    df_all = parse_chain_to_df(chain, params)
    if df_all.empty:
        print(f"[dt] parse_chain_to_df (all) → empty frame for ticker={ticker}")
        return df_all, 0.0, "empty"

    # Spot
    if spot_override is not None and spot_override > 0:
        spot = float(spot_override)
        spot_src = "manual"
    elif fc.use_spot_auto:
        s = fetch_spot(api, t)
        spot = float(s) if (s is not None and s > 0) else float(np.median(df_all["strike_price"]))
        spot_src = "auto" if s is not None else "median_strike"
    else:
        spot = float(np.median(df_all["strike_price"]))
        spot_src = "median_strike"

    try:
        _update_zero_gamma_cache(t, df_all, spot)
    except Exception:
        pass

    return df_all, spot, spot_src


def strike_weight(strike: pd.Series, spot: float, mode: WeightingMode, mny_band: float,
                  call_delta: pd.Series | None, put_delta: pd.Series | None) -> np.ndarray:
    if mode == "raw" or spot <= 0:
        base = np.ones(len(strike))
    else:
        dist = (strike - spot).abs() / max(spot, 1e-9)
        base = np.exp(- (dist / max(mny_band, 1e-6)) ** 2)
    if mode == "mny_delta":
        dmag = pd.concat([
            call_delta.abs().fillna(0) if call_delta is not None else pd.Series(np.zeros(len(strike))),
            put_delta.abs().fillna(0) if put_delta is not None else pd.Series(np.zeros(len(strike))),
        ], axis=1).max(axis=1).to_numpy()
        return base * np.clip(dmag, 0.0, 1.0)
    return base


def leg_weights(strike: pd.Series, spot: float, mode: WeightingMode, mny_band: float,
                call_delta: pd.Series | None, put_delta: pd.Series | None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-leg weights: returns (w_c, w_p).
    - raw: both 1
    - mny: both = mny Gaussian around spot
    - mny_delta: mny × |Δ_leg|
    """
    n = len(strike)
    if mode == 'raw' or spot <= 0:
        ones = np.ones(n, dtype=float)
        return ones, ones
    dist = (strike.astype(float) - float(spot)).abs() / max(float(spot), 1e-9)
    base = np.exp(- (dist / max(float(mny_band), 1e-6)) ** 2).to_numpy()
    if mode == 'mny':
        return base, base
    # mny_delta
    cdel = (call_delta.abs().fillna(0.0) if call_delta is not None else pd.Series(np.zeros(n))).astype(float).clip(0.0, 1.0).to_numpy()
    pdel = (put_delta.abs().fillna(0.0) if put_delta is not None else pd.Series(np.zeros(n))).astype(float).clip(0.0, 1.0).to_numpy()
    return base * cdel, base * pdel


def _dynamic_band_from_vix(vix_value: float) -> float:
    try:
        v = float(vix_value)
        band = v / 16.0 / 100.0
        return float(max(band, 1e-4))
    except Exception:
        return 0.01


def _expiry_decay_inv_sqrt(exp_series: pd.Series | None) -> np.ndarray:
    if exp_series is None:
        return np.ones(0, dtype=float)
    try:
        exp_dates = pd.to_datetime(exp_series, errors='coerce')
        today = now_ny().date()
        dte = exp_dates.dt.date.map(lambda d: max((d - today).days, 0) if pd.notna(d) else None)
        weights = []
        for val in dte:
            if val is None:
                weights.append(1.0)
            else:
                weights.append(1.0 / math.sqrt(float(val) + 1.0))
        return np.array(weights, dtype=float)
    except Exception:
        return np.ones(len(exp_series), dtype=float)


def compute_interval_flow_dyn(
    df: pd.DataFrame,
    interval_param: str,
    spot: float,
    mode: WeightingMode,
    mny_band_dyn: float,
    fallback_param: str | None = None,
) -> Tuple[float, float, float, float]:
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0

    def _compute(param: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        c_col = f"call_{param}"
        p_col = f"put_{param}"
        if (c_col not in df.columns) or (p_col not in df.columns):
            return np.zeros(len(df)), np.zeros(len(df)), False
        c_vals = df[c_col].astype(float).to_numpy()
        p_vals = df[p_col].astype(float).to_numpy()
        return c_vals, p_vals, True

    c_arr, p_arr, ok = _compute(interval_param)
    used_param = interval_param
    if not ok and fallback_param:
        c_arr, p_arr, ok = _compute(fallback_param)
        used_param = fallback_param if ok else interval_param
    if not ok:
        return 0.0, 0.0, 0.0, 0.0

    strike = df["strike_price"].astype(float)
    call_delta = df.get("call_delta", pd.Series(np.zeros(len(df))))
    put_delta = df.get("put_delta", pd.Series(np.zeros(len(df))))
    w_c, w_p = leg_weights(strike, float(spot), mode, float(mny_band_dyn), call_delta, put_delta)

    exp_weights = _expiry_decay_inv_sqrt(df.get('expiration_date'))
    if exp_weights.size == len(w_c):
        w_c = w_c * exp_weights
        w_p = w_p * exp_weights

    c_weighted = c_arr * w_c
    p_weighted = p_arr * w_p
    C = float(np.nansum(c_weighted))
    P = float(np.nansum(p_weighted))
    V = float(np.nansum(np.abs(c_weighted)) + np.nansum(np.abs(p_weighted)))
    denom = abs(C) + abs(P)
    bias = 0.0 if denom < 1e-12 else (C - P) / (denom + EPS)
    return C, P, V, bias
def compute_alt_flows(
    df: pd.DataFrame,
    spot: float,
    mode: WeightingMode,
    mny_band_dyn: float,
    intervals: Tuple[str, ...] = ("volmbs_5m", "volmbs_15m", "volmbs_30m", "volmbs_60m"),
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    exp_weights_alt = _expiry_decay_inv_sqrt(df.get('expiration_date'))
    for param in intervals:
        label = param.replace('volmbs_', '').replace('volm_bs', 'net')
        key = f"{label}_vix16"
        fallback = 'volm_bs' if param in ('volmbs_5m', 'volmbs_15m') else None
        try:
            C, P, V, bias = compute_interval_flow(
                df,
                param,
                spot,
                mode,
                mny_band_dyn,
                exp_weights_alt,
            )
            if V == 0.0 and fallback:
                C, P, V, bias = compute_interval_flow(
                    df,
                    fallback,
                    spot,
                    mode,
                    mny_band_dyn,
                    exp_weights_alt,
                )
        except Exception as exc:
            print(f"[cli] alt flow error ({param}): {exc}")
            C = P = V = bias = 0.0
        out[key] = {
            'C': C,
            'P': P,
            'V': V,
            'bias': bias,
        }
    return out


def compute_interval_flow(
    df: pd.DataFrame,
    interval_param: str,
    spot: float,
    mode: WeightingMode,
    mny_band: float,
    exp_weights: np.ndarray | None = None,
    return_breakdown: bool = False,
) -> Tuple[float, float, float, float] | Tuple[float, float, float, float, dict]:
    """Compute weighted C, P, V, bias for a given interval param name like 'volmbs_5m' or 'volm_bs'.
    Returns (C, P, V, bias).
    """
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0

    # Build dynamic column names present from parse
    c_col = f"call_{interval_param}"
    p_col = f"put_{interval_param}"
    if (c_col not in df.columns) or (p_col not in df.columns):
        return 0.0, 0.0, 0.0, 0.0

    strike = df["strike_price"].astype(float)
    call_delta = df.get("call_delta", pd.Series(np.zeros(len(df))))
    put_delta = df.get("put_delta", pd.Series(np.zeros(len(df))))
    w_c, w_p = leg_weights(strike, spot, mode, mny_band, call_delta, put_delta)
    ew = None
    if exp_weights is not None:
        try:
            ew = np.asarray(exp_weights, dtype=float)
            if ew.shape[0] == len(df):
                w_c = w_c * ew
                w_p = w_p * ew
            else:
                ew = None
        except Exception:
            ew = None
    c_arr = df[c_col].astype(float).to_numpy() * w_c
    p_arr = df[p_col].astype(float).to_numpy() * w_p
    C = float(np.nansum(c_arr))
    P = float(np.nansum(p_arr))
    V = float(np.nansum(np.abs(c_arr)) + np.nansum(np.abs(p_arr)))
    bias = 0.0 if (abs(C) + abs(P)) < 1e-12 else (C - P) / (abs(C) + abs(P) + EPS)

    if not return_breakdown:
        return C, P, V, bias

    breakdown: dict[str, dict[str, float | None]] = {}
    try:
        exp_series = df.get('expiration_date')
        if exp_series is not None:
            exp_dates = pd.to_datetime(exp_series, errors='coerce')
            today = now_ny().date()
            dte_days = exp_dates.dt.date.map(lambda d: (d - today).days if pd.notna(d) else None)
            dte_arr = np.full(len(df), 'd2p', dtype=object)
            dte_vals = dte_days.to_numpy()
            mask_valid = pd.notna(dte_vals)
            mask_d0 = mask_valid & (dte_vals == 0)
            mask_d1 = mask_valid & (dte_vals == 1)
            mask_d2p = mask_valid & (dte_vals >= 2)
            dte_arr[mask_d0] = 'd0'
            dte_arr[mask_d1] = 'd1'
            dte_arr[mask_d2p] = 'd2p'
        else:
            dte_arr = np.full(len(df), 'unknown', dtype=object)

        abs_c = np.abs(c_arr)
        abs_p = np.abs(p_arr)
        for bucket in ('d0', 'd1', 'd2p', 'unknown'):
            mask = dte_arr == bucket
            if not mask.any():
                continue
            C_bucket = float(np.nansum(c_arr[mask]))
            P_bucket = float(np.nansum(p_arr[mask]))
            V_bucket = float(np.nansum(abs_c[mask]) + np.nansum(abs_p[mask]))
            weight_val = float(np.nanmean(ew[mask])) if ew is not None else None
            breakdown[bucket] = {
                'C': C_bucket,
                'P': P_bucket,
                'V': V_bucket,
                'weight': weight_val,
            }
    except Exception:
        breakdown = {}

    return C, P, V, bias, breakdown


def compute_ladders(df: pd.DataFrame, spot: float, weighting: WeightingMode, mny_band: float,
                    intervals: Tuple[str, ...] = ("volmbs_5m", "volmbs_15m", "volmbs_30m", "volmbs_60m"),
                    view_band_pct: float | None = None,
                    exp_weights: np.ndarray | None = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-interval strike ladders limited to ±mny_band around spot.
    For each interval key, returns dict with keys: 'K' (strikes), 'net' (Cw-Pw), 'cw', 'pw'.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if df.empty:
        return out
    # Prepare strike and weights
    strike_ser = df['strike_price'].astype(float)
    call_delta = df.get('call_delta', pd.Series(np.zeros(len(df))))
    put_delta = df.get('put_delta', pd.Series(np.zeros(len(df))))
    w_c, w_p = leg_weights(strike_ser, spot, weighting, mny_band, call_delta, put_delta)
    if exp_weights is not None:
        try:
            ew = np.asarray(exp_weights, dtype=float)
            if ew.shape[0] == len(df):
                w_c = w_c * ew
                w_p = w_p * ew
        except Exception:
            pass
    strike_np = strike_ser.to_numpy()

    # Filter to ±band around spot for visualization (use pure numpy to avoid index alignment issues)
    # Use a separate "view" band for selection if provided (e.g., ±10%),
    # otherwise fall back to the weighting band
    band = max(view_band_pct if (view_band_pct is not None) else mny_band, 1e-6)
    denom = max(spot, 1e-9)
    mask_np = (np.abs((strike_np - spot) / denom) <= band)
    if not mask_np.any():
        mask_np = np.ones(len(df), dtype=bool)

    for key in intervals:
        c_col = f"call_{key}"
        p_col = f"put_{key}"
        if c_col not in df.columns or p_col not in df.columns:
            continue
        c_arr = df[c_col].astype(float).to_numpy() * w_c
        p_arr = df[p_col].astype(float).to_numpy() * w_p
        cw = c_arr[mask_np]
        pw = p_arr[mask_np]
        K = strike_np[mask_np]
        if K.size == 0:
            continue
        # Aggregate by unique strike
        order = np.argsort(K)
        # Ensure pure numpy to avoid pandas index alignment issues
        K = np.asarray(K)
        cw = np.asarray(cw)
        pw = np.asarray(pw)
        K_sorted = np.take(K, order)
        cw_sorted = np.take(cw, order)
        pw_sorted = np.take(pw, order)
        Ks, idx = np.unique(K_sorted, return_index=True)
        cw_sum = np.add.reduceat(cw_sorted, idx)
        pw_sum = np.add.reduceat(pw_sorted, idx)
        net = cw_sum - pw_sum
        out[key] = {"K": Ks, "net": net, "cw": cw_sum, "pw": pw_sum}
    return out


def compute_expiry_strip(df_all: pd.DataFrame, spot: float, weighting: WeightingMode, mny_band: float,
                         max_exps: int = 7,
                         only_zero_dte: bool = False) -> pd.DataFrame:
    """Compute since-open weighted net flow per expiration (up to max_exps).
    Returns a DataFrame with columns: expiration_date, C, P, V, net.
    """
    if df_all.empty:
        return pd.DataFrame(columns=["expiration_date","C","P","V","net"])
    # require since-open columns
    if ("call_volm_bs" not in df_all.columns) or ("put_volm_bs" not in df_all.columns):
        return pd.DataFrame(columns=["expiration_date","C","P","V","net"])

    if only_zero_dte and 'expiration_date' in df_all.columns:
        try:
            today = now_ny().date()
            mask_0dte = pd.to_datetime(df_all['expiration_date'], errors='coerce').dt.date == today
            df_all = df_all[mask_0dte].copy()
        except Exception:
            pass
    if df_all.empty:
        return pd.DataFrame(columns=["expiration_date","C","P","V","net"])

    strike = df_all['strike_price'].astype(float)
    call_delta = df_all.get('call_delta', pd.Series(np.zeros(len(df_all))))
    put_delta = df_all.get('put_delta', pd.Series(np.zeros(len(df_all))))
    w_c, w_p = leg_weights(strike, spot, weighting, mny_band, call_delta, put_delta)

    dfw = pd.DataFrame({
        'expiration_date': df_all['expiration_date'],
        'Cw': df_all['call_volm_bs'].astype(float).to_numpy() * w_c,
        'Pw': df_all['put_volm_bs'].astype(float).to_numpy() * w_p,
    })
    agg = dfw.groupby('expiration_date', as_index=False)[['Cw','Pw']].sum()
    agg = agg.sort_values('expiration_date').head(max_exps)
    agg['V'] = agg['Cw'].abs() + agg['Pw'].abs()
    agg['net'] = agg['Cw'] - agg['Pw']
    agg = agg.rename(columns={'Cw':'C','Pw':'P'})
    return agg


# SPX Zero-Gamma logic removed per request.


def pick_expiration_filter(df: pd.DataFrame, mode: str, next_n_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    today = now_ny().date()
    if mode == "0dte":
        # For index (e.g., SPX) we often have 0DTE; many stocks do not.
        # If no 0DTE rows are present, fall back to returning the unfiltered df
        # (which should already be constrained by the API exps list).
        out = df[df["expiration_date"] == today].copy()
        return out if not out.empty else df.copy()
    else:
        end = pd.Timestamp(today) + pd.Timedelta(days=max(1, int(next_n_days) - 1))
        return df[(df["expiration_date"] >= today) & (df["expiration_date"] <= end.date())].copy()


def signal_from_flows(C5: float, P5: float, V5: float, b5: float,
                      C15: float, P15: float, V15: float, b15: float,
                      sc: SignalConfig) -> Dict[str, object]:
    # Activity floors
    ok5 = (V5 >= sc.minV5) and (abs(b5) >= sc.thresh5)
    ok15 = (V15 >= sc.minV15) and (abs(b15) >= sc.thresh15) and (np.sign(b15) == np.sign(b5))
    signal = "flat"
    if ok5 and ok15:
        signal = "long" if b5 > 0 else "short"
    strength = float(max(0.0, min(1.0, 0.5 * (abs(b5) / max(sc.thresh5, 1e-6)) + 0.5 * (abs(b15) / max(sc.thresh15, 1e-6)))))
    return {
        "signal": signal,
        "agree": bool(np.sign(b15) == np.sign(b5)) if (V5 > 0 and V15 > 0) else False,
        "strength": strength,
        "metrics": {
            "5m": {"C": C5, "P": P5, "V": V5, "bias": b5},
            "15m": {"C": C15, "P": P15, "V": V15, "bias": b15},
        },
    }


def fetch_and_signal(api: ConvexApi, ticker: str, fc: FlowConfig, sc: SignalConfig,
                     spot_override: float | None = None,
                     do_iv_upsert: bool = False,
                     retry_on_zero_flow: bool = False,
                     max_zero_attempts: int = 2,
                     wall_guard_enabled: bool = True) -> Dict[str, object]:
    t = norm_ticker(ticker)
    attempts = 1
    if retry_on_zero_flow:
        attempts = max(1, int(max_zero_attempts))

    last_df: Optional[pd.DataFrame] = None
    spot = 0.0
    spot_src = "empty"
    exp_w = None
    iv_ctx_agg = None
    dte_shares = None
    anchor_val = None
    sig: Dict[str, Any] = {}
    zero_flow_flags: List[str] = []

    for attempt in range(attempts):
        df, spot, spot_src = get_chain_df_spot(api, t, fc, spot_override)
        if df.empty:
            return {"ticker": t, "error": "empty_chain"}
        last_df = df

        exp_w = None
        if t == 'SPX' and ('expiration_date' in df.columns):
            try:
                exp_w = _spx_expiry_weight_for_date(df['expiration_date'])
            except Exception:
                exp_w = None

        C5, P5, V5, b5, breakdown5 = compute_interval_flow(
            df,
            "volmbs_5m",
            spot,
            fc.weighting,
            fc.mny_band,
            exp_w,
            return_breakdown=True,
        )
        C15, P15, V15, b15, breakdown15 = compute_interval_flow(
            df,
            "volmbs_15m",
            spot,
            fc.weighting,
            fc.mny_band,
            exp_w,
            return_breakdown=True,
        )
        C30, P30, V30, b30 = compute_interval_flow(df, "volmbs_30m", spot, fc.weighting, fc.mny_band, exp_w)
        C60, P60, V60, b60 = compute_interval_flow(df, "volmbs_60m", spot, fc.weighting, fc.mny_band, exp_w)
        Cday, Pday, Vday, bday, breakdown_day = compute_interval_flow(
            df,
            "volm_bs",
            spot,
            fc.weighting,
            fc.mny_band,
            exp_w,
            return_breakdown=True,
        )

        if V5 == 0.0 and ("call_volmbs_5m" not in df.columns):
            C5, P5, V5, b5, breakdown5 = compute_interval_flow(
                df,
                "volm_bs",
                spot,
                fc.weighting,
                fc.mny_band,
                exp_w,
                return_breakdown=True,
            )
        if V15 == 0.0 and ("call_volmbs_15m" not in df.columns):
            C15, P15, V15, b15, breakdown15 = compute_interval_flow(
                df,
                "volm_bs",
                spot,
                fc.weighting,
                fc.mny_band,
                exp_w,
                return_breakdown=True,
            )

        sig = signal_from_flows(C5, P5, V5, b5, C15, P15, V15, b15, sc)
        metrics_map = sig.setdefault('metrics', {})
        try:
            metrics_map['30m'] = {"C": C30, "P": P30, "V": V30, "bias": b30}
            metrics_map['60m'] = {"C": C60, "P": P60, "V": V60, "bias": b60}
            metrics_map['day'] = {"C": Cday, "P": Pday, "V": Vday, "bias": bday}
        except Exception:
            pass

        if breakdown5:
            try:
                metrics_map.setdefault('5m', {})['dte_breakdown'] = breakdown5
            except Exception:
                pass
        if breakdown15:
            try:
                metrics_map.setdefault('15m', {})['dte_breakdown'] = breakdown15
            except Exception:
                pass
        if breakdown_day:
            try:
                metrics_map.setdefault('day', {})['dte_breakdown'] = breakdown_day
            except Exception:
                pass

        m5_metrics = metrics_map.get('5m', {}) if isinstance(metrics_map, dict) else {}
        m15_metrics = metrics_map.get('15m', {}) if isinstance(metrics_map, dict) else {}
        b5_now = float(m5_metrics.get('bias', 0.0))
        b15_now = float(m15_metrics.get('bias', 0.0))
        b5_pass_thresh = abs(b5_now) >= float(sc.thresh5)
        b15_pass_thresh = abs(b15_now) >= float(sc.thresh15)
        poll_time = now_ny()
        b5_fail_ratio_30m = _update_fail_window_ratio(t, '5m', not b5_pass_thresh, poll_time)
        b15_fail_ratio_30m = _update_fail_window_ratio(t, '15m', not b15_pass_thresh, poll_time)
        stats_bucket = _BIAS_FAIL_STATS.setdefault(t, {'total_polls': 0, 'b5_fail_count': 0, 'b15_fail_count': 0})
        stats_bucket['total_polls'] += 1
        if not b5_pass_thresh:
            stats_bucket['b5_fail_count'] += 1
        if not b15_pass_thresh:
            stats_bucket['b15_fail_count'] += 1
        total_polls = stats_bucket['total_polls']
        b5_fail_count = stats_bucket['b5_fail_count']
        b15_fail_count = stats_bucket['b15_fail_count']
        b5_fail_ratio = (b5_fail_count / total_polls) if total_polls > 0 else 0.0
        b15_fail_ratio = (b15_fail_count / total_polls) if total_polls > 0 else 0.0

        disagree_bucket = _SIGN_DISAGREE_STATS.setdefault(t, {'total_polls': 0, 'disagree_count': 0})
        disagree_bucket['total_polls'] += 1
        sign_b5 = np.sign(b5_now)
        sign_b15 = np.sign(b15_now)
        if sign_b5 != 0 and sign_b15 != 0 and sign_b5 != sign_b15:
            disagree_bucket['disagree_count'] += 1
        sign_disagree_count = disagree_bucket['disagree_count']

        alert_active = False
        alert_message = None
        min_polls = max(1, int(getattr(sc, 'b5_alert_min_polls', 12)))
        ratio_threshold = float(getattr(sc, 'b5_fail_ratio_alert', 0.20))
        if total_polls >= min_polls and b5_fail_ratio >= ratio_threshold:
            alert_active = True
            alert_message = (
                f"Choppy bias regime: b5 fails {b5_fail_count}/{total_polls} ({b5_fail_ratio:.1%})"
            )
        bias_stats_payload = {
            'total_polls': total_polls,
            'b5_fail_count': b5_fail_count,
            'b15_fail_count': b15_fail_count,
            'b5_fail_ratio': b5_fail_ratio,
            'b15_fail_ratio': b15_fail_ratio,
            'b5_fail_ratio_30m': b5_fail_ratio_30m,
            'b15_fail_ratio_30m': b15_fail_ratio_30m,
            'sign_disagree_count': sign_disagree_count,
        }
        bias_alert_payload = {
            'active': alert_active,
            'message': alert_message,
            'b5_fail_ratio': b5_fail_ratio,
            'b5_fail_ratio_30m': b5_fail_ratio_30m,
            'threshold': ratio_threshold,
            'min_polls': min_polls,
            'b5_fail_count': b5_fail_count,
            'total_polls': total_polls,
        }

        zero_flow_flags = []
        if _is_zero_flow_metric(metrics_map.get('5m')):
            zero_flow_flags.append('zero_flow_5m')
        if _is_zero_flow_metric(metrics_map.get('15m')):
            zero_flow_flags.append('zero_flow_15m')

        if zero_flow_flags and retry_on_zero_flow and (attempt + 1) < attempts:
            continue
        break

    if last_df is None:
        return {"ticker": t, "error": "empty_chain"}
    df = last_df
    attempt_count = attempt + 1

    if zero_flow_flags:
        sig['signal'] = 'flat'
        sig['agree'] = False
        sig['strength'] = 0.0
        for m in sig.get('metrics', {}).values():
            if isinstance(m, dict):
                m['C'] = 0.0
                m['P'] = 0.0
                m['V'] = 0.0
                m['bias'] = 0.0

    raw_signal_mode = str(sig.get('signal', 'flat'))

    wall_snapshot_dict: Optional[Dict[str, Any]] = None
    wall_guard_dict: Optional[Dict[str, Any]] = None
    wall_guard_applied = False
    wall_snapshot = None
    if t == 'SPX':
        try:
            detector = _WALL_DETECTORS.setdefault(t, WallDetector())
            wall_snapshot = detector.update(df, float(spot), now_ny())
        except Exception as wall_exc:
            if wall_guard_enabled:
                wall_guard_dict = {
                    "error": str(wall_exc),
                    "original_signal": raw_signal_mode,
                    "signal": raw_signal_mode,
                    "triggered": False,
                    "enabled": True,
                    "applied": False,
                }
        else:
            if wall_snapshot is not None:
                wall_snapshot_dict = wall_snapshot.to_dict()
                if wall_guard_enabled:
                    guard_decision = _WALL_GUARD.evaluate(raw_signal_mode, wall_snapshot)
                    wall_guard_dict = guard_decision.to_dict()
                    wall_guard_dict['enabled'] = True
                    sig['raw_signal'] = raw_signal_mode
                    sig.setdefault('guards', {})['wall'] = wall_guard_dict
                    if guard_decision.triggered:
                        sig['signal'] = guard_decision.signal
                        wall_guard_applied = True
                        wall_guard_dict['applied'] = True
                    else:
                        wall_guard_dict['applied'] = False
            elif wall_guard_enabled:
                wall_guard_dict = {
                    "original_signal": raw_signal_mode,
                    "signal": raw_signal_mode,
                    "triggered": False,
                    "enabled": True,
                    "applied": False,
                }
    else:
        wall_guard_enabled = False
    if 'raw_signal' not in sig:
        sig['raw_signal'] = raw_signal_mode

    if t == 'SPX' and exp_w is not None:
        try:
            dte_shares = compute_dte_shares(df, "volmbs_15m", spot, fc.weighting, fc.mny_band, exp_w)
        except Exception:
            dte_shares = None
        try:
            anchor_val = update_anchor('SPX', float(sig.get('metrics', {}).get('5m', {}).get('bias', 0.0)), half_life_min=30.0)
        except Exception:
            anchor_val = None

    try:
        if do_iv_upsert:
            iv_ctx_agg = update_iv_ctx_for_ticker(t, df, spot)
    except Exception:
        iv_ctx_agg = None

    w = strike_weight(df["strike_price"], spot, fc.weighting, fc.mny_band, df.get("call_delta"), df.get("put_delta"))

    def _series_or_zero(name_primary: str, name_fallback: str) -> pd.Series:
        primary = df.get(name_primary)
        if primary is not None:
            return pd.to_numeric(primary, errors='coerce').fillna(0.0)
        fallback = df.get(name_fallback)
        if fallback is not None:
            return pd.to_numeric(fallback, errors='coerce').fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index)

    call_act = _series_or_zero("call_volmbs_5m", "call_volm_bs").to_numpy()
    put_act = _series_or_zero("put_volmbs_5m", "put_volm_bs").to_numpy()
    act = (np.abs(call_act) + np.abs(put_act)) * w
    top_idx = np.argsort(act)[::-1][:8]
    tops = df.iloc[top_idx][[c for c in ["strike_price", "expiration_date"] if c in df.columns]].copy()
    tops["mny"] = (tops["strike_price"].astype(float) - spot) / max(spot, 1e-9)

    mny_band_dyn = _dynamic_band_from_vix(_ALT_VIX_ASSUMED)
    try:
        alt_intervals = compute_alt_flows(df, spot, fc.weighting, mny_band_dyn)
    except Exception as exc:
        print(f"[cli] alt flow compute error: {exc}")
        alt_intervals = {}
    dyn_config = {
        'vix_assumed': float(_ALT_VIX_ASSUMED),
        'mny_band_dyn': float(mny_band_dyn),
        'expiry_weight': _ALT_EXPIRY_WEIGHT_DESC,
    }
    if zero_flow_flags:
        for key in list(alt_intervals.keys()):
            alt_intervals[key] = {'C': 0.0, 'P': 0.0, 'V': 0.0, 'bias': 0.0}

    return {
        "ticker": t,
        "spot": spot,
        "spot_src": spot_src,
        "exp_count": getattr(fc, 'exp_count', 7),
        "weighting": fc.weighting,
        "mny_band": fc.mny_band,
        "intervals": ["5m", "15m", "30m", "60m", "day"],
        "signal": sig,
        "parse_rows": int(len(df)),
        "distinct_exps": int(df['expiration_date'].nunique()) if 'expiration_date' in df.columns else 0,
        "tops": tops.to_dict("records") if not tops.empty else [],
        "dte_shares": dte_shares if dte_shares else {},
        "anchor": float(anchor_val) if anchor_val is not None else None,
        "iv_ctx": iv_ctx_agg if iv_ctx_agg else {},
        "atm_iv_curve": compute_atm_iv_curve(df, spot, max_exps=6),
        "zero_flow_flags": zero_flow_flags,
        "zero_flow_attempts": attempt_count,
        "wall_snapshot": wall_snapshot_dict,
        "wall_guard": wall_guard_dict,
        "wall_guard_applied": wall_guard_applied,
        "alt_intervals": alt_intervals,
        "dyn_config": dyn_config,
        "bias_fail_stats": bias_stats_payload,
        "bias_alert": bias_alert_payload,
    }


def print_signal(res: Dict[str, object]) -> None:
    if "error" in res:
        print(f"{res['ticker']}: ERROR — {res['error']}")
        return
    sig = res["signal"]
    expc = res.get('exp_count', '?')
    print(f"\n{res['ticker']} — spot={res['spot']:.2f} ({res['spot_src']}); expiries={expc} weighting={res['weighting']} band={res['mny_band']}")
    print(f"  5m:  bias={sig['metrics']['5m']['bias']:.3f}  V={sig['metrics']['5m']['V']:.0f}  C={sig['metrics']['5m']['C']:.0f}  P={sig['metrics']['5m']['P']:.0f}")
    print(f"  15m: bias={sig['metrics']['15m']['bias']:.3f}  V={sig['metrics']['15m']['V']:.0f}  C={sig['metrics']['15m']['C']:.0f}  P={sig['metrics']['15m']['P']:.0f}")
    print(f"  agree={sig['agree']}  -> SIGNAL: {sig['signal'].upper()}  strength={sig['strength']:.2f}")
    wall_guard = res.get('wall_guard') if isinstance(res, dict) else None
    if isinstance(wall_guard, dict):
        wall = wall_guard.get('wall') or {}
        side = wall_guard.get('side', '')
        strike = wall.get('strike')
        density = wall.get('density')
        health_state = str(wall.get('health_state', 'n/a'))
        try:
            health_change = f"{float(wall.get('health_change', 0.0)):+.2f}"
        except Exception:
            health_change = "?"
        try:
            health_velocity = f"{float(wall.get('health_velocity', 0.0)):+.2f}"
        except Exception:
            health_velocity = "?"
        if wall_guard.get('triggered'):
            print(
                f"  wall guard: triggered ({side} @{strike}) -> {wall_guard.get('signal')} "
                f"health={health_state} Δ={health_change} vel={health_velocity}"
            )
        elif wall:
            try:
                density_txt = f"{float(density):.2f}"
            except Exception:
                density_txt = "?"
            print(
                f"  wall context: {side} wall @{strike} density={density_txt} "
                f"health={health_state} Δ={health_change} vel={health_velocity}"
            )
    if res.get("tops"):
        print("  top strikes (by weighted 5m activity):")
        for t in res["tops"]:
            print(f"    K={t['strike_price']:.1f}  exp={t.get('expiration_date','?')}  mny={t['mny']:+.3%}")
    zero_flags = res.get('zero_flow_flags')
    if zero_flags:
        attempts = res.get('zero_flow_attempts')
        tag = ','.join(zero_flags) if isinstance(zero_flags, (list, tuple)) else str(zero_flags)
        print(f"  NOTE: zero-flow fallback applied ({tag}) attempts={attempts}")


def make_api(creds: Credentials) -> ConvexApi:
    return ConvexApi(creds.email, creds.password, creds.env)


# ---------------------- CLI Logging helpers ----------------------

def _ny_now_iso_cli() -> str:
    try:
        return datetime.now(NY_TZ).isoformat()
    except Exception:
        return datetime.now().isoformat()


def _is_regular_market_time_cli(ref: datetime | None = None) -> bool:
    """Return True when within regular NY session (Mon-Fri, 09:30-16:00 ET)."""
    try:
        now_ref = ref or datetime.now(NY_TZ)
    except Exception:
        now_ref = ref or datetime.now()
    if now_ref.weekday() >= 5:  # Saturday/Sunday
        return False
    start = now_ref.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now_ref.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now_ref <= end


def _log_dir_for_today_cli(custom_dir: str | None = None) -> str:
    if custom_dir:
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir
    d = datetime.now(NY_TZ).strftime("%Y%m%d")
    path = os.path.join("logs", f"daytrade-{d}")
    os.makedirs(path, exist_ok=True)
    return path


def _append_jsonl_cli(path: str, obj: dict) -> None:
    def _json_default(o):
        try:
            import numpy as _np  # local import to avoid hard dep at import time
            import pandas as _pd
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, (_pd.Timestamp, datetime)):
                return o.isoformat()
        except Exception:
            pass
        if isinstance(o, set):
            return list(o)
        return str(o)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
    except Exception as e:
        print(f"[cli] JSONL write error {path}: {e}")


def _append_csv_cli(path: str, header: list[str], row: list) -> None:
    try:
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print(f"[cli] CSV write error {path}: {e}")


def main() -> None:
    args = parse_args()
    # UI mode: start the Dash server and let you enter ticker/poll in the UI
    if args.ui:
        # Try importing as a package (from repo root) and as a local module (when run inside daytrade/)
        try:
            import importlib
            import pathlib, sys as _sys
            pkg_dir = pathlib.Path(__file__).resolve().parent           # .../daytrade
            repo_root = pkg_dir.parent                                   # repo root
            # Ensure repo root is on sys.path so 'daytrade' package resolves even when running from the folder
            if str(repo_root) not in _sys.path:
                _sys.path.insert(0, str(repo_root))
            try:
                ui_mod = importlib.import_module("daytrade.ui")
            except ModuleNotFoundError:
                # Fallback: import local module if package form still not available
                if str(pkg_dir) not in _sys.path:
                    _sys.path.insert(0, str(pkg_dir))
                ui_mod = importlib.import_module("ui")
            print("Starting Daytrade UI on http://127.0.0.1:8052 …")
            ui_mod.app.run(debug=True, port=8052, use_reloader=False)
            return
        except Exception as e:
            print(f"Failed to start UI: {e}")
            return
    if not args.email or not args.password:
        raise SystemExit("Missing credentials. Provide --email/--password or set CONVEX_EMAIL/CONVEX_PASSWORD env vars.")
    creds = Credentials(args.email, args.password, args.convex_env)
    api = make_api(creds)

    fc = FlowConfig(weighting=args.weighting,
                    mny_band=args.mny_band,
                    exp_count=int(getattr(args, 'exp_count', 7) or 7),
                    use_spot_auto=bool(args.spot_auto))

    # Load thresholds.json if present
    thresholds_map = {}
    try:
        if args.thresholds and os.path.exists(args.thresholds):
            with open(args.thresholds, 'r', encoding='utf-8') as f:
                thresholds_map = json.load(f)
            print(f"[cli] Loaded thresholds from {args.thresholds} for {len(thresholds_map)} tickers")
    except Exception as e:
        print(f"[cli] thresholds load error: {e}")

    def run_once():
        if is_weekend():
            print("Weekend (market closed). Skipping run.")
            return
        for ticker in args.tickers:
            t = norm_ticker(ticker)
            try:
                # Per-ticker SignalConfig (apply thresholds override if present)
                th = thresholds_map.get(t, {}) if isinstance(thresholds_map, dict) else {}
                alert_cfg = th.get('bias_alert', {}) if isinstance(th, dict) else {}
                sc_t = SignalConfig(
                    thresh5=float(th.get('thresh5', args.thresh5)),
                    thresh15=float(th.get('thresh15', args.thresh15)),
                    minV5=float(th.get('minV5', args.minV5)),
                    minV15=float(th.get('minV15', args.minV15)),
                    b5_fail_ratio_alert=float(alert_cfg.get('b5_fail_ratio_alert', th.get('b5_fail_ratio_alert', 0.20))),
                    b5_alert_min_polls=int(alert_cfg.get('b5_alert_min_polls', th.get('b5_alert_min_polls', 12))),
                )
                res = fetch_and_signal(
                    api,
                    t,
                    fc,
                    sc_t,
                    spot_override=args.spot,
                    do_iv_upsert=True,
                    retry_on_zero_flow=True,
                    max_zero_attempts=2,
                )
            except Exception as e:
                res = {"ticker": t, "error": str(e)}
            print_signal(res)
            # Optional diagnostics logging to files
            try:
                if args.log and 'error' not in res:
                    now_cli = datetime.now(NY_TZ)
                    if not _is_regular_market_time_cli(now_cli):
                        continue
                    ts = now_cli.isoformat()
                    logdir = _log_dir_for_today_cli(args.log_dir)
                    sig = res.get('signal', {})
                    zero_flow_flags = list(res.get('zero_flow_flags') or [])
                    b5 = float(sig.get('metrics', {}).get('5m', {}).get('bias', 0.0))
                    v5 = float(sig.get('metrics', {}).get('5m', {}).get('V', 0.0))
                    b15 = float(sig.get('metrics', {}).get('15m', {}).get('bias', 0.0))
                    v15 = float(sig.get('metrics', {}).get('15m', {}).get('V', 0.0))
                    # gates
                    # Use the same sc_t thresholds for gate logging
                    b5_pass = abs(b5) >= float(sc_t.thresh5)
                    v5_pass = v5 >= float(sc_t.minV5)
                    b15_pass = abs(b15) >= float(sc_t.thresh15)
                    v15_pass = v15 >= float(sc_t.minV15)
                    agree_pass = (np.sign(b5) != 0) and (np.sign(b5) == np.sign(b15))
                    reasons = []
                    if not b5_pass: reasons.append('b5<thresh5')
                    if not v5_pass: reasons.append('V5<minV5')
                    if not b15_pass: reasons.append('b15<thresh15')
                    if not v15_pass: reasons.append('V15<minV15')
                    if not agree_pass: reasons.append('sign_disagree')
                    for flag in zero_flow_flags:
                        if flag not in reasons:
                            reasons.append(flag)

                    # guards (30m/60m sign) for logging
                    def _sgn(x):
                        return 1 if x>0 else (-1 if x<0 else 0)
                    try:
                        s30 = _sgn(float(res.get('signal',{}).get('metrics',{}).get('30m',{}).get('bias',0.0)))
                        s60 = _sgn(float(res.get('signal',{}).get('metrics',{}).get('60m',{}).get('bias',0.0)))
                    except Exception:
                        s30 = s60 = 0

                    metrics_sig = sig.get('metrics', {}) if isinstance(sig.get('metrics'), dict) else {}

                    def _serialize_interval(key: str) -> dict:
                        try:
                            mdict = metrics_sig.get(key, {}) if isinstance(metrics_sig, dict) else {}
                        except Exception:
                            mdict = {}

                        def _cast(val):
                            if val is None:
                                return None
                            try:
                                num = float(val)
                                if math.isnan(num):  # type: ignore[arg-type]
                                    return None
                                return num
                            except Exception:
                                return None

                        entry = {
                            'C': _cast(mdict.get('C')) or 0.0,
                            'P': _cast(mdict.get('P')) or 0.0,
                            'V': _cast(mdict.get('V')) or 0.0,
                            'bias': _cast(mdict.get('bias')) or 0.0,
                        }

                        breakdown = mdict.get('dte_breakdown') if isinstance(mdict, dict) else None
                        if isinstance(breakdown, dict) and breakdown:
                            bd_out: dict[str, dict[str, float | None]] = {}
                            for bucket, vals in breakdown.items():
                                if not isinstance(vals, dict):
                                    continue
                                bd_entry: dict[str, float | None] = {}
                                for k, v in vals.items():
                                    bd_entry[k] = _cast(v)
                                if bd_entry:
                                    bd_out[str(bucket)] = bd_entry
                            if bd_out:
                                entry['dte_breakdown'] = bd_out
                        return entry

                    alt_metrics = res.get('alt_intervals') if isinstance(res.get('alt_intervals'), dict) else {}
                    dyn_cfg = res.get('dyn_config') if isinstance(res.get('dyn_config'), dict) else {}

                    def _serialize_alt_intervals() -> dict:
                        out_alt: dict[str, dict[str, float]] = {}
                        for name, vals in alt_metrics.items():
                            if not isinstance(vals, dict):
                                continue
                            def _cast_alt(v):
                                try:
                                    num = float(v)
                                    if math.isnan(num):
                                        return 0.0
                                    return num
                                except Exception:
                                    return 0.0
                            out_alt[str(name)] = {
                                'C': _cast_alt(vals.get('C')),
                                'P': _cast_alt(vals.get('P')),
                                'V': _cast_alt(vals.get('V')),
                                'bias': _cast_alt(vals.get('bias')),
                            }
                        return out_alt

                    def _serialize_dyn_cfg() -> dict:
                        out_cfg: dict[str, object] = {}
                        for key, val in dyn_cfg.items():
                            if key in {'vix_assumed', 'mny_band_dyn'}:
                                try:
                                    out_cfg[key] = float(val)
                                except Exception:
                                    continue
                            else:
                                out_cfg[key] = val
                        if out_cfg:
                            out_cfg.setdefault('expiry_weight', _ALT_EXPIRY_WEIGHT_DESC)
                        return out_cfg

                    obj = {
                        'schema_version': '1.1',
                        'ts_ny': ts,
                        'ticker': res.get('ticker',''),
                        'app_version': 'dt_cli',
                        'spot': float(res.get('spot', 0.0)),
                        'spot_src': res.get('spot_src','auto'),
                        'config': {'expiries': int(res.get('exp_count', 7)), 'weighting': res.get('weighting'), 'mny_band': float(res.get('mny_band', 0.02)), 'poll_secs': int(args.poll or 0)},
                        'api': {'parse_rows': int(res.get('parse_rows', 0)), 'distinct_exps': int(res.get('distinct_exps', 0))},
                        'guards': {'s30': s30, 's60': s60},
                        'iv_ctx': (res.get('iv_ctx', {}) if isinstance(res.get('iv_ctx',{}), dict) else {}),
                        'zero_flow_flags': zero_flow_flags,
                        'zero_flow_attempts': int(res.get('zero_flow_attempts', 1) or 1),
                        'intervals': {
                            '5m': _serialize_interval('5m'),
                            '15m': _serialize_interval('15m'),
                            '30m': _serialize_interval('30m'),
                            '60m': _serialize_interval('60m'),
                            'day': _serialize_interval('day'),
                        },
                        'alt_intervals': _serialize_alt_intervals(),
                        'dyn_config': _serialize_dyn_cfg(),
                        'bias_fail_stats': res.get('bias_fail_stats', {}),
                        'bias_alert': res.get('bias_alert', {}),
                        'wall_guard_applied': bool(res.get('wall_guard_applied', False)),
                        'decision': {'gates': {'b5_pass': b5_pass, 'V5_pass': v5_pass, 'b15_pass': b15_pass, 'V15_pass': v15_pass, 'agree': agree_pass},
                                     'reasons': reasons, 'signal': sig.get('signal','flat'), 'strength': float(sig.get('strength', 0.0))}
                    }
                    _append_jsonl_cli(os.path.join(logdir, f"{t}.jsonl"), obj)
                    # Optional: also store in SQLite ctx DB
                    try:
                        if _insert_ctx_cli is not None:
                            _insert_ctx_cli(
                                ts_iso=ts,
                                ticker=t,
                                spot=float(res.get('spot', 0.0)),
                                b5=b5, v5=v5, b15=b15, v15=v15,
                                signal=str(sig.get('signal','flat')),
                                strength=float(sig.get('strength', 0.0)),
                                dte0_share=(float(res.get('dte_shares',{}).get('d0',0.0)) if isinstance(res.get('dte_shares',{}), dict) else None),
                                dte1_share=(float(res.get('dte_shares',{}).get('d1',0.0)) if isinstance(res.get('dte_shares',{}), dict) else None),
                                anchor=(float(res.get('anchor',0.0)) if res.get('anchor') is not None else None),
                            )
                    except Exception:
                        pass

                    csv_path = os.path.join(logdir, 'metrics.csv')
                    csv_header = ['ts_ny','ticker','spot','b5','b15','V5','V15','signal','strength','reasons']
                    csv_row = [ts, t, f"{float(res.get('spot',0.0)):.4f}", f"{b5:.4f}", f"{b15:.4f}", f"{v5:.0f}", f"{v15:.0f}", sig.get('signal','flat'), f"{float(sig.get('strength',0.0)):.2f}", ';'.join(reasons)]
                    _append_csv_cli(csv_path, csv_header, csv_row)
            except Exception as e:
                print(f"[cli] log error for {t}: {e}")

    if args.poll and args.poll > 0:
        try:
            while True:
                # Stop explicitly after 16:00 NY
                if _market_closed_ny() and not bool(getattr(args, 'after_hours', False)):
                    print("Market closed (>= 16:00 NY). Stopping polling.")
                    break
                print(f"\n[{datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}] polling...")
                run_once()
                time.sleep(args.poll)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        if _market_closed_ny() and not bool(getattr(args, 'after_hours', False)):
            print("Market closed (>= 16:00 NY). Skipping run.")
        else:
            run_once()


def _as_dec_iv(x: float) -> float:
    try:
        v = float(x)
        return v/100.0 if v > 3.0 else max(0.01, min(3.0, v))
    except Exception:
        return 0.0


def _bs_vega(S: float, K: float, sigma: float, T: float) -> float:
    try:
        import math
        S = float(max(S, 1e-9)); K = float(max(K, 1e-12)); sigma = float(max(sigma, 1e-6)); T = float(max(T, 1e-9))
        d1 = (math.log(S/K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        n_pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
        return S * n_pdf * math.sqrt(T)
    except Exception:
        return 0.0


def compute_atm_iv_curve(df: pd.DataFrame, spot: float, max_exps: int = 6, band: float = 0.02) -> List[Dict[str, float]]:
    try:
        dfx = df.copy()
        dfx['exp_date'] = pd.to_datetime(dfx.get('expiration_date'), errors='coerce')
        dfx = dfx[pd.notna(dfx['exp_date'])]
        if dfx.empty:
            return []
        dfx = dfx.sort_values('exp_date')
        S = float(spot)
        dfx['mny'] = (dfx['strike_price'].astype(float) - S) / max(S, 1e-9)
        dfx['c_iv'] = dfx.get('call_volatility', dfx.get('call_iv', 0)).apply(_as_dec_iv)
        dfx['p_iv'] = dfx.get('put_volatility', dfx.get('put_iv', 0)).apply(_as_dec_iv)
        today = now_ny().date()
        curve: List[Dict[str, float]] = []
        seen = 0
        for exp, grp in dfx.groupby('exp_date'):
            if seen >= max_exps:
                break
            atm = grp[grp['mny'].abs() <= band]
            if atm.empty:
                continue
            c_iv = float(atm['c_iv'].median())
            p_iv = float(atm['p_iv'].median())
            atm_iv = (c_iv + p_iv) / 2.0
            dte_days = max(0.0, (exp.date() - today).days)
            curve.append({
                'expiry': exp.date().isoformat(),
                'dte': float(dte_days),
                'atm_iv': atm_iv,
            })
            seen += 1
        return curve
    except Exception:
        return []


_IV_LAST_ATM: Dict[Tuple[str, str], float] = {}
_OPT_LAST_PRICE: Dict[Tuple[str, str, float, str], float] = {}
_LAST_SPOT: Dict[str, float] = {}
_OPTR_EWM: Dict[str, Tuple[float, datetime]] = {}


def update_iv_ctx_for_ticker(ticker: str, df: pd.DataFrame, spot: float) -> Dict[str, float] | None:
    """Compute IV context for a ticker and upsert into iv_ctx table per expiry.
    - atm_iv and change per expiry (rows with |mny|<=2%)
    - aggregated optR and EWM across 2–15 DTE, near‑ATM (|mny|<=5%)
    - PC_ATM, RR25 (mid‑dated), TS (front/back)
    Inserts one row per expiry and an 'ALL' aggregate row per date.
    """
    if _upsert_iv_ctx is None or df is None or len(df) == 0:
        return None
    try:
        today = now_ny().date().isoformat()
        ts = now_ny().isoformat()
        # Normalize columns
        dfe = df.copy()
        dfe['exp_date'] = pd.to_datetime(dfe.get('expiration_date'), errors='coerce').dt.date
        dfe = dfe[pd.notna(dfe['exp_date'])]
        if dfe.empty:
            return None
        S = float(spot)
        mny = (dfe['strike_price'].astype(float) - S) / max(S, 1e-9)
        dfe['mny'] = mny
        # IV columns
        dfe['c_iv'] = dfe.get('call_volatility', dfe.get('call_iv', 0)).apply(_as_dec_iv)
        dfe['p_iv'] = dfe.get('put_volatility', dfe.get('put_iv', 0)).apply(_as_dec_iv)
        dfe['c_delta'] = dfe.get('call_delta', 0.0).astype(float)
        dfe['p_delta'] = dfe.get('put_delta', 0.0).astype(float)

        # Per-expiry ATM IV and change
        for edate, grp in dfe.groupby('exp_date'):
            try:
                atm = grp[grp['mny'].abs() <= 0.02]
                if atm.empty:
                    continue
                c_iv = float(atm['c_iv'].median())
                p_iv = float(atm['p_iv'].median())
                atm_iv = (c_iv + p_iv) / 2.0
                key = (ticker, str(edate))
                prev = _IV_LAST_ATM.get(key, None)
                atm_ch = (atm_iv - prev) if (prev is not None) else 0.0
                _IV_LAST_ATM[key] = atm_iv
                _upsert_iv_ctx(
                    date=today,
                    ticker=ticker,
                    expiry=str(edate),
                    atm_iv=atm_iv,
                    atm_iv_ch=atm_ch,
                    ts_iso=ts,
                )
            except Exception:
                continue

        # Aggregated metrics across expiries
        # optR: 2–15 DTE, |mny|<=5%
        try:
            today_d = now_ny().date()
            dte_days = (pd.to_datetime(dfe['exp_date']) - pd.Timestamp(today_d)).dt.days.astype(float)
            mask = (dte_days >= 2) & (dte_days <= 15) & (dfe['mny'].abs() <= 0.05)
            slice_df = dfe[mask]
            if not slice_df.empty:
                # last spot and dt
                ls_prev = _LAST_SPOT.get(ticker, S)
                dS = S - ls_prev
                _LAST_SPOT[ticker] = S
                nowt = now_ny()
                prev_state = _OPTR_EWM.get(ticker, (0.0, nowt))
                # residuals per leg
                r_vals = []
                w_vals = []
                for _, row in slice_df.iterrows():
                    try:
                        K = float(row['strike_price'])
                        d_call = float(row['c_delta'])
                        d_put = float(row['p_delta'])
                        sigma_c = float(row['c_iv'])
                        sigma_p = float(row['p_iv'])
                        T = max(1.0/365.0, float((row['exp_date'] - today_d).days)/365.0)
                        # vega (use call vega as proxy)
                        vega = _bs_vega(S, K, max(sigma_c, sigma_p), T)
                        if vega <= 0:
                            continue
                        # prices and deltas
                        c_price = float(row.get('call_price', 0.0))
                        p_price = float(row.get('put_price', 0.0))
                        key_c = (ticker, str(row['exp_date']), float(K), 'c')
                        key_p = (ticker, str(row['exp_date']), float(K), 'p')
                        c_prev = _OPT_LAST_PRICE.get(key_c, c_price)
                        p_prev = _OPT_LAST_PRICE.get(key_p, p_price)
                        _OPT_LAST_PRICE[key_c] = c_price
                        _OPT_LAST_PRICE[key_p] = p_price
                        # residuals (calls, puts)
                        rc = ((c_price - c_prev) - d_call * dS) / max(vega, 1e-9)
                        rp = ((p_price - p_prev) - d_put * dS) / max(vega, 1e-9)
                        # weights (moneyness gaussian × vega)
                        w = vega * math.exp(- (float(row['mny'])/0.02)**2)
                        r_vals.extend([rc, rp])
                        w_vals.extend([w, w])
                    except Exception:
                        continue
                if r_vals and sum(w_vals) > 0:
                    import numpy as _np
                    R = float((_np.array(r_vals) * _np.array(w_vals)).sum() / max(1e-9, _np.array(w_vals).sum()))
                else:
                    R = 0.0
                # EWM update
                prev_val, prev_ts = prev_state
                dt_sec = max(1.0, (nowt - prev_ts).total_seconds())
                hl_sec = 30.0 * 60.0
                alpha = 1.0 - math.exp(-math.log(2.0) * dt_sec / hl_sec)
                R_ewm = (1 - alpha) * float(prev_val) + alpha * float(R)
                _OPTR_EWM[ticker] = (R_ewm, nowt)
            else:
                R = 0.0
                R_ewm = _OPTR_EWM.get(ticker, (0.0, now_ny()))[0]
        except Exception:
            R = 0.0
            R_ewm = _OPTR_EWM.get(ticker, (0.0, now_ny()))[0]

        # PC_ATM, RR25, TS
        pc_atm = None
        try:
            atm_all = dfe[dfe['mny'].abs() <= 0.02]
            if not atm_all.empty:
                pc_atm = float((atm_all['p_iv'] - atm_all['c_iv']).median())
        except Exception:
            pc_atm = None

        rr25 = None
        try:
            # choose expiry nearest 14D
            today_d = now_ny().date()
            dfe['dte'] = (pd.to_datetime(dfe['exp_date']) - pd.Timestamp(today_d)).dt.days.abs()
            ed_near = dfe.sort_values('dte').iloc[0]['exp_date']
            exp_df = dfe[dfe['exp_date'] == ed_near]
            if not exp_df.empty:
                # find nearest deltas
                c_row = exp_df.iloc[(exp_df['c_delta'] - 0.25).abs().argsort()[:1]]
                p_row = exp_df.iloc[(exp_df['p_delta'].abs() - 0.25).abs().argsort()[:1]]
                if not c_row.empty and not p_row.empty:
                    rr25 = float(c_row['c_iv'].values[0] - p_row['p_iv'].values[0])
        except Exception:
            rr25 = None

        ts_ratio = None
        try:
            today_d = now_ny().date()
            dfe['dte'] = (pd.to_datetime(dfe['exp_date']) - pd.Timestamp(today_d)).dt.days.astype(float)
            # nearest to 7D and 30D
            front_day = dfe.iloc[(dfe['dte'] - 7.0).abs().argsort()[:1]]
            back_day = dfe.iloc[(dfe['dte'] - 30.0).abs().argsort()[:1]]
            def _atm_iv_of(group_row):
                ed = group_row['exp_date'].values[0]
                g = dfe[dfe['exp_date'] == ed]
                g_atm = g[g['mny'].abs() <= 0.02]
                if g_atm.empty:
                    return None
                return float(((g_atm['c_iv'].median()) + (g_atm['p_iv'].median())) / 2.0)
            iv_f = _atm_iv_of(front_day) if len(front_day) else None
            iv_b = _atm_iv_of(back_day) if len(back_day) else None
            if iv_f and iv_b and iv_b > 0:
                ts_ratio = float(iv_f / iv_b)
        except Exception:
            ts_ratio = None

        # Upsert aggregate row
        try:
            _upsert_iv_ctx(
                date=today,
                ticker=ticker,
                expiry='ALL',
                atm_iv=None,
                atm_iv_ch=None,
                pc_atm=pc_atm,
                rr25=rr25,
                ts_ratio=ts_ratio,
                optR=R,
                optR_ewm=R_ewm,
                ts_iso=ts,
            )
        except Exception:
            pass
        # return aggregate metrics for UI use
        return {
            'pc_atm': float(pc_atm) if pc_atm is not None else None,
            'rr25': float(rr25) if rr25 is not None else None,
            'ts_ratio': float(ts_ratio) if ts_ratio is not None else None,
            'optR': float(R),
            'optR_ewm': float(R_ewm),
        }
    except Exception:
        return None


if __name__ == "__main__":
    main()
