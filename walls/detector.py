from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")


class WallSide(str, Enum):
    PUT = "put"
    CALL = "call"


@dataclass
class WallParams:
    ema_alpha: float = 0.35
    min_minutes_after_open: int = 60
    min_notional: float = 8_000.0
    density_threshold: float = 1.4
    max_distance_frac: float = 0.008  # 0.8%
    search_distance_frac: float = 0.02  # strikes within ±2% considered in density calc
    touch_band_points: float = 6.0
    touch_window_minutes: int = 45
    max_negative_decay: float = 0.30  # allow up to 30% shrink before considered weak
    min_touch_count: int = 2
    max_stale_minutes: int = 240
    health_window_minutes: int = 15
    health_decay_threshold: float = 0.25
    health_growth_threshold: float = 0.15
    health_velocity_threshold: float = 0.20


@dataclass
class FlowState:
    call_value: float = 0.0
    put_value: float = 0.0
    call_prev: float = 0.0
    put_prev: float = 0.0
    call_decay: float = 0.0
    put_decay: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=ZoneInfo("UTC")))
    call_history: deque[Tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=180))
    put_history: deque[Tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=180))


@dataclass
class WallLevel:
    side: WallSide
    strike: float
    value: float
    density: float
    decay_rate: float
    distance_pts: float
    distance_frac: float
    touch_count: int
    active: bool
    health_change: float
    health_velocity: float
    health_state: str

    def to_dict(self) -> Dict[str, float | int | bool | str]:
        return {
            "side": self.side.value,
            "strike": float(self.strike),
            "value": float(self.value),
            "density": float(self.density),
            "decay_rate": float(self.decay_rate),
            "distance_pts": float(self.distance_pts),
            "distance_frac": float(self.distance_frac),
            "touch_count": int(self.touch_count),
            "active": bool(self.active),
            "health_change": float(self.health_change),
            "health_velocity": float(self.health_velocity),
            "health_state": self.health_state,
        }


@dataclass
class WallSnapshot:
    timestamp: datetime
    spot: float
    ready: bool
    put_wall: Optional[WallLevel]
    call_wall: Optional[WallLevel]
    aggregates: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spot": float(self.spot),
            "ready": bool(self.ready),
            "put_wall": self.put_wall.to_dict() if self.put_wall else None,
            "call_wall": self.call_wall.to_dict() if self.call_wall else None,
            "aggregates": [
                {
                    "strike": float(entry["strike"]),
                    "call": float(entry["call"]),
                    "put": float(entry["put"]),
                }
                for entry in self.aggregates
            ],
        }


def _minutes_since_open(ts: datetime) -> float:
    ts = ts.astimezone(NY_TZ)
    open_ts = ts.replace(hour=9, minute=30, second=0, microsecond=0)
    if ts <= open_ts:
        return 0.0
    return (ts - open_ts).total_seconds() / 60.0


class WallDetector:
    """Maintains rolling strike-level flow aggregates to identify option walls."""

    def __init__(self, params: Optional[WallParams] = None) -> None:
        self.params = params or WallParams()
        self._flow_state: Dict[float, FlowState] = {}
        self._spot_history: deque[tuple[datetime, float]] = deque(maxlen=720)

    def reset(self) -> None:
        self._flow_state.clear()
        self._spot_history.clear()

    def update(
        self,
        df: pd.DataFrame,
        spot: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[WallSnapshot]:
        if df is None or len(df) == 0:
            return None

        timestamp = (timestamp or datetime.now(NY_TZ)).astimezone(NY_TZ)
        spot = float(spot or 0.0)
        if spot <= 0:
            return None

        self._spot_history.append((timestamp, spot))
        touch_cutoff = timestamp - timedelta(minutes=self.params.touch_window_minutes + 5)
        while self._spot_history and self._spot_history[0][0] < touch_cutoff:
            self._spot_history.popleft()

        if "strike_price" not in df.columns:
            return None

        df = self._filter_zero_dte(df, timestamp)
        if df is None or df.empty:
            return None

        try:
            strikes = pd.to_numeric(df["strike_price"], errors="coerce")
        except Exception:
            return None

        call_col = self._series(df, "call_volm_bs")
        put_col = self._series(df, "put_volm_bs")
        flows = pd.DataFrame({"strike": strikes, "call": call_col, "put": put_col})
        flows = flows.dropna(subset=["strike"])
        if flows.empty:
            return None
        grouped = flows.groupby("strike", as_index=True).sum()
        call_flows = grouped.get("call", pd.Series(dtype=float))
        put_flows = grouped.get("put", pd.Series(dtype=float))

        alpha = float(max(0.0, min(1.0, self.params.ema_alpha)))
        strikes_all = set(call_flows.index.astype(float).tolist()) | set(put_flows.index.astype(float).tolist()) | set(self._flow_state.keys())

        for strike in strikes_all:
            state = self._flow_state.get(strike)
            if state is None:
                state = FlowState()
                self._flow_state[strike] = state

            call_current = float(call_flows.get(strike, 0.0))
            put_current = float(put_flows.get(strike, 0.0))

            prev_call = state.call_value
            prev_put = state.put_value

            if state.last_update == datetime.min.replace(tzinfo=ZoneInfo("UTC")):
                state.call_value = call_current
                state.put_value = put_current
                state.call_prev = call_current
                state.put_prev = put_current
                state.call_decay = 0.0
                state.put_decay = 0.0
            else:
                new_call = (1.0 - alpha) * prev_call + alpha * call_current
                new_put = (1.0 - alpha) * prev_put + alpha * put_current
                state.call_prev = prev_call
                state.put_prev = prev_put
                state.call_value = new_call
                state.put_value = new_put
                state.call_decay = self._decay_ratio(prev_call, new_call)
                state.put_decay = self._decay_ratio(prev_put, new_put)

            state.last_update = timestamp
            self._update_state_history(state, timestamp)

        stale_cutoff = timestamp - timedelta(minutes=self.params.max_stale_minutes)
        for strike in list(self._flow_state.keys()):
            if self._flow_state[strike].last_update < stale_cutoff:
                del self._flow_state[strike]

        ready = _minutes_since_open(timestamp) >= self.params.min_minutes_after_open

        aggregates = self._build_aggregates()
        put_wall = None
        call_wall = None

        if self._flow_state:
            put_candidates = self._build_candidates(WallSide.PUT, spot, timestamp)
            call_candidates = self._build_candidates(WallSide.CALL, spot, timestamp)

            if put_candidates:
                put_wall = sorted(
                    put_candidates,
                    key=lambda w: (int(w.active), w.density, -w.distance_frac),
                    reverse=True,
                )[0]
                if not ready:
                    put_wall.active = False

            if call_candidates:
                call_wall = sorted(
                    call_candidates,
                    key=lambda w: (int(w.active), w.density, -w.distance_frac),
                    reverse=True,
                )[0]
                if not ready:
                    call_wall.active = False

        return WallSnapshot(
            timestamp=timestamp,
            spot=spot,
            ready=ready,
            put_wall=put_wall,
            call_wall=call_wall,
            aggregates=aggregates,
        )

    @staticmethod
    def _filter_zero_dte(df: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """Return only 0DTE rows when an expiration column is present."""

        if df is None or df.empty:
            return df

        expiration_cols = [
            "expiration_date",
            "exp_date",
            "expiration",
        ]
        today = timestamp.date()

        seen_exp_column = False
        for col in expiration_cols:
            if col not in df.columns:
                continue
            seen_exp_column = True
            exp_series = pd.to_datetime(df[col], errors="coerce")
            try:
                exp_series = exp_series.dt.tz_localize(None)
            except (TypeError, AttributeError):
                try:
                    exp_series = exp_series.dt.tz_convert(None)
                except (TypeError, AttributeError, ValueError):
                    pass
            exp_dates = exp_series.dt.date
            mask = exp_dates == today
            if mask.any():
                return df[mask].copy()
        if seen_exp_column:
            return df.iloc[0:0].copy()
        return df

    @staticmethod
    def _series(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index)

    @staticmethod
    def _decay_ratio(prev: float, new: float) -> float:
        prev_abs = abs(prev)
        new_abs = abs(new)
        if prev_abs < 1e-6:
            return 0.0
        return (new_abs - prev_abs) / max(prev_abs, 1e-6)

    def _build_candidates(self, side: WallSide, spot: float, timestamp: datetime) -> List[WallLevel]:
        params = self.params
        target_attr = "put_value" if side == WallSide.PUT else "call_value"
        decay_attr = "put_decay" if side == WallSide.PUT else "call_decay"

        magnitudes: List[float] = []
        for strike, state in self._flow_state.items():
            value = getattr(state, target_attr)
            strike_f = float(strike)
            distance_frac = abs(spot - strike_f) / max(abs(spot), 1e-9)
            if distance_frac <= params.search_distance_frac and value < 0:
                magnitudes.append(abs(value))

        if magnitudes:
            baseline = float(np.median(magnitudes))
            if baseline <= 1.0:
                baseline = float(max(magnitudes)) if magnitudes else 1.0
        else:
            baseline = 1.0

        candidates: List[WallLevel] = []
        for strike, state in self._flow_state.items():
            value = getattr(state, target_attr)
            if value >= -params.min_notional:
                continue
            strike_f = float(strike)
            distance_pts = abs(spot - strike_f)
            distance_frac = distance_pts / max(abs(spot), 1e-9)
            if distance_frac > params.search_distance_frac:
                continue
            density = abs(value) / max(baseline, 1.0)
            decay_rate = getattr(state, decay_attr)
            touch_count = self._touch_count(strike_f, timestamp)
            health_change, health_velocity, health_state = self._compute_health(state, side, timestamp)
            is_active = (
                density >= params.density_threshold
                and distance_frac <= params.max_distance_frac
                and decay_rate >= -params.max_negative_decay
                and touch_count >= params.min_touch_count
                and health_state in {"stable", "building"}
            )
            candidates.append(
                WallLevel(
                    side=side,
                    strike=strike_f,
                    value=value,
                    density=density,
                    decay_rate=decay_rate,
                    distance_pts=distance_pts,
                    distance_frac=distance_frac,
                    touch_count=touch_count,
                    active=is_active,
                    health_change=health_change,
                    health_velocity=health_velocity,
                    health_state=health_state,
                )
            )

        return candidates

    def _update_state_history(self, state: FlowState, ts: datetime) -> None:
        window = timedelta(minutes=max(1, self.params.health_window_minutes))
        for history, value in (
            (state.call_history, state.call_value),
            (state.put_history, state.put_value),
        ):
            history.append((ts, float(value)))
            while history and (ts - history[0][0]) > window:
                history.popleft()

    def _compute_health(
        self,
        state: FlowState,
        side: WallSide,
        timestamp: datetime,
    ) -> Tuple[float, float, str]:
        params = self.params
        history = state.put_history if side == WallSide.PUT else state.call_history
        if len(history) < 2:
            return 0.0, 0.0, "forming"

        oldest_ts, oldest_val = history[0]
        # ensure oldest sample within window; _update_state_history already prunes but guard anyway
        window = timedelta(minutes=max(1, params.health_window_minutes))
        cutoff = timestamp - window
        if oldest_ts < cutoff and len(history) > 1:
            # find first within window
            for idx, (ts_entry, _) in enumerate(history):
                if ts_entry >= cutoff:
                    oldest_ts, oldest_val = history[idx]
                    break

        current_val = history[-1][1]
        prev_val = history[-2][1]

        base = max(abs(oldest_val), 1.0)
        change = (abs(current_val) - abs(oldest_val)) / base
        base_prev = max(abs(prev_val), 1.0)
        velocity = (abs(current_val) - abs(prev_val)) / base_prev

        if change <= -params.health_decay_threshold or velocity <= -params.health_velocity_threshold:
            state_label = "eroding"
        elif change >= params.health_growth_threshold or velocity >= params.health_velocity_threshold:
            state_label = "building"
        else:
            state_label = "stable"

        return change, velocity, state_label

    def _touch_count(self, strike: float, reference: datetime) -> int:
        band = float(self.params.touch_band_points)
        reference = reference.astimezone(NY_TZ)
        cutoff = reference - timedelta(minutes=self.params.touch_window_minutes)
        count = 0
        in_band = False
        for ts, price in reversed(self._spot_history):
            if ts < cutoff:
                break
            if abs(price - strike) <= band:
                if not in_band:
                    count += 1
                    in_band = True
            else:
                in_band = False
        return count

    def _build_aggregates(self) -> List[Dict[str, float]]:
        entries: List[Dict[str, float]] = []
        for strike in sorted(self._flow_state.keys()):
            state = self._flow_state[strike]
            entries.append(
                {
                    "strike": float(strike),
                    "call": float(state.call_value),
                    "put": float(state.put_value),
                }
            )
        return entries[-160:]
