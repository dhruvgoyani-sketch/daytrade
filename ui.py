#!/usr/bin/env python3
"""
Daytrade UI — simple Dash app to show intraday flow signals (BUY/SELL/FLAT)
using the flow engine in daytrade/dt.py. Poll interval is selectable (15/30/60s).

Run:
  python3 daytrade/ui.py

Env:
  CONVEX_EMAIL, CONVEX_PASSWORD, CONVEX_ENV (live/pro)
"""
import os
import sys
import pathlib
import json
import pandas as pd
import csv
import io
import base64
import wave
import math
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Any, Dict, Tuple
from collections import defaultdict, deque

import dash
from dash import dcc, html, Input, Output, State, dash_table
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format
from dash import callback_context, no_update
import plotly.graph_objs as go

# Ensure repo root is on sys.path so both package and local runs work
try:
    # Running from repo root: already OK
    pass
except Exception:
    pass
repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

_LOG_ROOTS: list[pathlib.Path] = []
for candidate in [
    repo_root / 'daytrade' / 'logs',
    repo_root / 'logs',
    pathlib.Path('daytrade/logs'),
    pathlib.Path('logs'),
]:
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    if resolved not in _LOG_ROOTS:
        _LOG_ROOTS.append(resolved)

from daytrade.dt import (
    FlowConfig, SignalConfig, Credentials, fetch_and_signal, make_api,
    get_chain_both, compute_ladders, compute_expiry_strip, update_iv_ctx_for_ticker,
    compute_interval_flow, update_anchor,
    now_ny, is_weekend, _spx_expiry_weight_for_date,
    get_zero_gamma_snapshot,
)
try:
    from daytrade.store import insert_intraday_ctx as _insert_ctx
except Exception:
    _insert_ctx = None


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
app.title = "Daytrade Flow Bot"
_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), 'thresholds.json')
_THRESHOLDS = None

def _load_thresholds_ui():
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    try:
        if os.path.exists(_THRESHOLDS_PATH):
            with open(_THRESHOLDS_PATH, 'r', encoding='utf-8') as f:
                _THRESHOLDS = json.load(f)
        else:
            _THRESHOLDS = {}
    except Exception as e:
        print(f"[ui] thresholds load error: {e}")
        _THRESHOLDS = {}
    return _THRESHOLDS


def _ny_now_iso() -> str:
    try:
        return datetime.now(ZoneInfo("America/New_York")).isoformat()
    except Exception:
        return datetime.now().isoformat()


def _in_final_session_window(ts: Optional[datetime] = None) -> bool:
    """Return True when the reference time lies within the final 30 minutes before the 16:00 NY close."""

    try:
        ref = ts or now_ny()
    except Exception:
        try:
            ref = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            ref = datetime.now()
    try:
        ref_ny = ref.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        ref_ny = ref
    cutoff_start = ref_ny.replace(hour=15, minute=30, second=0, microsecond=0)
    close = ref_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return cutoff_start <= ref_ny < close


def _log_dir_for_today() -> str:
    try:
        d = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    except Exception:
        d = datetime.now().strftime("%Y%m%d")
    for root in _LOG_ROOTS:
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"daytrade-{d}"
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        except Exception:
            continue
    # fallback to local logs directory
    path = pathlib.Path("logs") / f"daytrade-{d}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _append_jsonl(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[dt] JSONL write error {path}: {e}")


def _append_csv(path: str, header: list[str], row: list) -> None:
    try:
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print(f"[dt] CSV write error {path}: {e}")


_THEME_OPTIONS = {'dark', 'light'}
DEFAULT_THEME = os.environ.get("DAYTRADE_UI_THEME", "dark")
if not isinstance(DEFAULT_THEME, str):
    DEFAULT_THEME = "dark"
DEFAULT_THEME = DEFAULT_THEME.strip().lower()
if DEFAULT_THEME not in _THEME_OPTIONS:
    DEFAULT_THEME = "dark"


def _normalize_theme(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return DEFAULT_THEME
    val = value.strip().lower()
    return val if val in _THEME_OPTIONS else DEFAULT_THEME


def _plotly_template(theme: str) -> str:
    return 'plotly_dark' if _normalize_theme(theme) == 'dark' else 'plotly_white'


def _theme_font_color(theme: str) -> str:
    return '#ecf0f6' if _normalize_theme(theme) == 'dark' else '#1e2836'


def _apply_fig_theme(fig: go.Figure, theme: str, *, height: Optional[int] = None) -> go.Figure:
    normalized = _normalize_theme(theme)
    if height is not None:
        fig.update_layout(height=height)
    fig.update_layout(
        template=_plotly_template(normalized),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=_theme_font_color(normalized)),
    )
    return fig


def _empty_fig(theme: str, height: Optional[int] = None) -> go.Figure:
    fig = go.Figure()
    return _apply_fig_theme(fig, theme, height=height)


class ZeroGammaPanel:
    def __init__(self, base_ticker: str = "SPX", max_points: int = 120):
        self.base_ticker = (base_ticker or "SPX").strip().upper() or "SPX"
        self.max_points = max_points
        self._history: deque[Dict[str, Any]] = deque(maxlen=max_points)
        self._tz = ZoneInfo("America/New_York")

    @staticmethod
    def _norm_ticker(ticker: Optional[str], default: str) -> str:
        try:
            val = str(ticker or default).strip().upper()
            return val if val else default
        except Exception:
            return default

    @staticmethod
    def _coerce(value: Any) -> Optional[float]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def reset(self) -> None:
        self._history.clear()

    def tick(self, ticker: Optional[str] = None) -> None:
        target = self._norm_ticker(ticker, self.base_ticker)
        if target != self.base_ticker:
            return
        snap = get_zero_gamma_snapshot(target)
        ts_dt = datetime.fromtimestamp(snap.ts, tz=self._tz)
        spot_val = self._coerce(snap.spot)
        zg_val = self._coerce(snap.zero_gamma)
        entry = {
            "ts": ts_dt,
            "spot": spot_val,
            "zg": zg_val,
            "has_root": bool(snap.has_root),
            "n_opts": int(snap.n_opts_used),
        }
        self._history.append(entry)

    def build_figure(self, theme: str, ticker: Optional[str]) -> go.Figure:
        fig = _empty_fig(theme, height=160)
        fig.update_layout(
            title="SPX 0DTE — Zero-Gamma",
            margin=dict(l=36, r=18, t=36, b=32),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        )

        active = self._norm_ticker(ticker, self.base_ticker)
        if active != self.base_ticker:
            fig.add_annotation(
                text="Zero-gamma view available for SPX only.",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(title='Index level')
            return fig

        if not self._history:
            fig.add_annotation(
                text="Waiting for zero-gamma data…",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
            )
            fig.update_xaxes(title='Time (NY)')
            fig.update_yaxes(title='Index level')
            return fig

        times = [entry["ts"] for entry in self._history]
        spot_vals = [entry["spot"] for entry in self._history]
        if any(val is not None for val in spot_vals):
            fig.add_trace(go.Scatter(
                x=times,
                y=spot_vals,
                mode="lines",
                name="SPX spot",
                line=dict(color="#95a5a6", width=1.3),
                opacity=0.75,
                connectgaps=False,
            ))

        zg_vals = [entry["zg"] for entry in self._history]
        if any(val is not None for val in zg_vals):
            custom = [[entry["has_root"], entry["n_opts"]] for entry in self._history]
            fig.add_trace(go.Scatter(
                x=times,
                y=zg_vals,
                mode="lines",
                name="Zero-Gamma (S*)",
                line=dict(color="#f1c40f", width=2.2),
                connectgaps=True,
                customdata=custom,
                hovertemplate=(
                    "Time=%{x|%H:%M:%S}<br>"
                    "S*=%{y:.1f}<br>"
                    "Root=%{customdata[0]}<br>"
                    "Opts=%{customdata[1]}<extra></extra>"
                ),
            ))

        last = self._history[-1]
        status_txt = "root resolved" if last["has_root"] else "no root (using prior)"
        fig.add_annotation(
            text=f"Latest status: {status_txt} • opts used={last['n_opts']}",
            showarrow=False,
            x=0.0,
            y=1.08,
            xref='paper',
            yref='paper',
            align='left',
            font=dict(size=11),
        )
        fig.update_yaxes(title='Index level')
        fig.update_xaxes(title='Time (NY)')
        return fig


ZERO_GAMMA_PANEL = ZeroGammaPanel()


def _default_pos_state():
    return {
        "mode": "flat",
        "enter_long": 0,
        "enter_short": 0,
        "hold_fail": 0,
        "disagree_streak": 0,
        "v5_fail_count": 0,
        "v15_fail_count": 0,
        "b5_hold_fail": 0,
        "entry_price": None,
        "entry_ts": None,
        "pnl_realized": 0.0,
        "pnl_unrealized": 0.0,
        "pnl_total": 0.0,
        "flip_dir": None,
        "flip_count": 0,
    }


def _pos_with_defaults(pos_data):
    base = _default_pos_state()
    if isinstance(pos_data, dict):
        base.update(pos_data)
    return base


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _list_today_logged_tickers() -> list[str]:
    return _list_logged_tickers_for_date(None)


def _load_trades_from_log(ticker: str, date_str: Optional[str] = None) -> list[dict]:
    if not ticker:
        return []
    if date_str:
        base = _resolve_log_dir_for_date(date_str)
        if base is None:
            return []
    else:
        base = pathlib.Path(_log_dir_for_today())
    path = base / f"{ticker}.jsonl"
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = obj.get('ts_ny')
                decision = obj.get('decision') or {}
                signal = (decision.get('signal') or '').lower()
                if signal not in {'long', 'short', 'flat'} or not ts:
                    continue
                spot_val = obj.get('spot')
                if spot_val is None:
                    continue
                decision = obj.get('decision') or {}
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                intervals = (obj.get('signal') or {}).get('metrics', {}) or obj.get('intervals', {}) or {}
                m5_raw = intervals.get('5m', {}) if isinstance(intervals, dict) else {}
                m15_raw = intervals.get('15m', {}) if isinstance(intervals, dict) else {}
                records.append({
                    'dt': dt,
                    'signal': signal,
                    'reasons': list(decision.get('reasons') or []),
                    'spot': float(spot_val),
                    'm5': {
                        'V': _safe_float(m5_raw.get('V')),
                        'bias': _safe_float(m5_raw.get('bias')),
                    },
                    'm15': {
                        'V': _safe_float(m15_raw.get('V')),
                        'bias': _safe_float(m15_raw.get('bias')),
                    },
                })
    except Exception as exc:  # pragma: no cover - I/O guard
        print(f"[ui] trade log read error {path}: {exc}")
        return []

    records.sort(key=lambda r: r['dt'])
    trades: list[dict] = []
    state = 'flat'
    entry = None
    for rec in records:
        sig = rec['signal']
        if sig in {'long', 'short'}:
            if state == sig:
                continue
            if state in {'long', 'short'} and entry is not None:
                pnl = (rec['spot'] - entry['spot']) if state == 'long' else (entry['spot'] - rec['spot'])
                trades.append({
                    'side': state,
                    'entry_dt': entry['dt'],
                    'exit_dt': rec['dt'],
                    'entry_spot': entry['spot'],
                    'exit_spot': rec['spot'],
                    'pnl': pnl,
                    'duration_min': (rec['dt'] - entry['dt']).total_seconds() / 60.0,
                    'entry_v5': entry['m5'].get('V'),
                    'entry_b5': entry['m5'].get('bias'),
                    'entry_v15': entry['m15'].get('V'),
                    'entry_b15': entry['m15'].get('bias'),
                    'exit_v5': rec['m5'].get('V'),
                    'exit_b5': rec['m5'].get('bias'),
                    'exit_v15': rec['m15'].get('V'),
                    'exit_b15': rec['m15'].get('bias'),
                    'entry_reasons': entry.get('reasons', []),
                    'exit_reasons': rec.get('reasons', []),
                })
            state = sig
            entry = rec
        elif sig == 'flat':
            if state in {'long', 'short'} and entry is not None:
                pnl = (rec['spot'] - entry['spot']) if state == 'long' else (entry['spot'] - rec['spot'])
                trades.append({
                    'side': state,
                    'entry_dt': entry['dt'],
                    'exit_dt': rec['dt'],
                    'entry_spot': entry['spot'],
                    'exit_spot': rec['spot'],
                    'pnl': pnl,
                    'duration_min': (rec['dt'] - entry['dt']).total_seconds() / 60.0,
                    'entry_v5': entry['m5'].get('V'),
                    'entry_b5': entry['m5'].get('bias'),
                    'entry_v15': entry['m15'].get('V'),
                    'entry_b15': entry['m15'].get('bias'),
                    'exit_v5': rec['m5'].get('V'),
                    'exit_b5': rec['m5'].get('bias'),
                    'exit_v15': rec['m15'].get('V'),
                    'exit_b15': rec['m15'].get('bias'),
                    'entry_reasons': entry.get('reasons', []),
                    'exit_reasons': rec.get('reasons', []),
                })
                state = 'flat'
                entry = None
    return trades


def _summarize_trades(trades: list[dict]) -> dict:
    tol = 1e-9
    total = len(trades)
    net = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > tol]
    losses = [t for t in trades if t['pnl'] < -tol]
    flats = total - len(wins) - len(losses)
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0.0
    best = max((t['pnl'] for t in trades), default=0.0)
    worst = min((t['pnl'] for t in trades), default=0.0)
    avg_hold = sum(t['duration_min'] for t in trades) / total if total else 0.0
    win_rate = (len(wins) / total) * 100.0 if total else 0.0
    cum = 0.0
    equity = []
    for t in trades:
        cum += t['pnl']
        equity.append({'exit_dt': t['exit_dt'], 'equity': cum})
    return {
        'total': total,
        'net': net,
        'wins': len(wins),
        'losses': len(losses),
        'flats': flats,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best': best,
        'worst': worst,
        'avg_hold': avg_hold,
        'win_rate': win_rate,
        'equity': equity,
    }


def _format_trades_for_table(trades: list[dict], limit: Optional[int] = None):
    rows = trades if limit is None else trades[-limit:]
    ny = ZoneInfo("America/New_York")

    def _fmt_time(dt):
        return dt.astimezone(ny).strftime('%H:%M:%S')

    def _fmt_float(val, digits=2):
        return '' if val is None else f"{val:.{digits}f}"

    columns = [
        {"name": "Side", "id": "side"},
        {"name": "Entry", "id": "entry_time"},
        {"name": "Exit", "id": "exit_time"},
        {"name": "Entry Spot", "id": "entry_spot"},
        {"name": "Exit Spot", "id": "exit_spot"},
        {"name": "PnL", "id": "pnl"},
        {"name": "Hold (min)", "id": "hold_min"},
        {"name": "Entry V5", "id": "entry_v5"},
        {"name": "Entry V15", "id": "entry_v15"},
        {"name": "Exit V5", "id": "exit_v5"},
        {"name": "Exit V15", "id": "exit_v15"},
        {"name": "Entry b5", "id": "entry_b5"},
        {"name": "Entry b15", "id": "entry_b15"},
        {"name": "Exit b5", "id": "exit_b5"},
        {"name": "Exit b15", "id": "exit_b15"},
    ]

    data = []
    for tr in reversed(rows):
        data.append({
            'side': tr['side'].upper(),
            'entry_time': _fmt_time(tr['entry_dt']),
            'exit_time': _fmt_time(tr['exit_dt']),
            'entry_spot': f"{tr['entry_spot']:.2f}",
            'exit_spot': f"{tr['exit_spot']:.2f}",
            'pnl': f"{tr['pnl']:+.2f}",
            'hold_min': f"{tr['duration_min']:.2f}",
            'entry_v5': _fmt_float(tr.get('entry_v5')),
            'entry_v15': _fmt_float(tr.get('entry_v15')),
            'exit_v5': _fmt_float(tr.get('exit_v5')),
            'exit_v15': _fmt_float(tr.get('exit_v15')),
            'entry_b5': _fmt_float(tr.get('entry_b5'), digits=3),
            'entry_b15': _fmt_float(tr.get('entry_b15'), digits=3),
            'exit_b5': _fmt_float(tr.get('exit_b5'), digits=3),
            'exit_b15': _fmt_float(tr.get('exit_b15'), digits=3),
        })

    return columns, data


def _build_wall_fig(snapshot: Optional[dict], theme: str) -> go.Figure:
    fig = _empty_fig(theme)
    fig.update_layout(
        #title="Strike Walls",
        height=200,
        margin=dict(l=28, r=18, t=38, b=32),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
    )

    if not snapshot or not snapshot.get('aggregates'):
        fig.update_layout(
            annotations=[
                dict(
                    text="No wall data",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                )
            ]
        )
        fig.update_xaxes(title='Strike')
        fig.update_yaxes(title='Net flow')
        return _apply_fig_theme(fig, theme)

    aggregates_all = sorted(snapshot.get('aggregates') or [], key=lambda item: item.get('strike', 0.0))
    spot = float(snapshot.get('spot', 0.0) or 0.0)
    if spot > 0:
        span = max(spot * 0.02, 10.0)
        aggregates = [
            item for item in aggregates_all
            if (spot - span) <= float(item.get('strike', 0.0)) <= (spot + span)
        ]
        if not aggregates:
            aggregates = aggregates_all[-40:]
    else:
        aggregates = aggregates_all[-40:]
    strikes = [float(item.get('strike', 0.0)) for item in aggregates]
    calls = [float(item.get('call', 0.0)) for item in aggregates]
    puts = [float(item.get('put', 0.0)) for item in aggregates]

    fig.add_bar(name='Calls', x=strikes, y=calls, marker_color='#27ae60')
    fig.add_bar(name='Puts', x=strikes, y=puts, marker_color='#c0392b')

    if spot > 0:
        fig.add_vline(
            x=spot,
            line=dict(color='#2980b9', dash='dot', width=2),
            annotation_text=f"spot {spot:.1f}",
            annotation_position="top",
        )

    for key, color, position in (
        ("put_wall", '#c0392b', "top left"),
        ("call_wall", '#27ae60', "top right"),
    ):
        wall = snapshot.get(key)
        if not wall:
            continue
        strike = wall.get('strike')
        if strike is None:
            continue
        label = f"{key.replace('_', ' ')} @{float(strike):.0f}"
        fig.add_vline(
            x=float(strike),
            line=dict(color=color, dash='dash', width=1.5),
            annotation_text=label,
            annotation_position=position,
        )

    fig.update_layout(barmode='group')
    fig.update_xaxes(title='Strike')
    fig.update_yaxes(title='Volume Balance')
    return _apply_fig_theme(fig, theme)


def _make_tone(freq: float, duration: float = 0.25) -> str:
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.4 * np.sin(2 * np.pi * freq * t)
        audio = (tone * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())
        return 'data:audio/wav;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    except Exception:
        return ''


def _minutes_to_label(mins: int) -> str:
    mins = int(mins)
    base_minutes = 9 * 60 + 30
    total_minutes = base_minutes + mins
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    suffix = 'AM'
    if hour >= 12:
        suffix = 'PM'
    display_hour = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
    return f"{display_hour}:{minute:02d} {suffix}"


_TONE_MAP = {
    'long': _make_tone(880.0),
    'short': _make_tone(440.0),
    'flat': _make_tone(660.0),
}


def _resolve_log_dir_for_date(date_str: str) -> Optional[pathlib.Path]:
    candidates = [root / f"daytrade-{date_str}" for root in _LOG_ROOTS]
    for path in candidates:
        if path.exists():
            return path
    return None


def _list_logged_tickers_for_date(date_str: Optional[str]) -> list[str]:
    try:
        if date_str:
            base = _resolve_log_dir_for_date(date_str)
            if base is None:
                return []
        else:
            base = pathlib.Path(_log_dir_for_today())
        tickers = {p.stem.upper() for p in base.glob('*.jsonl')}
        return sorted(tickers)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[ui] ticker list error for {date_str}: {exc}")
        return []


def _serve_layout():
    return html.Div(
        id="page-root",
        className=f"theme-{DEFAULT_THEME}",
        style={
            "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial",
            "padding": "6px",
            "maxWidth": "100%",
            "margin": "0",
            "fontSize": "12px",
            "minHeight": "100vh",
        },
        children=[
            html.Div([
                html.H3("Daytrade — Intraday", style={"margin": "0", "fontSize": "16px"}),
                html.Div([
                    html.Label("Theme", style={"marginRight": "6px", "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="theme-select",
                        options=[
                            {"label": "Dark", "value": "dark"},
                            {"label": "Light", "value": "light"},
                        ],
                        value=DEFAULT_THEME,
                        clearable=False,
                        style={"width": "120px", "fontSize": "12px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"}),
            dcc.Store(id="theme-store", data=DEFAULT_THEME),
            dcc.Tabs(
                id='dt-tabs',
                value='trade',
                children=[
                    dcc.Tab(
                        label='Trade',
                        value='trade',
                        children=[
                            html.Div([
                                html.Label("Ticker"),
                                dcc.Input(
                                    id="dt-ticker",
                                    type="text",
                                    value="SPX",
                                    style={"width": "56px", "height": "22px", "fontSize": "12px"},
                                ),
                                html.Label("Expiries", style={"marginLeft": "10px"}),
                                dcc.Input(
                                    id="dt-exp-count",
                                    type="number",
                                    value=7,
                                    min=1,
                                    step=1,
                                    style={"width": "48px", "height": "22px", "fontSize": "12px"},
                                ),
                                html.Label("Weighting", style={"marginLeft": "10px"}),
                                dcc.Dropdown(
                                    id="dt-weighting",
                                    options=[
                                        {"label": "raw", "value": "raw"},
                                        {"label": "mny", "value": "mny"},
                                        {"label": "mny_delta", "value": "mny_delta"},
                                    ],
                                    value="mny_delta",
                                    clearable=False,
                                    style={"width": "110px", "display": "inline-block", "verticalAlign": "middle", "fontSize": "12px"},
                                ),
                                html.Label("Kernel σ (%)", style={"marginLeft": "10px"}),
                                dcc.Input(
                                    id="dt-mny-band",
                                    type="number",
                                    value=1.0,
                                    min=0.1,
                                    step=0.1,
                                    style={"width": "64px", "height": "22px", "fontSize": "12px"},
                                ),
                                html.Label("Wall guard", style={"marginLeft": "10px"}),
                                dcc.Checklist(
                                    id="wall-toggle",
                                    options=[{"label": "", "value": "ON"}],
                                    value=['ON'],
                                    inputStyle={"marginRight": "0.4rem", "width": "18px", "height": "18px"},
                                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                                    style={"display": "inline-flex", "alignItems": "center"},
                                ),
                            ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "alignItems": "center", "marginTop": "4px"}),
                            html.Div([
                                html.Label("5m thresh"),
                                dcc.Input(
                                    id="dt-thresh5",
                                    type="number",
                                    value=0.30,
                                    step=0.05,
                                    style={"width": "48px", "height": "20px", "fontSize": "12px"},
                                ),
                                html.Label("15m thresh", style={"marginLeft": "10px"}),
                                dcc.Input(
                                    id="dt-thresh15",
                                    type="number",
                                    value=0.20,
                                    step=0.05,
                                    style={"width": "48px", "height": "20px", "fontSize": "12px"},
                                ),
                                html.Label("minV5", style={"marginLeft": "10px"}),
                                dcc.Input(
                                    id="dt-minV5",
                                    type="number",
                                    value=20000,
                                    step=1000,
                                    style={"width": "64px", "height": "20px", "fontSize": "12px"},
                                ),
                                html.Label("minV15", style={"marginLeft": "10px"}),
                                dcc.Input(
                                    id="dt-minV15",
                                    type="number",
                                    value=40000,
                                    step=1000,
                                    style={"width": "64px", "height": "20px", "fontSize": "12px"},
                                ),
                                html.Label("Poll", style={"marginLeft": "10px"}),
                                dcc.Dropdown(
                                    id="dt-poll",
                                    options=[
                                        {"label": "15s", "value": 15},
                                        {"label": "30s", "value": 30},
                                        {"label": "60s", "value": 60},
                                    ],
                                    value=15,
                                    clearable=False,
                                    style={"width": "64px", "display": "inline-block", "verticalAlign": "middle", "fontSize": "12px"},
                                ),
                                dcc.Checklist(
                                    id="dt-override",
                                    options=[{"label": " After-hours", "value": "ah"}],
                                    value=[],
                                    style={"marginLeft": "6px", "display": "inline-block", "fontSize": "12px"},
                                ),
                                html.Button(
                                    "Go / Stop",
                                    id="dt-start",
                                    n_clicks=0,
                                    style={"marginLeft": "4px", "height": "24px", "fontSize": "12px", "padding": "2px 6px"},
                                ),
                                html.Button(
                                    "Force Flat",
                                    id="dt-force-flat",
                                    n_clicks=0,
                                    disabled=True,
                                    title="Immediately flatten any open position",
                                    style={"marginLeft": "6px", "height": "24px", "fontSize": "12px", "padding": "2px 6px"},
                                ),
                            ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "alignItems": "center", "marginTop": "4px"}),
                            html.Div([
                                html.Label("Strategy", title="Select which decision engine drives trades: baseline (current), experimental (new), shadow (run experimental in background)."),
                                html.Span(
                                    dcc.Dropdown(
                                        id="dt-strategy",
                                        options=[
                                            {"label": "baseline", "value": "baseline"},
                                            {"label": "experimental", "value": "experimental"},
                                            {"label": "shadow", "value": "shadow"},
                                        ],
                                        value="baseline",
                                        clearable=False,
                                        style={"width": "120px", "display": "inline-block", "verticalAlign": "middle", "fontSize": "12px"},
                                    ),
                                    title="Baseline uses current logic; experimental uses the new filters; shadow logs the alternative without switching.",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Features", style={"marginLeft": "8px"}, title="Experimental-only filters/gates. Toggle on to apply."),
                                dcc.Checklist(
                                    id="dt-features",
                                    options=[
                                        {"label": html.Span("anchor", title="Anchor regime: use since-open bias as a directional filter."), "value": "anchor"},
                                        {"label": html.Span("transition", title="Transition: allow flips only after short-anchor confirms for N polls."), "value": "transition"},
                                        {"label": html.Span("trend30/60", title="Trend30/60: require 30m/60m bias to align with the entry."), "value": "trend"},
                                        {"label": html.Span("near-spot", title="Near-spot: confirm bias using a tighter moneyness band."), "value": "near"},
                                        {"label": html.Span("ToD", title="ToD: increase thresholds during the lunch window to reduce noise."), "value": "tod"},
                                    ],
                                    value=[],
                                    style={"display": "inline-flex", "gap": "8px", "fontSize": "12px"},
                                    inputStyle={"marginRight": "4px"},
                                ),
                                html.Label("Anchor|min", style={"marginLeft": "8px"}, title="Minimum absolute day-anchor bias to treat the regime as strong."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-anchor-min",
                                        type="number",
                                        value=0.20,
                                        step=0.05,
                                        style={"width": "52px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Minimum absolute day-anchor bias to treat the regime as strong.",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Anchor HL", style={"marginLeft": "6px"}, title="Half-life (minutes) for the since-open anchor EMA."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-anchor-hl",
                                        type="number",
                                        value=45,
                                        step=5,
                                        style={"width": "52px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Half-life (minutes) for the since-open anchor EMA.",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Short HL", style={"marginLeft": "6px"}, title="Half-life (minutes) for the short-term anchor (15m bias)."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-anchor-short-hl",
                                        type="number",
                                        value=15,
                                        step=5,
                                        style={"width": "48px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Half-life (minutes) for the short-term anchor (15m bias).",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Flip N", style={"marginLeft": "6px"}, title="Number of polls required to confirm a regime transition."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-transition-polls",
                                        type="number",
                                        value=3,
                                        min=1,
                                        step=1,
                                        style={"width": "40px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Number of polls required to confirm a regime transition.",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Flip x", style={"marginLeft": "6px"}, title="Threshold multiplier applied during a transition."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-transition-mult",
                                        type="number",
                                        value=1.3,
                                        step=0.1,
                                        style={"width": "52px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Threshold multiplier applied during a transition.",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("Near x", style={"marginLeft": "6px"}, title="Multiplier on the moneyness band for near-spot confirmation (smaller = tighter)."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-near-mult",
                                        type="number",
                                        value=0.5,
                                        step=0.1,
                                        style={"width": "48px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Multiplier on the moneyness band for near-spot confirmation (smaller = tighter).",
                                    style={"display": "inline-block"},
                                ),
                                html.Label("ToD x", style={"marginLeft": "6px"}, title="Threshold multiplier during the lunch window."),
                                html.Span(
                                    dcc.Input(
                                        id="dt-tod-mult",
                                        type="number",
                                        value=1.2,
                                        step=0.1,
                                        style={"width": "48px", "height": "20px", "fontSize": "12px"},
                                    ),
                                    title="Threshold multiplier during the lunch window.",
                                    style={"display": "inline-block"},
                                ),
                            ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "alignItems": "center", "marginTop": "4px"}),
                            dcc.Store(id="dt-running", data=False),
                            dcc.Interval(id="dt-interval", interval=60000, n_intervals=0, disabled=True),
                            dcc.Store(id="df-store"),
                            dcc.Store(id="spot-store"),
                            dcc.Store(id="cfg-store"),
                            dcc.Store(id="wall-store"),
                            dcc.Store(id="dt-pos", data=_default_pos_state()),
                            dcc.Store(id="dt-pos-exp", data=_default_pos_state()),
                            dcc.Store(id="ctx-series", data={}),
                            dcc.Store(id="ctx-yesterday", data={}),
                            dcc.Store(id="iv-prev", data={}),
                            dcc.Store(id="dt-dyn", data={"armed": False, "poll_ms": None}),
                            html.Hr(),
                            html.Div(id="dt-status", style={"marginTop": "4px", "fontSize": "1.05em"}),
                            html.Div(id="dt-regime", style={"marginTop": "2px", "fontSize": "12px"}),
                            html.Audio(id="dt-tone", src="", autoPlay=True, controls=False, style={"display": "none"}),
                            dash_table.DataTable(
                                id="dt-metrics",
                                columns=[],
                                data=[],
                                page_size=5,
                                style_table={"overflowX": "auto", "marginTop": "4px"},
                                style_cell={
                                    "fontFamily": "monospace",
                                    "fontSize": "11px",
                                    "padding": "4px",
                                    "minWidth": "80px",
                                    "width": "80px",
                                    "maxWidth": "120px",
                                    "textAlign": "center",
                                },
                            ),
                            html.Details(
                                id="dt-trade-details",
                                children=[
                                    html.Summary("Recent Trades"),
                                    html.Div(id="dt-trade-summary", style={"fontSize": "12px", "marginTop": "6px"}),
                                    dash_table.DataTable(
                                        id="dt-trade-table",
                                        columns=[],
                                        data=[],
                                        page_size=10,
                                        style_table={"overflowX": "auto", "marginTop": "6px"},
                                        style_cell={"fontFamily": "monospace", "fontSize": "11px", "padding": "4px"},
                                        sort_action='native'
                                    ),
                                ],
                                open=False,
                                style={"marginTop": "6px"},
                            ),
                            html.Div([
                                dcc.Graph(
                                    id="dt-mini",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-zero-gamma",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="wall-figure",
                                    figure=_empty_fig(DEFAULT_THEME, height=220),
                                    config={"responsive": True},
                                    style={"height": "220px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-iv-delta",
                                    figure=_empty_fig(DEFAULT_THEME, height=220),
                                    config={"responsive": True},
                                    style={"height": "220px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-graph-5m",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-graph-15m",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-graph-30m",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                                dcc.Graph(
                                    id="dt-graph-60m",
                                    figure=_empty_fig(DEFAULT_THEME, height=160),
                                    config={"responsive": True},
                                    style={"height": "160px", "width": "100%", "minWidth": 0},
                                ),
                            ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "6px", "marginTop": "6px"}),
                        ],
                    ),
                ],
            ),
        ],
    )



app.layout = _serve_layout


@app.callback(
    Output("page-root", "className"),
    Output("theme-store", "data"),
    Input("theme-select", "value"),
    State("theme-store", "data"),
)
def on_theme_select(theme_value, stored_theme):
    selected = _normalize_theme(theme_value or stored_theme)
    return f"theme-{selected}", selected


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _update_pnl_state(prev: dict, pos: dict, prev_mode: str, curr_mode: str, spot_val: Optional[float]) -> None:
    prev_mode = (prev_mode or 'flat').lower()
    curr_mode = (curr_mode or 'flat').lower()
    pnl_realized = float(prev.get('pnl_realized', 0.0) or 0.0)
    entry_price = prev.get('entry_price', None)
    entry_ts = prev.get('entry_ts', None)

    if spot_val is None:
        # No fresh price; carry prior state forward
        pos['pnl_realized'] = pnl_realized
        pos['entry_price'] = entry_price
        pos['entry_ts'] = entry_ts
        pos['pnl_unrealized'] = float(prev.get('pnl_unrealized', 0.0) or 0.0)
        pos['pnl_total'] = float(prev.get('pnl_total', pnl_realized + pos['pnl_unrealized']))
        return

    spot_float = float(spot_val)

    # Handle position exits and entries
    if prev_mode in ('long', 'short') and curr_mode == 'flat':
        if entry_price is not None:
            entry_float = float(entry_price)
            delta = spot_float - entry_float if prev_mode == 'long' else entry_float - spot_float
            pnl_realized += delta
        entry_price = None
        entry_ts = None
    elif prev_mode == 'flat' and curr_mode in ('long', 'short'):
        entry_price = spot_float
        entry_ts = _ny_now_iso()
    elif prev_mode in ('long', 'short') and curr_mode in ('long', 'short') and curr_mode != prev_mode:
        # Direct flip: book prior leg then seed new entry
        if entry_price is not None:
            entry_float = float(entry_price)
            delta = spot_float - entry_float if prev_mode == 'long' else entry_float - spot_float
            pnl_realized += delta
        entry_price = spot_float
        entry_ts = _ny_now_iso()

    pos['pnl_realized'] = float(pnl_realized)
    pos['entry_price'] = None if entry_price is None else float(entry_price)
    pos['entry_ts'] = entry_ts

    if curr_mode == 'long' and entry_price is not None:
        pnl_unreal = spot_float - float(entry_price)
    elif curr_mode == 'short' and entry_price is not None:
        pnl_unreal = float(entry_price) - spot_float
    else:
        pnl_unreal = 0.0

    pos['pnl_unrealized'] = float(pnl_unreal)
    pos['pnl_total'] = float(pos['pnl_realized'] + pnl_unreal)

@app.callback(
    Output("dt-running", "data"),
    Input("dt-start", "n_clicks"),
    State("dt-running", "data"),
    prevent_initial_call=True,
)
def on_toggle_dt(n, running):
    try:
        return not bool(running)
    except Exception:
        return True


@app.callback(
    Output("dt-force-flat", "disabled"),
    Input("dt-pos", "data"),
)
def on_force_flat_availability(pos_data):
    mode = (pos_data or {}).get('mode', 'flat') if isinstance(pos_data, dict) else 'flat'
    try:
        return str(mode).lower() not in {'long', 'short'}
    except Exception:
        return True


@app.callback(
    Output("dt-features", "value"),
    Input("dt-strategy", "value"),
    State("dt-features", "value"),
    prevent_initial_call=True,
)
def on_strategy_features(strategy, feature_vals):
    if str(strategy or '').lower() == 'shadow':
        return ['anchor', 'transition', 'trend', 'near', 'tod']
    return no_update


# Populate thresholds fields from thresholds.json when starting or when ticker changes
@app.callback(
    Output("dt-thresh5", "value"),
    Output("dt-thresh15", "value"),
    Output("dt-minV5", "value"),
    Output("dt-minV15", "value"),
    Output("dt-mny-band", "value"),
    Output("ctx-series", "data", allow_duplicate=True),
    Output("ctx-yesterday", "data", allow_duplicate=True),
    Input("dt-start", "n_clicks"),
    Input("dt-ticker", "value"),
    prevent_initial_call=True,
)
def populate_threshold_fields(n_clicks, ticker):
    try:
        TH = _load_thresholds_ui()
        t = (str(ticker or 'SPX')).strip().upper()
        th = TH.get(t, {}) if isinstance(TH, dict) else {}
        # Fallback defaults if not present
        band_default_pct = 1.0 if t == 'SPX' else 3.0
        raw_band = th.get('mny_band') if isinstance(th, dict) else None
        try:
            if raw_band is None:
                band_value = band_default_pct
            else:
                band_float = float(raw_band)
                band_value = band_float * 100.0 if band_float < 0.5 else band_float
        except Exception:
            band_value = band_default_pct
        # Reset ctx-series; load yesterday reference for SPX
        yday = {}
        if t == 'SPX':
            try:
                import json, os
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo as _Zone
                d = datetime.now(_Zone("America/New_York")).date() - timedelta(days=1)
                logdir = os.path.join("logs", f"daytrade-{d.strftime('%Y%m%d')}")
                path = os.path.join(logdir, "SPX.jsonl")
                xs, ys = [], []
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                obj = json.loads(line.strip())
                            except Exception:
                                continue
                            ts = obj.get('ts_ny')
                            spx_exp = obj.get('spx_exp', {}) if isinstance(obj, dict) else {}
                            anchor = spx_exp.get('anchor', None)
                            if ts is None or anchor is None:
                                continue
                            try:
                                tdt = datetime.fromisoformat(ts)
                            except Exception:
                                continue
                            # map to minutes since 09:30
                            mo = tdt.astimezone(_Zone("America/New_York")).replace(second=0, microsecond=0)
                            start = mo.replace(hour=9, minute=30)
                            if mo < start:
                                continue
                            mins = int((mo - start).total_seconds() // 60)
                            if 0 <= mins <= 390:
                                xs.append(mins)
                                ys.append(float(anchor))
                if xs:
                    yday = {"x": xs, "anchor": ys}
            except Exception:
                yday = {}
        return (
            float(th.get('thresh5', 0.30)),
            float(th.get('thresh15', 0.20)),
            float(th.get('minV5', 20000)),
            float(th.get('minV15', 40000)),
            band_value,
            {},
            yday,
        )
    except Exception:
        return 0.30, 0.20, 20000, 40000, 3.0, {}, {}


# Reset decision state when start button pressed
@app.callback(
    Output("dt-pos", "data", allow_duplicate=True),
    Input("dt-start", "n_clicks"),
    prevent_initial_call=True,
)
def reset_dt_pos(n):
    return _default_pos_state()


@app.callback(
    Output("dt-trade-summary", "children"),
    Output("dt-trade-table", "columns"),
    Output("dt-trade-table", "data"),
    Input("dt-interval", "n_intervals"),
    Input("dt-start", "n_clicks"),
    State("dt-ticker", "value"),
)
def update_trade_details(n_intervals, n_clicks, ticker):
    ticker_val = (str(ticker or "").strip().upper()) or "SPX"
    trades = _load_trades_from_log(ticker_val)
    if not trades:
        return html.Div(f"No trades logged for {ticker_val} today."), [], []
    stats = _summarize_trades(trades)
    summary_lines = [
        f"Trades: {stats['total']} — Wins: {stats['wins']} ({stats['win_rate']:.1f}%) • Losses: {stats['losses']} • Flats: {stats['flats']}",
        f"Net PnL: {stats['net']:+.2f} — Avg win: {stats['avg_win']:+.2f} • Avg loss: {stats['avg_loss']:+.2f}",
        f"Best: {stats['best']:+.2f} • Worst: {stats['worst']:+.2f} • Avg hold: {stats['avg_hold']:.1f} min",
    ]
    last_trade = trades[-1]
    ny = ZoneInfo("America/New_York")
    summary_lines.append(
        f"Last trade: {last_trade['side'].upper()} {last_trade['entry_dt'].astimezone(ny).strftime('%H:%M:%S')} → "
        f"{last_trade['exit_dt'].astimezone(ny).strftime('%H:%M:%S')}  PnL={last_trade['pnl']:+.2f}"
    )
    summary = html.Ul([html.Li(line) for line in summary_lines])
    columns, data = _format_trades_for_table(trades, limit=30)
    return summary, columns, data


@app.callback(
    Output("dt-interval", "interval"),
    Output("dt-interval", "disabled"),
    Input("dt-poll", "value"),
    Input("dt-running", "data"),
    Input("dt-override", "value"),
    State("dt-dyn", "data"),
)
def on_dt_interval(poll_val, running, override_val, dyn):
    from datetime import datetime, time as _time
    dyn = dyn or {}
    dyn_ms = dyn.get('poll_ms') if isinstance(dyn, dict) else None
    base_ms = int(max(1, int(poll_val or 60))) * 1000
    ms = int(dyn_ms) if dyn_ms else base_ms
    # Stop polling after 4:00 PM America/New_York
    try:
        now_ny_time = datetime.now(ZoneInfo("America/New_York")).time()
        market_closed = now_ny_time >= _time(16, 0)
    except Exception:
        market_closed = False
    override = bool(override_val) and ('ah' in (override_val or []))
    disabled = (not bool(running)) or (market_closed and not override)
    return ms, disabled


@app.callback(
    Output("dt-status", "children"),
    Output("dt-regime", "children"),
    Output("dt-tone", "src"),
    Output("dt-metrics", "columns"),
    Output("dt-metrics", "data"),
    Output("dt-metrics", "style_data_conditional"),
    Output("df-store", "data"),
    Output("spot-store", "data"),
    Output("cfg-store", "data"),
    Output("wall-store", "data"),
    Output("wall-figure", "figure"),
    Output("wall-figure", "style"),
    Output("dt-pos", "data", allow_duplicate=True),
    Output("dt-pos-exp", "data", allow_duplicate=True),
    Output("ctx-series", "data", allow_duplicate=True),
    Output("dt-mini", "figure", allow_duplicate=True),
    Output("dt-mini", "style", allow_duplicate=True),
    Output("dt-zero-gamma", "figure", allow_duplicate=True),
    Output("dt-dyn", "data", allow_duplicate=True),
    Input("dt-interval", "n_intervals"),
    Input("dt-start", "n_clicks"),
    Input("dt-force-flat", "n_clicks"),
    State("dt-ticker", "value"),
    State("dt-exp-count", "value"),
    State("dt-weighting", "value"),
    State("dt-mny-band", "value"),
    State("dt-thresh5", "value"),
    State("dt-thresh15", "value"),
    State("dt-minV5", "value"),
    State("dt-minV15", "value"),
    State("dt-strategy", "value"),
    State("dt-features", "value"),
    State("dt-anchor-min", "value"),
    State("dt-anchor-hl", "value"),
    State("dt-anchor-short-hl", "value"),
    State("dt-transition-polls", "value"),
    State("dt-transition-mult", "value"),
    State("dt-near-mult", "value"),
    State("dt-tod-mult", "value"),
    State("dt-pos", "data"),
    State("dt-pos-exp", "data"),
    State("ctx-series", "data"),
    State("ctx-yesterday", "data"),
    State("wall-toggle", "value"),
    State("dt-override", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def on_dt_tick(n_intervals, n_clicks, force_clicks, ticker, exp_count, weighting, mny_band, thresh5, thresh15, minV5, minV15, strategy_mode, feature_flags, anchor_min, anchor_hl, anchor_short_hl, transition_polls, transition_mult, near_mult, tod_mult, pos_store, pos_exp_store, ctx_series, ctx_yday, wall_toggle_val, override_val, theme_value):
    theme = _normalize_theme(theme_value)
    wall_style_visible = {"height": "220px", "width": "100%", "minWidth": 0}
    wall_style_hidden = {"display": "none"}
    t_upper = (str(ticker or 'SPX')).strip().upper()
    def _dyn_payload(pos_data):
        try:
            mode = (pos_data or {}).get('mode', 'flat')
            return {'armed': mode in ('long', 'short'), 'poll_ms': None}
        except Exception:
            return {'armed': False, 'poll_ms': None}
    triggered_ids = {item['prop_id'].split('.')[0] for item in (callback_context.triggered or [])}
    manual_force_flat = 'dt-force-flat' in triggered_ids
    if is_weekend():
        pos_next = _pos_with_defaults(pos_store)
        pos_exp_next = _pos_with_defaults(pos_exp_store)
        pnl_realized = float(pos_next.get('pnl_realized', 0.0))
        pnl_unreal = float(pos_next.get('pnl_unrealized', 0.0))
        pnl_total = float(pos_next.get('pnl_total', pnl_realized + pnl_unreal))
        status_weekend = html.Span([
            "Weekend • market closed • ",
            f"PnL={pnl_total:+.2f} (R={pnl_realized:+.2f}, U={pnl_unreal:+.2f})",
        ])
        ctx_next = ctx_series if isinstance(ctx_series, dict) else {}
        return (
            status_weekend,
            "",
            '',
            [],
            [],
            [],
            None,
            None,
            {'theme': theme},
            None,
            _empty_fig(theme, height=200),
            wall_style_hidden,
            pos_next,
            pos_exp_next,
            ctx_next,
            _empty_fig(theme, height=160),
            {'display': 'none'},
            _empty_fig(theme, height=200),
            _dyn_payload(pos_next),
        )
    pos_prev = _pos_with_defaults(pos_store)
    pos_exp_prev = _pos_with_defaults(pos_exp_store)
    force_close = False
    market_closed = False
    override = False
    now_dt_ny: Optional[datetime] = None
    # If market closed (>= 16:00 NY), flatten any open position then stop polling
    from datetime import datetime, time as _time
    try:
        now_dt_ny = datetime.now(ZoneInfo("America/New_York"))
        now_ny_time = now_dt_ny.time()
        override = bool(override_val) and ('ah' in (override_val or []))
        market_closed = now_ny_time >= _time(16, 0)
        if market_closed and (not override):
            if str(pos_prev.get('mode', 'flat')).lower() in {'long', 'short'}:
                force_close = True
            else:
                pos_next = pos_prev
                pnl_realized = float(pos_next.get('pnl_realized', 0.0))
                pnl_unreal = float(pos_next.get('pnl_unrealized', 0.0))
                pnl_total = float(pos_next.get('pnl_total', pnl_realized + pnl_unreal))
                status_closed = (
                    "Market closed • polling stopped at 16:00 NY "
                    f"• PnL={pnl_total:+.2f} (R={pnl_realized:+.2f}, U={pnl_unreal:+.2f})"
                )
                ctx_next = ctx_series if isinstance(ctx_series, dict) else {}
                return (
                    status_closed,
                    "",
                    '',
                    [],
                    [],
                    [],
                    None,
                    None,
                    {'theme': theme},
                    None,
                    _empty_fig(theme, height=200),
                    wall_style_hidden,
                    pos_next,
                    pos_exp_prev,
                    ctx_next,
                    _empty_fig(theme, height=160),
                    {'display': 'none'},
                    _empty_fig(theme, height=200),
                    _dyn_payload(pos_next),
                )
    except Exception:
        pass
    # Build API from env
    try:
        # use env creds for convex
        email = os.environ.get("CONVEX_EMAIL")
        password = os.environ.get("CONVEX_PASSWORD")
        env = os.environ.get("CONVEX_ENV", "live")
        if not email or not password:
            raise RuntimeError("Missing CONVEX_EMAIL/CONVEX_PASSWORD in environment")
        api = make_api(Credentials(email, password, env))
    except Exception as e:
        # keep pos state unchanged
        pos_next = _pos_with_defaults(pos_store)
        ctx_next = ctx_series if isinstance(ctx_series, dict) else {}
        return (
            f"API init error: {e}",
            "",
            '',
            [],
            [],
            [],
            None,
            None,
            {'theme': theme},
            None,
            _empty_fig(theme, height=200),
            wall_style_hidden,
            pos_next,
            pos_exp_prev,
            ctx_next,
            _empty_fig(theme, height=160),
            {'display': 'none'},
            _empty_fig(theme, height=200),
            _dyn_payload(pos_next),
        )

    # Build configs
    # Cap to closest 7 expiries to reduce noise and cost
    try:
        exp_n = int(exp_count or 7)
    except Exception:
        exp_n = 7
    exp_n = max(1, min(7, exp_n))
    band_default = 0.01 if t_upper == 'SPX' else 0.03
    try:
        band_input = float(mny_band) if mny_band not in (None, '') else None
        band_val = (band_input / 100.0) if band_input is not None else band_default
    except Exception:
        band_val = band_default
    if not (band_val and band_val > 0):
        band_val = band_default
    fc = FlowConfig(
        weighting=weighting or 'mny_delta',
        mny_band=band_val,
        exp_count=exp_n,
        use_spot_auto=True,
    )
    # Apply thresholds.json overrides if present
    TH = _load_thresholds_ui()
    t = t_upper
    th_t = TH.get(t_upper, {}) if isinstance(TH, dict) else {}
    alert_cfg = th_t.get('bias_alert', {}) if isinstance(th_t, dict) else {}
    sc = SignalConfig(
        thresh5=float(th_t.get('thresh5', (thresh5 or 0.30))),
        thresh15=float(th_t.get('thresh15', (thresh15 or 0.20))),
        minV5=float(th_t.get('minV5', (minV5 or 20000))),
        minV15=float(th_t.get('minV15', (minV15 or 40000))),
        b5_fail_ratio_alert=float(alert_cfg.get('b5_fail_ratio_alert', th_t.get('b5_fail_ratio_alert', 0.20))),
        b5_alert_min_polls=int(alert_cfg.get('b5_alert_min_polls', th_t.get('b5_alert_min_polls', 12))),
    )
    view_pct_default = 0.02 if t == 'SPX' else 0.10
    raw_view = None
    if isinstance(th_t, dict):
        raw_view = th_t.get('view_band_pct')
        if raw_view is None:
            raw_view = th_t.get('view_pct') or th_t.get('view_band_percent')
    try:
        if raw_view is None:
            view_pct = view_pct_default
        else:
            view_candidate = float(raw_view)
            view_pct = view_candidate / 100.0 if view_candidate > 1.0 else view_candidate
            if view_pct <= 0:
                view_pct = view_pct_default
    except Exception:
        view_pct = view_pct_default
    thresh5_val = float(sc.thresh5)
    thresh15_val = float(sc.thresh15)
    minV5_val = float(sc.minV5)
    minV15_val = float(sc.minV15)
    try:
        persist_n = int(th_t.get('persistence', 2))
    except Exception:
        persist_n = 2
    hold_thresh15_val = float(th_t.get('hold_thresh15', max(0.0, thresh15_val * 0.75)))
    hold_minV15_val = float(th_t.get('hold_minV15', max(0.0, minV15_val * 0.75)))
    hold_persist_n = int(th_t.get('hold_persistence', persist_n))

    zero_flow_flags: List[str] = []
    zero_flow_attempts = 1
    ivm = None
    wall_guard_enabled = bool(wall_toggle_val and 'ON' in wall_toggle_val) and (t == 'SPX')

    try:
        print(
            f"[dt] fetch ticker={t} exp_count={exp_count} weighting={weighting} band={band_val}"
            f" n_int={n_intervals} clicks={n_clicks}"
        )
        res = fetch_and_signal(
            api,
            t,
            fc,
            sc,
            spot_override=None,
            retry_on_zero_flow=True,
            max_zero_attempts=2,
            wall_guard_enabled=wall_guard_enabled,
        )
        zero_flow_flags = list(res.get('zero_flow_flags') or [])
        zero_flow_attempts = int(res.get('zero_flow_attempts') or 1)
        gb = get_chain_both(api, t, fc, spot_override=None)
        print(f"[dt] get_chain_both type={type(gb)} len={len(gb) if isinstance(gb, tuple) else 'n/a'}")
        if isinstance(gb, tuple) and len(gb) == 4:
            df_all, df_filt, spot_f, spot_src = gb
            print("[dt] NOTE: get_chain_both returned legacy 4-tuple; using df_all, spot_f, spot_src")
        elif isinstance(gb, tuple) and len(gb) == 3:
            df_all, spot_f, spot_src = gb
        else:
            raise RuntimeError(f"Unexpected get_chain_both return: {type(gb)}")
        print(f"[dt] df_all rows={0 if df_all is None else len(df_all)} spot={spot_f} src={spot_src} zero_flow={zero_flow_flags}")
        try:
            if isinstance(n_intervals, int) and (n_intervals % 3 == 0):
                ivm = update_iv_ctx_for_ticker(t, df_all, float(spot_f))
        except Exception:
            ivm = None
    except Exception as e:
        pos_next = _pos_with_defaults(pos_store)
        ctx_next = ctx_series if isinstance(ctx_series, dict) else {}
        return (
            f"Fetch error: {e}",
            "",
            '',
            [],
            [],
            [],
            None,
            None,
            {'theme': theme},
            None,
            _empty_fig(theme, height=200),
            wall_style_hidden,
            pos_next,
            pos_exp_prev,
            ctx_next,
            _empty_fig(theme, height=160),
            {'display': 'none'},
            _empty_fig(theme, height=200),
            _dyn_payload(pos_next),
        )

    if 'error' in res:
        pos_next = _pos_with_defaults(pos_store)
        ctx_next = ctx_series if isinstance(ctx_series, dict) else {}
        return (
            f"{res['ticker']}: {res['error']}",
            "",
            '',
            [],
            [],
            [],
            None,
            None,
            {'theme': theme},
            None,
            _empty_fig(theme, height=200),
            wall_style_hidden,
            pos_next,
            pos_exp_prev,
            ctx_next,
            _empty_fig(theme, height=160),
            {'display': 'none'},
            _empty_fig(theme, height=200),
            _dyn_payload(pos_next),
        )

    sig = res.get('signal', {})
    calc_signal = str(sig.get('signal', 'flat')).lower()
    strength = float(sig.get('strength', 0.0))
    agree = sig.get('agree', False)
    spot = float(res.get('spot', 0.0))
    bias_stats = res.get('bias_fail_stats', {}) if isinstance(res, dict) else {}
    total_polls = int(bias_stats.get('total_polls', 0))
    b5_fail_count = int(bias_stats.get('b5_fail_count', 0))
    b15_fail_count = int(bias_stats.get('b15_fail_count', 0))
    b5_fail_ratio = float(bias_stats.get('b5_fail_ratio', 0.0))
    b15_fail_ratio = float(bias_stats.get('b15_fail_ratio', 0.0))
    b5_fail_ratio_30m = float(bias_stats.get('b5_fail_ratio_30m', 0.0))
    b15_fail_ratio_30m = float(bias_stats.get('b15_fail_ratio_30m', 0.0))
    sign_disagree_count = int(bias_stats.get('sign_disagree_count', 0))
    bias_alert_info = res.get('bias_alert', {}) if isinstance(res, dict) else {}
    wall_guard_applied = bool(res.get('wall_guard_applied', False)) if t_upper == 'SPX' else False
    metrics_style = []
    if isinstance(bias_alert_info, dict) and bias_alert_info.get('active'):
        metrics_style.append({
            'if': {'filter_query': '{window} = "5m"', 'column_id': 'bias_fail_ratio'},
            'backgroundColor': '#e74c3c',
            'color': '#ffffff',
        })
        metrics_style.append({
            'if': {'filter_query': '{window} = "5m"', 'column_id': 'bias_fail_count'},
            'backgroundColor': '#e74c3c',
            'color': '#ffffff',
        })
        metrics_style.append({
            'if': {'filter_query': '{window} = "5m"', 'column_id': 'bias_fail_ratio_30m'},
            'backgroundColor': '#e74c3c',
            'color': '#ffffff',
        })
    wall_snapshot = res.get('wall_snapshot') if isinstance(res, dict) else None
    wall_guard_info = res.get('wall_guard') if (isinstance(res, dict) and t_upper == 'SPX') else None
    t_upper = (str(ticker or 'SPX')).strip().upper()
    # Entry/hold gating with persistence (UI-level)
    try:
        session_cutoff = _in_final_session_window(now_dt_ny)
    except Exception:
        session_cutoff = False
    if override:
        session_cutoff = False

    features = set(feature_flags or [])
    strategy_mode = str(strategy_mode or "baseline").lower()
    if strategy_mode not in {"baseline", "experimental", "shadow"}:
        strategy_mode = "baseline"
    use_anchor = "anchor" in features
    use_transition = "transition" in features
    use_trend = "trend" in features
    use_near = "near" in features
    use_tod = "tod" in features

    anchor_min_val = float(anchor_min or 0.20)
    anchor_hl_val = float(anchor_hl or 45.0)
    anchor_short_hl_val = float(anchor_short_hl or 15.0)
    transition_polls_val = max(1, int(transition_polls or 3))
    transition_mult_val = float(transition_mult or 1.3)
    near_mult_val = float(near_mult or 0.5)
    tod_mult_val = float(tod_mult or 1.2)

    # Metrics for gates
    m5 = sig.get('metrics', {}).get('5m', {})
    m15 = sig.get('metrics', {}).get('15m', {})
    m30 = sig.get('metrics', {}).get('30m', {})
    m60 = sig.get('metrics', {}).get('60m', {})
    mday = sig.get('metrics', {}).get('day', {})
    b5 = float(m5.get('bias', 0.0)); v5 = float(m5.get('V', 0.0))
    b15 = float(m15.get('bias', 0.0)); v15 = float(m15.get('V', 0.0))
    b30 = float(m30.get('bias', 0.0)) if isinstance(m30, dict) else 0.0
    b60 = float(m60.get('bias', 0.0)) if isinstance(m60, dict) else 0.0
    bday = float(mday.get('bias', 0.0)) if isinstance(mday, dict) else 0.0
    vday = float(mday.get('V', 0.0)) if isinstance(mday, dict) else 0.0

    exp_w = None
    if (t_upper == 'SPX') and (df_all is not None) and ('expiration_date' in df_all.columns):
        try:
            exp_w = _spx_expiry_weight_for_date(df_all['expiration_date'])
        except Exception:
            exp_w = None

    if (not isinstance(mday, dict)) and (df_all is not None):
        try:
            _Cday, _Pday, _Vday, _bday = compute_interval_flow(
                df_all,
                "volm_bs",
                spot,
                weighting or 'mny_delta',
                float(band_val),
                exp_w,
            )
            bday = float(_bday)
            vday = float(_Vday)
        except Exception:
            pass

    def _calc_signal(b5_val, b15_val, v5_val, v15_val, th5, th15, v5min, v15min):
        ok5 = (v5_val >= v5min) and (abs(b5_val) >= th5)
        ok15 = (v15_val >= v15min) and (abs(b15_val) >= th15) and (_sign(b15_val) == _sign(b5_val))
        sig_out = "flat"
        if ok5 and ok15:
            sig_out = "long" if b5_val > 0 else "short"
        strength_out = float(max(0.0, min(1.0, 0.5 * (abs(b5_val) / max(th5, 1e-6)) + 0.5 * (abs(b15_val) / max(th15, 1e-6)))))
        return sig_out, strength_out

    # Select baseline/experimental state stores
    if strategy_mode == 'experimental':
        pos_baseline_prev = pos_exp_prev
        pos_experimental_prev = pos_prev
    else:
        pos_baseline_prev = pos_prev
        pos_experimental_prev = pos_exp_prev

    calc_signal_base = calc_signal
    strength_base = strength

    # Time-of-day multiplier (default lunch window)
    tod_mult_applied = 1.0
    if use_tod and now_dt_ny is not None:
        try:
            t_now = now_dt_ny.time()
            if _time(11, 0) <= t_now < _time(12, 30):
                tod_mult_applied = max(1.0, tod_mult_val)
        except Exception:
            tod_mult_applied = 1.0

    th5_eff = thresh5_val * tod_mult_applied
    th15_eff = thresh15_val * tod_mult_applied
    v5_eff = minV5_val * tod_mult_applied
    v15_eff = minV15_val * tod_mult_applied

    calc_signal_exp, strength_exp = _calc_signal(b5, b15, v5, v15, th5_eff, th15_eff, v5_eff, v15_eff)

    reasons_base: list[str] = []
    reasons_exp: list[str] = []
    if isinstance(wall_guard_info, dict) and wall_guard_info.get('triggered'):
        if wall_guard_applied:
            reasons_base.append('wall_guard')
            reasons_exp.append('wall_guard')
        else:
            reasons_base.append('wall_guard_suppressed')
            reasons_exp.append('wall_guard_suppressed')

    # Near-spot bias gate
    if use_near and (calc_signal_exp in {'long', 'short'}) and (df_all is not None):
        try:
            near_band = max(float(band_val) * max(near_mult_val, 0.05), 1e-6)
            _, _, _, b5_near = compute_interval_flow(
                df_all, "volmbs_5m", spot, weighting or 'mny_delta', near_band, exp_w
            )
            _, _, _, b15_near = compute_interval_flow(
                df_all, "volmbs_15m", spot, weighting or 'mny_delta', near_band, exp_w
            )
            cand_dir = 1 if calc_signal_exp == 'long' else (-1 if calc_signal_exp == 'short' else 0)
            if cand_dir != 0:
                if (_sign(b5_near) != cand_dir) or (_sign(b15_near) != cand_dir) or (abs(b5_near) < th5_eff) or (abs(b15_near) < th15_eff):
                    calc_signal_exp = 'flat'
                    reasons_exp.append('near_spot_block')
        except Exception:
            pass

    # Trend filter (30m/60m)
    trend_signs = [s for s in (_sign(b30), _sign(b60)) if s != 0]
    if use_trend and (calc_signal_exp in {'long', 'short'}):
        cand_dir = 1 if calc_signal_exp == 'long' else -1
        if trend_signs and all(s != cand_dir for s in trend_signs):
            calc_signal_exp = 'flat'
            reasons_exp.append('trend_conflict')

    # Regime anchor + transition
    anchor_val = None
    short_anchor = None
    anchor_dir = 0
    anchor_strong = False
    regime_state = "neutral"
    transition_active = False
    flip_count = int(pos_experimental_prev.get('flip_count', 0) or 0)
    if use_anchor and (vday > 0.0):
        anchor_val = update_anchor(f"{t_upper}-day", bday, half_life_min=anchor_hl_val)
        anchor_dir = _sign(anchor_val)
        anchor_strong = abs(anchor_val) >= anchor_min_val
        if anchor_strong:
            regime_state = "bull" if anchor_dir > 0 else "bear"
        if use_transition:
            short_anchor = update_anchor(f"{t_upper}-short", b15, half_life_min=anchor_short_hl_val)

    pos_exp_work = dict(pos_experimental_prev)
    if use_anchor and anchor_strong and (calc_signal_exp in {'long', 'short'}):
        cand_dir = 1 if calc_signal_exp == 'long' else -1
        if anchor_dir != 0 and cand_dir != anchor_dir:
            transition_ok = False
            if use_transition and short_anchor is not None and _sign(short_anchor) == cand_dir:
                if (not use_trend) or (not trend_signs) or all(s == cand_dir for s in trend_signs):
                    transition_ok = True
            if transition_ok:
                if pos_exp_work.get('flip_dir') == cand_dir:
                    flip_count = int(pos_exp_work.get('flip_count', 0) or 0) + 1
                else:
                    flip_count = 1
                pos_exp_work['flip_dir'] = cand_dir
                pos_exp_work['flip_count'] = flip_count
            else:
                pos_exp_work['flip_dir'] = None
                pos_exp_work['flip_count'] = 0
            flip_count = int(pos_exp_work.get('flip_count', 0) or 0)
            transition_active = flip_count >= transition_polls_val
            if transition_active:
                regime_state = "transition"
                if transition_mult_val > 1.0:
                    calc_signal_exp, strength_exp = _calc_signal(
                        b5, b15, v5, v15,
                        th5_eff * transition_mult_val,
                        th15_eff * transition_mult_val,
                        v5_eff * transition_mult_val,
                        v15_eff * transition_mult_val,
                    )
                    if calc_signal_exp == 'flat':
                        reasons_exp.append('transition_block')
                    else:
                        reasons_exp.append('transition_active')
                else:
                    reasons_exp.append('transition_active')
            else:
                calc_signal_exp = 'flat'
                reasons_exp.append('anchor_conflict')
        else:
            pos_exp_work['flip_dir'] = None
            pos_exp_work['flip_count'] = 0
    else:
        pos_exp_work['flip_dir'] = None
        pos_exp_work['flip_count'] = 0
    flip_count = int(pos_exp_work.get('flip_count', 0) or 0)

    if use_anchor and anchor_strong and (calc_signal_exp in {'long', 'short'}) and (anchor_dir == (1 if calc_signal_exp == 'long' else -1)):
        reasons_exp.append('anchor_align')

    def _apply_position_logic(calc_signal_in, pos_prev_in, agree_in, b5_in, b15_in, v5_in, v15_in,
                              th5_in, th15_in, v5min_in, v15min_in,
                              persist_in, hold_th15_in, hold_v15_in, hold_persist_in,
                              session_cutoff_in, manual_force_flat_in, zero_flow_flags_in, force_close_in,
                              reasons_init):
        pos_out = dict(pos_prev_in)
        reasons_out = list(reasons_init or [])
        manual_action_taken_out = False
        manual_noop_out = False

        try:
            v5_pass = float(v5_in) >= float(v5min_in)
        except Exception:
            v5_pass = False
        try:
            v15_pass = float(v15_in) >= float(v15min_in)
        except Exception:
            v15_pass = False
        pos_out['v5_fail_count'] = int(pos_out.get('v5_fail_count', 0) or 0)
        pos_out['v15_fail_count'] = int(pos_out.get('v15_fail_count', 0) or 0)
        if not v5_pass:
            pos_out['v5_fail_count'] += 1
        if not v15_pass:
            pos_out['v15_fail_count'] += 1

        long_entry = (calc_signal_in == 'long')
        short_entry = (calc_signal_in == 'short')
        hold15_ok_long = (b15_in >= hold_th15_in) and (v15_in >= hold_v15_in)
        hold15_ok_short = (b15_in <= -hold_th15_in) and (v15_in >= hold_v15_in)
        b5_ok_long = (b5_in >= th5_in)
        b5_ok_short = (b5_in <= -th5_in)

        final_signal_out = 'flat'
        if pos_out.get('mode', 'flat') == 'flat':
            pos_out['disagree_streak'] = 0
            if session_cutoff_in and (long_entry or short_entry):
                pos_out['enter_long'] = 0
                pos_out['enter_short'] = 0
                final_signal_out = 'flat'
                reasons_out.append('late_session_block')
            elif long_entry:
                pos_out['enter_long'] = int(pos_out.get('enter_long', 0)) + 1
                pos_out['enter_short'] = 0
                if pos_out['enter_long'] >= max(1, persist_in):
                    final_signal_out = 'long'
                    pos_out['mode'] = 'long'
                    pos_out['enter_long'] = 0
                    pos_out['hold_fail'] = 0
                    pos_out['disagree_streak'] = 0
                else:
                    final_signal_out = 'flat'
                    reasons_out.append('enter_persisting')
            elif short_entry:
                pos_out['enter_short'] = int(pos_out.get('enter_short', 0)) + 1
                pos_out['enter_long'] = 0
                if pos_out['enter_short'] >= max(1, persist_in):
                    final_signal_out = 'short'
                    pos_out['mode'] = 'short'
                    pos_out['enter_short'] = 0
                    pos_out['hold_fail'] = 0
                    pos_out['disagree_streak'] = 0
                else:
                    final_signal_out = 'flat'
                    reasons_out.append('enter_persisting')
            else:
                pos_out['enter_long'] = 0
                pos_out['enter_short'] = 0
                final_signal_out = 'flat'
        else:
            if pos_out['mode'] == 'long':
                pos_out.setdefault('disagree_streak', 0)
                pos_out.setdefault('b5_hold_fail', 0)
                if agree_in:
                    pos_out['disagree_streak'] = 0
                else:
                    pos_out['disagree_streak'] = int(pos_out.get('disagree_streak', 0)) + 1
                    if pos_out['disagree_streak'] > 2:
                        final_signal_out = 'flat'
                        pos_out['mode'] = 'flat'
                        pos_out['hold_fail'] = 0
                        pos_out['disagree_streak'] = 0
                        pos_out['b5_hold_fail'] = 0
                        reasons_out.append('sign_disagree_exit')
                if pos_out['mode'] == 'long':
                    if not b5_ok_long:
                        pos_out['b5_hold_fail'] = int(pos_out.get('b5_hold_fail', 0) or 0) + 1
                        if pos_out['b5_hold_fail'] >= 3:
                            pos_out['hold_fail'] = 0
                            pos_out['disagree_streak'] = 0
                            final_signal_out = 'flat'
                            pos_out['mode'] = 'flat'
                            pos_out['b5_hold_fail'] = 0
                            reasons_out.append('b5_hold_exit')
                        else:
                            final_signal_out = 'long'
                            reasons_out.append('b5_hold_warn')
                    elif hold15_ok_long:
                        final_signal_out = 'long'
                        pos_out['hold_fail'] = 0
                        pos_out['b5_hold_fail'] = 0
                    else:
                        pos_out['hold_fail'] = int(pos_out.get('hold_fail', 0)) + 1
                        if pos_out['hold_fail'] >= max(1, hold_persist_in):
                            final_signal_out = 'flat'
                            pos_out['mode'] = 'flat'
                            pos_out['hold_fail'] = 0
                            pos_out['disagree_streak'] = 0
                            pos_out['b5_hold_fail'] = 0
                            reasons_out.append('hold_exit')
                        else:
                            final_signal_out = 'long'
                            reasons_out.append('hold_persisting')
            elif pos_out['mode'] == 'short':
                pos_out.setdefault('disagree_streak', 0)
                pos_out.setdefault('b5_hold_fail', 0)
                if agree_in:
                    pos_out['disagree_streak'] = 0
                else:
                    pos_out['disagree_streak'] = int(pos_out.get('disagree_streak', 0)) + 1
                    if pos_out['disagree_streak'] > 2:
                        final_signal_out = 'flat'
                        pos_out['mode'] = 'flat'
                        pos_out['hold_fail'] = 0
                        pos_out['disagree_streak'] = 0
                        pos_out['b5_hold_fail'] = 0
                        reasons_out.append('sign_disagree_exit')
                if pos_out['mode'] == 'short':
                    if not b5_ok_short:
                        pos_out['b5_hold_fail'] = int(pos_out.get('b5_hold_fail', 0) or 0) + 1
                        if pos_out['b5_hold_fail'] >= 3:
                            pos_out['hold_fail'] = 0
                            pos_out['disagree_streak'] = 0
                            final_signal_out = 'flat'
                            pos_out['mode'] = 'flat'
                            pos_out['b5_hold_fail'] = 0
                            reasons_out.append('b5_hold_exit')
                        else:
                            final_signal_out = 'short'
                            reasons_out.append('b5_hold_warn')
                    elif hold15_ok_short:
                        final_signal_out = 'short'
                        pos_out['hold_fail'] = 0
                        pos_out['b5_hold_fail'] = 0
                    else:
                        pos_out['hold_fail'] = int(pos_out.get('hold_fail', 0)) + 1
                        if pos_out['hold_fail'] >= max(1, hold_persist_in):
                            final_signal_out = 'flat'
                            pos_out['mode'] = 'flat'
                            pos_out['hold_fail'] = 0
                            pos_out['disagree_streak'] = 0
                            pos_out['b5_hold_fail'] = 0
                            reasons_out.append('hold_exit')
                        else:
                            final_signal_out = 'short'
                            reasons_out.append('hold_persisting')

        if manual_force_flat_in:
            if pos_out.get('mode', 'flat') != 'flat':
                manual_action_taken_out = True
                pos_out['mode'] = 'flat'
                pos_out['hold_fail'] = 0
                pos_out['enter_long'] = 0
                pos_out['enter_short'] = 0
                pos_out['disagree_streak'] = 0
                pos_out['b5_hold_fail'] = 0
                final_signal_out = 'flat'
                reasons_out.append('user_flat')
            else:
                manual_noop_out = True
                final_signal_out = 'flat'
                reasons_out.append('user_flat_no_pos')

        zero_flow_applied = bool(zero_flow_flags_in)
        if zero_flow_applied:
            if pos_out.get('mode', 'flat') != 'flat':
                pos_out['mode'] = 'flat'
                pos_out['hold_fail'] = 0
                pos_out['enter_long'] = 0
                pos_out['enter_short'] = 0
                pos_out['disagree_streak'] = 0
                pos_out['b5_hold_fail'] = 0
            final_signal_out = 'flat'
            for flag in zero_flow_flags_in:
                if flag not in reasons_out:
                    reasons_out.append(flag)
            if not zero_flow_flags_in and 'zero_flow' not in reasons_out:
                reasons_out.append('zero_flow')

        if force_close_in:
            if pos_out.get('mode', 'flat') != 'flat':
                pos_out['mode'] = 'flat'
                pos_out['hold_fail'] = 0
                pos_out['enter_long'] = 0
                pos_out['enter_short'] = 0
                pos_out['disagree_streak'] = 0
                pos_out['b5_hold_fail'] = 0
            final_signal_out = 'flat'
            reasons_out.append('market_close')

        return final_signal_out, pos_out, reasons_out, manual_action_taken_out, manual_noop_out

    final_base, pos_base_next, reasons_base, manual_action_base, manual_noop_base = _apply_position_logic(
        calc_signal_base, pos_baseline_prev, agree, b5, b15, v5, v15,
        thresh5_val, thresh15_val, minV5_val, minV15_val,
        persist_n, hold_thresh15_val, hold_minV15_val, hold_persist_n,
        session_cutoff, manual_force_flat, zero_flow_flags, force_close,
        reasons_base,
    )

    final_exp, pos_exp_next, reasons_exp, manual_action_exp, manual_noop_exp = _apply_position_logic(
        calc_signal_exp, pos_exp_work, agree, b5, b15, v5, v15,
        thresh5_val, thresh15_val, minV5_val, minV15_val,
        persist_n, hold_thresh15_val, hold_minV15_val, hold_persist_n,
        session_cutoff, manual_force_flat, zero_flow_flags, force_close,
        reasons_exp,
    )

    prev_mode_base = str(pos_baseline_prev.get('mode', 'flat'))
    prev_mode_exp = str(pos_experimental_prev.get('mode', 'flat'))
    curr_mode_base = str(pos_base_next.get('mode', 'flat'))
    curr_mode_exp = str(pos_exp_next.get('mode', 'flat'))
    _update_pnl_state(pos_baseline_prev, pos_base_next, prev_mode_base, curr_mode_base, spot)
    _update_pnl_state(pos_experimental_prev, pos_exp_next, prev_mode_exp, curr_mode_exp, spot)

    if strategy_mode == 'experimental':
        final_signal = final_exp
        strength_active = strength_exp
        pos_next = pos_exp_next
        pos_shadow = pos_base_next
        manual_action_taken = manual_action_exp
        manual_noop = manual_noop_exp
        reasons_active = reasons_exp
    else:
        final_signal = final_base
        strength_active = strength_base
        pos_next = pos_base_next
        pos_shadow = pos_exp_next
        manual_action_taken = manual_action_base
        manual_noop = manual_noop_base
        reasons_active = reasons_base

    active_strategy = 'experimental' if strategy_mode == 'experimental' else 'baseline'
    shadow_signal = final_base if active_strategy == 'experimental' else final_exp
    shadow_strength = strength_base if active_strategy == 'experimental' else strength_exp
    zero_flow_applied = bool(zero_flow_flags)
    mode_label = active_strategy if strategy_mode != 'shadow' else f"{active_strategy} (shadow)"
    prev_mode_active = prev_mode_exp if strategy_mode == 'experimental' else prev_mode_base

    signal_txt = final_signal.upper()
    print(f"[dt] mode={strategy_mode} calc={calc_signal_base} final={final_signal} b5={b5:.2f} v5={v5:.0f} b15={b15:.2f} v15={v15:.0f} spot={spot:.2f}")
    curr_mode = str(pos_next.get('mode', 'flat'))
    pnl_realized = float(pos_next.get('pnl_realized', 0.0))
    pnl_unreal = float(pos_next.get('pnl_unrealized', 0.0))
    pnl_total = float(pos_next.get('pnl_total', pnl_realized + pnl_unreal))
    signal_colors = {
        'long': '#2ecc71',
        'short': '#e74c3c',
        'flat': '#2d3436',
    }
    signal_span = html.Span(signal_txt, style={'color': signal_colors.get(final_signal, '#2d3436'), 'fontWeight': '700'})
    # SPX-only: show DTE shares and anchor if present
    dte_badge = ""
    try:
        if t_upper == 'SPX':
            shares = res.get('dte_shares', {}) if isinstance(res, dict) else {}
            d0 = float(shares.get('d0', 0.0))
            d1 = float(shares.get('d1', 0.0))
            anchor = res.get('anchor', None)
            dte_badge = f" • dte0={d0:.0%} dte1={d1:.0%}"
            if anchor is not None:
                dte_badge += f" anchor={float(anchor):+.2f}"
    except Exception:
        dte_badge = ""
    # IV context badge for non-SPX (if available)
    iv_badge = ""
    if (t_upper != 'SPX') and isinstance(ivm, dict):
        iv_badge = ""
    logdir_status = _log_dir_for_today()
    status_children = [
        f"{res['ticker']} • SIGNAL: ", 
        signal_span,
        f"  strength={strength_active:.2f}  agree={agree}  spot(auto)={spot:.2f} ",
        f"• mode={mode_label} ",
        f"• expiries={int(exp_count or 7)}{dte_badge}{iv_badge} ",
        f"• PnL={pnl_total:+.2f} (R={pnl_realized:+.2f}, U={pnl_unreal:+.2f}) ",
        #f"• logs: {logdir_status}",
    ]
    if total_polls > 0:
        status_children.append(
            f" • b5_fail={b5_fail_count}/{total_polls} ({b5_fail_ratio:.0%})"
        )
        status_children.append(
            f" • b15_fail={b15_fail_count}/{total_polls} ({b15_fail_ratio:.0%})"
        )
        status_children.append(
            f" • sign_disagree={sign_disagree_count}"
        )
    if manual_action_taken:
        status_children.append(" • user_flat (manual exit)")
    elif manual_noop:
        status_children.append(" • user_flat (no position)")
    if zero_flow_applied:
        tag = '/'.join(zero_flow_flags) if zero_flow_flags else 'zero_flow'
        if zero_flow_attempts > 1:
            tag = f"{tag} retry={zero_flow_attempts}"
        status_children.append(f" • {tag}")
    if strategy_mode == 'shadow':
        status_children.append(f" • shadow={shadow_signal.upper()} ({shadow_strength:.2f})")
    if isinstance(wall_guard_info, dict):
        wall = wall_guard_info.get('wall') or {}
        side = wall_guard_info.get('side', '') or ''
        strike = wall.get('strike')
        health_state = wall.get('health_state')
        applied = bool(wall_guard_info.get('applied', False))
        try:
            delta_txt = f"{float(wall.get('health_change', 0.0)):+.2f}"
        except Exception:
            delta_txt = "n/a"
        try:
            vel_txt = f"{float(wall.get('health_velocity', 0.0)):+.2f}"
        except Exception:
            vel_txt = "n/a"
        health_suffix = ""
        if health_state:
            health_suffix = f" {health_state} Δ={delta_txt} vel={vel_txt}"
        if wall_guard_info.get('triggered'):
            label = 'wall_guard'
            if not applied:
                label = 'wall_guard(disabled)'
            status_children.append(
                f" • {label}({side}@{strike}{health_suffix})"
            )
        elif wall and wall.get('active'):
            status_children.append(
                f" • wall_watch({side}@{strike}{health_suffix})"
            )
    if bias_alert_info and bias_alert_info.get('active'):
        alert_msg = bias_alert_info.get('message') or 'Bias alert triggered'
        status_children.append(html.Span(alert_msg, style={'backgroundColor': '#e67e22', 'color': '#ffffff', 'padding': '0 6px', 'borderRadius': '4px', 'marginLeft': '6px'}))
    status = html.Span(status_children)

    regime_bits = []
    if use_anchor:
        regime_bits.append(f"regime={regime_state}")
        if anchor_val is not None:
            regime_bits.append(f"anchor={float(anchor_val):+.2f}")
        if short_anchor is not None:
            regime_bits.append(f"short={float(short_anchor):+.2f}")
        if vday:
            regime_bits.append(f"day_bias={bday:+.2f} V={vday:.0f}")
        if use_transition:
            regime_bits.append(f"flip={int(flip_count)}/{transition_polls_val}")
    if use_tod and tod_mult_applied > 1.0:
        regime_bits.append(f"tod_x={tod_mult_applied:.2f}")
    if use_near:
        regime_bits.append(f"near_x={near_mult_val:.2f}")
    regime_children = " • ".join(regime_bits) if regime_bits else ""

    # Update mini context series (SPX only) and build the compact figure
    mini_fig = _empty_fig(theme, height=160)
    cs_new = ctx_series if isinstance(ctx_series, dict) else {}
    try:
        if t_upper == 'SPX':
            # minute index since 09:30
            from datetime import datetime as _dt
            now = _dt.now(ZoneInfo("America/New_York")).replace(second=0, microsecond=0)
            start = now.replace(hour=9, minute=30)
            mins = int(max(0, min(390, (now - start).total_seconds() // 60)))
            cs = ctx_series if isinstance(ctx_series, dict) else {}
            xs = list(cs.get('x', []))
            ya = list(cs.get('anchor', []))
            yd0 = list(cs.get('d0', []))
            # append only if increasing time
            if (not xs) or (mins > xs[-1]):
                xs.append(int(mins))
                ya.append(float(res.get('anchor', 0.0) or 0.0))
                shares = res.get('dte_shares', {}) if isinstance(res, dict) else {}
                yd0.append(float(shares.get('d0', 0.0) or 0.0))
            cs_new = {'x': xs[-400:], 'anchor': ya[-400:], 'd0': yd0[-400:]}
            # plot today's anchor
            time_axis = cs_new['x']
            mini_fig.add_trace(go.Scatter(x=time_axis, y=cs_new['anchor'], mode='lines+markers', name='anchor', line=dict(color='#1f77b4'), marker=dict(size=4)))
            # yesterday overlay
            if isinstance(ctx_yday, dict) and ctx_yday.get('x') and ctx_yday.get('anchor'):
                mini_fig.add_trace(go.Scatter(x=ctx_yday['x'], y=ctx_yday['anchor'], mode='lines', name='yday', line=dict(color='#888', dash='dot')))
            # dte0 share on secondary axis scaled to [-1,1] around 0 via (d0-0.5)*2
            if cs_new['d0']:
                d0_scaled = [(v - 0.5) * 2.0 for v in cs_new['d0']]
                mini_fig.add_trace(go.Scatter(x=time_axis, y=d0_scaled, mode='lines+markers', name='dte0(±)', line=dict(color='#f1c40f'), marker=dict(size=3)))
            mini_fig.update_layout(height=160, margin=dict(l=20, r=10, t=10, b=20), showlegend=False)
            mini_fig.update_yaxes(title='anchor', range=[-1,1])
            tick_candidates = [30, 90, 150, 210, 270, 330, 390]
            tick_vals = [v for v in tick_candidates if time_axis and (time_axis[0] <= v <= time_axis[-1])]
            if tick_vals:
                mini_fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=[_minutes_to_label(v) for v in tick_vals],
                    range=[0, 390],
                )
            else:
                mini_fig.update_xaxes(range=[0, 390])
        else:
            cs_new = ctx_series if isinstance(ctx_series, dict) else {}
            mini_fig = None
    except Exception:
        cs_new = ctx_series

    # Metrics table (5m & 15m)
    m5 = sig.get('metrics', {}).get('5m', {})
    m15 = sig.get('metrics', {}).get('15m', {})
    cols = [
        {"name": "window", "id": "window"},
        {"name": "bias", "id": "bias"},
        {"name": "V", "id": "V"},
        {"name": "C", "id": "C"},
        {"name": "P", "id": "P"},
        {"name": "sign_disagreement", "id": "sign_disagreement"},
        {"name": "bias_fail_count", "id": "bias_fail_count"},
        {"name": "bias_fail_ratio", "id": "bias_fail_ratio", "type": "numeric", "format": FormatTemplate.percentage(1)},
        {"name": "bias_fail_ratio_30m", "id": "bias_fail_ratio_30m", "type": "numeric", "format": FormatTemplate.percentage(1)},
        {"name": "V<min count", "id": "vol_fail"},
    ]
    enter_long = int(pos_next.get('enter_long', 0))
    enter_short = int(pos_next.get('enter_short', 0))
    persist_req = max(1, persist_n)
    data = [
        {
            "window": "5m",
            "bias": round(float(m5.get('bias', 0.0)), 3),
            "V": round(float(m5.get('V', 0.0)), 0),
            "C": round(float(m5.get('C', 0.0)), 0),
            "P": round(float(m5.get('P', 0.0)), 0),
            "sign_disagreement": sign_disagree_count,
            "bias_fail_count": b5_fail_count,
            "bias_fail_ratio": b5_fail_ratio,
            "bias_fail_ratio_30m": b5_fail_ratio_30m,
            "vol_fail": int(pos_next.get('v5_fail_count', 0) or 0),
        },
        {
            "window": "15m",
            "bias": round(float(m15.get('bias', 0.0)), 3),
            "V": round(float(m15.get('V', 0.0)), 0),
            "C": round(float(m15.get('C', 0.0)), 0),
            "P": round(float(m15.get('P', 0.0)), 0),
            "sign_disagreement": '',
            "bias_fail_count": b15_fail_count,
            "bias_fail_ratio": b15_fail_ratio,
            "bias_fail_ratio_30m": b15_fail_ratio_30m,
            "vol_fail": int(pos_next.get('v15_fail_count', 0) or 0),
        },
    ]
    # Diagnostics logging (JSONL + CSV)
    try:
        b5 = float(sig.get('metrics', {}).get('5m', {}).get('bias', 0.0))
        v5 = float(sig.get('metrics', {}).get('5m', {}).get('V', 0.0))
        b15 = float(sig.get('metrics', {}).get('15m', {}).get('bias', 0.0))
        v15 = float(sig.get('metrics', {}).get('15m', {}).get('V', 0.0))

        b5_pass = abs(b5) >= thresh5_val
        v5_pass = v5 >= minV5_val
        b15_pass = abs(b15) >= thresh15_val
        v15_pass = v15 >= minV15_val
        agree_pass = (_sign(b5) != 0) and (_sign(b5) == _sign(b15))
        reasons_base_gate = []
        if not b5_pass: reasons_base_gate.append('b5<thresh5')
        if not v5_pass: reasons_base_gate.append('V5<minV5')
        if not b15_pass: reasons_base_gate.append('b15<thresh15')
        if not v15_pass: reasons_base_gate.append('V15<minV15')
        if not agree_pass: reasons_base_gate.append('sign_disagree')

        b5_pass_exp = abs(b5) >= th5_eff
        v5_pass_exp = v5 >= v5_eff
        b15_pass_exp = abs(b15) >= th15_eff
        v15_pass_exp = v15 >= v15_eff
        agree_pass_exp = agree_pass
        reasons_exp_gate = []
        if not b5_pass_exp: reasons_exp_gate.append('b5<thresh5')
        if not v5_pass_exp: reasons_exp_gate.append('V5<minV5')
        if not b15_pass_exp: reasons_exp_gate.append('b15<thresh15')
        if not v15_pass_exp: reasons_exp_gate.append('V15<minV15')
        if not agree_pass_exp: reasons_exp_gate.append('sign_disagree')

        reasons_base_full = reasons_base_gate + list(reasons_base)
        reasons_exp_full = reasons_exp_gate + list(reasons_exp)
        reasons_active_full = reasons_base_full if active_strategy == 'baseline' else reasons_exp_full

        ts = _ny_now_iso()
        logdir = _log_dir_for_today()
        params_used = ['oi','oi_ch','volmbs_5m','volmbs_15m','volmbs_30m','volmbs_60m','volm_bs','expiration_ts','price','delta','volatility']
        exps_used = list(range(0, int(exp_count or 7)))
        try:
            distinct_exps = int(df_all['expiration_date'].nunique()) if 'expiration_date' in df_all.columns else 0
        except Exception:
            distinct_exps = 0

        gates_base = {
            'b5_pass': b5_pass,
            'V5_pass': v5_pass,
            'b15_pass': b15_pass,
            'V15_pass': v15_pass,
            'agree': agree_pass,
        }
        gates_exp = {
            'b5_pass': b5_pass_exp,
            'V5_pass': v5_pass_exp,
            'b15_pass': b15_pass_exp,
            'V15_pass': v15_pass_exp,
            'agree': agree_pass_exp,
        }
        decision_base = {
            'gates': gates_base,
            'reasons': reasons_base_full,
            'signal': final_base,
            'strength': float(strength_base),
        }
        decision_exp = {
            'gates': gates_exp,
            'reasons': reasons_exp_full,
            'signal': final_exp,
            'strength': float(strength_exp),
        }
        decision_active = decision_base if active_strategy == 'baseline' else decision_exp

        obj = {
            'schema_version': '1.0',
            'ts_ny': ts,
            'ticker': res.get('ticker',''),
            'app_version': 'dt_ui',
            'spot': spot,
            'spot_src': 'auto',
            'config': {'expiries': int(exp_count or 7), 'weighting': weighting, 'mny_band': float(band_val),
                       'view_band_pct': float(view_pct), 'poll_secs': 15, 'theme': theme},
            'strategy': {
                'mode': strategy_mode,
                'active': active_strategy,
                'features': sorted(list(features)),
                'anchor_min': float(anchor_min_val),
                'anchor_hl': float(anchor_hl_val),
                'anchor_short_hl': float(anchor_short_hl_val),
                'transition_polls': int(transition_polls_val),
                'transition_mult': float(transition_mult_val),
                'near_mult': float(near_mult_val),
                'tod_mult': float(tod_mult_val),
                'tod_mult_applied': float(tod_mult_applied),
            },
            'api': {'params': params_used, 'exps': exps_used, 'rng': 100, 'parse_rows': (0 if df_all is None else len(df_all)), 'distinct_exps': distinct_exps},
            'spx_exp': ( {'d0': float(res.get('dte_shares',{}).get('d0',0.0)), 'd1': float(res.get('dte_shares',{}).get('d1',0.0)), 'basis': '15m', 'anchor': (float(res.get('anchor',0.0)) if res.get('anchor') is not None else None)} if (str(ticker or '').strip().upper()=='SPX') else {} ),
            'iv_ctx': ( {'pc_atm': (None if (not isinstance(ivm, dict) or (ivm.get('pc_atm') is None)) else float(ivm.get('pc_atm'))),
                         'rr25': (None if (not isinstance(ivm, dict) or (ivm.get('rr25') is None)) else float(ivm.get('rr25'))),
                         'ts_ratio': (None if (not isinstance(ivm, dict) or (ivm.get('ts_ratio') is None)) else float(ivm.get('ts_ratio')))} if (str(ticker or '').strip().upper()!='SPX') else {} ),
            'zero_flow_flags': zero_flow_flags,
            'zero_flow_attempts': zero_flow_attempts,
            'intervals': {
                '5m': {'C': float(sig.get('metrics', {}).get('5m', {}).get('C', 0.0)),
                       'P': float(sig.get('metrics', {}).get('5m', {}).get('P', 0.0)),
                       'V': v5, 'bias': b5},
                '15m': {'C': float(sig.get('metrics', {}).get('15m', {}).get('C', 0.0)),
                        'P': float(sig.get('metrics', {}).get('15m', {}).get('P', 0.0)),
                        'V': v15, 'bias': b15},
                '30m': {'C': float(sig.get('metrics', {}).get('30m', {}).get('C', 0.0)),
                        'P': float(sig.get('metrics', {}).get('30m', {}).get('P', 0.0)),
                        'V': float(sig.get('metrics', {}).get('30m', {}).get('V', 0.0) or 0.0), 'bias': b30},
                '60m': {'C': float(sig.get('metrics', {}).get('60m', {}).get('C', 0.0)),
                        'P': float(sig.get('metrics', {}).get('60m', {}).get('P', 0.0)),
                        'V': float(sig.get('metrics', {}).get('60m', {}).get('V', 0.0) or 0.0), 'bias': b60},
                'day': {'C': float(sig.get('metrics', {}).get('day', {}).get('C', 0.0)),
                        'P': float(sig.get('metrics', {}).get('day', {}).get('P', 0.0)),
                        'V': vday, 'bias': bday},
            },
            'alt_intervals': res.get('alt_intervals', {}),
            'dyn_config': res.get('dyn_config', {}),
            'bias_fail_stats': res.get('bias_fail_stats', {}),
            'bias_alert': res.get('bias_alert', {}),
            'wall_guard_applied': bool(res.get('wall_guard_applied', False)),
            'decision': decision_active,
            'decision_baseline': decision_base,
            'decision_candidate': decision_exp,
            'regime': {
                'state': regime_state,
                'anchor': (float(anchor_val) if anchor_val is not None else None),
                'short_anchor': (float(short_anchor) if short_anchor is not None else None),
                'day_bias': float(bday),
                'day_V': float(vday),
                'flip_count': int(flip_count),
            },
        }
        _append_jsonl(os.path.join(logdir, f"{t}.jsonl"), obj)
        # Store intraday context in SQLite (SPX + others). Safe no-op if store missing.
        try:
            if _insert_ctx is not None:
                d0 = float(res.get('dte_shares', {}).get('d0', 0.0)) if isinstance(res.get('dte_shares', {}), dict) else None
                d1 = float(res.get('dte_shares', {}).get('d1', 0.0)) if isinstance(res.get('dte_shares', {}), dict) else None
                _insert_ctx(
                    ts_iso=ts,
                    ticker=t,
                    spot=float(res.get('spot', 0.0)),
                    b5=b5, v5=v5, b15=b15, v15=v15,
                    signal=final_signal,
                    strength=float(strength_active),
                    dte0_share=d0,
                    dte1_share=d1,
                    anchor=(float(res.get('anchor', 0.0)) if res.get('anchor') is not None else None),
                )
        except Exception as _e:
            # Do not fail UI on DB write
            pass

        csv_path = os.path.join(logdir, 'metrics.csv')
        csv_header = ['ts_ny','ticker','spot','b5','b15','V5','V15','signal','strength','reasons']
        csv_row = [ts, t, f"{spot:.4f}", f"{b5:.4f}", f"{b15:.4f}", f"{v5:.0f}", f"{v15:.0f}", signal_txt.lower(), f"{strength_active:.2f}", ';'.join(reasons_active_full)]
        _append_csv(csv_path, csv_header, csv_row)
    except Exception as e:
        print(f"[dt] log write error: {e}")

    # Pack stores for lightweight view updates
    if t_upper == 'SPX':
        wall_store_data = wall_snapshot if isinstance(wall_snapshot, dict) else None
        wall_fig = _build_wall_fig(wall_store_data, theme)
        wall_fig_style = dict(wall_style_visible)
    else:
        wall_store_data = None
        wall_fig = _empty_fig(theme, height=220)
        wall_fig_style = dict(wall_style_hidden)
    try:
        # Preserve readable datetime encoding
        df_json = df_all.to_json(orient='split', date_format='iso') if df_all is not None else None
    except Exception:
        df_json = None
    cfg = {
        'weighting': weighting or 'mny_delta',
        'mny_band': float(band_val),
        'exp_count': int(exp_count or 7),
        'ticker': t,
        'view_pct': float(view_pct),
        'theme': theme,
        'strategy_mode': strategy_mode,
        'strategy_active': active_strategy,
        'features': sorted(list(features)),
        'anchor_min': float(anchor_min_val),
        'anchor_hl': float(anchor_hl_val),
        'anchor_short_hl': float(anchor_short_hl_val),
        'transition_polls': int(transition_polls_val),
        'transition_mult': float(transition_mult_val),
        'near_mult': float(near_mult_val),
        'tod_mult': float(tod_mult_val),
    }
    if t_upper == 'SPX':
        try:
            ZERO_GAMMA_PANEL.tick(t_upper)
        except Exception:
            pass
    zg_fig = ZERO_GAMMA_PANEL.build_figure(theme, t_upper)
    tone_src = _TONE_MAP.get(curr_mode, '') if curr_mode != prev_mode_active else ''

    ctx_next = cs_new if isinstance(cs_new, dict) else {}
    dyn_next = _dyn_payload(pos_next)
    if mini_fig is None:
        mini_output = _empty_fig(theme, height=160)
        mini_style = {"display": "none"}
    else:
        mini_output = _apply_fig_theme(mini_fig, theme)
        mini_style = {"height": "160px", "marginTop": "4px", "flex": "1"}
    return (
        status,
        regime_children,
        tone_src,
        cols,
        data,
        metrics_style,
        df_json,
        float(spot_f or spot),
        cfg,
        wall_store_data,
        wall_fig,
        wall_fig_style,
        pos_next,
        pos_shadow,
        ctx_next,
        mini_output,
        mini_style,
        zg_fig,
        dyn_next,
    )


# Lightweight view update: recompute figures from stored df + slider, no network calls
@app.callback(
    Output("dt-graph-5m", "figure"),
    Output("dt-graph-15m", "figure"),
    Output("dt-graph-30m", "figure"),
    Output("dt-graph-60m", "figure"),
    Input("df-store", "data"),
    Input("spot-store", "data"),
    State("cfg-store", "data"),
)
def update_figs(df_json, spot_val, cfg):
    cfg = cfg if isinstance(cfg, dict) else {}
    theme = _normalize_theme(cfg.get('theme', DEFAULT_THEME))
    blank = _empty_fig(theme, height=140)
    try:
        if not df_json or spot_val is None:
            return blank, blank, blank, blank
        df_all = pd.read_json(df_json, orient='split')
        # Ensure expiration_date is a readable date (handle epoch-ms, ISO strings, etc.)
        if 'expiration_date' in df_all.columns:
            try:
                col = df_all['expiration_date']
                if pd.api.types.is_numeric_dtype(col):
                    df_all['expiration_date'] = pd.to_datetime(col, unit='ms', errors='coerce').dt.date
                else:
                    df_all['expiration_date'] = pd.to_datetime(col, errors='coerce').dt.date
            except Exception:
                pass
        spot_f = float(spot_val)
        weighting = cfg.get('weighting', 'mny_delta')
        ticker = cfg.get('ticker', 'SPX')
        ticker_upper = str(ticker).upper()
        default_band = 0.015 if ticker_upper == 'SPX' else 0.03
        try:
            mny_band = float(cfg.get('mny_band', default_band))
        except Exception:
            mny_band = default_band
        view_pct_default = 0.02 if ticker_upper == 'SPX' else 0.10
        cfg_view = cfg.get('view_pct')
        try:
            if cfg_view is None:
                view_pct = view_pct_default
            else:
                view_candidate = float(cfg_view)
                view_pct = view_candidate / 100.0 if view_candidate > 1.0 else view_candidate
                if view_pct <= 0:
                    view_pct = view_pct_default
        except Exception:
            view_pct = view_pct_default

        exp_weights = None
        if ticker_upper == 'SPX':
            try:
                exp_weights = _spx_expiry_weight_for_date(df_all.get('expiration_date'))
            except Exception:
                exp_weights = None
        ladders = compute_ladders(
            df_all,
            float(spot_f),
            weighting or 'mny_delta',
            float(mny_band or default_band),
            view_band_pct=view_pct,
            exp_weights=exp_weights,
        )

        def _dte_customdata(param: str) -> dict:
            try:
                c_col = f"call_{param}"
                p_col = f"put_{param}"
                if c_col not in df_all.columns or p_col not in df_all.columns:
                    return {}
                strike_series = df_all['strike_price'].astype(float)
                c_arr = df_all[c_col].astype(float).to_numpy()
                p_arr = df_all[p_col].astype(float).to_numpy()
                exp_dates = pd.to_datetime(df_all.get('expiration_date'), errors='coerce').dt.date
                today = now_ny().date()
                buckets = []
                for ed in exp_dates:
                    try:
                        diff = (ed - today).days if ed is not None else 99
                    except Exception:
                        diff = 99
                    if diff <= 0:
                        buckets.append('0')
                    elif diff == 1:
                        buckets.append('1')
                    else:
                        buckets.append('2+')
                tmp = pd.DataFrame({
                    'strike': strike_series,
                    'bucket': buckets,
                    'cw': c_arr,
                    'pw': p_arr,
                })
                grouped = tmp.groupby(['strike', 'bucket'], as_index=False).sum()
                out: dict[float, dict[str, tuple[float, float]]] = {}
                for row in grouped.itertuples(index=False):
                    strike_val = float(row.strike)
                    bucket = str(row.bucket)
                    entry = out.setdefault(strike_val, {'0': (0.0, 0.0), '1': (0.0, 0.0), '2+': (0.0, 0.0)})
                    entry[bucket] = (float(row.cw), float(row.pw))
                return out
            except Exception:
                return {}

        dte_map = {
            'volmbs_5m': _dte_customdata('volmbs_5m'),
            'volmbs_15m': _dte_customdata('volmbs_15m'),
            'volmbs_30m': _dte_customdata('volmbs_30m'),
            'volmbs_60m': _dte_customdata('volmbs_60m'),
        }

        strip_df = compute_expiry_strip(
            df_all,
            float(spot_f),
            weighting or 'mny_delta',
            float(mny_band or default_band),
            max_exps=int(cfg.get('exp_count', 7)),
            only_zero_dte=True,
        )

        def make_fig(key: str, title: str) -> go.Figure:
            if key not in ladders:
                fig_blank = _empty_fig(theme, height=140)
                fig_blank.update_layout(title=title)
                fig_blank.update_xaxes(title='Strike')
                fig_blank.update_yaxes(title='Net flow (w)')
                return fig_blank
            K = np.asarray(ladders[key]['K'])
            net = np.asarray(ladders[key]['net'])
            cw_weighted = np.asarray(ladders[key]['cw'])
            pw_weighted = np.asarray(ladders[key]['pw'])
            colors = ['#f1c40f' if v >= 0 else '#00bcd4' for v in net]
            custom_rows = []
            by_strike = dte_map.get(key, {})
            for idx, strike_val in enumerate(K):
                info = by_strike.get(float(strike_val), {'0': (0.0, 0.0), '1': (0.0, 0.0), '2+': (0.0, 0.0)})
                c0, p0 = info.get('0', (0.0, 0.0))
                c1, p1 = info.get('1', (0.0, 0.0))
                c2, p2 = info.get('2+', (0.0, 0.0))
                custom_rows.append([
                    float(cw_weighted[idx]),
                    float(pw_weighted[idx]),
                    c0, p0, c1, p1, c2, p2,
                ])
            hover_template = (
                "Strike %{x:.2f}<br>Net %{y:.0f}"
                "<br>Weighted C=%{customdata[0]:.0f}, P=%{customdata[1]:.0f}"
                "<br>0D raw: C=%{customdata[2]:.0f}, P=%{customdata[3]:.0f}"
                "<br>1D raw: C=%{customdata[4]:.0f}, P=%{customdata[5]:.0f}"
                "<br>2D+ raw: C=%{customdata[6]:.0f}, P=%{customdata[7]:.0f}<extra></extra>"
            )
            fig = go.Figure(data=[go.Bar(x=K, y=net, marker_color=colors, customdata=custom_rows, hovertemplate=hover_template)])
            fig.add_vline(x=float(spot_f), line=dict(color='red', width=2))
            fig.update_layout(title=title, height=140, bargap=0.30, margin=dict(l=14,r=8,t=28,b=20))
            fig.update_xaxes(title='Strike')
            fig.update_yaxes(title='Net flow (w)')
            return _apply_fig_theme(fig, theme)

        f5 = make_fig('volmbs_5m', '5m (net)')
        f15 = make_fig('volmbs_15m', '15m (net)')
        f30 = make_fig('volmbs_30m', '30m (net)')
        f60 = make_fig('volmbs_60m', '60m (net)')

        return f5, f15, f30, f60
    except Exception:
        return blank, blank, blank, blank


# IV delta ladder (per strike) for front expiry — non-blocking, uses df-store
@app.callback(
    Output("dt-iv-delta", "figure"),
    Output("iv-prev", "data"),
    Input("df-store", "data"),
    Input("spot-store", "data"),
    State("iv-prev", "data"),
    State("theme-store", "data"),
)
def on_iv_delta(df_json, spot_val, prev, theme_value):
    theme = _normalize_theme(theme_value)
    fig = _empty_fig(theme, height=200)
    prev = prev or {}
    try:
        import pandas as pd
        if not df_json or spot_val is None:
            return fig, prev
        df = pd.read_json(df_json, orient='split')
        if df is None or len(df) == 0:
            return fig, prev
        if 'expiration_date' not in df.columns:
            return fig, prev
        spot = float(spot_val)
        # Find front expiry
        exp_series = pd.to_datetime(df['expiration_date'], errors='coerce')
        exp_min = exp_series.min()
        if pd.isna(exp_min):
            return fig, prev
        dff = df[exp_series == exp_min].copy()
        if dff.empty:
            return fig, prev
        # near spot band
        dff['mny'] = (dff['strike_price'].astype(float) - spot) / max(spot, 1e-9)
        dff = dff[dff['mny'].abs() <= 0.10]
        if dff.empty:
            return fig, prev
        # IVs (decimals), choose volatility fields robustly
        import numpy as _np
        c_iv = dff.get('call_volatility', dff.get('call_iv', 0.0)).astype(float).to_numpy()
        p_iv = dff.get('put_volatility', dff.get('put_iv', 0.0)).astype(float).to_numpy()
        # Normalize if in percent
        c_iv = _np.where(c_iv > 3.0, c_iv/100.0, c_iv)
        p_iv = _np.where(p_iv > 3.0, p_iv/100.0, p_iv)
        K = dff['strike_price'].astype(float).to_numpy()
        exp_s = str(exp_min.date()) if hasattr(exp_min, 'date') else str(exp_min)
        # compute deltas vs prev store
        d_c = []
        d_p = []
        for k, civ, piv in zip(K, c_iv, p_iv):
            kc = f"{exp_s}|{k:.2f}|c"
            kp = f"{exp_s}|{k:.2f}|p"
            prev_c = float(prev.get(kc, civ))
            prev_p = float(prev.get(kp, piv))
            d_c.append(float(civ - prev_c))
            d_p.append(float(piv - prev_p))
            prev[kc] = float(civ)
            prev[kp] = float(piv)
        # Convert to vol points (×100)
        d_c_pts = [v*100.0 for v in d_c]
        d_p_pts = [v*100.0 for v in d_p]
        # Build grouped bars
        fig = _empty_fig(theme, height=200)
        fig.add_trace(go.Bar(name='Call ΔIV', x=K, y=d_c_pts, marker_color='#2ecc71'))
        fig.add_trace(go.Bar(name='Put ΔIV', x=K, y=d_p_pts, marker_color='#e74c3c'))
        fig.update_layout(barmode='group',
                          #title=f"ΔIV(front {exp_s})",
                          margin=dict(l=20,r=10,t=28,b=20), showlegend=True,
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title='ΔIV (pts)')
        fig.update_xaxes(title='Strike')
        return fig, prev
    except Exception:
        return fig, prev


if __name__ == "__main__":
    # Fixed port 8052 as requested
    app.run(debug=True, port=8052, use_reloader=False)
