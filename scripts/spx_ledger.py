#!/usr/bin/env python3
"""
SPX ledger maintenance
======================

This script keeps a lightweight dealer/customer exposure ledger for SPX 0DTE
strikes.  It stores the running customer positioning alongside the raw chain
data in ``pg_database_spx.db`` and supports two workflows:

* ``--mode open``  (default 08:00 ET): fetch fresh OI from Convex, blend the
  prior closing positions down to the new OI, and seed the opening baseline.
* ``--mode close`` (default 17:05 ET): combine the opening baseline with the
  day’s signed flow (volmbs) to produce a closing ledger.

Both workflows persist their results in a dedicated helper table
``spx_ledger`` *and* mirror the latest values back onto the ``SPX`` chain rows
when they exist (so downstream tooling can query one place).
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from convexlib.api import ConvexApi
except Exception as exc:  # pragma: no cover - convenience message
    raise SystemExit("convexlib.api not available. Install convexlib first.") from exc


DEFAULT_DB_PATH = os.environ.get(
    "SPX_LEDGER_DB",
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "pg_database_spx.db")
    ),
)
DEFAULT_EMAIL = os.environ.get("CONVEX_EMAIL")
DEFAULT_PASSWORD = os.environ.get("CONVEX_PASSWORD")
DEFAULT_ENV = os.environ.get("CONVEX_ENV", "live")


# ---------------------------------------------------------------------------
# Helpers


def _ny_now() -> datetime:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0)


def _today_ny() -> date:
    return _ny_now().date()


def _date_to_code(val: date) -> str:
    return val.strftime("%y%m%d")


def _today_code() -> str:
    return _date_to_code(_today_ny())


def _trade_date_code(val: date) -> str:
    return _date_to_code(val)


def _clamp(val: float, low: float, high: float) -> float:
    if val < low:
        return low
    if val > high:
        return high
    return val


def _to_float(val: Any) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _parse_expiration(exp_val: Any) -> Optional[datetime]:
    if exp_val is None:
        return None
    if isinstance(exp_val, datetime):
        return exp_val
    if isinstance(exp_val, (int, float)):
        ts = float(exp_val)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(exp_val, str):
        txt = exp_val.strip()
        if not txt:
            return None
        try:
            # allow both "YYYY-mm-ddTHH:MM:SS" and with timezone offset
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except ValueError:
            try:
                ts = float(txt)
                return _parse_expiration(ts)
            except (TypeError, ValueError):
                return None
    return None


def _exp_to_iso(exp_dt: Optional[datetime]) -> Optional[str]:
    if exp_dt is None:
        return None
    if exp_dt.tzinfo is None:
        return exp_dt.strftime("%Y-%m-%dT%H:%M:%S")
    return exp_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# SQLite schema management


LEDGER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spx_ledger (
    trade_date TEXT NOT NULL,
    expiration TEXT NOT NULL,
    strike_price REAL NOT NULL,
    cust_call_open REAL DEFAULT 0,
    cust_put_open REAL DEFAULT 0,
    cust_call_close REAL DEFAULT 0,
    cust_put_close REAL DEFAULT 0,
    call_oi_open REAL DEFAULT 0,
    put_oi_open REAL DEFAULT 0,
    call_oi_close REAL DEFAULT 0,
    put_oi_close REAL DEFAULT 0,
    open_ts TEXT,
    close_ts TEXT,
    PRIMARY KEY (trade_date, expiration, strike_price)
);
"""


SPX_EXTRA_COLUMNS: Dict[str, str] = {
    "cust_call_pos": "REAL DEFAULT 0",
    "cust_put_pos": "REAL DEFAULT 0",
    "call_oi_open": "REAL DEFAULT 0",
    "put_oi_open": "REAL DEFAULT 0",
    "call_oi_close": "REAL DEFAULT 0",
    "put_oi_close": "REAL DEFAULT 0",
    "ledger_ts": "TEXT",
    "recon_ts": "TEXT"
}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(LEDGER_TABLE_SQL)
    cur = conn.execute("PRAGMA table_info('SPX')")
    existing = {row[1] for row in cur.fetchall()}
    for col, ddl in SPX_EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE SPX ADD COLUMN {col} {ddl}")
    conn.commit()


# ---------------------------------------------------------------------------
# Chain flattening for OI snapshots


@dataclass
class ChainRow:
    strike: float
    kind: str  # 'C' or 'P'
    symbol: Optional[str]
    oi: float
    volm_bs: float
    expiration: Optional[datetime]


def _flatten_chain_for_oi(raw: Dict[str, Any],
                          params: List[str]) -> List[ChainRow]:
    idx = {name: i + 1 for i, name in enumerate(params)}
    rows: List[ChainRow] = []

    def vec_to_row(strike_val: Any, vec: Any, kind: str) -> Optional[ChainRow]:
        if not isinstance(vec, (list, tuple)) or len(vec) == 0:
            return None
        strike = _to_float(strike_val)
        symbol = vec[0] if isinstance(vec[0], str) else None
        oi = _to_float(vec[idx.get("oi", -1)]) if idx.get("oi") is not None and idx["oi"] < len(vec) else 0.0
        volm = _to_float(vec[idx.get("volm_bs", -1)]) if idx.get("volm_bs") is not None and idx["volm_bs"] < len(vec) else 0.0
        exp_raw = vec[idx.get("expiration_ts", -1)] if idx.get("expiration_ts") is not None and idx["expiration_ts"] < len(vec) else None
        exp_dt = _parse_expiration(exp_raw)
        return ChainRow(strike=strike, kind=kind, symbol=symbol, oi=oi, volm_bs=volm, expiration=exp_dt)

    data = raw.get("data")
    if not isinstance(data, list):
        return rows
    for container in data:
        chain = container.get("chain") if isinstance(container, dict) else None
        if not isinstance(chain, list):
            continue
        for entry in chain:
            # entry may be [day_id, [...], ...] or [strike, call_vec, put_vec]
            option_groups: Iterable[Any] = []
            if isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], list) and isinstance(entry[0], (int, float)):
                option_groups = entry[1]
            elif isinstance(entry, list) and len(entry) >= 3:
                option_groups = [entry]
            else:
                continue
            for og in option_groups:
                if not isinstance(og, list) or len(og) < 3:
                    continue
                strike_val, call_vec, put_vec = og[0], og[1], og[2]
                c_row = vec_to_row(strike_val, call_vec, "C")
                p_row = vec_to_row(strike_val, put_vec, "P")
                if c_row is not None:
                    rows.append(c_row)
                if p_row is not None:
                    rows.append(p_row)
    return rows


def fetch_open_chain(email: str, password: str, env: str, exp_count: int = 7) -> List[ChainRow]:
    api = ConvexApi(email, password, env)
    params = ["oi", "volm_bs", "expiration_ts"]
    exps = list(range(0, max(1, exp_count)))
    raw = api.get_chain("SPX", params=params, exps=exps, rng=100)  # type: ignore
    return _flatten_chain_for_oi(raw, params)


# ---------------------------------------------------------------------------
# Ledger operations


def upsert_open_ledger(conn: sqlite3.Connection,
                       trade_date: date,
                       rows: List[ChainRow]) -> None:
    """
    Create/overwrite the opening baseline for ``trade_date``.
    """
    today_date = trade_date
    trade_date_code = _date_to_code(trade_date)
    call_map: Dict[Tuple[float, str], ChainRow] = {}
    put_map: Dict[Tuple[float, str], ChainRow] = {}

    for row in rows:
        if row.expiration is None or row.expiration.date() != today_date:
            continue
        exp_iso = _exp_to_iso(row.expiration)
        key = (row.strike, exp_iso or "")
        if row.kind == "C":
            call_map[key] = row
        else:
            put_map[key] = row

    # Load previous day's closing ledger (if available)
    cur = conn.cursor()
    existing_rows = cur.execute(
        """
        SELECT strike_price, expiration,
               cust_call_open, cust_put_open,
               cust_call_close, cust_put_close,
               call_oi_close, put_oi_close
        FROM spx_ledger
        WHERE trade_date = ?
        """,
        (trade_date_code,)
    ).fetchall()

    existing_map: Dict[Tuple[float, str], Tuple[float, float, float, float]] = {}
    for strike_price, expiration, c_open_prev, p_open_prev, c_close_prev, p_close_prev, c_oi_close_prev, p_oi_close_prev in existing_rows:
        key = (float(strike_price), expiration)
        base_call = _to_float(c_close_prev) if _to_float(c_close_prev) != 0.0 else _to_float(c_open_prev)
        base_put = _to_float(p_close_prev) if _to_float(p_close_prev) != 0.0 else _to_float(p_open_prev)
        existing_map[key] = (base_call, base_put, _to_float(c_oi_close_prev), _to_float(p_oi_close_prev))

    if not existing_map:
        prev_trade_date_row = cur.execute(
            "SELECT trade_date FROM spx_ledger WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1",
            (trade_date_code,)
        ).fetchone()
        prev_date = prev_trade_date_row[0] if prev_trade_date_row else None

        if prev_date:
            prev_rows = cur.execute(
                """
                SELECT strike_price, expiration, cust_call_close, cust_put_close,
                       call_oi_close, put_oi_close
                FROM spx_ledger
                WHERE trade_date = ?
                """,
                (prev_date,)
            ).fetchall()
        else:
            prev_rows = []

        for strike_price, expiration, c_close, p_close, c_oi_close, p_oi_close in prev_rows:
            existing_map[(float(strike_price), expiration)] = (
                _to_float(c_close),
                _to_float(p_close),
                _to_float(c_oi_close),
                _to_float(p_oi_close),
            )

    now_iso = _ny_now().isoformat()
    inserts: List[Tuple[Any, ...]] = []

    for key, call_info in call_map.items():
        strike, exp_iso = key
        put_info = put_map.get(key)
        prev = existing_map.get(key)
        call_oi_open = call_info.oi
        put_oi_open = put_info.oi if put_info else 0.0
        if prev:
            c_prev, p_prev, c_oi_prev, p_oi_prev = prev
            c_ratio = (call_oi_open / c_oi_prev) if c_oi_prev and c_oi_prev > 0 else 1.0
            p_ratio = (put_oi_open / p_oi_prev) if p_oi_prev and p_oi_prev > 0 else 1.0
            cust_call_open = c_prev * c_ratio
            cust_put_open = p_prev * p_ratio
            if call_oi_open > 0:
                cust_call_open = _clamp(cust_call_open, -call_oi_open, call_oi_open)
            if put_oi_open > 0:
                cust_put_open = _clamp(cust_put_open, -put_oi_open, put_oi_open)
        else:
            cust_call_open = 0.0
            cust_put_open = 0.0
        inserts.append((
            trade_date_code,
            exp_iso,
            strike,
            cust_call_open,
            cust_put_open,
            0.0,
            0.0,
            call_oi_open,
            put_oi_open,
            0.0,
            0.0,
            now_iso,
            None,
        ))

    # Insert rows; replace on conflict
    cur.executemany(
        """
        INSERT INTO spx_ledger (
            trade_date, expiration, strike_price,
            cust_call_open, cust_put_open,
            cust_call_close, cust_put_close,
            call_oi_open, put_oi_open,
            call_oi_close, put_oi_close,
            open_ts, close_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_date, expiration, strike_price)
        DO UPDATE SET
            cust_call_open=excluded.cust_call_open,
            cust_put_open=excluded.cust_put_open,
            call_oi_open=excluded.call_oi_open,
            put_oi_open=excluded.put_oi_open,
            open_ts=excluded.open_ts
        """,
        inserts,
    )

    # Mirror into SPX table when rows for today already exist
    cur.executemany(
        """
        UPDATE SPX
        SET
            cust_call_pos = :cust_call_open,
            cust_put_pos  = :cust_put_open,
            call_oi_open  = :call_oi_open,
            put_oi_open   = :put_oi_open,
            recon_ts      = :ts
        WHERE date = :trade_date
          AND ABS(strike_price - :strike_price) < 1e-6
          AND expiration = :expiration
        """,
        [
            {
                "cust_call_open": row[3],
                "cust_put_open": row[4],
                "call_oi_open": row[7],
                "put_oi_open": row[8],
                "ts": now_iso,
            "trade_date": trade_date_code,
            "strike_price": row[2],
            "expiration": row[1],
        }
        for row in inserts
        ],
    )

    conn.commit()


def build_closing_ledger(conn: sqlite3.Connection,
                         trade_date: date,
                         data_date_code: str) -> None:
    """
    Compute closing exposures by combining opening ledger rows and the daily
    ``SPX`` chain snapshot (call_volm_bs / put_volm_bs).
    """
    cur = conn.cursor()
    trade_date_code = _date_to_code(trade_date)
    ledger_rows = cur.execute(
        """
        SELECT strike_price, expiration, cust_call_open, cust_put_open,
               call_oi_open, put_oi_open
        FROM spx_ledger
        WHERE trade_date = ?
        """,
        (trade_date_code,)
    ).fetchall()
    if not ledger_rows:
        raise SystemExit(f"No opening ledger found for trade_date={trade_date_code}. Run --mode open first.")

    spx_rows = cur.execute(
        """
        SELECT strike_price, expiration,
               call_volm_bs, put_volm_bs,
               call_oi, put_oi,
               rowid
        FROM SPX
        WHERE date = ?
        """,
        (data_date_code,)
    ).fetchall()
    if not spx_rows:
        raise SystemExit(
            f"No SPX chain rows found for date={trade_date}. "
            "Run convexfetchchain.py before the closing ledger job."
        )

    spx_map: Dict[Tuple[float, str], Tuple[float, float, float, float, int]] = {}
    spx_map_by_date: Dict[Tuple[float, str], Tuple[float, float, float, float, int]] = {}
    for strike_price, expiration, call_volm_bs, put_volm_bs, call_oi, put_oi, rowid in spx_rows:
        key = (float(strike_price), expiration)
        spx_map[key] = (
            _to_float(call_volm_bs),
            _to_float(put_volm_bs),
            _to_float(call_oi),
            _to_float(put_oi),
            int(rowid),
        )
        if isinstance(expiration, str) and "T" in expiration:
            exp_date = expiration.split("T", 1)[0]
            spx_map_by_date[(float(strike_price), exp_date)] = spx_map[key]

    now_iso = _ny_now().isoformat()
    updates_ledger: List[Tuple[float, float, float, float, float, float, str, float]] = []
    updates_spx: List[Tuple[float, float, float, float, str, int]] = []

    for strike_price, expiration, cust_call_open, cust_put_open, call_oi_open, put_oi_open in ledger_rows:
        key = (float(strike_price), expiration)
        spx_info = spx_map.get(key)
        if not spx_info:
            exp_text = expiration if isinstance(expiration, str) else str(expiration)
            exp_date = exp_text.split("T", 1)[0]
            spx_info = spx_map_by_date.get((float(strike_price), exp_date))
        if not spx_info:
            # no matching flow rows found; skip
            continue
        call_flow, put_flow, call_oi_close, put_oi_close, rowid = spx_info
        call_new = _to_float(cust_call_open) + call_flow
        put_new = _to_float(cust_put_open) + put_flow

        updates_ledger.append((
            call_new,
            put_new,
            call_oi_close,
            put_oi_close,
            now_iso,
            strike_price,
            expiration,
            trade_date_code,
        ))

        updates_spx.append((
            call_new,
            put_new,
            call_oi_close,
            put_oi_close,
            now_iso,
            rowid,
        ))

    if updates_ledger:
        cur.executemany(
            """
            UPDATE spx_ledger
            SET
                cust_call_close = ?,
                cust_put_close  = ?,
                call_oi_close   = ?,
                put_oi_close    = ?,
                close_ts        = ?
            WHERE strike_price = ?
              AND expiration   = ?
              AND trade_date   = ?
            """,
            updates_ledger,
        )

    if updates_spx:
        cur.executemany(
            """
            UPDATE SPX
            SET
                cust_call_pos = ?,
                cust_put_pos  = ?,
                call_oi_close = ?,
                put_oi_close  = ?,
                ledger_ts     = ?
            WHERE rowid = ?
            """,
            updates_spx,
        )

    conn.commit()


def _parse_date_arg(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    txt = value.strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt).date()
    except ValueError:
        try:
            return datetime.strptime(txt, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {txt}") from exc


def determine_target_date(conn: sqlite3.Connection, mode: str, explicit: Optional[str]) -> date:
    explicit_date = _parse_date_arg(explicit)
    if explicit_date is not None:
        return explicit_date

    today_date = _today_ny()
    if mode == "open":
        return today_date

    # mode == close: pick next available expiration after today
    today_code = _date_to_code(today_date)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT expiration FROM SPX WHERE date = ?",
        (today_code,)
    ).fetchall()
    expiries: List[date] = []
    for (exp,) in rows:
        exp_dt = _parse_expiration(exp)
        if exp_dt is None:
            continue
        expiries.append(exp_dt.date())
    expiries = sorted({d for d in expiries if d >= today_date})
    for candidate in expiries:
        if candidate > today_date:
            return candidate
    # fallback: use today
    return today_date


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Maintain SPX customer/dealer ledger.")
    ap.add_argument("--mode", choices=["open", "close", "auto"], default="auto",
                    help="Which workflow to run (default: auto based on time).")
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help="Path to pg_database_spx.db (default: %(default)s).")
    ap.add_argument("--email", default=DEFAULT_EMAIL)
    ap.add_argument("--password", default=DEFAULT_PASSWORD)
    ap.add_argument("--convex-env", default=DEFAULT_ENV)
    ap.add_argument("--exp-count", type=int, default=7,
                    help="Number of expirations to request when fetching OI.")
    ap.add_argument("--target-date", default=None,
                    help="Trade date (YYYY-mm-dd) to prepare (default: today for open, next expiry for close).")
    ap.add_argument("--chain-date", default=None,
                    help="Date (YYYY-mm-dd) of SPX chain rows to use for closing (default: today).")
    return ap.parse_args()


def decide_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    hour = _ny_now().hour
    # rough heuristic: before noon => open, otherwise close
    return "open" if hour < 12 else "close"


def main() -> None:
    args = parse_args()
    mode = decide_mode(args.mode)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    target_date = determine_target_date(conn, mode, args.target_date)
    chain_date_arg = _parse_date_arg(args.chain_date)
    chain_date = chain_date_arg or (target_date if mode == "open" else _today_ny())

    if mode == "open":
        if not args.email or not args.password:
            raise SystemExit("Convex credentials required for --mode open.")
        chain_rows = fetch_open_chain(args.email, args.password, args.convex_env, args.exp_count)
        upsert_open_ledger(conn, target_date, chain_rows)
        print(f"[ledger] Opening baseline stored for {target_date.isoformat()} (rows={len(chain_rows)}).")
    else:
        data_date_code = _trade_date_code(chain_date) or _today_code()
        build_closing_ledger(conn, target_date, data_date_code)
        print(f"[ledger] Closing ledger updated for {target_date.isoformat()} using chain date {data_date_code}.")

    conn.close()


if __name__ == "__main__":
    main()
