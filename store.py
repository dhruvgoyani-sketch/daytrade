#!/usr/bin/env python3
"""
SQLite storage for intraday context metrics (anchor, DTE shares, flow metrics).

Default DB path: daytrade/ctx.db (next to this file).

Schema:
  intraday_ctx(ts TEXT, date TEXT, ticker TEXT,
               spot REAL, b5 REAL, v5 REAL, b15 REAL, v15 REAL,
               signal TEXT, strength REAL,
               dte0_share REAL, dte1_share REAL, anchor REAL)

This module keeps the API tiny and dependency‑free so it can be called from
both UI and CLI code without fuss.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional


def _default_db_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, 'ctx.db')


def _ensure(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_ctx (
            ts TEXT NOT NULL,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            spot REAL,
            b5 REAL, v5 REAL, b15 REAL, v15 REAL,
            signal TEXT, strength REAL,
            dte0_share REAL, dte1_share REAL, anchor REAL
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_intraday_ctx
        ON intraday_ctx(date, ticker, ts)
        """
    )
    conn.commit()


def _ensure_iv_ctx(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS iv_ctx (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiry TEXT NOT NULL,
            atm_iv REAL,
            atm_iv_ch REAL,
            pc_atm REAL,
            rr25 REAL,
            ts_ratio REAL,
            optR REAL,
            optR_ewm REAL,
            updated_ts TEXT,
            PRIMARY KEY (date, ticker, expiry)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_iv_ctx_by_ticker
        ON iv_ctx(date, ticker)
        """
    )
    conn.commit()


def upsert_iv_ctx(
    *,
    date: str,
    ticker: str,
    expiry: str,
    atm_iv: float | None = None,
    atm_iv_ch: float | None = None,
    pc_atm: float | None = None,
    rr25: float | None = None,
    ts_ratio: float | None = None,
    optR: float | None = None,
    optR_ewm: float | None = None,
    ts_iso: str | None = None,
    db_path: str | None = None,
) -> None:
    path = db_path or _default_db_path()
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _ensure_iv_ctx(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO iv_ctx
            (date, ticker, expiry, atm_iv, atm_iv_ch, pc_atm, rr25, ts_ratio, optR, optR_ewm, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker, expiry) DO UPDATE SET
                atm_iv=excluded.atm_iv,
                atm_iv_ch=excluded.atm_iv_ch,
                pc_atm=excluded.pc_atm,
                rr25=excluded.rr25,
                ts_ratio=excluded.ts_ratio,
                optR=excluded.optR,
                optR_ewm=excluded.optR_ewm,
                updated_ts=excluded.updated_ts
            """,
            (
                str(date), str(ticker), str(expiry),
                None if atm_iv is None else float(atm_iv),
                None if atm_iv_ch is None else float(atm_iv_ch),
                None if pc_atm is None else float(pc_atm),
                None if rr25 is None else float(rr25),
                None if ts_ratio is None else float(ts_ratio),
                None if optR is None else float(optR),
                None if optR_ewm is None else float(optR_ewm),
                str(ts_iso or ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_expired_iv_rows(current_date: str, db_path: str | None = None) -> int:
    """Delete iv_ctx rows whose expiry is a real date < current_date (keep 'ALL').
    Returns number of rows deleted.
    """
    path = db_path or _default_db_path()
    conn = sqlite3.connect(path)
    try:
        _ensure_iv_ctx(conn)
        cur = conn.cursor()
        # Only delete rows where expiry looks like YYYY-MM-DD and is older than current_date
        cur.execute(
            """
            DELETE FROM iv_ctx
            WHERE expiry != 'ALL' AND date(expiry) < date(?)
            """,
            (current_date,),
        )
        n = cur.rowcount if hasattr(cur, 'rowcount') else 0
        conn.commit()
        return int(n or 0)
    finally:
        conn.close()



def insert_intraday_ctx(
    *,
    ts_iso: str,
    ticker: str,
    spot: float,
    b5: float,
    v5: float,
    b15: float,
    v15: float,
    signal: str,
    strength: float,
    dte0_share: Optional[float] = None,
    dte1_share: Optional[float] = None,
    anchor: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    """Insert a single intraday context row.

    ts_iso: ISO timestamp in NY (we also derive date=YYYY-MM-DD from it).
    db_path: optional custom DB path; default is daytrade/ctx.db
    """
    try:
        dt = datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))
        date_s = dt.date().isoformat()
    except Exception:
        date_s = ts_iso[:10]
    path = db_path or _default_db_path()
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _ensure(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO intraday_ctx
            (ts, date, ticker, spot, b5, v5, b15, v15, signal, strength,
             dte0_share, dte1_share, anchor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(ts_iso), str(date_s), str(ticker or ''),
                float(spot), float(b5), float(v5), float(b15), float(v15),
                str(signal or ''), float(strength),
                (None if dte0_share is None else float(dte0_share)),
                (None if dte1_share is None else float(dte1_share)),
                (None if anchor is None else float(anchor)),
            ),
        )
        conn.commit()
    finally:
        conn.close()
