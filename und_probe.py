#!/usr/bin/env python3
"""
Quick underlying probe utility.

Calls Convex get_und() for one or more symbols and prints the raw payload
plus a light normalization preview so you can see what the API returns for
each name (price, volm_* etc.).

Usage examples:
  python3 daytrade/und_probe.py SPX
  python3 daytrade/und_probe.py TSLA NVDA --params price volm_buy volm_sell volm_bs volm_und

Env (recommended):
  CONVEX_EMAIL, CONVEX_PASSWORD, CONVEX_ENV (live/pro)
or pass via flags: --email/--password/--convex_env
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Any

try:
    from convexlib.api import ConvexApi
except Exception:
    print("convexlib.api not found. Please install convexlib and retry.", file=sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe Convex get_und() for symbols")
    ap.add_argument("symbols", nargs="+", help="Symbols to query (e.g., SPX TSLA)")
    ap.add_argument("--params", nargs="*",
                    default=["price", "value", "volm_buy", "volm_sell", "volm_bs", "volm_und"],
                    help="Params to request (default: price value volm_buy volm_sell volm_bs volm_und)")
    ap.add_argument("--email", default=os.environ.get("CONVEX_EMAIL"))
    ap.add_argument("--password", default=os.environ.get("CONVEX_PASSWORD"))
    ap.add_argument("--convex_env", default=os.environ.get("CONVEX_ENV", "live"))
    ap.add_argument("--raw", action="store_true", help="Only print raw JSON response")
    return ap.parse_args()


def make_api(args: argparse.Namespace) -> ConvexApi:
    if not args.email or not args.password:
        raise SystemExit("Missing CONVEX_EMAIL/CONVEX_PASSWORD (env or flags)")
    return ConvexApi(args.email, args.password, args.convex_env)


def normalize_preview(payload: Dict[str, Any]) -> List[List[Any]]:
    """Return a normalized preview list of rows for printing.
    Tries to handle both observed shapes:
      {'data': [['SPX', 4500.6, ...]]}
      {'data': [[['SPX', 4500.6]], ...]}
    """
    out: List[List[Any]] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        return out
    first = data[0]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (str, bytes)):
        # Flat rows
        return [list(r) if isinstance(r, (list, tuple)) else [r] for r in data]
    # Nested rows
    for item in data:
        try:
            inner = item[0]
            out.append(list(inner) if isinstance(inner, (list, tuple)) else [inner])
        except Exception:
            continue
    return out


def main() -> None:
    args = parse_args()
    api = make_api(args)

    print(f"Params requested: {args.params}")
    syms = [str(s).strip().upper() for s in args.symbols]
    for sym in syms:
        print("\n=== get_und ===")
        print(f"symbol={sym}")
        try:
            resp = api.get_und(symbols=[sym], params=args.params)  # type: ignore
        except Exception as e:
            print(f"ERROR: get_und failed for {sym}: {e}")
            continue
        print("raw:")
        try:
            print(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception:
            print(str(resp))
        if args.raw:
            continue
        try:
            rows = normalize_preview(resp if isinstance(resp, dict) else {})
            if rows:
                print("preview rows (first 5):")
                for r in rows[:5]:
                    print("  ", r)
            else:
                print("preview rows: (none)")
        except Exception as e:
            print(f"preview error: {e}")


if __name__ == "__main__":
    main()

