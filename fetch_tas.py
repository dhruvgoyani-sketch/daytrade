#!/usr/bin/env python3
"""
Quick TAS fetch example using ConvexApi.make_request.

Usage:
  CONVEX_EMAIL=you@example.com CONVEX_PASSWORD=secret \\
    python3 daytrade/fetch_tas.py --symbol TSLA

Notes:
- Adjust --endpoint/--payload flags if your TAS endpoint expects a different
  shape; this uses a simple symbols/since/limit payload.
- Use the python interpreter that has convexlib installed (e.g., python3.13).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

try:
    from convexlib.api import ConvexApi
except Exception:
    print("convexlib.api not found for this interpreter. Try python3.13 or install convexlib.", file=sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch TAS data via ConvexApi.make_request")
    ap.add_argument("--symbol", default="TSLA", help="Underlying symbol to query (default: TSLA)")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max rows to request if supported by the endpoint (default: 500)")
    ap.add_argument("--endpoint", default="/api/data/tas",
                    help="Endpoint path to call (default: /api/data/tas)")
    ap.add_argument("--email", default=os.environ.get("CONVEX_EMAIL"), help="Convex email (env: CONVEX_EMAIL)")
    ap.add_argument("--password", default=os.environ.get("CONVEX_PASSWORD"), help="Convex password (env: CONVEX_PASSWORD)")
    ap.add_argument("--convex-env", default=os.environ.get("CONVEX_ENV", "live"),
                    help="Convex environment (passed through to ConvexApi, default: live)")
    return ap.parse_args()


def build_payload(symbol: str, limit: int) -> Dict[str, Any]:
    # Minimal payload: just symbols and limit. Add filters if the API supports them.
    return {"symbols": [symbol], "limit": limit}


def main() -> None:
    args = parse_args()
    if not args.email or not args.password:
        raise SystemExit("Missing CONVEX_EMAIL/CONVEX_PASSWORD (env or flags)")

    api = ConvexApi(args.email, args.password, args.convex_env)
    payload = build_payload(args.symbol.upper(), args.limit)

    print(f"Requesting TAS: endpoint={args.endpoint} payload={json.dumps(payload)}")
    try:
        resp = api.make_request(args.endpoint, method="POST", data=payload)
    except Exception as e:
        print(f"TAS request failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception:
        print(resp)


if __name__ == "__main__":
    main()
