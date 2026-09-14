"""Minimal Tradier REST client for equity orders."""
from __future__ import annotations

import requests
from typing import Optional


class TradierClient:
    """Lightweight wrapper around Tradier accounts/orders endpoints."""

    def __init__(self, token: str, account_id: str, sandbox: bool = True):
        if not token or not account_id:
            raise ValueError("Tradier token and account_id are required")
        self.base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
        self.account_id = account_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            }
        )

    def place_equity_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        duration: str = "day",
        tag: Optional[str] = None,
    ) -> dict:
        """Submit an equity order to Tradier.

        Returns a dict with keys `ok`, `response`, and optionally `error`.
        """
        payload = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": int(quantity),
            "type": order_type,
            "duration": duration,
        }
        if tag:
            payload["tag"] = tag
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        resp = self.session.post(url, data=payload, timeout=10)
        result: dict = {"ok": resp.ok, "status_code": resp.status_code}
        try:
            result["response"] = resp.json()
        except ValueError:
            result["response"] = resp.text
        if not resp.ok:
            result["error"] = resp.text
        return result
