from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class AlpacaRestClient:
    base_url: str
    key_id: str
    secret_key: str

    @classmethod
    def from_env(cls, base_url: str | None = None) -> "AlpacaRestClient":
        resolved_base_url = base_url or os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or ""
        key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if not resolved_base_url:
            raise RuntimeError("Missing APCA_API_BASE_URL/ALPACA_BASE_URL environment variable.")
        if not key_id or not secret_key:
            raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY environment variables.")
        return cls(base_url=resolved_base_url.rstrip("/"), key_id=key_id, secret_key=secret_key)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        import requests

        response = requests.request(method, f"{self.base_url}{path}", headers=self.headers, timeout=30, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path} failed {response.status_code}: {response.text[:1000]}")
        if not response.text:
            return {}
        return response.json()

    def account(self) -> dict[str, Any]:
        return self.request("GET", "/v2/account")

    def clock(self) -> dict[str, Any]:
        return self.request("GET", "/v2/clock")

    def positions(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/v2/positions")
        return data if isinstance(data, list) else []

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        encoded = quote(client_order_id, safe="")
        try:
            return self.request("GET", f"/v2/orders:by_client_order_id?client_order_id={encoded}")
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v2/orders", json=payload)
