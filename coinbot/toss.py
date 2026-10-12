"""
Toss Securities Open API client: auth, generic GET, and a candle fetcher
with a parquet cache.

Toss stamps 1m candles with their CLOSE time ([timestamp - 1m, timestamp))
and 1d candles with local midnight of the trading day. This module adds
`open_time`/`close_time` (UTC) and drops unfinished candles. Minutes with no
trades are NOT filled: for stocks, gaps (overnight, halts, illiquid minutes)
are real and flat filler bars would be fiction.

Only one access token is valid per client; issuing a new one revokes the old.
The token is cached on disk so separate processes share it instead of
revoking each other.

Data from this API is for the account holder's own trading only; the terms
forbid distributing it to third parties, even non-commercially.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://openapi.tossinvest.com"
ROOT = Path(__file__).resolve().parents[1]
CRED_FILE = ROOT / "toss_api.txt"
CACHE_DIR = ROOT / "data" / "toss"
TOKEN_FILE = CACHE_DIR / "token.json"
MAX_COUNT = 200          # Toss hard limit per candle request
REQUEST_DELAY = 0.06     # MARKET_DATA_CHART group allows 20 req/s per client
SETTLE = pd.Timedelta(seconds=10)  # a just-closed candle can still change for a few seconds

# interval label -> bar length
INTERVALS = {
    "1m": pd.Timedelta(minutes=1),
    "1d": pd.Timedelta(days=1),
}

COLUMNS = ["open_time", "close_time", "open", "high", "low", "close", "volume"]


class TossError(RuntimeError):
    pass


def credentials() -> tuple[str, str]:
    """TOSS_CLIENT_ID / TOSS_CLIENT_SECRET env vars, else toss_api.txt (label line, value line)."""
    cid, secret = os.environ.get("TOSS_CLIENT_ID"), os.environ.get("TOSS_CLIENT_SECRET")
    if cid and secret:
        return cid, secret
    lines = [ln.strip() for ln in CRED_FILE.read_text().splitlines() if ln.strip()]
    values = {lines[i].lower(): lines[i + 1] for i in range(0, len(lines) - 1, 2)}
    return values["client_id"], values["client_secret"]


def _read_cached_token() -> str | None:
    try:
        cached = json.loads(TOKEN_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if pd.Timestamp(cached["expires_at"]) - pd.Timedelta(minutes=5) <= pd.Timestamp.now(tz="UTC"):
        return None
    return cached["access_token"]


def _issue_token() -> str:
    cid, secret = credentials()
    resp = requests.post(f"{BASE_URL}/oauth2/token", timeout=15, data={
        "grant_type": "client_credentials", "client_id": cid, "client_secret": secret})
    if resp.status_code != 200:
        raise TossError(f"token issue failed: {resp.status_code} {resp.text}")
    body = resp.json()
    expires_at = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=body["expires_in"])
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"access_token": body["access_token"],
                                      "expires_at": expires_at.isoformat()}))
    TOKEN_FILE.chmod(0o600)
    logger.info("issued new Toss access token")
    return body["access_token"]


def token(rejected: str | None = None) -> str:
    """
    A valid access token. Pass the token the server just rejected to force a
    new one, unless another process has already replaced it in the cache.
    """
    cached = _read_cached_token()
    if cached and cached != rejected:
        return cached
    return _issue_token()


def get(path: str, **params) -> dict | list:
    """GET an API path and return the `result` payload, retrying 429/5xx/network errors."""
    tok = token()
    for attempt in range(8):
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=15,
                                headers={"Authorization": f"Bearer {tok}"})
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning(f"{path}: {type(exc).__name__}, retrying")
            time.sleep(min(30.0, 2.0 ** attempt))
            continue
        if resp.status_code == 401:
            tok = token(rejected=tok)
            continue
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 1)))
            continue
        if resp.status_code >= 500:
            time.sleep(min(30.0, 2.0 ** attempt))
            continue
        if resp.status_code != 200:
            # 403 here usually means this IP is not in the Open API allow-list
            raise TossError(f"{path} {params}: {resp.status_code} {resp.text}")
        return resp.json()["result"]
    raise TossError(f"request failed after retries: {path} {params}")


def _get_page(symbol: str, interval: str, before: str | None) -> dict:
    """One request: up to 200 candles at or before `before` (newest first) plus `nextBefore`."""
    params = {"symbol": symbol, "interval": interval, "count": MAX_COUNT}
    if before is not None:
        params["before"] = before
    return get("/api/v1/candles", **params)


def to_frame(raw: list[dict], interval: str) -> pd.DataFrame:
    """Convert raw Toss candles to a sorted frame with open/close times (UTC)."""
    if not raw:
        return pd.DataFrame(columns=COLUMNS)
    stamp = pd.to_datetime([c["timestamp"] for c in raw], utc=True)
    step = INTERVALS[interval]
    open_time = stamp - step if interval == "1m" else stamp
    df = pd.DataFrame({
        "open_time": open_time,
        "open": [float(c["openPrice"]) for c in raw],
        "high": [float(c["highPrice"]) for c in raw],
        "low": [float(c["lowPrice"]) for c in raw],
        "close": [float(c["closePrice"]) for c in raw],
        "volume": [float(c["volume"]) for c in raw],
    })
    # A daily bar includes after-market trades, so treat it as known only at the next local midnight.
    df["close_time"] = df["open_time"] + step
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[COLUMNS]


def fetch(symbol: str, interval: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Page backwards from now until `since` (or the start of history)."""
    pages: list[dict] = []
    before = None
    while True:
        page = _get_page(symbol, interval, before)
        pages.extend(page["candles"])
        before = page.get("nextBefore")
        if before is None or not page["candles"]:
            break
        if since is not None and to_frame(page["candles"], interval)["open_time"].iloc[0] <= since:
            break
        time.sleep(REQUEST_DELAY)
    df = to_frame(pages, interval)
    if since is not None:
        df = df[df["open_time"] >= since]
    return df.reset_index(drop=True)


def load(symbol: str, interval: str = "1m", refresh: bool = False,
         start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Closed-bar history from cache, fetching only bars newer than the cache.
    Without a cache, history begins at `start` (default: all available).
    Prices are split/dividend adjusted, so a corporate action rewrites history:
    if the overlapping bar no longer matches the cache, everything is refetched.
    """
    path = CACHE_DIR / f"{symbol}_{interval}.parquet"
    df = None
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        last = cached.iloc[-1]
        new = fetch(symbol, interval, since=last["open_time"])
        overlap = new[new["open_time"] == last["open_time"]]
        if len(overlap) and overlap["close"].iloc[0] == last["close"]:
            df = pd.concat([cached, new]).drop_duplicates("open_time", keep="last")
        else:
            logger.info(f"{symbol} {interval}: cached prices no longer match (adjustment?), refetching")
    if df is None:
        logger.info(f"Fetching {symbol} {interval} history since {start or 'the beginning'}")
        df = fetch(symbol, interval, since=start)
    df = df.sort_values("open_time")
    df = df[df["close_time"] <= pd.Timestamp.now(tz="UTC") - SETTLE].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
