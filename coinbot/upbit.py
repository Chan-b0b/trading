"""
Upbit candle fetcher with a parquet cache.

Upbit stamps candles with their OPEN time and returns the still-forming
candle as the newest row. This module adds `close_time`, drops unfinished
candles, and fills hours with no trades (Upbit omits those) so every row
is a bar that was fully known at `close_time`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upbit.com/v1/candles"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "upbit"
MAX_COUNT = 200          # Upbit hard limit per request
REQUEST_DELAY = 0.12     # candle API allows 10 req/s per IP
SETTLE = pd.Timedelta(seconds=10)  # a just-closed candle can still change for a few seconds

# interval label -> (endpoint path, bar length)
INTERVALS = {
    "5m": ("minutes/5", pd.Timedelta(minutes=5)),
    "1h": ("minutes/60", pd.Timedelta(hours=1)),
    "4h": ("minutes/240", pd.Timedelta(hours=4)),
    "1d": ("days", pd.Timedelta(days=1)),
}

COLUMNS = ["open_time", "close_time", "open", "high", "low", "close", "volume", "value"]


def _get_page(market: str, interval: str, to: pd.Timestamp | None) -> list[dict]:
    """One request: up to 200 candles strictly before `to` (newest first)."""
    params = {"market": market, "count": MAX_COUNT}
    if to is not None:
        params["to"] = to.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{BASE_URL}/{INTERVALS[interval][0]}"
    for attempt in range(8):
        try:
            resp = requests.get(url, params=params, timeout=15)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning(f"{interval}: {type(exc).__name__}, retrying")
            time.sleep(min(30.0, 2.0 ** attempt))
            continue
        if resp.status_code == 429:
            time.sleep(1.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Upbit candle request failed after retries: {url} {params}")


def to_frame(raw: list[dict], interval: str) -> pd.DataFrame:
    """Convert raw Upbit candles to a sorted frame with open/close times (UTC)."""
    if not raw:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame({
        "open_time": pd.to_datetime([c["candle_date_time_utc"] for c in raw], utc=True),
        "open": [float(c["opening_price"]) for c in raw],
        "high": [float(c["high_price"]) for c in raw],
        "low": [float(c["low_price"]) for c in raw],
        "close": [float(c["trade_price"]) for c in raw],
        "volume": [float(c["candle_acc_trade_volume"]) for c in raw],
        "value": [float(c["candle_acc_trade_price"]) for c in raw],
    })
    df["close_time"] = df["open_time"] + INTERVALS[interval][1]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[COLUMNS]


def finalize(df: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Drop bars not yet closed at `now` and fill bars Upbit skipped (no trades)."""
    df = df[df["close_time"] <= now]
    if df.empty:
        return df.reset_index(drop=True)
    step = INTERVALS[interval][1]
    full = pd.date_range(df["open_time"].iloc[0], df["open_time"].iloc[-1], freq=step)
    missing = len(full) - len(df)
    if missing:
        logger.info(f"{interval}: filling {missing} bars with no trades")
    df = df.set_index("open_time").reindex(full)
    df.index.name = "open_time"
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df[["volume", "value"]] = df[["volume", "value"]].fillna(0.0)
    df = df.reset_index()
    df["close_time"] = df["open_time"] + step
    return df[COLUMNS]


def fetch(market: str, interval: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Page backwards from now until `since` (or the start of history)."""
    pages: list[dict] = []
    to = None
    while True:
        page = _get_page(market, interval, to)
        pages.extend(page)
        if len(page) < MAX_COUNT:
            break
        oldest = pd.Timestamp(min(c["candle_date_time_utc"] for c in page), tz="UTC")
        if since is not None and oldest <= since:
            break
        to = oldest
        time.sleep(REQUEST_DELAY)
    df = to_frame(pages, interval)
    if since is not None:
        df = df[df["open_time"] >= since]
    return df.reset_index(drop=True)


def load(market: str = "KRW-BTC", interval: str = "1h", refresh: bool = False) -> pd.DataFrame:
    """Full closed-bar history from cache, fetching only bars newer than the cache."""
    path = CACHE_DIR / f"{market}_{interval}.parquet"
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        new = fetch(market, interval, since=cached["open_time"].iloc[-1])
        df = pd.concat([cached, new]).drop_duplicates("open_time", keep="last")
    else:
        logger.info(f"Fetching full {market} {interval} history")
        df = fetch(market, interval)
    df = finalize(df.sort_values("open_time"), interval, now=pd.Timestamp.now(tz="UTC") - SETTLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
