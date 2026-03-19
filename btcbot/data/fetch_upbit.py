"""
Upbit OHLCV Candle Fetcher for KRW-BTC

Upbit public REST API returns max 200 candles per call.
This module paginates automatically to fetch arbitrarily long histories
and caches the result as a Parquet file.

Supported timeframes (Upbit API endpoint names):
    minutes1, minutes3, minutes5, minutes10, minutes15, minutes30,
    minutes60, minutes240, days, weeks, months
"""

import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from btcbot.config import DATA_DIR, SYMBOL

logger = logging.getLogger(__name__)

UPBIT_BASE = "https://api.upbit.com/v1"
MAX_PER_REQUEST = 200   # Upbit API hard limit
DEFAULT_DELAY    = 0.12  # seconds between requests (~8 req/s, under 10 req/s limit)


# ── Low-level candle API call ────────────────────────────────────────────────

def _fetch_page(
    tf: str,
    to: str | None = None,
    count: int = 200,
    market: str = SYMBOL,
) -> list[dict]:
    """
    Fetch one page (up to 200 candles) from the Upbit REST API.

    Args:
        tf:     Timeframe endpoint fragment, e.g. "minutes60" or "days"
        to:     ISO-8601 datetime string; fetch candles ending AT or BEFORE this time.
                Pass None to get the most recent candles.
        count:  Number of candles (1-200)
        market: Upbit market code, e.g. "KRW-BTC"

    Returns:
        List of raw candle dicts from Upbit
    """
    # Build URL:  /candles/minutes60  OR  /candles/days
    if tf.startswith("minutes"):
        unit = tf.replace("minutes", "")
        url = f"{UPBIT_BASE}/candles/minutes/{unit}"
    else:
        url = f"{UPBIT_BASE}/candles/{tf}"

    params: dict = {"market": market, "count": min(count, MAX_PER_REQUEST)}
    if to:
        params["to"] = to

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── Parse candles → DataFrame ────────────────────────────────────────────────

def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert Upbit candle list to a tidy DataFrame."""
    rows = []
    for c in candles:
        rows.append({
            "time":   pd.to_datetime(c["candle_date_time_utc"], utc=True),
            "open":   float(c["opening_price"]),
            "high":   float(c["high_price"]),
            "low":    float(c["low_price"]),
            "close":  float(c["trade_price"]),
            "volume": float(c["candle_acc_trade_volume"]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ── Paginated history fetch ──────────────────────────────────────────────────

def fetch_ohlcv(
    tf: str = "minutes60",
    n_candles: int = 2000,
    market: str = SYMBOL,
    delay: float = DEFAULT_DELAY,
) -> pd.DataFrame:
    """
    Fetch N candles of OHLCV history for *market* at timeframe *tf*.

    Pages backwards through time until n_candles are collected.

    Args:
        tf:         Upbit timeframe string (e.g. "minutes60", "days")
        n_candles:  Total number of candles desired
        market:     Upbit market code
        delay:      Sleep between paginated requests (rate-limit safety)

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
                sorted ascending by time, deduplicated.
    """
    all_candles: list[dict] = []
    to: str | None = None          # start from most recent

    logger.info(f"Fetching {n_candles} candles  tf={tf}  market={market}")

    while len(all_candles) < n_candles:
        remaining = n_candles - len(all_candles)
        page = _fetch_page(tf, to=to, count=min(remaining, MAX_PER_REQUEST), market=market)
        if not page:
            logger.warning("Empty page received; stopping pagination.")
            break

        all_candles.extend(page)
        logger.debug(f"  fetched {len(page)} candles  total={len(all_candles)}")

        # Oldest candle in this page → use as next `to` cursor
        oldest_time = min(c["candle_date_time_utc"] for c in page)
        to = oldest_time   # Upbit returns candles STRICTLY BEFORE `to`

        if len(page) < MAX_PER_REQUEST:
            break   # reached beginning of history

        time.sleep(delay)

    df = _candles_to_df(all_candles)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    logger.info(f"  ✅  {len(df)} candles  {df['time'].iloc[0]}  →  {df['time'].iloc[-1]}")
    return df


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(tf: str, market: str = SYMBOL) -> Path:
    safe_market = market.replace("-", "_")
    return DATA_DIR / f"{safe_market}_{tf}.parquet"


def load_or_fetch(
    tf: str = "minutes60",
    n_candles: int = 2000,
    market: str = SYMBOL,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load cached candles from disk, or fetch (and cache) if missing / refresh=True.

    When the cache already exists the function only fetches the delta
    (candles newer than the last cached timestamp) and appends them.

    Args:
        tf:        Upbit timeframe string
        n_candles: Minimum history to keep
        market:    Upbit market code
        refresh:   Force full re-fetch ignoring cache

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    path = _cache_path(tf, market)

    if not refresh and path.exists():
        df_cached = pd.read_parquet(path)
        last_ts = df_cached["time"].max()
        logger.info(f"Cache hit: {path.name}  last={last_ts}")

        # Fetch only incremental update (up to 200 new candles)
        try:
            new_page = _fetch_page(tf, to=None, count=MAX_PER_REQUEST, market=market)
            df_new = _candles_to_df(new_page)
            df_new = df_new[df_new["time"] > last_ts]
            if not df_new.empty:
                logger.info(f"  Appending {len(df_new)} new candles")
                df_cached = pd.concat([df_cached, df_new], ignore_index=True)
                df_cached = df_cached.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Incremental update failed: {exc}")

        df_cached.to_parquet(path, index=False)
        return df_cached

    # Full fetch
    df = fetch_ohlcv(tf=tf, n_candles=n_candles, market=market)
    df.to_parquet(path, index=False)
    logger.info(f"Cached → {path}")
    return df


# ── Multi-timeframe convenience ───────────────────────────────────────────────

def fetch_all_timeframes(
    timeframes: dict | None = None,
    n_candles: int = 2000,
    market: str = SYMBOL,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all configured timeframes and return a dict of DataFrames.

    Args:
        timeframes: {label: upbit_tf_string}  defaults to btcbot.config.TIMEFRAMES
        n_candles:  candles per timeframe
        market:     Upbit market code
        refresh:    Force re-fetch

    Returns:
        {"1h": df_1h, "4h": df_4h, ...}
    """
    if timeframes is None:
        from btcbot.config import TIMEFRAMES
        timeframes = TIMEFRAMES

    result = {}
    for label, tf in timeframes.items():
        logger.info(f"── {label} ({tf}) ──")
        result[label] = load_or_fetch(tf=tf, n_candles=n_candles, market=market, refresh=refresh)
        time.sleep(DEFAULT_DELAY)
    return result


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = fetch_ohlcv(tf="minutes60", n_candles=500)
    print(df.tail())
    print(f"\nShape: {df.shape}")
