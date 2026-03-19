"""
Fear & Greed Index fetcher for BTC.

Uses the free Alternative.me API (no API key required):
    https://api.alternative.me/fng/

Returns a daily time series of the Fear & Greed value (0–100) and
its classification label ('Extreme Fear', 'Fear', 'Neutral',
'Greed', 'Extreme Greed').

The value is normalized to [-1, +1] before being added to the
feature matrix so it has the same scale as other normalized signals:
    fng_norm = (value - 50) / 50
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FNG_URL   = "https://api.alternative.me/fng/"
MAX_LIMIT = 500    # API max per request


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_fear_greed(n_days: int = 500) -> pd.DataFrame:
    """
    Download the most recent *n_days* values of the Fear & Greed Index.

    Returns a DataFrame with columns:
        time      – UTC midnight of each day (timezone-aware)
        fng_value – raw 0-100 score
        fng_norm  – normalized score in [-1, +1]  (0 → -1, 50 → 0, 100 → +1)
        fng_label – string classification

    Raises requests.HTTPError on API failure.
    """
    pages = []
    fetched = 0
    limit = min(n_days, MAX_LIMIT)

    while fetched < n_days:
        params = {"limit": limit, "format": "json"}
        resp = requests.get(FNG_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        pages.extend(data)
        fetched += len(data)
        if len(data) < limit:
            break   # reached history start
        time.sleep(0.2)

    if not pages:
        logger.warning("Fear & Greed API returned no data")
        return pd.DataFrame(columns=["time", "fng_value", "fng_norm", "fng_label"])

    rows = []
    for entry in pages:
        ts = int(entry["timestamp"])
        value = int(entry["value"])
        rows.append({
            "time":      pd.Timestamp(ts, unit="s", tz="UTC").normalize(),  # UTC midnight
            "fng_value": value,
            "fng_norm":  round((value - 50) / 50, 4),
            "fng_label": entry.get("value_classification", ""),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    logger.info(f"Fear & Greed: {len(df)} days  {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}")
    return df


# ── Align to OHLCV timeframe ──────────────────────────────────────────────────

def align_fear_greed(df_ohlcv: pd.DataFrame, df_fng: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Fear & Greed (daily) onto an OHLCV DataFrame of any timeframe.

    Uses the most-recent COMPLETED day (backward merge → no look-ahead).

    Args:
        df_ohlcv: OHLCV DataFrame with 'time' column
        df_fng:   Output of fetch_fear_greed()

    Returns:
        df_ohlcv with 'fng_value', 'fng_norm', 'fng_label' appended.
    """
    if df_fng.empty:
        logger.warning("Fear & Greed DataFrame empty; returning unenriched OHLCV")
        return df_ohlcv

    df = df_ohlcv.copy()

    # Ensure timezone-aware
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    df_fng_clean = df_fng[["time", "fng_value", "fng_norm", "fng_label"]].copy()
    if df_fng_clean["time"].dt.tz is None:
        df_fng_clean["time"] = df_fng_clean["time"].dt.tz_localize("UTC")

    fng_cols = ["fng_value", "fng_norm", "fng_label"]

    # Normalise both sides to the same unit (pandas 2+ uses ns/us/s depending on source)
    df["time"]              = df["time"].astype("datetime64[us, UTC]")
    df_fng_clean["time"]    = df_fng_clean["time"].astype("datetime64[us, UTC]")

    df_merged = pd.merge_asof(
        df.sort_values("time"),
        df_fng_clean.sort_values("time"),
        on="time",
        direction="backward",
    )
    # Fill any leading NaN with neutral value
    df_merged["fng_value"] = df_merged["fng_value"].ffill().fillna(50)
    df_merged["fng_norm"]  = df_merged["fng_norm"].ffill().fillna(0.0)
    df_merged["fng_label"] = df_merged["fng_label"].ffill().fillna("Neutral")

    return df_merged


# ── Convenience one-shot ──────────────────────────────────────────────────────

def get_current_fng() -> dict:
    """Return today's Fear & Greed value as a dict."""
    resp = requests.get(FNG_URL, params={"limit": 1, "format": "json"}, timeout=10)
    resp.raise_for_status()
    entry = resp.json()["data"][0]
    return {
        "value": int(entry["value"]),
        "norm":  round((int(entry["value"]) - 50) / 50, 4),
        "label": entry.get("value_classification", ""),
        "time":  pd.Timestamp(int(entry["timestamp"]), unit="s", tz="UTC"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = fetch_fear_greed(n_days=100)
    print(df.tail(10))
    print(f"\nCurrent: {get_current_fng()}")
