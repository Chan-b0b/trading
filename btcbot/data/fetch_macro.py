"""
Macro feature fetcher for BTC correlation context.

Fetches daily data for:
    ^VIX   - CBOE Volatility Index  (risk-off gauge)
    DX-Y.NYB / DX=F  - US Dollar Index  (USD strength)
    ^GSPC  - S&P 500  (risk-on / risk-off)

All sourced via yfinance (free, no API key needed).
Data is resampled and forward-filled to align with any OHLCV timeframe.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed – macro features will be zeros. "
                   "Install with: pip install yfinance")

MACRO_TICKERS = {
    "vix":  "^VIX",
    "dxy":  "DX=F",      # Dollar Index Futures (more reliable than DX-Y.NYB)
    "spx":  "^GSPC",
}

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_macro(
    start: str | None = None,
    end:   str | None = None,
    years_back: int = 3,
) -> pd.DataFrame:
    """
    Download daily OHLCV for all macro tickers and merge into one DataFrame.

    Returns a DataFrame indexed by UTC date with columns:
        vix_close, dxy_close, spx_close, dxy_ret, spx_ret, vix_ret
    covering the requested period.

    Args:
        start:      ISO date string, e.g. "2021-01-01". Defaults to years_back ago.
        end:        ISO date string. Defaults to today.
        years_back: Used only when start is None.
    """
    if not _YF_AVAILABLE:
        logger.warning("yfinance unavailable; returning empty macro DataFrame")
        return pd.DataFrame()

    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.utcnow() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    logger.info(f"Downloading macro data  {start} → {end}")

    frames = {}
    for label, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"  {ticker}: no data returned")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            s = raw["close"].rename(f"{label}_close")
            frames[label] = s
            logger.info(f"  ✅ {ticker}: {len(s)} days")
        except Exception as exc:
            logger.warning(f"  {ticker}: fetch failed – {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index().ffill()

    # Add log-returns
    for col in list(df.columns):
        key = col.replace("_close", "")
        df[f"{key}_ret"] = np.log(df[col]).diff().fillna(0.0)

    logger.info(f"Macro DataFrame: {df.shape}  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── Align macro to OHLCV timeframe ─────────────────────────────────────────────

def align_macro(df_ohlcv: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro (daily) features onto an OHLCV DataFrame of any timeframe.

    Strategy: for each OHLCV row take the macro values from the
    most-recent COMPLETED trading day (no look-ahead).

    Args:
        df_ohlcv:  Must have a 'time' column (timezone-aware or naive UTC)
        df_macro:  Output of fetch_macro()

    Returns:
        df_ohlcv with macro columns appended, NaN rows forward-filled.
    """
    if df_macro.empty:
        logger.warning("Macro DataFrame is empty; skipping alignment")
        return df_ohlcv

    df = df_ohlcv.copy()

    # Ensure timezone-aware
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    # Macro index is already UTC
    macro_cols = df_macro.columns.tolist()

    # merge_asof: for each ohlcv row grab latest macro row <= that time
    df_macro_reset = df_macro.reset_index().rename(columns={"index": "time", "Date": "time"})
    if "time" not in df_macro_reset.columns:
        df_macro_reset = df_macro_reset.rename(columns={df_macro_reset.columns[0]: "time"})

    # Ensure macro time is also UTC
    if df_macro_reset["time"].dt.tz is None:
        df_macro_reset["time"] = df_macro_reset["time"].dt.tz_localize("UTC")

    # Normalise both sides to the same unit (pandas 2+ uses ns/us/s depending on source)
    df["time"]               = df["time"].astype("datetime64[us, UTC]")
    df_macro_reset["time"]   = df_macro_reset["time"].astype("datetime64[us, UTC]")

    df_merged = pd.merge_asof(
        df.sort_values("time"),
        df_macro_reset.sort_values("time"),
        on="time",
        direction="backward",
    )
    # Forward-fill any remaining NaN in macro columns
    df_merged[macro_cols] = df_merged[macro_cols].ffill().fillna(0.0)

    return df_merged


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = fetch_macro(years_back=2)
    print(df.tail(10))
    print(f"\nColumns: {list(df.columns)}")
