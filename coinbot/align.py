"""Combine bars of different intervals without look-ahead."""

from __future__ import annotations

import pandas as pd


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate bars into longer bars (e.g. 1h -> "4h" / "1D"), aligned to UTC.
    A trailing incomplete group gets its full-length close_time, so `attach`
    will not expose it before it would have closed.
    """
    g = bars.groupby(bars["open_time"].dt.floor(rule))
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
    }).rename_axis("open_time").reset_index()
    out["close_time"] = out["open_time"] + pd.Timedelta(rule)
    return out


def attach(base: pd.DataFrame, other: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add `other`'s columns to each `base` row, using the latest `other` bar
    that closed at or before the base bar's close.

    Both frames need a `close_time` column. Compute indicators on `other`
    first, then attach. A 1d bar closing at 00:00 is visible to the 1h bar
    closing at 00:00, but never to the 1h bars inside that day.
    """
    cols = [c for c in other.columns if c != "close_time"]
    right = other[["close_time", *cols]].rename(columns={c: f"{prefix}{c}" for c in cols})
    right[f"{prefix}close_time"] = right["close_time"]
    return pd.merge_asof(
        base.sort_values("close_time"),
        right.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )
