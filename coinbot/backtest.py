"""
Bar-by-bar long/flat backtest.

`signal[t]` is the target weight (0 = cash, 1 = fully long) decided at the
CLOSE of bar t. It is executed at the OPEN of bar t+1, so a signal can never
earn the return of the bar it was computed on.

Per bar t, with w_prev held into the bar and w set at its open:
    equity *= (1 + w_prev * (open_t / close_{t-1} - 1))   # gap since last close
    equity *= (1 - |w - w_prev| * (fee + slippage))       # rebalance at open
    equity *= (1 + w * (close_t / open_t - 1))            # intrabar move

Exact for 0/1 weights. For fractional weights, drift inside a bar is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEE = 0.0005       # Upbit KRW market taker fee, per side
SLIPPAGE = 0.0002  # per side


@dataclass
class Result:
    bars: pd.DataFrame    # open_time, close_time, weight, ret, equity
    trades: pd.DataFrame  # entry_time, exit_time, bars_held, ret


def run(bars: pd.DataFrame, signal: pd.Series, fee: float = FEE,
        slippage: float = SLIPPAGE, initial: float = 1.0) -> Result:
    """Backtest target weights `signal` (aligned row-by-row with `bars`)."""
    if len(signal) != len(bars):
        raise ValueError("signal must have one value per bar")
    target = np.asarray(signal, dtype=float)
    if np.isnan(target).any() or (target < 0).any() or (target > 1).any():
        raise ValueError("signal must be in [0, 1] with no NaN")

    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    w = np.concatenate([[0.0], target[:-1]])          # weight held during each bar
    w_prev = np.concatenate([[0.0], w[:-1]])
    prev_close = np.concatenate([[o[0]], c[:-1]])

    growth = ((1 + w_prev * (o / prev_close - 1))
              * (1 - np.abs(w - w_prev) * (fee + slippage))
              * (1 + w * (c / o - 1)))
    equity = initial * np.cumprod(growth)

    out = pd.DataFrame({
        "open_time": bars["open_time"].to_numpy(),
        "close_time": bars["close_time"].to_numpy(),
        "weight": w,
        "ret": growth - 1,
        "equity": equity,
    })
    return Result(bars=out, trades=_trades(out, growth))


def _trades(out: pd.DataFrame, growth: np.ndarray) -> pd.DataFrame:
    """One trade per contiguous run of bars with weight > 0, entry and exit costs included."""
    held = out["weight"].to_numpy() > 0
    rows = []
    i, n = 0, len(held)
    while i < n:
        if not held[i]:
            i += 1
            continue
        j = i
        while j < n and held[j]:
            j += 1
        # the exit fill happens at bar j's open, so bar j carries the gap and exit cost
        rows.append({
            "entry_time": out["open_time"].iloc[i],
            "exit_time": out["open_time"].iloc[j] if j < n else pd.NaT,
            "bars_held": j - i,
            "ret": float(np.prod(growth[i:min(j + 1, n)]) - 1),
        })
        i = j
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "bars_held", "ret"])

