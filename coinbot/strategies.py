"""
Long/flat strategies on 1h bars. Each returns target weights decided at
each bar's close (see coinbot.backtest) and must only use past bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coinbot.align import attach, resample


def tsmom(bars: pd.DataFrame, lookback: int) -> pd.Series:
    """Time-series momentum: long while close is above the close `lookback` bars ago."""
    c = bars["close"]
    return (c > c.shift(lookback)).astype(float)


def sma_cross(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Long while the fast moving average is above the slow one."""
    c = bars["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def vol_target(bars: pd.DataFrame, lookback: int, target_vol: float,
               vol_window: int = 24 * 7, band: float = 0.1) -> pd.Series:
    """
    TSMOM sized by volatility: weight = min(1, target_vol / realized vol) while
    the trend is up. Realized vol is the EWMA std of 1h returns, annualized.
    The weight only moves when the change exceeds `band`, to save fees.
    """
    c = bars["close"]
    vol = np.log(c).diff().ewm(span=vol_window, min_periods=vol_window).std() * np.sqrt(24 * 365)
    raw = ((target_vol / vol).clip(upper=1.0) * tsmom(bars, lookback)).fillna(0.0).to_numpy()
    out = np.zeros(len(raw))
    w = 0.0
    for i, x in enumerate(raw):
        if x == 0.0 or abs(x - w) > band:
            w = x
        out[i] = w
    return pd.Series(out, index=bars.index)


def donchian(bars: pd.DataFrame, entry: int, exit: int) -> pd.Series:
    """Long when close breaks above the prior `entry`-bar high, flat below the prior `exit`-bar low."""
    c = bars["close"]
    up = c > bars["high"].rolling(entry).max().shift(1)
    down = c < bars["low"].rolling(exit).min().shift(1)
    state = pd.Series(np.where(up, 1.0, np.where(down, 0.0, np.nan)), index=bars.index)
    return state.ffill().fillna(0.0)


def _rsi(c: pd.Series, period: int = 14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50.0) / 100.0


def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return pd.DataFrame({
        "close_time": df["close_time"],
        "macd": (macd - macd.ewm(span=9, adjust=False).mean()) / c,
        "ma_diff": (c.ewm(span=20, adjust=False).mean() - c.ewm(span=50, adjust=False).mean()) / c,
        "rsi": _rsi(c),
    })


def rules(bars: pd.DataFrame, stop_loss: float = 0.03, take_profit: float = 0.05,
          max_hold: int = 12, rsi_max: float = 0.75, rsi_min: float = 0.45,
          mom_bars: int = 3, cooldown: int = 1) -> pd.Series:
    """
    The old 5m/15m/1h rule strategy moved up one scale to 1h/4h/1d:
    enter when the 1d and 4h trends are up, 4h MACD is positive, 4h RSI is in
    range and 1h momentum is positive; exit on stop-loss / take-profit
    (checked at the close), max holding time, or when momentum/trend turns.
    """
    f = bars[["open_time", "close_time", "open", "close"]].copy()
    f["mom"] = f["close"].pct_change(mom_bars)
    f = attach(f, _trend_features(resample(bars, "4h")), "h4_")
    f = attach(f, _trend_features(resample(bars, "1D")), "d1_")

    entry = ((f["d1_ma_diff"] > 0) & (f["h4_ma_diff"] > 0) & (f["h4_macd"] > 0) & (f["mom"] > 0)
             & f["h4_rsi"].between(rsi_min, rsi_max)).to_numpy()
    early_exit = ((f["mom"] < 0) | (f["h4_macd"] < 0) | (f["h4_ma_diff"] < 0)).to_numpy()
    o, c = f["open"].to_numpy(), f["close"].to_numpy()

    n = len(f)
    target = np.zeros(n)
    holding, entry_px, held, cooldown_until = False, 0.0, 0, -1
    for i in range(n):
        if holding:
            held += 1
            r = c[i] / entry_px - 1
            if r <= -stop_loss or r >= take_profit or held >= max_hold or early_exit[i]:
                holding, cooldown_until = False, i + cooldown
        elif i > cooldown_until and entry[i]:
            # filled at the next open; only used by exit checks from bar i+1 on
            holding, held = True, 0
            entry_px = o[i + 1] if i + 1 < n else c[i]
        target[i] = float(holding)
    return pd.Series(target, index=bars.index)


GRIDS = {
    "tsmom": [{"lookback": 24 * d} for d in (7, 14, 30, 60, 120, 240)],
    "sma_cross": [{"fast": 24 * f, "slow": 24 * s}
                  for f in (5, 10, 20) for s in (50, 100, 200)],
    "vol_target": [{"lookback": 24 * d, "target_vol": v}
                   for d in (7, 14, 30, 60, 120, 240) for v in (0.3, 0.5, 0.8)],
    "donchian": [{"entry": 24 * e, "exit": 24 * x}
                 for e, x in ((10, 5), (20, 5), (20, 10), (55, 10), (55, 20), (100, 20))],
    "rules": [{"stop_loss": sl, "take_profit": tp, "max_hold": mh}
              for sl in (0.02, 0.03, 0.05) for tp in (0.03, 0.05, 0.08) for mh in (6, 12, 24)],
}

STRATEGIES = {"tsmom": tsmom, "sma_cross": sma_cross, "rules": rules,
              "vol_target": vol_target, "donchian": donchian}
