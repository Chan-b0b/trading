"""Performance and risk metrics on per-bar strategy returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coinbot.backtest import Result


def sharpe(ret: np.ndarray, bars_per_year: float) -> float:
    sd = np.std(ret, ddof=1)
    return float(np.mean(ret) / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0


def sortino(ret: np.ndarray, bars_per_year: float) -> float:
    downside = np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))
    return float(np.mean(ret) / downside * np.sqrt(bars_per_year)) if downside > 0 else 0.0


def max_drawdown(ret: np.ndarray) -> float:
    eq = np.cumprod(1 + np.asarray(ret))
    peak = np.maximum.accumulate(np.maximum(eq, 1.0))
    return float((eq / peak - 1).min())


def cvar(ret: np.ndarray, level: float = 0.95) -> float:
    """Mean of the worst (1 - level) share of bar returns (a negative number)."""
    r = np.sort(np.asarray(ret))
    k = max(1, int(np.ceil(len(r) * (1 - level))))
    return float(r[:k].mean())


def report(result: Result, bars_per_year: float) -> dict:
    ret = result.bars["ret"].to_numpy()
    trades = result.trades
    growth = float(np.prod(1 + ret))
    cagr = growth ** (bars_per_year / len(ret)) - 1
    mdd = max_drawdown(ret)
    wins, losses = trades["ret"][trades["ret"] > 0], trades["ret"][trades["ret"] <= 0]
    return {
        "total_return": growth - 1,
        "cagr": cagr,
        "sharpe": sharpe(ret, bars_per_year),
        "sortino": sortino(ret, bars_per_year),
        "max_drawdown": mdd,
        "calmar": cagr / -mdd if mdd < 0 else float("nan"),
        "cvar_95": cvar(ret),
        "exposure": float((result.bars["weight"] > 0).mean()),
        "turnover_per_year": float(result.bars["weight"].diff().abs().sum() * bars_per_year / len(ret)),
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan"),
    }


def bars_per_year(bars: pd.DataFrame) -> float:
    """24/7 market: bars per 365 days, from the bar length."""
    step = bars["close_time"].iloc[0] - bars["open_time"].iloc[0]
    return pd.Timedelta(days=365) / step
