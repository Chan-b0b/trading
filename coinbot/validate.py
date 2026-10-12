"""
Tools to decide whether a strategy has an edge.

A strategy is a function `fn(bars, **params) -> pd.Series` of target weights,
one per bar, where weight[t] may only use bars[0..t].
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from coinbot import backtest
from coinbot.metrics import sharpe

Strategy = Callable[..., pd.Series]


def check_causal(fn: Strategy, bars: pd.DataFrame, params: dict | None = None,
                 n_cuts: int = 5) -> None:
    """Raise if the signal at any bar changes when later bars are removed (look-ahead)."""
    params = params or {}
    full = np.asarray(fn(bars, **params), dtype=float)
    for cut in np.linspace(len(bars) // 4, len(bars) - 1, n_cuts, dtype=int):
        part = np.asarray(fn(bars.iloc[:cut].reset_index(drop=True), **params), dtype=float)
        bad = np.flatnonzero(~np.isclose(part, full[:cut], equal_nan=True))
        if len(bad):
            raise AssertionError(
                f"look-ahead: signal at bar {bad[0]} differs when data is cut at bar {cut}")


def select_params(fn: Strategy, bars: pd.DataFrame, grid: Iterable[dict], start: int,
                  train: int, score: Callable[[backtest.Result], float]) -> tuple[dict, float]:
    """Params with the best `score` on bars[start - train:start], using only bars before `start`."""
    seen = bars.iloc[:start]
    best, best_score = None, -np.inf
    for params in grid:
        sig = fn(seen, **params)
        res = backtest.run(seen.iloc[start - train:].reset_index(drop=True),
                           sig.iloc[start - train:].reset_index(drop=True))
        s = score(res)
        if s > best_score:
            best, best_score = params, s
    return best, best_score


def walk_forward(fn: Strategy, bars: pd.DataFrame, grid: Iterable[dict],
                 train: int, test: int, score: Callable[[backtest.Result], float]
                 ) -> tuple[pd.Series, pd.DataFrame]:
    """
    Rolling refit: for each fold pick the params with the best `score` on the
    previous `train` bars, then use them for the next `test` bars.

    Returns the stitched out-of-sample signal (0 before the first test window)
    and a table of the params chosen per fold. Backtest the signal once so
    costs at fold boundaries are counted.
    """
    grid = list(grid)
    oos = pd.Series(0.0, index=bars.index)
    folds = []
    for start in range(train, len(bars), test):
        end = min(start + test, len(bars))
        best, best_score = select_params(fn, bars, grid, start, train, score)
        seen = bars.iloc[:end]                 # nothing after the test window
        oos.iloc[start:end] = np.asarray(fn(seen, **best), dtype=float)[start:end]
        folds.append({"test_start": bars["open_time"].iloc[start],
                      "test_end": bars["close_time"].iloc[end - 1],
                      "train_score": best_score, **best})
    return oos, pd.DataFrame(folds)


def shift_test(bars: pd.DataFrame, signal: pd.Series, bars_per_year: float,
               n: int = 500, seed: int = 0) -> dict:
    """
    Is the timing better than chance? Compare the strategy's Sharpe with the
    same signal circularly shifted by random offsets. Shifting keeps exposure,
    trade count and holding periods, and only breaks the alignment with prices.
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float)
    observed = sharpe(backtest.run(bars, signal).bars["ret"].to_numpy(), bars_per_year)
    lo = max(1, len(sig) // 20)
    null = np.array([
        sharpe(backtest.run(bars, np.roll(sig, k)).bars["ret"].to_numpy(), bars_per_year)
        for k in rng.integers(lo, len(sig) - lo, size=n)
    ])
    return {
        "sharpe": observed,
        "null_mean": float(null.mean()),
        "null_95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= observed).sum()) / (1 + n)),
    }


def bootstrap_sharpe(ret: np.ndarray, bars_per_year: float, n: int = 1000,
                     block: int = 24 * 7, seed: int = 0) -> tuple[float, float]:
    """95% interval of the Sharpe ratio by stationary block bootstrap."""
    rng = np.random.default_rng(seed)
    ret = np.asarray(ret)
    T = len(ret)
    sr = []
    for size in [100] * (n // 100) + ([n % 100] if n % 100 else []):   # chunks bound memory
        idx = np.empty((size, T), dtype=np.int64)
        idx[:, 0] = rng.integers(0, T, size)
        new_block = rng.random((size, T)) < 1 / block
        jumps = rng.integers(0, T, (size, T))
        for t in range(1, T):
            idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
        s = ret[idx]
        sr.append(s.mean(1) / s.std(1, ddof=1) * np.sqrt(bars_per_year))
    lo, hi = np.quantile(np.concatenate(sr), [0.025, 0.975])
    return float(lo), float(hi)
