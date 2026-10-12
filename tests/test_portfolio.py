import numpy as np
import pandas as pd
import pytest

from coinbot import backtest, portfolio


def grid(n):
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def test_single_coin_matches_backtest_engine():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    open_ = np.concatenate([[100.0], close[:-1] * (1 + rng.normal(0, 0.001, 299))])
    sig = (rng.random(300) > 0.5).astype(float)
    idx = grid(300)
    bars = pd.DataFrame({"open_time": idx, "close_time": idx + pd.Timedelta(hours=1),
                         "open": open_, "close": close})
    exact = backtest.run(bars, pd.Series(sig)).bars["ret"].to_numpy()
    approx = portfolio.returns(pd.DataFrame({"A": open_}, index=idx), pd.DataFrame({"A": close}, index=idx),
                               pd.DataFrame({"A": sig}, index=idx))
    assert np.abs(approx - exact).max() < 1e-4                      # only cross terms differ
    assert np.prod(1 + approx) == pytest.approx(np.prod(1 + exact), rel=1e-2)


def test_two_coins_half_each_and_unlisted_coin_is_ignored():
    idx = grid(3)
    open_ = pd.DataFrame({"A": [100, 100, 110], "B": [np.nan, 50, 50]}, index=idx)
    close = pd.DataFrame({"A": [100, 110, 110], "B": [np.nan, 50, 60]}, index=idx)
    target = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.0, 0.5, 0.5]}, index=idx)
    r = portfolio.returns(open_, close, target, cost=0.0)
    assert r.tolist() == pytest.approx([0.0, 0.05, 0.1])            # bar 2: B +20% on half


def test_membership_uses_trailing_value_of_eligible_coins_only():
    idx = grid(portfolio.REBALANCE * 2)
    value = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 3.0}, index=idx)
    value.iloc[portfolio.REBALANCE - 100:, 0] = 100.0                # A surges 100h before rebalance
    eligible = pd.DataFrame(True, index=idx, columns=value.columns)
    eligible["C"] = False                                            # C never eligible
    m = portfolio.membership(value, eligible, n=1, start=0)
    assert (m.iloc[: portfolio.REBALANCE]["B"] == 1).all()
    t = portfolio.REBALANCE
    assert m.iloc[t]["A"] == 1 and m.iloc[t]["B"] == 0               # switched at the rebalance bar
    assert (m["C"] == 0).all()
