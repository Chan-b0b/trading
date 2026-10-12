import numpy as np
import pandas as pd
import pytest

from coinbot import backtest

COST = backtest.FEE + backtest.SLIPPAGE


def make_bars(opens, closes) -> pd.DataFrame:
    open_time = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": open_time, "close_time": open_time + pd.Timedelta(hours=1),
                         "open": opens, "close": closes})


def test_buy_and_hold_matches_hand_calculation():
    bars = make_bars([100, 102, 99, 105], [101, 98, 104, 110])
    res = backtest.run(bars, pd.Series([1.0] * 4))
    # signal at bar 0 close -> buy at bar 1 open (102), still holding at bar 3 close (110)
    assert res.bars["equity"].iloc[-1] == pytest.approx((1 - COST) * 110 / 102)
    assert res.bars["weight"].tolist() == [0, 1, 1, 1]


def test_signal_cannot_earn_its_own_bar():
    # bar 2 jumps +50% intrabar; a signal raised at bar 2's close must not capture it
    bars = make_bars([100, 100, 100, 150, 150], [100, 100, 150, 150, 150])
    sig = pd.Series([0, 0, 1, 0, 0], dtype=float)
    res = backtest.run(bars, sig)
    assert res.bars["equity"].iloc[2] == 1.0
    assert res.bars["equity"].iloc[-1] == pytest.approx((1 - COST) ** 2)


def test_gap_is_earned_by_previous_weight():
    # close 100 -> next open 110: only a position held into the gap earns it
    bars = make_bars([100, 110, 110], [100, 110, 110])
    held = backtest.run(bars, pd.Series([1, 1, 1], dtype=float))
    assert held.bars["equity"].iloc[-1] == pytest.approx(1 - COST)       # bought at 110, gap missed
    bars = make_bars([100, 100, 110], [100, 100, 110])
    held = backtest.run(bars, pd.Series([1, 1, 1], dtype=float))
    assert held.bars["equity"].iloc[-1] == pytest.approx((1 - COST) * 1.1)


def test_round_trips_cost_two_sides_each():
    bars = make_bars([100] * 9, [100] * 9)
    sig = pd.Series([1, 0] * 4 + [0], dtype=float)
    res = backtest.run(bars, sig)
    assert res.bars["equity"].iloc[-1] == pytest.approx((1 - COST) ** 8)
    assert len(res.trades) == 4
    assert res.trades["ret"].tolist() == pytest.approx([(1 - COST) ** 2 - 1] * 4)


def test_trade_record_times_and_return():
    bars = make_bars([100, 100, 110, 121, 121], [100, 110, 121, 121, 121])
    res = backtest.run(bars, pd.Series([1, 1, 0, 0, 0], dtype=float))
    t = res.trades.iloc[0]
    assert t["entry_time"] == bars["open_time"].iloc[1]
    assert t["exit_time"] == bars["open_time"].iloc[3]
    assert t["bars_held"] == 2
    assert t["ret"] == pytest.approx((1 - COST) ** 2 * 1.21 - 1)


def test_open_trade_at_end_has_no_exit():
    bars = make_bars([100, 100, 100], [100, 100, 120])
    res = backtest.run(bars, pd.Series([0, 1, 1], dtype=float))
    assert pd.isna(res.trades["exit_time"].iloc[0])
    assert res.trades["ret"].iloc[0] == pytest.approx((1 - COST) * 1.2 - 1)


@pytest.mark.parametrize("bad", [[0, 1.5, 0], [0, -0.1, 0], [0, np.nan, 0], [0, 1]])
def test_rejects_invalid_signal(bad):
    bars = make_bars([100] * 3, [100] * 3)
    with pytest.raises(ValueError):
        backtest.run(bars, pd.Series(bad, dtype=float))
