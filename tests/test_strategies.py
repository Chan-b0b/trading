import numpy as np
import pandas as pd
import pytest

from coinbot import strategies, validate
from coinbot.align import resample


def random_walk(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    open_time = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": open_time, "close_time": open_time + pd.Timedelta(hours=1),
                         "open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close})


def test_resample_to_daily():
    bars = random_walk(48)
    d = resample(bars, "1D")
    assert len(d) == 2
    assert d["open"].iloc[0] == bars["open"].iloc[0]
    assert d["close"].iloc[0] == bars["close"].iloc[23]
    assert d["high"].iloc[1] == bars["high"].iloc[24:].max()
    assert d["close_time"].iloc[0] == bars["close_time"].iloc[23]


@pytest.mark.parametrize("name", list(strategies.STRATEGIES))
def test_strategies_are_causal(name):
    bars = random_walk(24 * 120)
    fn = strategies.STRATEGIES[name]
    params = strategies.GRIDS[name][0]
    validate.check_causal(fn, bars, params, n_cuts=7)


def test_rules_respects_max_hold_and_stop():
    bars = random_walk(24 * 120, seed=1)
    w = strategies.rules(bars, max_hold=6).to_numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], w, [0]]))))[::2]
    assert w.sum() > 0 and runs.max() <= 6


def test_vol_target_weights_bounded_and_banded():
    # alternate calm (~28%/yr) and wild (~140%/yr) regimes in a rising market
    rng = np.random.default_rng(2)
    sigma = np.tile(np.repeat([0.003, 0.015], 500), 3)
    bars = random_walk(len(sigma))
    bars["close"] = 100 * np.exp(np.cumsum(rng.normal(0.0005, sigma)))
    w = strategies.vol_target(bars, lookback=48, target_vol=0.3).to_numpy()
    assert ((w >= 0) & (w <= 1)).all() and w.max() == 1.0 and 0 < w[w > 0].min() < 0.5
    resized = (w[1:] != w[:-1]) & (w[1:] > 0) & (w[:-1] > 0)   # size changes while long
    assert resized.any() and (np.abs(np.diff(w))[resized] > 0.1).all()


def test_donchian_enters_on_breakout_and_exits_on_breakdown():
    close = np.array([100, 101, 100, 101, 105, 106, 104, 98, 99], dtype=float)
    bars = random_walk(len(close))
    bars["close"], bars["high"], bars["low"] = close, close, close
    w = strategies.donchian(bars, entry=3, exit=2).tolist()
    assert w == [0, 0, 0, 0, 1, 1, 0, 0, 0]   # 104 < prior 2-bar low 105
