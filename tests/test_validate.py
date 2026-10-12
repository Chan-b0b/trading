import numpy as np
import pandas as pd
import pytest

from coinbot import backtest, metrics, validate


def random_walk(n: int, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    open_time = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": open_time, "close_time": open_time + pd.Timedelta(hours=1),
                         "open": open_, "close": close})


def sma_strategy(bars, fast=5, slow=20):
    c = bars["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def peeking_strategy(bars):
    return (bars["close"].shift(-1) > bars["close"]).astype(float)   # uses tomorrow's close


# ── metrics ──────────────────────────────────────────────────────────────────

def test_sharpe_and_sortino_known_values():
    ret = np.array([0.01, -0.01, 0.02, 0.0])
    assert metrics.sharpe(ret, 1) == pytest.approx(0.005 / np.std(ret, ddof=1))
    assert metrics.sortino(ret, 1) == pytest.approx(0.005 / np.sqrt(0.0001 / 4))


def test_max_drawdown_and_cvar():
    ret = np.array([0.1, -0.5, 0.2])         # 1.1 -> 0.55 -> 0.66
    assert metrics.max_drawdown(ret) == pytest.approx(-0.5)
    assert metrics.cvar(np.arange(-10, 10) / 100, 0.9) == pytest.approx(-0.095)


def test_report_and_bars_per_year():
    bars = random_walk(500)
    res = backtest.run(bars, sma_strategy(bars).fillna(0))
    r = metrics.report(res, metrics.bars_per_year(bars))
    assert metrics.bars_per_year(bars) == 365 * 24
    assert r["n_trades"] == len(res.trades) > 0
    assert r["total_return"] == pytest.approx(res.bars["equity"].iloc[-1] - 1)


# ── look-ahead detector ─────────────────────────────────────────────────────

def test_check_causal_passes_causal_strategy():
    validate.check_causal(sma_strategy, random_walk(400))


def test_check_causal_catches_peeking_strategy():
    with pytest.raises(AssertionError, match="look-ahead"):
        validate.check_causal(peeking_strategy, random_walk(400))


# ── walk-forward ─────────────────────────────────────────────────────────────

def test_walk_forward_picks_params_from_train_only():
    # constant weight strategies: up-trend in first half favours 1, down-trend later favours 0
    up, down = random_walk(300, seed=1, drift=0.01), random_walk(300, seed=2, drift=-0.01)
    close = np.concatenate([up["close"], down["close"] * up["close"].iloc[-1] / 100])
    bars = random_walk(600)
    bars["close"] = close
    bars["open"] = np.concatenate([[100.0], close[:-1]])

    def const(b, w):
        return pd.Series(float(w), index=b.index)

    sig, folds = validate.walk_forward(const, bars, [{"w": 0}, {"w": 1}], train=300, test=100,
                                       score=lambda r: r.bars["equity"].iloc[-1])
    assert (sig.iloc[:300] == 0).all()                # nothing traded before the first test window
    assert folds["w"].iloc[0] == 1                    # chosen on the up-trend only
    assert folds["w"].iloc[-1] == 0                   # later train windows contain the down-trend
    assert len(folds) == 3


# ── edge tests ───────────────────────────────────────────────────────────────

def test_shift_test_random_signal_has_no_edge():
    bars = random_walk(3000, seed=3, drift=0.0005)   # rising market: random long earns > 0
    rng = np.random.default_rng(4)
    sig = pd.Series(np.repeat(rng.integers(0, 2, 300), 10), dtype=float)
    out = validate.shift_test(bars, sig, 365 * 24, n=200)
    assert out["p_value"] > 0.05


def test_shift_test_detects_real_timing():
    bars = random_walk(3000, seed=5)
    sig = peeking_strategy(bars).fillna(0)          # cheating signal = maximal "edge"
    out = validate.shift_test(bars, sig, 365 * 24, n=200)
    assert out["p_value"] < 0.01


def test_bootstrap_interval_covers_true_sharpe():
    rng = np.random.default_rng(6)
    ret = rng.normal(0.001, 0.01, 5000)              # true per-bar Sharpe 0.1
    lo, hi = validate.bootstrap_sharpe(ret, bars_per_year=1, n=200, block=20)
    assert lo < 0.1 < hi and hi - lo < 0.1
