import pandas as pd
import pytest

from coinbot import backtest, live

COST_IN = (1 - backtest.FEE) / (1 + backtest.SLIPPAGE)


@pytest.fixture
def signal(monkeypatch):
    """Replace the strategy with a settable constant target and count param selections."""
    box = {"target": 1.0, "selections": 0}

    def fake(bars, **params):
        return pd.Series(box["target"], index=bars.index)

    def fake_select(fn, bars, grid, start, train, score):
        box["selections"] += 1
        return {"k": box["selections"]}, 1.0

    monkeypatch.setattr(live, "STRATEGIES", {live.STRATEGY: fake})
    monkeypatch.setattr(live.validate, "select_params", fake_select)
    return box


def bars_until(close_time: str, n: int = 10) -> pd.DataFrame:
    close = pd.date_range(end=close_time, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": close - pd.Timedelta(hours=1), "close_time": close,
                         "open": 100.0, "close": 100.0})


NOW = pd.Timestamp("2026-01-01 10:00:30", tz="UTC")


def test_buys_up_to_cap_and_records_bar(signal):
    state = live.State(cash=5_000_000)
    rec = live.step(state, live.PaperBroker(state), bars_until("2026-01-01 10:00"), 100.0, NOW)
    assert rec["action"] == "order" and rec["order"]["side"] == "buy"
    assert state.cash == pytest.approx(5_000_000 - live.MAX_POSITION_KRW)
    assert state.coin == pytest.approx(live.MAX_POSITION_KRW * COST_IN / 100.0)
    assert state.last_bar == str(pd.Timestamp("2026-01-01 10:00", tz="UTC"))


def test_same_bar_is_not_traded_twice(signal):
    state = live.State()
    bars = bars_until("2026-01-01 10:00")
    live.step(state, live.PaperBroker(state), bars, 100.0, NOW)
    coin = state.coin
    rec = live.step(state, live.PaperBroker(state), bars, 100.0, NOW + pd.Timedelta(minutes=5))
    assert rec["action"] == "skip" and state.coin == coin


def test_stale_data_blocks_trading(signal):
    state = live.State()
    rec = live.step(state, live.PaperBroker(state), bars_until("2026-01-01 07:00"), 100.0, NOW)
    assert rec["reason"] == "stale data" and state.coin == 0 and state.last_bar is None


def test_halt_places_no_order(signal):
    state = live.State()
    rec = live.step(state, live.PaperBroker(state), bars_until("2026-01-01 10:00"), 100.0, NOW, halted=True)
    assert rec["action"] == "halted" and state.coin == 0 and state.cash == 1_000_000


def test_exit_sells_everything(signal):
    state = live.State(cash=0.0, coin=1_000.0)
    signal["target"] = 0.0
    rec = live.step(state, live.PaperBroker(state), bars_until("2026-01-01 10:00"), 100.0, NOW)
    assert rec["order"]["side"] == "sell" and state.coin == 0
    assert state.cash == pytest.approx(1_000 * 100 * (1 - backtest.SLIPPAGE) * (1 - backtest.FEE))


def test_no_order_below_minimum(signal):
    state = live.State(cash=1_000_000, coin=(live.MAX_POSITION_KRW - 1_000) / 100.0)
    rec = live.step(state, live.PaperBroker(state), bars_until("2026-01-01 10:00"), 100.0, NOW)
    assert rec["action"] == "hold"


def test_params_reselected_only_after_interval(signal):
    state = live.State()
    t = NOW
    for hours in (0, 1, 2):
        now = t + pd.Timedelta(hours=hours)
        live.step(state, live.PaperBroker(state), bars_until(str(now.floor("h"))), 100.0, now)
    assert signal["selections"] == 1
    later = t + live.RESELECT
    live.step(state, live.PaperBroker(state), bars_until(str(later.floor("h"))), 100.0, later)
    assert signal["selections"] == 2 and state.params == {"k": 2}
