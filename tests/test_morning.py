import numpy as np
import pandas as pd

from coinbot import morning

KST = morning.KST


def minute_bars(days: list[str], start: str = "08:00", end: str = "20:00") -> pd.DataFrame:
    """1m bars over whole extended sessions, price = running bar count."""
    opens = pd.DatetimeIndex([])
    for d in days:
        opens = opens.append(pd.date_range(f"{d} {start}", f"{d} {end}", freq="1min",
                                           inclusive="left", tz=KST))
    price = np.arange(len(opens), dtype=float) + 100
    return pd.DataFrame({"open_time": opens.tz_convert("UTC"), "close_time": (opens + pd.Timedelta("1min")).tz_convert("UTC"),
                         "open": price, "high": price, "low": price, "close": price, "volume": 1.0})


DAYS = [f"2026-09-{d:02d}" for d in (21, 22, 23, 24, 25, 28, 29, 30)] + ["2026-10-01"]


def test_session_keeps_regular_hours_and_closing_auction():
    bars = morning.session_bars(minute_bars(["2026-09-30"]))
    times = bars["open_time"].dt.strftime("%H:%M")
    assert times.iloc[0] == "09:00" and times.iloc[-1] == "15:30"
    assert len(bars) == morning.BARS_PER_DAY
    assert bars["volume"].iloc[-1] == 1.0          # 15:30 bin holds only the auction minute


def test_context_never_sees_bars_after_asof():
    bars = minute_bars(DAYS)
    asof = pd.Timestamp("2026-10-01 09:20", tz=KST)
    full = morning.split(bars, asof)
    cut = morning.split(bars[bars["close_time"] <= asof], asof)

    pd.testing.assert_frame_equal(full["context"], cut["context"])
    assert full["context"]["close_time"].max() == asof
    assert len(full["context"]) == morning.CONTEXT
    assert full["price"] == cut["price"]
    assert np.isnan(cut["actual"]) and full["actual_final"]


def test_unaligned_asof_drops_partial_bin():
    bars = minute_bars(DAYS)
    info = morning.split(bars, pd.Timestamp("2026-10-01 09:22", tz=KST))
    assert info["context"]["close_time"].max() == pd.Timestamp("2026-10-01 09:20", tz=KST)


def test_split_skips_stock_without_trades_today():
    bars = minute_bars(DAYS[:-1])
    assert morning.split(bars, pd.Timestamp("2026-10-01 09:20", tz=KST)) is None


def test_horizon_ends_at_closing_auction():
    future = morning.horizon(pd.Timestamp("2026-10-01 09:20", tz=KST))
    assert future.iloc[0].strftime("%H:%M") == "09:20" and future.iloc[-1].strftime("%H:%M") == "15:30"
    assert len(future) == 75


def test_summarize_and_render():
    stocks = pd.DataFrame({"symbol": ["A", "B"], "name": ["에이", "비"], "price": [100.0, 100.0],
                           "prev_close": [95.0, 105.0], "actual": [103.0, 98.0], "actual_final": [True, True]})
    closes = np.array([[101.0, 103.0, 99.0, 105.0], [97.0, 99.0, 98.0, 101.0]])
    table = morning.summarize(stocks, closes)

    assert list(table["symbol"]) == ["A", "B"]
    assert np.isclose(table.loc[0, "exp_ret"], 0.02) and table.loc[0, "p_up"] == 0.75
    asof = pd.Timestamp("2026-10-01 09:20", tz=KST)
    text = morning.render(table, asof, asof + pd.Timedelta(hours=7), "small", 4, 100)
    assert "에이 (A)" in text and "종가" in text and "사후 생성" in text
