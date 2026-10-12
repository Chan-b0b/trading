import pandas as pd

from coinbot import upbit
from coinbot.align import attach


def raw_candle(open_time: str, price: float, volume: float = 1.0) -> dict:
    return {
        "candle_date_time_utc": open_time,
        "opening_price": price,
        "high_price": price + 1,
        "low_price": price - 1,
        "trade_price": price,
        "candle_acc_trade_volume": volume,
        "candle_acc_trade_price": price * volume,
    }


def hourly(n: int, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    open_time = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": open_time, "close_time": open_time + pd.Timedelta(hours=1),
                         "close": range(n)})


def test_close_time_is_open_plus_interval():
    df = upbit.to_frame([raw_candle("2024-01-01T00:00:00", 100)], "1h")
    assert df["close_time"].iloc[0] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_finalize_drops_unfinished_bar():
    raw = [raw_candle("2024-01-01T00:00:00", 100), raw_candle("2024-01-01T01:00:00", 101)]
    df = upbit.to_frame(raw, "1h")
    out = upbit.finalize(df, "1h", now=pd.Timestamp("2024-01-01 01:30", tz="UTC"))
    assert list(out["close"]) == [100.0]


def test_finalize_fills_skipped_bars_flat():
    raw = [raw_candle("2024-01-01T00:00:00", 100), raw_candle("2024-01-01T03:00:00", 103)]
    out = upbit.finalize(upbit.to_frame(raw, "1h"), "1h", now=pd.Timestamp("2025-01-01", tz="UTC"))
    assert len(out) == 4
    filled = out.iloc[1]
    assert filled[["open", "high", "low", "close"]].tolist() == [100.0] * 4
    assert filled["volume"] == 0.0
    assert filled["close_time"] == pd.Timestamp("2024-01-01 02:00", tz="UTC")


def test_fetch_paginates_and_stops_at_since(monkeypatch):
    times = pd.date_range("2024-01-01", periods=450, freq="1h", tz="UTC")
    candles = [raw_candle(t.strftime("%Y-%m-%dT%H:%M:%S"), float(i)) for i, t in enumerate(times)]
    calls = []

    def fake_page(market, interval, to):
        calls.append(to)
        older = [c for c in candles if to is None or pd.Timestamp(c["candle_date_time_utc"], tz="UTC") < to]
        return list(reversed(older))[: upbit.MAX_COUNT]

    monkeypatch.setattr(upbit, "_get_page", fake_page)
    monkeypatch.setattr(upbit, "REQUEST_DELAY", 0)

    full = upbit.fetch("KRW-BTC", "1h")
    assert len(full) == 450 and full["open_time"].is_monotonic_increasing
    assert len(calls) == 3

    calls.clear()
    recent = upbit.fetch("KRW-BTC", "1h", since=times[300])
    assert recent["open_time"].iloc[0] == times[300] and len(recent) == 150
    assert len(calls) == 1


def test_attach_never_uses_unclosed_daily_bar():
    base = hourly(72)
    daily_open = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    daily = pd.DataFrame({"close_time": daily_open + pd.Timedelta(days=1), "close": [10.0, 20.0, 30.0]})

    out = attach(base, daily, "d_")

    assert (out["d_close_time"].dropna() <= out["close_time"][out["d_close_time"].notna()]).all()
    by_close = out.set_index("close_time")["d_close"]
    assert pd.isna(by_close[pd.Timestamp("2024-01-01 23:00", tz="UTC")])   # day 1 still open
    assert by_close[pd.Timestamp("2024-01-02 00:00", tz="UTC")] == 10.0    # day 1 just closed
    assert by_close[pd.Timestamp("2024-01-02 23:00", tz="UTC")] == 10.0    # day 2 still open
    assert by_close[pd.Timestamp("2024-01-03 00:00", tz="UTC")] == 20.0

