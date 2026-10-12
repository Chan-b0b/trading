import pandas as pd

from coinbot import collect

OB = {"type": "orderbook", "code": "KRW-BTC", "timestamp": 1790830096000,
      "total_ask_size": 3.0, "total_bid_size": 4.0,
      "orderbook_units": [{"ask_price": 101.0, "bid_price": 100.0, "ask_size": 1.0, "bid_size": 2.0}]}
TRADE = {"type": "trade", "code": "KRW-BTC", "trade_timestamp": 1790830096107, "trade_price": 100.5,
         "trade_volume": 0.01, "ask_bid": "BID", "sequential_id": 17, "stream_type": "REALTIME"}


def test_orderbook_row_pads_missing_levels():
    row = collect.orderbook_row(OB, 1)
    assert row["ask_price_0"] == 101.0 and row["bid_size_0"] == 2.0
    assert pd.isna(row[f"ask_price_{collect.LEVELS - 1}"])
    assert len([k for k in row if k.startswith("ask_price_")]) == collect.LEVELS


def test_buffer_writes_one_file_per_type_and_market(tmp_path):
    buf = collect.Buffer(tmp_path)
    recv = 1790830096500                                   # 2026-10-01 04:48:16.5 UTC
    buf.add(OB, recv)
    buf.add(TRADE, recv)
    buf.add({**TRADE, "code": "KRW-ETH"}, recv)
    buf.add({"status": "UP"}, recv)                        # non-data messages are ignored
    paths = sorted(buf.flush())
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "orderbook/KRW-BTC/2026-10-01/044816.parquet",
        "trade/KRW-BTC/2026-10-01/044816.parquet",
        "trade/KRW-ETH/2026-10-01/044816.parquet",
    ]
    t = pd.read_parquet(tmp_path / "trade/KRW-BTC/2026-10-01/044816.parquet")
    assert t[["price", "side", "seq"]].iloc[0].tolist() == [100.5, "BID", 17]
    assert buf.flush() == []                               # buffer emptied
