"""
Does Upbit order-book / trade flow predict the next 10s-5m mid-price move,
and is the move big enough to pay the 0.05% per-side fee?

Reads the websocket dump under data/ws (orderbook + trade parquet files),
builds a 1-second grid per market, and for each feature reports:
  - Spearman IC with the future mid return, on non-overlapping samples
  - mean future return when the feature is in its top / bottom decile,
    both mid-to-mid and as a taker round trip (buy ask, sell bid) net of fees

Trade `side`: BID = buyer-initiated (prints at the ask), ASK = seller-initiated.

    python research/orderflow.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[1] / "data" / "ws"
MARKETS = ("KRW-BTC", "KRW-ETH", "KRW-XRP")
HORIZONS = (10, 60, 300)          # seconds
FEE_BP = 5.0                      # Upbit, per side
STALE_MS = 2000                   # grid points with an older book are dropped


def read(kind: str, market: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(WS / kind / market / "*" / "*.parquet")))
    df = pd.concat(pd.read_parquet(f) for f in files).sort_values("ts").reset_index(drop=True)
    return df[df["stream_type"] != "SNAPSHOT"] if "stream_type" in df else df


def features(ob: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """1-second grid of book state, flow features, and future mid returns."""
    pb, qb, pa, qa = ob["bid_price_0"], ob["bid_size_0"], ob["ask_price_0"], ob["ask_size_0"]
    # Cont-Kukanov-Stoikov order flow imbalance, one term per book update
    e = ((pb >= pb.shift()) * qb - (pb <= pb.shift()) * qb.shift()
         - (pa <= pa.shift()) * qa + (pa >= pa.shift()) * qa.shift()).fillna(0.0)
    bid5 = ob[[f"bid_size_{i}" for i in range(5)]].sum(axis=1)
    ask5 = ob[[f"ask_size_{i}" for i in range(5)]].sum(axis=1)
    book = pd.DataFrame({
        "ts": ob["ts"], "bid": pb, "ask": pa, "ofi_cum": e.cumsum(),
        "imb_l1": (qb - qa) / (qb + qa),
        "imb_5": (bid5 - ask5) / (bid5 + ask5),
        "imb_30": (ob["total_bid_size"] - ob["total_ask_size"]) / (ob["total_bid_size"] + ob["total_ask_size"]),
    })
    signed = np.where(tr["side"] == "BID", tr["volume"], -tr["volume"])
    flow = pd.DataFrame({"ts": tr["ts"], "signed_cum": signed.cumsum(), "vol_cum": tr["volume"].cumsum()})

    grid = pd.DataFrame({"ts": np.arange(ob["ts"].iloc[0] // 1000 * 1000 + 1000, ob["ts"].iloc[-1], 1000)})
    g = pd.merge_asof(grid, book.rename(columns={"ts": "book_ts"}), left_on="ts", right_on="book_ts")
    g = pd.merge_asof(g, flow, on="ts")
    g[["signed_cum", "vol_cum"]] = g[["signed_cum", "vol_cum"]].fillna(0.0)
    g["fresh"] = g["ts"] - g["book_ts"] <= STALE_MS
    g["mid"] = (g["bid"] + g["ask"]) / 2

    for w in (10, 60):
        g[f"ofi_{w}"] = g["ofi_cum"] - g["ofi_cum"].shift(w)
        vol = g["vol_cum"] - g["vol_cum"].shift(w)
        g[f"trade_imb_{w}"] = (g["signed_cum"] - g["signed_cum"].shift(w)) / vol.where(vol > 0)
        g[f"fresh_past_{w}"] = g["fresh"].astype(float).rolling(w + 1).min().astype(bool)
    for h in HORIZONS:
        g[f"ret_{h}"] = (g["mid"].shift(-h) / g["mid"] - 1) * 1e4
        g[f"long_{h}"] = (g["bid"].shift(-h) / g["ask"] - 1) * 1e4 - 2 * FEE_BP
        g[f"short_{h}"] = (g["bid"] / g["ask"].shift(-h) - 1) * 1e4 - 2 * FEE_BP
        g[f"fresh_fut_{h}"] = g["fresh"].astype(float)[::-1].rolling(h + 1).min()[::-1].astype(bool)
    return g


FEATURES = ["imb_l1", "imb_5", "imb_30", "ofi_10", "ofi_60", "trade_imb_10", "trade_imb_60"]


def evaluate(g: pd.DataFrame, market: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        for f in FEATURES:
            past = g["fresh_past_60"] if f.endswith("_60") else g["fresh_past_10"]
            ok = g[g["fresh"] & past & g[f"fresh_fut_{h}"]].dropna(subset=[f, f"ret_{h}"])
            sample = ok[ok["ts"] // 1000 % h == 0]          # non-overlapping horizons
            ic = sample[f].corr(sample[f"ret_{h}"], method="spearman")
            lo, hi = ok[f].quantile([0.1, 0.9])
            top, bot = ok[ok[f] >= hi], ok[ok[f] <= lo]
            rows.append({
                "market": market, "h": h, "feature": f, "n": len(sample),
                "ic": ic, "t": ic * np.sqrt(len(sample)),
                "top10_mid": top[f"ret_{h}"].mean(), "bot10_mid": bot[f"ret_{h}"].mean(),
                "top10_long_net": top[f"long_{h}"].mean(), "bot10_short_net": bot[f"short_{h}"].mean(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    for m in MARKETS:
        g = features(read("orderbook", m), read("trade", m))
        hours = len(g) / 3600
        spread = ((g["ask"] - g["bid"]) / g["mid"] * 1e4).median()
        absmove = {h: g[f"ret_{h}"].abs().median() for h in HORIZONS}
        print(f"\n== {m}: {hours:.1f}h, median spread {spread:.2f}bp, "
              + ", ".join(f"|move {h}s| median {v:.1f}bp" for h, v in absmove.items()))
        print(evaluate(g, m).drop(columns="market").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
