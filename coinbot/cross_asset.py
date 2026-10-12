"""
Out-of-sample check on other coins: same strategies, same grids, no retuning.

    python -m coinbot.cross_asset [--fetch] [--shifts 500]

Coins are picked by a fixed rule, not by results: every KRW market that is
listed today and already traded before LISTED_BEFORE (so walk-forward has a
2-year training window). Delisted coins are unavailable from Upbit, so the
universe has survivorship bias.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from coinbot import upbit
from coinbot.evaluate import evaluate

LISTED_BEFORE = "2019-01-01T00:00:00Z"
STRATEGIES = ["tsmom", "donchian"]
LIQUID_KRW = 1e9          # 24h trade value for the "liquid" subgroup
OUT = Path(__file__).resolve().parents[1] / "research" / "cross_asset.csv"


def select_markets() -> pd.DataFrame:
    """KRW markets listed before LISTED_BEFORE, with today's 24h trade value."""
    markets = [m["market"] for m in requests.get("https://api.upbit.com/v1/market/all", timeout=15).json()
               if m["market"].startswith("KRW-")]
    old = []
    for m in markets:
        for _ in range(5):
            r = requests.get(f"{upbit.BASE_URL}/days", timeout=15,
                             params={"market": m, "count": 1, "to": LISTED_BEFORE})
            if r.status_code != 429:
                break
            time.sleep(1)
        r.raise_for_status()
        if r.json():
            old.append(m)
        time.sleep(upbit.REQUEST_DELAY)
    tick = requests.get("https://api.upbit.com/v1/ticker", params={"markets": ",".join(old)}, timeout=15).json()
    return pd.DataFrame([{"market": t["market"], "value_24h": t["acc_trade_price_24h"]} for t in tick])


def _one(market: str, shifts: int) -> pd.DataFrame:
    bars = upbit.load(market, "1h")
    table = evaluate(bars, STRATEGIES, shifts, verbose=False).reset_index()
    table.insert(0, "market", market)
    table["oos_start"] = bars["open_time"].iloc[0] + pd.Timedelta(days=730)
    return table


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: how often it beats chance and buy-and-hold across coins."""
    bh = df[df["strategy"] == "buy_and_hold"].set_index("market")
    rows = []
    for name in STRATEGIES:
        s = df[df["strategy"] == name].set_index("market")
        rows.append({
            "strategy": name,
            "coins": len(s),
            "p<0.05": int((s["p_value"] < 0.05).sum()),
            "median_p": s["p_value"].median(),
            "sharpe>bh": int((s["sharpe"] > bh.loc[s.index, "sharpe"]).sum()),
            "median_sharpe": s["sharpe"].median(),
            "median_bh_sharpe": bh.loc[s.index, "sharpe"].median(),
            "mdd_better": int((s["max_drawdown"] > bh.loc[s.index, "max_drawdown"]).sum()),
            "median_mdd": s["max_drawdown"].median(),
            "median_bh_mdd": bh.loc[s.index, "max_drawdown"].median(),
        })
    return pd.DataFrame(rows).set_index("strategy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download/refresh 1h candles first")
    ap.add_argument("--shifts", type=int, default=500)
    args = ap.parse_args()

    universe = select_markets()
    others = universe[universe["market"] != "KRW-BTC"]
    print(f"{len(others)} coins listed before {LISTED_BEFORE[:10]} (excluding BTC)")

    if args.fetch:
        with ThreadPoolExecutor(3) as pool:   # ~10 req/s shared candle limit
            for m, df in zip(others["market"], pool.map(lambda m: upbit.load(m, "1h"), others["market"])):
                print(f"  {m}: {len(df)} bars from {df['open_time'].iloc[0].date()}")

    with ProcessPoolExecutor() as pool:
        tables = list(pool.map(_one, others["market"], [args.shifts] * len(others)))
    df = pd.concat(tables).merge(universe, on="market")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print("\nAll coins\n" + summarize(df).round(3).to_string())
    liquid = df[df["value_24h"] >= LIQUID_KRW]
    print(f"\nLiquid coins (24h value >= {LIQUID_KRW / 1e9:.0f}bn KRW, n={liquid['market'].nunique()})\n"
          + summarize(liquid).round(3).to_string())
    print(f"\nPer-coin results: {OUT}")


if __name__ == "__main__":
    main()
