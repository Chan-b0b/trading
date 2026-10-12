"""
Can a classifier on Upbit order-book / trade-flow features find long entries
that beat the round-trip cost?  Spot market, so long only.

Label: buy at the ask now, sell at the bid H seconds later; positive if the
return net of 2 x 0.05% fees is > 0.  Decision points every 5 s.
Walk-forward by KST day: train on all earlier days (H-second purge), test on the next day.
Trades in the test period never overlap (one position per market at a time).

    python research/orderflow_model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import orderflow as of

HORIZONS = (60, 300, 900)
STEP = 5                 # seconds between decision points
FEATURES = ["imb_l1", "imb_5", "imb_30", "ofi_10n", "ofi_60n", "trade_imb_10", "trade_imb_60",
            "intensity_60", "spread_bp", "ret_past_10", "ret_past_60", "ret_past_300",
            "rv_60", "rv_300", "market_id"]


def build(market_id: int, market: str) -> pd.DataFrame:
    ob = of.read("orderbook", market)
    g = of.features(ob, of.read("trade", market))
    l1 = pd.merge_asof(g[["ts"]], pd.DataFrame({"ts": ob["ts"], "l1": (ob["bid_size_0"] + ob["ask_size_0"]) / 2}), on="ts")["l1"]
    depth = l1.rolling(300, min_periods=60).mean()
    g["ofi_10n"] = g["ofi_10"] / depth
    g["ofi_60n"] = g["ofi_60"] / depth
    vol60 = g["vol_cum"] - g["vol_cum"].shift(60)
    g["intensity_60"] = vol60 / vol60.rolling(3600, min_periods=600).mean()
    g["spread_bp"] = (g["ask"] - g["bid"]) / g["mid"] * 1e4
    r1 = np.log(g["mid"]).diff() * 1e4
    for w in (10, 60, 300):
        g[f"ret_past_{w}"] = (g["mid"] / g["mid"].shift(w) - 1) * 1e4
    g["rv_60"], g["rv_300"] = r1.rolling(60).std(), r1.rolling(300).std()
    g["market_id"] = market_id
    g["market"] = market
    g["past_ok"] = g["fresh"].astype(float).rolling(301).min().astype(bool)
    return g[g["ts"] // 1000 % STEP == 0]


def trades(test: pd.DataFrame, prob: np.ndarray, h: int, threshold: float) -> pd.Series:
    """Net returns (bp) of non-overlapping entries where prob >= threshold, per market."""
    out = []
    for _, d in test.assign(prob=prob).groupby("market"):
        free_at = -np.inf
        for ts, p, r in zip(d["ts"].to_numpy(), d["prob"].to_numpy(), d[f"long_{h}"].to_numpy()):
            if p >= threshold and ts >= free_at:
                out.append(r)
                free_at = ts + h * 1000
    return pd.Series(out, dtype=float)


def fit(train: pd.DataFrame, h: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0,
                                           categorical_features=[FEATURES.index("market_id")],
                                           random_state=0)
    return model.fit(train[FEATURES], train[f"long_{h}"] > 0)


def main() -> None:
    of.HORIZONS = HORIZONS   # orderflow builds labels for these
    data = pd.concat([build(i, m) for i, m in enumerate(of.MARKETS)], ignore_index=True)
    data["day"] = pd.to_datetime(data["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.date
    days = sorted(data["day"].unique())
    print(f"{(data['ts'].max() - data['ts'].min()) / 3.6e6:.1f}h of data, days {days[0]} .. {days[-1]}")
    rv = data[data["market"] == "KRW-BTC"].groupby("day")["rv_60"].median()
    print("BTC median 1-min realized vol by day (bp/s): " + ", ".join(f"{d:%m-%d} {v:.2f}" for d, v in rv.items()))

    for h in HORIZONS:
        d = data[data["fresh"] & data["past_ok"] & data[f"fresh_fut_{h}"]].dropna(subset=FEATURES + [f"long_{h}"])
        print(f"\n== H={h}s: walk-forward, train on all earlier days, test on the next day")
        print(f"   {'test day':>8} {'base':>6} {'AUC':>6} {'AUC l1':>6} | {'top10% n':>8} {'win':>5} {'mean':>7} | "
              f"{'top1% n':>7} {'win':>5} {'mean':>7} | {'always n':>8} {'win':>5} {'mean':>7}")
        pooled = {"top10%": [], "top1%": [], "always": []}
        for day in days[2:]:                       # need at least two days of training data
            test = d[d["day"] == day]
            train = d[(d["day"] < day) & (d["ts"] < test["ts"].min() - h * 1000)]
            y_te = test[f"long_{h}"] > 0
            if (train[f"long_{h}"] > 0).sum() < 50 or len(test) == 0:
                continue
            model = fit(train, h)
            prob = model.predict_proba(test[FEATURES])[:, 1]
            p_train = model.predict_proba(train[FEATURES])[:, 1]
            both = 0 < y_te.sum() < len(y_te)
            auc = roc_auc_score(y_te, prob) if both else np.nan
            auc_l1 = roc_auc_score(y_te, test["imb_l1"]) if both else np.nan
            line = f"   {day:%m-%d}   {y_te.mean():>6.1%} {auc:>6.3f} {auc_l1:>6.3f}"
            for name, q in (("top10%", 0.9), ("top1%", 0.99)):
                r = trades(test, prob, h, np.quantile(p_train, q))   # threshold from train only
                pooled[name].append(r)
                line += f" | {len(r):>8} {(r > 0).mean() if len(r) else np.nan:>5.0%} {r.mean() if len(r) else np.nan:>+6.1f}"
            r = trades(test, np.ones(len(test)), h, 0.5)
            pooled["always"].append(r)
            line += f" | {len(r):>8} {(r > 0).mean():>5.0%} {r.mean():>+6.1f}"
            print(line)
        for name, rs in pooled.items():
            r = pd.concat(rs) if rs else pd.Series(dtype=float)
            if len(r):
                print(f"   pooled {name:>7}: {len(r)} trades, win {(r > 0).mean():.1%}, mean {r.mean():+.1f}bp, total {r.sum():+.0f}bp")


if __name__ == "__main__":
    main()
