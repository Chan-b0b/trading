"""
Multi-coin portfolio: top-N coins by trailing traded value, each slot gets
1/N of capital and is long or in cash by its own strategy signal.

    python -m coinbot.portfolio [--n 5] [--shifts 1000]

Membership is decided every REBALANCE bars from the trailing 30-day KRW
value known at that bar, among coins with at least TRAIN bars of history
(so walk-forward has picked their params). Delisted coins are unavailable
from Upbit, so the candidate pool has survivorship bias.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from coinbot import backtest, metrics, upbit, validate
from coinbot.cross_asset import OUT as CROSS_ASSET_CSV
from coinbot.evaluate import TEST, TRAIN
from coinbot.strategies import GRIDS, STRATEGIES

REBALANCE = TEST           # quarterly
VALUE_WINDOW = 24 * 30     # trailing 30 days
BPY = 24 * 365


def panel(markets: list[str]) -> dict[str, pd.DataFrame]:
    """Wide open/close/value frames on one hourly grid (NaN before listing)."""
    frames = {m: upbit.load(m, "1h").set_index("open_time") for m in markets}
    index = pd.date_range(min(f.index[0] for f in frames.values()),
                          max(f.index[-1] for f in frames.values()), freq="1h")
    return {col: pd.DataFrame({m: f[col] for m, f in frames.items()}).reindex(index)
            for col in ("open", "close", "value")}


def membership(value: pd.DataFrame, eligible: pd.DataFrame, n: int, start: int) -> pd.DataFrame:
    """1.0 for the top-n eligible coins by trailing value, refreshed every REBALANCE bars."""
    trailing = value.fillna(0).rolling(VALUE_WINDOW, min_periods=1).sum().where(eligible)
    member = pd.DataFrame(np.nan, index=value.index, columns=value.columns)
    for t in range(start, len(value), REBALANCE):
        top = trailing.iloc[t].nlargest(n).index
        member.iloc[t] = 0.0
        member.iloc[t, member.columns.get_indexer(top)] = 1.0
    return member.ffill().fillna(0.0)


def returns(open_: pd.DataFrame, close: pd.DataFrame, target: pd.DataFrame,
            cost: float = backtest.FEE + backtest.SLIPPAGE) -> np.ndarray:
    """
    Per-bar portfolio return: target[t] (decided at close t) is held from the
    open of t+1, as in coinbot.backtest. Contributions are summed across coins.
    """
    o, c = open_.to_numpy(), close.to_numpy()
    prev_c = np.vstack([o[:1], c[:-1]])
    gap = np.nan_to_num(o / prev_c - 1)
    intra = np.nan_to_num(c / o - 1)
    tgt = target.to_numpy()
    w = np.vstack([np.zeros((1, tgt.shape[1])), tgt[:-1]])
    w_prev = np.vstack([np.zeros((1, tgt.shape[1])), w[:-1]])
    return (w_prev * gap + w * intra - np.abs(w - w_prev) * cost).sum(axis=1)


def _wf_signal(market: str, strategy: str) -> pd.Series:
    bars = upbit.load(market, "1h")
    score = lambda r: metrics.sharpe(r.bars["ret"].to_numpy(), BPY)  # noqa: E731
    sig, _ = validate.walk_forward(STRATEGIES[strategy], bars, GRIDS[strategy], TRAIN, TEST, score)
    return pd.Series(sig.to_numpy(), index=bars["open_time"], name=market)


def report(ret: np.ndarray, target: pd.DataFrame) -> dict:
    exposure = target.sum(axis=1).to_numpy()
    res = backtest.Result(
        bars=pd.DataFrame({"ret": ret, "weight": exposure}),
        trades=pd.DataFrame(columns=["entry_time", "exit_time", "bars_held", "ret"]))
    r = metrics.report(res, BPY)
    r["sharpe_lo"], r["sharpe_hi"] = validate.bootstrap_sharpe(ret, BPY, n=200)
    r["avg_exposure"] = float(exposure.mean())
    r["turnover_per_year"] = float(target.diff().abs().sum(axis=1).sum() * BPY / len(target))
    return {k: r[k] for k in ("cagr", "sharpe", "sharpe_lo", "sharpe_hi", "sortino", "max_drawdown",
                              "calmar", "cvar_95", "avg_exposure", "turnover_per_year")}


def shift_test(p: dict, member: pd.DataFrame, signal: pd.DataFrame, n_slots: int,
               start: int, n: int, seed: int = 0) -> dict:
    """Shift every coin's strategy signal by the same random offset; keep membership."""
    rng = np.random.default_rng(seed)
    sl = slice(start, None)
    observed = metrics.sharpe(returns(p["open"], p["close"], member * signal / n_slots)[sl], BPY)
    T = len(signal) - start
    null = []
    for k in rng.integers(T // 20, T - T // 20, size=n):
        shifted = signal.copy()
        shifted.iloc[start:] = np.roll(signal.iloc[start:].to_numpy(), k, axis=0)
        null.append(metrics.sharpe(returns(p["open"], p["close"], member * shifted / n_slots)[sl], BPY))
    null = np.array(null)
    return {"sharpe": observed, "null_95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + n))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--strategy", default="donchian")
    ap.add_argument("--shifts", type=int, default=1000)
    args = ap.parse_args()

    markets = sorted(pd.read_csv(CROSS_ASSET_CSV)["market"].unique().tolist() + ["KRW-BTC"])
    p = panel(markets)
    with ProcessPoolExecutor() as pool:
        sigs = list(pool.map(_wf_signal, markets, [args.strategy] * len(markets)))
    signal = pd.DataFrame({s.name: s for s in sigs}).reindex(p["close"].index).fillna(0.0)

    first = {m: p["close"][m].first_valid_index() for m in markets}
    eligible = pd.DataFrame({m: p["close"].index >= first[m] + pd.Timedelta(hours=TRAIN)
                             for m in markets}, index=p["close"].index)
    start = int(np.argmax(eligible.sum(axis=1).to_numpy() >= args.n))
    member = membership(p["value"], eligible, args.n, start)
    sl = slice(start, None)
    print(f"Period: {p['close'].index[start]} -> {p['close'].index[-1]}  "
          f"(top {args.n} by 30d value, rebalanced every {REBALANCE // 24} days)")

    held = member.iloc[start:]
    counts = (held > 0).sum().sort_values(ascending=False)
    print("Share of time in portfolio:\n" + (counts[counts > 0] / len(held)).round(2).to_string())

    btc = pd.DataFrame(0.0, index=member.index, columns=markets)
    btc["KRW-BTC"] = 1.0
    cases = {
        f"top{args.n}_{args.strategy}": member * signal / args.n,
        f"top{args.n}_buy_and_hold": member / args.n,
        f"btc_{args.strategy}": btc * signal,
        "btc_buy_and_hold": btc,
    }
    rows = {}
    for name, target in cases.items():
        rows[name] = report(returns(p["open"], p["close"], target)[sl], target.iloc[sl])
    table = pd.DataFrame(rows).T

    t = shift_test(p, member, signal, args.n, start, args.shifts)
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string())
    print(f"\nShift test top{args.n}_{args.strategy}: sharpe {t['sharpe']:.3f}  "
          f"null_95 {t['null_95']:.3f}  p {t['p_value']:.4f}")


if __name__ == "__main__":
    main()
