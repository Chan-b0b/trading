"""
Walk-forward evaluation of every strategy on KRW-BTC 1h bars.

    python -m coinbot.evaluate [--strategies tsmom rules] [--shifts 200]

Params are picked on the previous `train` bars by Sharpe and applied to the
next `test` bars. Only the stitched out-of-sample period is judged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from coinbot import backtest, metrics, upbit, validate
from coinbot.strategies import GRIDS, STRATEGIES

TRAIN = 24 * 365 * 2   # 2 years
TEST = 24 * 90         # 3 months
TRIALS = Path(__file__).resolve().parents[1] / "research" / "trials.csv"


def log_trials(table: pd.DataFrame, bars: pd.DataFrame, shifts: int) -> int:
    """Append every evaluated strategy to the trial log; return distinct strategies tried."""
    rows = table.drop(index="buy_and_hold").reset_index()[
        ["strategy", "sharpe", "p_value", "cagr", "max_drawdown"]]
    rows.insert(0, "run_at", pd.Timestamp.now(tz="UTC").floor("s"))
    rows["grid_size"] = [len(GRIDS[s]) for s in rows["strategy"]]
    rows["oos_start"] = bars["open_time"].iloc[TRAIN]
    rows["oos_end"] = bars["close_time"].iloc[-1]
    rows["shifts"] = shifts
    TRIALS.parent.mkdir(exist_ok=True)
    rows.to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False)
    return pd.read_csv(TRIALS)["strategy"].nunique()


def evaluate(bars: pd.DataFrame, names: list[str], shifts: int,
             verbose: bool = True) -> pd.DataFrame:
    bpy = metrics.bars_per_year(bars)
    oos = bars.iloc[TRAIN:].reset_index(drop=True)
    score = lambda r: metrics.sharpe(r.bars["ret"].to_numpy(), bpy)  # noqa: E731

    signals = {"buy_and_hold": pd.Series(1.0, index=oos.index)}
    for name in names:
        validate.check_causal(STRATEGIES[name], bars, GRIDS[name][0])
        sig, folds = validate.walk_forward(STRATEGIES[name], bars, GRIDS[name],
                                           TRAIN, TEST, score)
        signals[name] = sig.iloc[TRAIN:].reset_index(drop=True)
        if verbose:
            print(f"\n{name}: params chosen per fold")
            print(folds.drop(columns=["test_end"]).to_string(index=False))

    rows = []
    for name, sig in signals.items():
        res = backtest.run(oos, sig)
        r = metrics.report(res, bpy)
        r["sharpe_lo"], r["sharpe_hi"] = validate.bootstrap_sharpe(res.bars["ret"].to_numpy(), bpy, n=200)
        if name != "buy_and_hold":
            t = validate.shift_test(oos, sig, bpy, n=shifts)
            r["null_95"], r["p_value"] = t["null_95"], t["p_value"]
        rows.append({"strategy": name, **r})
    return pd.DataFrame(rows).set_index("strategy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--shifts", type=int, default=200)
    args = ap.parse_args()

    bars = upbit.load("KRW-BTC", "1h")
    print(f"OOS period: {bars['open_time'].iloc[TRAIN]} -> {bars['close_time'].iloc[-1]}")
    table = evaluate(bars, args.strategies, args.shifts)
    cols = ["cagr", "sharpe", "sharpe_lo", "sharpe_hi", "null_95", "p_value", "max_drawdown",
            "calmar", "exposure", "n_trades", "win_rate", "profit_factor", "turnover_per_year"]
    pd.set_option("display.width", 200)
    print("\n" + table[cols].round(3).to_string())
    n = log_trials(table, bars, args.shifts)
    print(f"\n{n} distinct strategies in {TRIALS.name}: Bonferroni threshold p < {0.05 / n:.4f}")


if __name__ == "__main__":
    main()
