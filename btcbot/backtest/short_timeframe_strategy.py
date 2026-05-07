"""
Rule-based BTC short-timeframe backtest for Upbit spot trading.

The strategy is long/flat only:
  - 5m candles drive execution and quick momentum confirmation.
  - 15m candles define short-term trend/momentum.
  - 1h candles act as the broader trend filter.

Examples
--------
Run one backtest and save charts:
    python -m btcbot.backtest.short_timeframe_strategy --candles 5000 --plot

Search a small parameter grid:
    python -m btcbot.backtest.short_timeframe_strategy --candles 8000 --optimize --plot
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from btcbot.config import BACKTEST, LOG_DIR, SYMBOL
from btcbot.data.fetch_upbit import fetch_all_timeframes

logger = logging.getLogger(__name__)

RESULT_DIR = LOG_DIR / "short_timeframe_backtests"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StrategyParams:
    """Tunable parameters for the 5m/15m BTC strategy."""

    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.008
    max_hold_bars: int = 6  # 6 * 5m = 30 minutes
    rsi_min: float = 0.45
    rsi_max: float = 0.75
    mom_5m_min: float = 0.0
    macd_15m_min: float = 0.0
    ma_diff_15m_min: float = 0.0
    ma_diff_1h_min: float = 0.0
    cooldown_bars: int = 1


@dataclass(frozen=True)
class BacktestParams:
    initial_capital: float = float(BACKTEST["initial_capital"])
    fee: float = float(BACKTEST["fee"])
    slippage: float = float(BACKTEST["slippage"])


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = (-delta.clip(upper=0)).fillna(0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0) / 100.0


def _feature_frame(df: pd.DataFrame, prefix: str, mom_period: int) -> pd.DataFrame:
    d = df[["time", "open", "high", "low", "close", "volume"]].copy()
    d = d.sort_values("time").reset_index(drop=True)
    c = d["close"]

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    out = pd.DataFrame({"time": d["time"]})
    out[f"{prefix}_ret"] = np.log(c).diff().fillna(0.0)
    out[f"{prefix}_mom"] = c.pct_change(mom_period).fillna(0.0)
    out[f"{prefix}_rsi"] = _rsi(c, 14)
    out[f"{prefix}_macd"] = ((macd - macd_signal) / c).fillna(0.0)
    out[f"{prefix}_ma_diff"] = ((ema20 - ema50) / c).fillna(0.0)
    return out


def _ensure_utc_time(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d["time"].dt.tz is None:
        d["time"] = d["time"].dt.tz_localize("UTC")
    return d


def prepare_strategy_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge 5m execution candles with latest closed 15m and 1h signals."""
    required = {"5m", "15m", "1h"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required timeframes: {sorted(missing)}")

    anchor = _ensure_utc_time(data["5m"]).sort_values("time").reset_index(drop=True)
    frame = anchor[["time", "open", "high", "low", "close", "volume"]].copy()

    features = [
        _feature_frame(anchor, "5m", mom_period=3),
        _feature_frame(_ensure_utc_time(data["15m"]), "15m", mom_period=2),
        _feature_frame(_ensure_utc_time(data["1h"]), "1h", mom_period=1),
    ]

    for feat in features:
        feat = _ensure_utc_time(feat).sort_values("time")
        frame = pd.merge_asof(
            frame.sort_values("time"),
            feat.sort_values("time"),
            on="time",
            direction="backward",
        )

    frame = frame.dropna().reset_index(drop=True)
    return frame


def entry_signal(row: pd.Series, params: StrategyParams) -> bool:
    return bool(
        row["1h_ma_diff"] > params.ma_diff_1h_min
        and row["15m_ma_diff"] > params.ma_diff_15m_min
        and row["15m_macd"] > params.macd_15m_min
        and row["5m_mom"] > params.mom_5m_min
        and params.rsi_min <= row["15m_rsi"] <= params.rsi_max
    )


def early_exit_signal(row: pd.Series, params: StrategyParams) -> bool:
    return bool(
        row["5m_mom"] < 0.0
        or row["15m_macd"] < params.macd_15m_min
        or row["15m_ma_diff"] < params.ma_diff_15m_min
    )


def run_backtest(
    frame: pd.DataFrame,
    strategy: StrategyParams | None = None,
    costs: BacktestParams | None = None,
) -> dict:
    strategy = strategy or StrategyParams()
    costs = costs or BacktestParams()

    equity = float(costs.initial_capital)
    units = 0.0
    position = 0
    entry_price = 0.0
    entry_equity = 0.0
    entry_time = None
    entry_idx = -1
    cooldown_until = -1

    trades: list[dict] = []
    curve: list[dict] = []
    actions: list[dict] = []

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal equity, units, position, entry_price, entry_equity, entry_time, entry_idx, cooldown_until

        exit_exec = price * (1 - costs.slippage)
        gross = units * exit_exec
        equity = gross * (1 - costs.fee)
        pnl_krw = equity - entry_equity
        pnl_pct = (equity / entry_equity - 1) if entry_equity > 0 else 0.0

        row = frame.iloc[i]
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": row["time"],
                "entry_price": entry_price,
                "exit_price": exit_exec,
                "hold_bars": i - entry_idx,
                "hold_minutes": (i - entry_idx) * 5,
                "pnl_pct": pnl_pct,
                "pnl_krw": pnl_krw,
                "reason": reason,
            }
        )
        actions.append({"time": row["time"], "action": "SELL", "price": exit_exec, "reason": reason})

        units = 0.0
        position = 0
        entry_price = 0.0
        entry_equity = 0.0
        entry_time = None
        entry_idx = -1
        cooldown_until = i + strategy.cooldown_bars

    for i in range(1, len(frame)):
        prev = frame.iloc[i - 1]
        row = frame.iloc[i]

        if position == 1:
            stop_price = entry_price * (1 - strategy.stop_loss_pct)
            target_price = entry_price * (1 + strategy.take_profit_pct)
            held_bars = i - entry_idx

            if row["low"] <= stop_price:
                close_position(i, stop_price, "stop_loss")
            elif row["high"] >= target_price:
                close_position(i, target_price, "take_profit")
            elif held_bars >= strategy.max_hold_bars:
                close_position(i, float(row["open"]), "max_hold")
            elif early_exit_signal(prev, strategy):
                close_position(i, float(row["open"]), "momentum_exit")

        if position == 0 and i > cooldown_until and entry_signal(prev, strategy):
            entry_exec = float(row["open"]) * (1 + costs.slippage)
            entry_equity = equity
            units = equity * (1 - costs.fee) / entry_exec
            entry_price = entry_exec
            entry_time = row["time"]
            entry_idx = i
            position = 1
            actions.append({"time": row["time"], "action": "BUY", "price": entry_exec, "reason": "entry"})

        marked_equity = units * float(row["close"]) if position == 1 else equity
        curve.append(
            {
                "time": row["time"],
                "close": row["close"],
                "equity": marked_equity,
                "position": position,
            }
        )

    if position == 1:
        close_position(len(frame) - 1, float(frame.iloc[-1]["close"]), "end_of_data")
        if curve:
            curve[-1]["equity"] = equity
            curve[-1]["position"] = 0

    equity_curve = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    actions_df = pd.DataFrame(actions)
    metrics = compute_metrics(equity_curve, trades_df, costs.initial_capital)

    return {
        "frame": frame,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "actions": actions_df,
        "metrics": metrics,
        "strategy": asdict(strategy),
        "costs": asdict(costs),
    }


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict:
    if equity_curve.empty:
        return {"note": "no equity curve"}

    eq = equity_curve["equity"].astype(float)
    returns = eq.pct_change().fillna(0.0)
    peak = eq.cummax()
    drawdown = eq / peak - 1
    total_return = eq.iloc[-1] / initial_capital - 1

    bars_per_year = 365 * 24 * 12  # 5m bars
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

    if trades.empty:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
    else:
        pnl = trades["pnl_krw"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float((pnl > 0).mean())
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else 0.0

    return {
        "final_equity_krw": round(float(eq.iloc[-1]), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "sharpe_ratio": round(sharpe, 3),
        "n_trades": int(len(trades)),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_win_krw": round(avg_win, 2),
        "avg_loss_krw": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
    }


def parameter_grid() -> Iterable[StrategyParams]:
    grid = {
        "stop_loss_pct": [0.004, 0.005, 0.006],
        "take_profit_pct": [0.006, 0.008, 0.01],
        "max_hold_bars": [3, 6],
        "rsi_max": [0.7, 0.75, 0.8],
        "mom_5m_min": [0.0, 0.001],
    }
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield replace(StrategyParams(), **dict(zip(keys, values)))


def optimize_parameters(frame: pd.DataFrame, costs: BacktestParams | None = None) -> pd.DataFrame:
    rows = []
    for params in parameter_grid():
        result = run_backtest(frame, strategy=params, costs=costs)
        rows.append({**asdict(params), **result["metrics"]})

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    # Prefer strong return, but penalize drawdown and tiny trade counts.
    trade_penalty = np.where(results["n_trades"] < 5, 100.0, 0.0)
    results["score"] = (
        results["total_return_pct"]
        + results["sharpe_ratio"] * 2.0
        + results["max_drawdown_pct"] * 0.5
        - trade_penalty
    )
    return results.sort_values("score", ascending=False).reset_index(drop=True)


def save_outputs(result: dict, label: str, plot: bool = False) -> dict[str, Path]:
    paths = {
        "equity": RESULT_DIR / f"{label}_equity.csv",
        "trades": RESULT_DIR / f"{label}_trades.csv",
        "actions": RESULT_DIR / f"{label}_actions.csv",
    }
    result["equity_curve"].to_csv(paths["equity"], index=False)
    result["trades"].to_csv(paths["trades"], index=False)
    result["actions"].to_csv(paths["actions"], index=False)

    if plot:
        paths["plot"] = plot_backtest(result, RESULT_DIR / f"{label}_plot.png")
    return paths


def plot_backtest(result: dict, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    frame = result["frame"]
    equity = result["equity_curve"]
    actions = result["actions"]
    trades = result["trades"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    axes[0].plot(frame["time"], frame["close"], color="black", linewidth=0.9, label="BTC close")
    if not actions.empty:
        buys = actions[actions["action"] == "BUY"]
        sells = actions[actions["action"] == "SELL"]
        axes[0].scatter(buys["time"], buys["price"], marker="^", color="green", s=45, label="Buy")
        axes[0].scatter(sells["time"], sells["price"], marker="v", color="red", s=45, label="Sell")
    axes[0].set_title(f"{SYMBOL} 5m Strategy Actions")
    axes[0].set_ylabel("Price (KRW)")
    axes[0].legend(loc="best")

    axes[1].plot(equity["time"], equity["equity"], color="blue", linewidth=1.0, label="Equity")
    axes[1].set_ylabel("Equity (KRW)")
    axes[1].legend(loc="best")

    eq = equity["equity"].astype(float)
    drawdown = eq / eq.cummax() - 1
    axes[2].fill_between(equity["time"], drawdown * 100, color="red", alpha=0.25, label="Drawdown")
    if not trades.empty:
        axes[2].scatter(trades["exit_time"], trades["pnl_pct"] * 100, c=np.where(trades["pnl_pct"] > 0, "green", "red"), s=25, label="Trade PnL")
    axes[2].set_ylabel("Drawdown / PnL (%)")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def print_summary(metrics: dict) -> None:
    print("\nBacktest results")
    print("----------------")
    for key, value in metrics.items():
        print(f"{key:>20}: {value}")


def load_data(candles: int, refresh: bool) -> dict[str, pd.DataFrame]:
    return fetch_all_timeframes(
        timeframes={"5m": "minutes5", "15m": "minutes15", "1h": "minutes60"},
        n_candles=candles,
        refresh=refresh,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a BTC 5-30 minute rule-based strategy.")
    parser.add_argument("--candles", type=int, default=5000, help="Candles to load per timeframe.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached candles and fetch fresh data.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG chart with price, actions, equity, and drawdown.")
    parser.add_argument("--optimize", action="store_true", help="Run a small parameter search before the final backtest.")
    parser.add_argument("--label", default="latest", help="Output filename label.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()

    data = load_data(candles=args.candles, refresh=args.refresh)
    frame = prepare_strategy_frame(data)
    costs = BacktestParams()

    strategy = StrategyParams()
    if args.optimize:
        search = optimize_parameters(frame, costs)
        search_path = RESULT_DIR / f"{args.label}_parameter_search.csv"
        search.to_csv(search_path, index=False)
        print(f"Saved parameter search: {search_path}")
        if not search.empty:
            best = search.iloc[0]
            strategy = StrategyParams(
                stop_loss_pct=float(best["stop_loss_pct"]),
                take_profit_pct=float(best["take_profit_pct"]),
                max_hold_bars=int(best["max_hold_bars"]),
                rsi_max=float(best["rsi_max"]),
                mom_5m_min=float(best["mom_5m_min"]),
            )
            print("Best parameters:", asdict(strategy))

    result = run_backtest(frame, strategy=strategy, costs=costs)
    print_summary(result["metrics"])
    paths = save_outputs(result, label=args.label, plot=args.plot)
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
