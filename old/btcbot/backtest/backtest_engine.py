"""
BTC Backtesting Engine — Upbit / Spot Trading

Adapted from tradingbot/backtest/backtest_engine.py for KRW-BTC:
  • Flat 0.05 % taker fee per side (no pip scaling)
  • Optional slippage model
  • Walk-forward validation
  • Standard performance metrics (Sharpe, Sortino, Max DD, Win-rate)

Usage
-----
    from btcbot.backtest.backtest_engine import BTCBacktester
    backtester = BTCBacktester(agent=my_dreamer_agent,
                               feats=feats, rets=rets, df=df_anchor)
    results = backtester.run()
    wf      = backtester.walk_forward(train_steps=4000, test_steps=1000)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    fee:             float = 0.0005    # Upbit taker fee (0.05 % per side)
    slippage:        float = 0.0002    # conservative slippage estimate
    initial_capital: float = 1_000_000 # KRW
    # Walk-forward parameters
    train_steps:     int   = 4000      # ~166 days at 1h
    test_steps:      int   = 1000      # ~41 days at 1h


# ── Main backtester ───────────────────────────────────────────────────────────

class BTCBacktester:
    """
    Rigorous backtester for KRW-BTC spot trading.

    The agent's act() method is called at every step and should accept:
        obs  – (obs_dim,) float32 observation vector
        h, z – DreamerV3 latent state (None on first call)
    and return:
        action (int: 0=flat, 1=long),  new (h, z)

    Alternatively, for simple policy-less baselines, act(obs) → int is also
    supported (the backtester checks the signature).
    """

    def __init__(
        self,
        agent,
        feats:  np.ndarray,
        rets:   np.ndarray,
        df:     pd.DataFrame,
        config: BacktestConfig | None = None,
        window: int = 64,
    ):
        """
        Args:
            agent:  Any object with an act(obs, h, z) method
            feats:  (T, F) feature matrix from make_features
            rets:   (T,)   1h log-returns
            df:     (T,)   aligned 1h OHLCV DataFrame (for timestamps / prices)
            config: BacktestConfig (uses defaults if None)
            window: observation look-back window
        """
        self.agent  = agent
        self.feats  = feats
        self.rets   = rets
        self.df     = df.reset_index(drop=True)
        self.cfg    = config or BacktestConfig()
        self.window = window
        self.T      = len(rets)

    # ── Observation helper ─────────────────────────────────────────────────────

    def _obs(self, step: int, position: int) -> np.ndarray:
        start = max(0, step - self.window + 1)
        chunk = self.feats[start: step + 1]
        if len(chunk) < self.window:
            pad   = np.zeros((self.window - len(chunk), self.feats.shape[1]), dtype=np.float32)
            chunk = np.vstack([pad, chunk])
        return np.append(chunk.flatten(), float(position)).astype(np.float32)

    # ── Cost model ────────────────────────────────────────────────────────────

    def _round_trip_cost(self, position_change: int) -> float:
        """Total % cost for entering or exiting a position."""
        if position_change == 0:
            return 0.0
        return self.cfg.fee + self.cfg.slippage   # one-way cost

    # ── Core backtest loop ────────────────────────────────────────────────────

    def run(self, start: int = 0, end: int | None = None) -> dict:
        """
        Run the agent over feats[start:end] and collect statistics.

        Returns
        -------
        dict with keys:
            trades        – list of trade dicts
            equity_curve  – list of equity values (KRW)
            step_returns  – list of per-step returns
            metrics       – summary dict (Sharpe, Sortino, MaxDD, etc.)
        """
        end = end or self.T
        equity     = float(self.cfg.initial_capital)
        position   = 0
        entry_price = 0.0
        entry_time  = None
        h, z       = None, None

        trades:       list[dict] = []
        equity_curve: list[float] = []
        step_returns: list[float] = []

        for t in range(start, end):
            obs = self._obs(t, position)

            # Support both DreamerV3 (returns h,z) and simple agents
            try:
                action, (h, z) = self.agent.act(obs, h, z)
                # DreamerV3 returns one-hot; convert to int
                if hasattr(action, "__len__"):
                    action = int(np.argmax(action))
                else:
                    action = int(action)
            except (TypeError, ValueError):
                action = int(self.agent.act(obs))

            # ── Execute position change ─────────────────────────────────
            if action != position:
                cost = self._round_trip_cost(abs(action - position))

                # Close existing position
                if position == 1 and t < self.T:
                    exit_price = float(self.df.loc[t, "close"])
                    pnl_pct    = np.log(exit_price / entry_price) - cost
                    equity    *= np.exp(pnl_pct)
                    trades.append({
                        "entry_time":  entry_time,
                        "exit_time":   self.df.loc[t, "time"],
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "pnl_pct":     pnl_pct,
                        "pnl_krw":     equity * (np.exp(pnl_pct) - 1),
                    })

                # Open new position
                if action == 1:
                    entry_price = float(self.df.loc[t, "close"])
                    entry_time  = self.df.loc[t, "time"]
                    equity     *= (1 - cost)   # entry cost

                position = action

            # ── Step return (mark-to-market) ───────────────────────────
            step_ret   = position * self.rets[t]
            step_returns.append(float(step_ret))
            equity_curve.append(equity * np.exp(sum(step_returns)))

        results = {
            "trades":       trades,
            "equity_curve": equity_curve,
            "step_returns": np.array(step_returns, dtype=np.float32),
            "metrics":      self._metrics(equity_curve, step_returns, trades),
        }
        self._print_summary(results["metrics"])
        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics(equity_curve: list, step_returns: list, trades: list) -> dict:
        r  = np.array(step_returns, dtype=np.float64)
        eq = np.array(equity_curve, dtype=np.float64)

        if len(r) < 2 or r.std() == 0:
            return {"note": "insufficient data"}

        # Annualise assuming 1h bars → 8760 bars/year
        ann_factor  = np.sqrt(8760)
        sharpe      = r.mean() / (r.std() + 1e-10) * ann_factor

        downside    = r[r < 0]
        sortino     = (r.mean() / (downside.std() + 1e-10)) * ann_factor if len(downside) > 0 else 0.0

        # Max drawdown
        peak        = np.maximum.accumulate(eq)
        drawdown    = (eq - peak) / peak
        max_dd      = float(drawdown.min())

        total_ret   = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0

        # Trade stats
        if trades:
            pnls     = [t["pnl_pct"] for t in trades]
            win_rate = float(np.mean([p > 0 for p in pnls]))
            avg_win  = float(np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0)
            avg_loss = float(np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0)
        else:
            win_rate = avg_win = avg_loss = 0.0

        return {
            "total_return_pct": round(total_ret * 100, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "sortino_ratio":    round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_trades":         len(trades),
            "win_rate_pct":     round(win_rate * 100, 1),
            "avg_win_pct":      round(avg_win * 100, 3),
            "avg_loss_pct":     round(avg_loss * 100, 3),
        }

    @staticmethod
    def _print_summary(m: dict):
        logger.info("── Backtest Results ─────────────────────────────────")
        for k, v in m.items():
            logger.info(f"   {k:<22} {v}")
        logger.info("─────────────────────────────────────────────────────")

    # ── Walk-forward validation ───────────────────────────────────────────────

    def walk_forward(
        self,
        train_steps: int | None = None,
        test_steps:  int | None = None,
        retrain_fn=None,
    ) -> list[dict]:
        """
        Time-series walk-forward cross-validation.

        For each window:
          1. Call retrain_fn(agent, feats[train], rets[train]) if provided
          2. Run backtest on feats[test]

        Args:
            train_steps: bars used for each training window
            test_steps:  bars used for each test window
            retrain_fn:  optional callable(agent, feats, rets) → None

        Returns:
            List of result dicts, one per walk-forward window.
        """
        train_steps = train_steps or self.cfg.train_steps
        test_steps  = test_steps  or self.cfg.test_steps

        all_results = []
        cursor = 0
        window_idx = 0

        while cursor + train_steps + test_steps <= self.T:
            train_end = cursor + train_steps
            test_end  = train_end + test_steps

            logger.info(
                f"Walk-forward window {window_idx + 1}: "
                f"train [{cursor}:{train_end}]  test [{train_end}:{test_end}]"
            )

            if retrain_fn is not None:
                retrain_fn(
                    self.agent,
                    self.feats[cursor:train_end],
                    self.rets[cursor:train_end],
                )

            result = self.run(start=train_end, end=test_end)
            result["window"] = window_idx
            result["train_range"] = (cursor, train_end)
            result["test_range"]  = (train_end, test_end)
            all_results.append(result)

            cursor     += test_steps
            window_idx += 1

        logger.info(
            f"Walk-forward complete: {window_idx} windows  "
            f"avg Sharpe = {np.mean([r['metrics'].get('sharpe_ratio', 0) for r in all_results]):.3f}"
        )
        return all_results
