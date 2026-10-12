"""
Hourly trading loop for one coin. Paper trading only for now.

    python -m coinbot.live            # run forever, wakes after every 1h close
    python -m coinbot.live --once     # one decision, then exit (cron / testing)

Each run: load closed 1h bars -> (re)select params every RESELECT on the last
TRAIN bars, as in walk-forward -> target weight from the last closed bar ->
trade the difference. Touch data/live/HALT to stop placing orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

from coinbot import backtest, metrics, upbit, validate
from coinbot.evaluate import TEST, TRAIN
from coinbot.strategies import GRIDS, STRATEGIES

logger = logging.getLogger(__name__)

LIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "live"
MARKET = "KRW-BTC"
STRATEGY = "donchian"
RESELECT = pd.Timedelta(hours=TEST)       # re-pick params every 90 days
MAX_POSITION_KRW = 1_000_000              # cap on capital the bot may put in the coin
MIN_ORDER_KRW = 5_000                     # Upbit minimum
MAX_BAR_AGE = pd.Timedelta(hours=2)       # older last bar = broken feed, don't trade
WAKE_AFTER_CLOSE = 25                     # seconds after the hour


@dataclass
class State:
    cash: float = 1_000_000.0
    coin: float = 0.0
    params: dict = field(default_factory=dict)
    params_at: str | None = None
    last_bar: str | None = None           # close_time of the last bar acted on


class PaperBroker:
    """Fills market orders at `price` with slippage and fee, against State balances."""

    def __init__(self, state: State, fee: float = backtest.FEE, slippage: float = backtest.SLIPPAGE):
        self.state, self.fee, self.slippage = state, fee, slippage

    def buy(self, krw: float, price: float) -> dict:
        krw = min(krw, self.state.cash)
        fill = price * (1 + self.slippage)
        qty = krw * (1 - self.fee) / fill
        self.state.cash -= krw
        self.state.coin += qty
        return {"side": "buy", "krw": krw, "qty": qty, "fill": fill}

    def sell(self, qty: float, price: float) -> dict:
        qty = min(qty, self.state.coin)
        fill = price * (1 - self.slippage)
        krw = qty * fill * (1 - self.fee)
        self.state.coin -= qty
        self.state.cash += krw
        return {"side": "sell", "krw": krw, "qty": qty, "fill": fill}


def _score(res: backtest.Result) -> float:
    return metrics.sharpe(res.bars["ret"].to_numpy(), 24 * 365)


def step(state: State, broker: PaperBroker, bars: pd.DataFrame, price: float,
         now: pd.Timestamp, halted: bool = False) -> dict:
    """One decision on the latest closed bar. Mutates state; returns a log record."""
    bar_close = bars["close_time"].iloc[-1]
    rec = {"now": str(now), "bar_close": str(bar_close), "price": price}
    if state.last_bar == str(bar_close):
        return {**rec, "action": "skip", "reason": "bar already handled"}
    if now - bar_close > MAX_BAR_AGE:
        return {**rec, "action": "skip", "reason": "stale data"}

    if state.params_at is None or now - pd.Timestamp(state.params_at) >= RESELECT:
        state.params, score = validate.select_params(
            STRATEGIES[STRATEGY], bars, GRIDS[STRATEGY], len(bars), TRAIN, _score)
        state.params_at = str(now)
        rec["reselected"] = {"params": state.params, "train_sharpe": score}

    target = float(STRATEGIES[STRATEGY](bars, **state.params).iloc[-1])
    equity = state.cash + state.coin * price
    desired = target * min(equity, MAX_POSITION_KRW)
    diff = desired - state.coin * price
    rec.update(target=target, equity=equity, cash=state.cash, coin=state.coin)
    state.last_bar = str(bar_close)

    if halted:
        return {**rec, "action": "halted"}
    if target == 0.0 and state.coin * price >= MIN_ORDER_KRW:
        return {**rec, "action": "order", "order": broker.sell(state.coin, price)}
    if abs(diff) < MIN_ORDER_KRW:
        return {**rec, "action": "hold"}
    order = broker.buy(diff, price) if diff > 0 else broker.sell(-diff / price, price)
    return {**rec, "action": "order", "order": order}


def _ticker(market: str) -> float:
    resp = requests.get("https://api.upbit.com/v1/ticker", params={"markets": market}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()[0]["trade_price"])


def run_once() -> dict:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LIVE_DIR / "state.json"
    state = State(**json.loads(state_path.read_text())) if state_path.exists() else State()
    bars = upbit.load(MARKET, "1h")
    rec = step(state, PaperBroker(state), bars, _ticker(MARKET),
               pd.Timestamp.now(tz="UTC"), halted=(LIVE_DIR / "HALT").exists())
    state_path.write_text(json.dumps(asdict(state), indent=2))
    with open(LIVE_DIR / "log.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    while True:
        try:
            rec = run_once()
            logger.info(json.dumps(rec, default=str))
        except Exception:
            logger.exception("run failed; will retry next hour")
        if args.once:
            return
        now = time.time()
        time.sleep(3600 - now % 3600 + WAKE_AFTER_CLOSE)


if __name__ == "__main__":
    main()
