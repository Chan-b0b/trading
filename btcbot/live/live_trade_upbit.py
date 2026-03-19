"""
Live Trading Bot — KRW-BTC on Upbit using DreamerV3 + MCTS

Flow (runs every POLL_INTERVAL seconds, aligned to candle close)
----------------------------------------------------------------
1.  Fetch latest N candles for each timeframe from Upbit REST API
2.  Re-compute feature matrix from scratch (rolling, no lookahead)
3.  Build observation for the current step
4.  Run MCTS (if use_mcts=True) or plain DreamerV3 actor to get action
5.  Compare desired position with current position
6.  If a trade is needed:
        - Cancel any open orders
        - Place a limit buy/sell order (or market as fallback)
7.  Log everything; sleep until next candle close

Installation
------------
    pip install pyupbit

Run
---
    cd /Users/chlee/vscode/trading
    python -m btcbot.live.live_trade_upbit

Safety
------
    • Positions are long or flat only (no leverage, no short)
    • Max position size is capped by LIVE["max_position_krw"]
    • The bot will NOT trade if the model file does not exist
    • Dry-run mode (DRY_RUN=True) logs orders without placing them
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Add tradingbot to Python path ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[3] / "tradingbot"))

from btcbot.config import (
    UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY,
    SYMBOL, LIVE, MCTS as MCTS_CFG, DREAMER, WINDOW,
    MODEL_DIR, LOG_DIR,
)
from btcbot.data.fetch_upbit      import fetch_all_timeframes
from btcbot.data.fetch_macro      import fetch_macro
from btcbot.data.fetch_fear_greed import fetch_fear_greed, get_current_fng
from btcbot.features.make_features import compute_features, build_observation

from ..models.dreamer_agent import DreamerV3Agent
from ..models.mcts          import MCTS

try:
    import pyupbit
    _PYUPBIT = True
except ImportError:
    _PYUPBIT = False
    print("⚠️  pyupbit not installed. Run: pip install pyupbit")
    print("    Continuing in DRY-RUN mode.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "live_trade.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Dry-run flag (set to False only when you are ready to trade real money) ───
DRY_RUN: bool = not (_PYUPBIT and bool(UPBIT_ACCESS_KEY) and bool(UPBIT_SECRET_KEY))

ACTION_NAMES = {0: "FLAT", 1: "LONG"}


# ── Upbit order helpers ───────────────────────────────────────────────────────

class UpbitTrader:
    """
    Thin wrapper around pyupbit for placing and managing orders.
    All methods are no-ops when DRY_RUN=True.
    """

    def __init__(self, access_key: str, secret_key: str, dry_run: bool = True):
        self.dry_run = dry_run
        if not dry_run and _PYUPBIT:
            self.upbit = pyupbit.Upbit(access_key, secret_key)
            logger.info("✅ Connected to Upbit API")
        else:
            self.upbit = None
            logger.info("🔶 DRY-RUN mode — no real orders will be placed")

    def get_balance_krw(self) -> float:
        if self.dry_run or self.upbit is None:
            return float(LIVE["max_position_krw"]) * 2
        return float(self.upbit.get_balance("KRW") or 0)

    def get_balance_btc(self) -> float:
        if self.dry_run or self.upbit is None:
            return 0.0
        return float(self.upbit.get_balance("BTC") or 0)

    def get_current_price(self) -> float:
        if not _PYUPBIT:
            return 0.0
        return float(pyupbit.get_current_price(SYMBOL) or 0)

    def buy_market(self, krw_amount: float) -> dict | None:
        logger.info(f"BUY  MARKET  {krw_amount:,.0f} KRW  [{SYMBOL}]")
        if self.dry_run or self.upbit is None:
            return {"type": "dry_run_buy", "krw": krw_amount}
        try:
            return self.upbit.buy_market_order(SYMBOL, krw_amount)
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return None

    def sell_market(self, btc_amount: float) -> dict | None:
        logger.info(f"SELL MARKET  {btc_amount:.8f} BTC  [{SYMBOL}]")
        if self.dry_run or self.upbit is None:
            return {"type": "dry_run_sell", "btc": btc_amount}
        try:
            return self.upbit.sell_market_order(SYMBOL, btc_amount)
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return None

    def cancel_all_orders(self):
        if self.dry_run or self.upbit is None:
            return
        try:
            orders = self.upbit.get_order(SYMBOL)
            if orders:
                for o in orders:
                    self.upbit.cancel_order(o["uuid"])
        except Exception as e:
            logger.warning(f"Could not cancel orders: {e}")


# ── Agent loader ──────────────────────────────────────────────────────────────

def load_agent(obs_dim: int, device: str = "cpu") -> DreamerV3Agent:
    cfg = DREAMER.copy()
    agent = DreamerV3Agent(
        obs_dim        = obs_dim,
        action_dim     = cfg["action_dim"],
        device         = device,
        embed_dim      = cfg["embed_dim"],
        hidden_dim     = cfg["hidden_dim"],
        stoch_dim      = cfg["stoch_dim"],
        num_categories = cfg["num_categories"],
    )
    model_path = LIVE["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train first: python -m btcbot.train.train_dreamer"
        )
    agent.load(model_path)
    logger.info(f"Model loaded from {model_path}  (step {agent.training_step:,})")
    return agent


# ── Main trading loop ─────────────────────────────────────────────────────────

def run_live():
    trader = UpbitTrader(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY, dry_run=DRY_RUN)
    poll   = LIVE["poll_interval_sec"]
    use_mcts = LIVE["use_mcts"]
    max_krw  = LIVE["max_position_krw"]

    # ── Initial data pull ────────────────────────────────────────────────────
    logger.info("── Initial data fetch ──────────────────────────────────")
    data     = fetch_all_timeframes(n_candles=2000)
    df_macro = fetch_macro(years_back=2)
    df_fng   = fetch_fear_greed(n_days=500)

    df_anchor, feats, rets = compute_features(data, df_macro, df_fng)
    T, F    = feats.shape
    obs_dim = WINDOW * F + 1

    agent   = load_agent(obs_dim)
    mcts    = MCTS(agent, **MCTS_CFG) if use_mcts else None

    h, z    = None, None
    position = 0   # 0=flat, 1=long
    step     = T - 1

    logger.info(f"Ready.  obs_dim={obs_dim}  use_mcts={use_mcts}  dry_run={DRY_RUN}")
    logger.info(f"Symbol: {SYMBOL}   Poll interval: {poll}s")

    while True:
        loop_start = time.time()

        try:
            # ── Refresh data (incremental) ───────────────────────────────────
            data     = fetch_all_timeframes(n_candles=300, refresh=False)
            df_fng_new = fetch_fear_greed(n_days=5)

            df_anchor, feats, rets = compute_features(data, df_macro, df_fng_new)
            step = len(feats) - 1

            # ── Build observation ─────────────────────────────────────────────
            obs = build_observation(feats, step, WINDOW)
            obs = np.append(obs, float(position)).astype(np.float32)

            # ── Get action ───────────────────────────────────────────────────
            with_latent, (h, z) = agent.act(obs, h, z, deterministic=True)

            if use_mcts and mcts is not None:
                action_oh = mcts.search(h, z)
                action = int(np.argmax(action_oh))
            else:
                action = int(np.argmax(with_latent))

            # ── Current portfolio state ───────────────────────────────────────
            current_price = trader.get_current_price()
            btc_balance   = trader.get_balance_btc()
            krw_balance   = trader.get_balance_krw()

            logger.info(
                f"step={step}  action={ACTION_NAMES[action]}  "
                f"position={ACTION_NAMES[position]}  "
                f"price={current_price:,.0f}  "
                f"btc={btc_balance:.6f}  krw={krw_balance:,.0f}"
            )

            # ── Execute trade ─────────────────────────────────────────────────
            if action != position:
                trader.cancel_all_orders()

                if action == 1 and position == 0:
                    # Go LONG: spend up to max_position_krw
                    krw_to_spend = min(krw_balance * 0.99, max_krw)
                    if krw_to_spend > 5_000:   # Upbit minimum ~5,000 KRW
                        result = trader.buy_market(krw_to_spend)
                        if result:
                            position = 1
                            logger.info(f"  → Bought BTC  (~{krw_to_spend:,.0f} KRW)")

                elif action == 0 and position == 1:
                    # Go FLAT: sell all BTC
                    if btc_balance > 0.00001:
                        result = trader.sell_market(btc_balance)
                        if result:
                            position = 0
                            logger.info(f"  → Sold {btc_balance:.6f} BTC")

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as exc:
            logger.exception(f"Loop error (will retry): {exc}")

        # ── Sleep until next candle ───────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_s = max(0.0, poll - elapsed)
        logger.info(f"Sleeping {sleep_s:.1f}s …\n")
        time.sleep(sleep_s)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_live()
