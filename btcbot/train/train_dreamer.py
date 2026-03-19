"""
Train DreamerV3 on KRW-BTC historical data from Upbit.

Pipeline
--------
1. Fetch / load all timeframe candles (5m, 15m, 1h, 4h, 1d)
2. Fetch macro (VIX, DXY, SPX via yfinance)
3. Fetch Fear & Greed Index
4. Compute multi-timeframe feature matrix
5. Train DreamerV3 agent on BTCTradingEnv
6. Periodically checkpoint and log training metrics
7. Optionally run a quick backtest after training

Run
---
    cd /Users/chlee/vscode/trading
    python -m btcbot.train.train_dreamer [--steps 200000] [--device cpu]

Arguments (all optional, defaults from config.py)
    --steps   N        Total env steps              (default: 500_000)
    --device  str      "cpu" / "mps" / "cuda"       (default: "cpu")
    --refresh          Force re-fetch of all data   (flag)
    --no-mcts          Skip MCTS during eval        (flag)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Add tradingbot to Python path so we can import DreamerV3 ─────────────────
sys.path.insert(0, str(Path(__file__).parents[3] / "tradingbot"))

from btcbot.config import (
    DATA_DIR, MODEL_DIR, LOG_DIR,
    DREAMER, MCTS as MCTS_CFG, TRAIN, WINDOW,
)
from btcbot.data.fetch_upbit    import fetch_all_timeframes
from btcbot.data.fetch_macro    import fetch_macro
from btcbot.data.fetch_fear_greed import fetch_fear_greed
from btcbot.features.make_features import compute_features, build_observation
from btcbot.env.btc_env         import BTCTradingEnv

from ..models.dreamer_agent import DreamerV3Agent
from ..models.mcts          import MCTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train_dreamer.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_device(requested: str) -> str:
    import torch
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_agent(obs_dim: int, device: str) -> DreamerV3Agent:
    cfg = DREAMER.copy()
    return DreamerV3Agent(
        obs_dim        = obs_dim,
        action_dim     = cfg["action_dim"],
        device         = device,
        embed_dim      = cfg["embed_dim"],
        hidden_dim     = cfg["hidden_dim"],
        stoch_dim      = cfg["stoch_dim"],
        num_categories = cfg["num_categories"],
        lr_world_model = cfg["lr_world_model"],
        lr_actor       = cfg["lr_actor"],
        lr_critic      = cfg["lr_critic"],
        gamma          = cfg["gamma"],
        lambda_        = cfg["lambda_"],
        horizon        = cfg["horizon"],
        free_nats      = cfg["free_nats"],
        kl_balance     = cfg["kl_balance"],
    )


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    total_steps: int  = TRAIN["total_steps"],
    batch_size:  int  = TRAIN["batch_size"],
    train_every: int  = TRAIN["train_every"],
    ckpt_every:  int  = TRAIN["checkpoint_every"],
    device_req:  str  = TRAIN["device"],
    refresh:     bool = False,
):
    device = _detect_device(device_req)
    logger.info(f"Device: {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    logger.info("── Fetching data ────────────────────────────────────────")
    data     = fetch_all_timeframes(n_candles=5000, refresh=refresh)
    df_macro = fetch_macro(years_back=4)
    df_fng   = fetch_fear_greed(n_days=500)

    # ── 2. Features ───────────────────────────────────────────────────────────
    logger.info("── Computing features ───────────────────────────────────")
    df_anchor, feats, rets = compute_features(data, df_macro, df_fng)
    T, F = feats.shape
    obs_dim = WINDOW * F + 1   # +1 for position flag
    logger.info(f"Feature matrix: {T} × {F}   obs_dim={obs_dim}")

    # ── 3. Environment ────────────────────────────────────────────────────────
    env = BTCTradingEnv(
        features         = feats,
        returns          = rets,
        window           = WINDOW,
        cost_per_trade   = TRAIN.get("cost_per_trade", 0.0005),
        max_episode_steps = 2000,
    )

    # ── 4. Agent ──────────────────────────────────────────────────────────────
    agent = _build_agent(obs_dim, device)

    ckpt_file = MODEL_DIR / "dreamer_btc_latest.pt"
    if ckpt_file.exists():
        logger.info(f"Resuming from checkpoint: {ckpt_file}")
        agent.load(str(ckpt_file))

    # ── 5. Training loop ──────────────────────────────────────────────────────
    logger.info(f"── Training  ({total_steps:,} steps) ─────────────────────")

    obs, _         = env.reset()
    h, z           = None, None
    episode_reward = 0.0
    episode_steps  = 0
    n_episodes     = 0
    global_step    = agent.training_step   # resume counting

    t0 = time.time()

    while global_step < total_steps:
        # ── Act ───────────────────────────────────────────────────────────────
        action_oh, (h, z) = agent.act(obs, h, z, deterministic=False)
        action_int = int(np.argmax(action_oh))

        next_obs, reward, terminated, truncated, info = env.step(action_int)

        # Store transition in replay buffer
        agent.replay_buffer.add(
            obs    = obs,
            action = action_oh,
            reward = reward,
            done   = float(terminated or truncated),
        )

        episode_reward += reward
        episode_steps  += 1
        global_step    += 1
        obs             = next_obs

        # ── Train ─────────────────────────────────────────────────────────────
        if global_step % train_every == 0 and len(agent.replay_buffer) >= DREAMER["seq_len"] * 2:
            losses = agent.train_step(batch_size=batch_size)

        # ── Episode end ───────────────────────────────────────────────────────
        if terminated or truncated:
            n_episodes += 1
            elapsed     = time.time() - t0
            sps         = global_step / elapsed
            logger.info(
                f"ep={n_episodes:4d}  step={global_step:7,d}  "
                f"ep_rew={episode_reward:+.4f}  "
                f"equity={info['equity']:.4f}  "
                f"sps={sps:.0f}"
            )
            obs, _         = env.reset()
            h, z           = None, None
            episode_reward = 0.0
            episode_steps  = 0

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % ckpt_every == 0:
            path = MODEL_DIR / f"dreamer_btc_{global_step:07d}.pt"
            agent.save(str(path))
            agent.save(str(ckpt_file))   # overwrite "latest"
            logger.info(f"Checkpoint saved → {path}")

    # ── Final save ────────────────────────────────────────────────────────────
    agent.save(str(ckpt_file))
    logger.info(f"Training complete. Model saved → {ckpt_file}")

    # ── Quick backtest on the last 20 % of data ───────────────────────────────
    logger.info("── Quick out-of-sample backtest ─────────────────────────")
    from btcbot.backtest.backtest_engine import BTCBacktester
    split = int(T * 0.8)
    bt = BTCBacktester(agent=agent, feats=feats, rets=rets, df=df_anchor, window=WINDOW)
    results = bt.run(start=split)
    return agent, results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamerV3 on KRW-BTC")
    parser.add_argument("--steps",   type=int, default=TRAIN["total_steps"],
                        help="Total training steps")
    parser.add_argument("--device",  type=str, default=TRAIN["device"],
                        help="cpu / mps / cuda")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch of all market data")
    args = parser.parse_args()

    train(
        total_steps = args.steps,
        device_req  = args.device,
        refresh     = args.refresh,
    )
