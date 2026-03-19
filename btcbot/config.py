"""
Central configuration for the KRW-BTC DreamerV3/MCTS trading system.

Edit the values here before running any script.
"""

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # .../trading/btcbot
DATA_DIR   = BASE_DIR / "data" / "cache"   # downloaded candle cache
MODEL_DIR  = BASE_DIR / "checkpoints"      # saved model weights
LOG_DIR    = BASE_DIR / "logs"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Upbit API Keys  (loaded from api_keys.txt if env vars not set) ───────────
def _load_api_keys() -> tuple[str, str]:
    key_file = Path(__file__).parents[1] / "api_keys.txt"
    access_key = os.getenv("UPBIT_ACCESS_KEY", "")
    secret_key = os.getenv("UPBIT_SECRET_KEY", "")
    if not access_key and key_file.exists():
        lines = [l.strip() for l in key_file.read_text().splitlines() if l.strip()]
        # Format expected:  "Access Key\n<key>\nSecret Key\n<key>"
        try:
            access_key = lines[lines.index("Access Key") + 1]
            secret_key = lines[lines.index("Secret Key") + 1]
        except (ValueError, IndexError):
            pass
    return access_key, secret_key

UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY = _load_api_keys()

# ── Symbol / Market ───────────────────────────────────────────────────────────
SYMBOL      = "KRW-BTC"          # Upbit market symbol
BASE_TF     = "minutes60"        # Primary candle timeframe (Upbit API name)
                                  # minutes1 / minutes3 / minutes5 / minutes15 /
                                  # minutes30 / minutes60 / minutes240 / days

# Multi-scale timeframes resampled from 1-minute candles
# Keys = short label, Values = Upbit API endpoint fragment
TIMEFRAMES = {
    "1m":  "minutes1",
    "5m":  "minutes5",
    "15m": "minutes15",
    "1h":  "minutes60",
    "4h":  "minutes240",
    "1d":  "days",
}

# How many candles to fetch per timeframe during live inference
LIVE_CANDLE_COUNT = {
    "1m":  500,
    "5m":  500,
    "15m": 300,
    "1h":  300,
    "4h":  200,
    "1d":  120,
}

# ── Feature / Training Config ─────────────────────────────────────────────────
WINDOW    = 64          # observation look-back window (steps)
OBS_TF    = "1h"        # primary timeframe for the RL observation

# ── Environment Config ────────────────────────────────────────────────────────
COST_PER_TRADE   = 0.0005   # Upbit 0.05 % taker fee (each side)
TURNOVER_COEF    = 0.0002   # extra penalty for flip-flopping
FLAT_PENALTY     = 0.00002
HOLD_BONUS       = 0.00002

# ── DreamerV3 Config ──────────────────────────────────────────────────────────
DREAMER = dict(
    embed_dim     = 256,
    hidden_dim    = 512,
    stoch_dim     = 32,
    num_categories = 32,
    action_dim    = 2,          # 0=flat, 1=long  (spot-only, no shorting)
    lr_world_model = 3e-4,
    lr_actor      = 1e-4,
    lr_critic     = 3e-4,
    gamma         = 0.99,
    lambda_       = 0.95,
    horizon       = 15,
    free_nats     = 1.0,
    kl_balance    = 0.8,
    seq_len       = 64,
    replay_capacity = 200_000,
)

# ── MCTS Config ───────────────────────────────────────────────────────────────
MCTS = dict(
    num_simulations = 50,    # lower = faster live inference
    c_puct          = 1.0,
    gamma           = 0.99,
)

# ── Training Config ───────────────────────────────────────────────────────────
TRAIN = dict(
    total_steps         = 500_000,
    batch_size          = 16,
    train_every         = 4,        # train every N env steps
    checkpoint_every    = 10_000,
    eval_every          = 25_000,
    device              = "cpu",    # "cpu" / "mps" / "cuda"
)

# ── Backtest Config ───────────────────────────────────────────────────────────
BACKTEST = dict(
    fee             = 0.0005,       # 0.05 % Upbit taker fee
    slippage        = 0.0002,       # 0.02 % conservative slippage
    initial_capital = 1_000_000,    # KRW
)

# ── Live Trading Config ───────────────────────────────────────────────────────
LIVE = dict(
    poll_interval_sec  = 60,        # check/act every 60 s (1-min candle close)
    order_type         = "limit",   # "market" or "limit"
    limit_offset_pct   = 0.001,     # 0.1 % inside spread for limit orders
    max_position_krw   = 500_000,   # max KRW per trade
    use_mcts           = True,
    model_path         = str(MODEL_DIR / "dreamer_btc_latest.pt"),
)
