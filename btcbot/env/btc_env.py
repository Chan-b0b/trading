"""
BTC Spot Trading Gym Environment

Instrument: KRW-BTC on Upbit (spot-only → long or flat, no shorting)

Actions
-------
    0  Flat   – hold no position
    1  Long   – hold 1 BTC equivalent (fully invested)

Reward
------
    At each step t the agent holds position p_t (0 or 1).
    Position is applied to the NEXT bar to avoid look-ahead.

    r_t = p_{t-1} * ret_t
          - cost_per_trade  * |p_t - p_{t-1}|   (Upbit 0.05 % per side)
          - turnover_coef   * |p_t - p_{t-1}|   (extra stability incentive)
          - flat_penalty    * (p_t == 0)         (nudges agent to stay invested)
          + hold_bonus      * (p_t == p_{t-1})   (encourages steady positions)

Observation
-----------
    Flattened (WINDOW * n_features,) float32 vector.
    Built by btcbot.features.make_features.build_observation().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BTCTradingEnv(gym.Env):
    """
    Spot BTC trading environment (long / flat only).

    Parameters
    ----------
    features : np.ndarray  shape (T, F)
        Pre-computed, normalised feature matrix from make_features.
    returns  : np.ndarray  shape (T,)
        1h log-returns aligned with the feature rows.
    window   : int
        Look-back window (number of rows flattened into one observation).
    cost_per_trade : float
        Fee per side expressed as a fraction of position value.
        Upbit taker fee = 0.0005 (0.05 %).
    turnover_coef  : float
        Extra penalty for changing position (reduces over-trading).
    flat_penalty   : float
        Small constant penalty while holding no position.
    hold_bonus     : float
        Small constant bonus for NOT changing position.
    max_episode_steps : int | None
        If set, episode ends after this many steps (useful during training).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features:         np.ndarray,
        returns:          np.ndarray,
        window:           int   = 64,
        cost_per_trade:   float = 0.0005,
        turnover_coef:    float = 0.0002,
        flat_penalty:     float = 0.00002,
        hold_bonus:       float = 0.00002,
        max_episode_steps: int | None = None,
    ):
        super().__init__()

        assert features.ndim == 2,          "features must be 2-D (T, F)"
        assert returns.ndim  == 1,          "returns must be 1-D (T,)"
        assert len(features) == len(returns), "features and returns must have same length"

        self.X    = features.astype(np.float32)
        self.r    = returns.astype(np.float32)
        self.T    = len(self.r)
        self.window        = int(window)
        self.cost          = float(cost_per_trade)
        self.turnover_coef = float(turnover_coef)
        self.flat_penalty  = float(flat_penalty)
        self.hold_bonus    = float(hold_bonus)
        self.max_episode_steps = max_episode_steps

        obs_dim = self.window * self.X.shape[1] + 1   # +1 for current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)         # 0=flat, 1=long

        self._step         = 0
        self._position     = 0
        self._episode_step = 0
        self._equity       = 1.0                        # normalised equity

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a random position (ensures diverse training episodes)
        min_start = self.window
        max_start = max(self.window, self.T - 200)
        if seed is not None:
            self._step = self.window
        else:
            self._step = int(self.np_random.integers(min_start, max_start + 1))

        self._position     = 0
        self._episode_step = 0
        self._equity       = 1.0

        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_position = self._position
        self._position = int(action)

        # Apply previous position to this bar's return (no look-ahead)
        trade_return = prev_position * self.r[self._step]

        # Transaction cost (both sides)
        trade_cost    = self.cost * abs(self._position - prev_position)
        turnover_cost = self.turnover_coef * abs(self._position - prev_position)

        # Flat and hold bonuses / penalties
        f_penalty = self.flat_penalty if self._position == 0 else 0.0
        h_bonus   = self.hold_bonus   if self._position == prev_position else 0.0

        reward = trade_return - trade_cost - turnover_cost - f_penalty + h_bonus

        # Update equity for monitoring
        self._equity *= np.exp(trade_return - trade_cost - turnover_cost)

        self._step         += 1
        self._episode_step += 1

        terminated = self._step >= self.T
        truncated  = (
            self.max_episode_steps is not None
            and self._episode_step >= self.max_episode_steps
        )

        info = {
            "step":           self._step,
            "position":       self._position,
            "equity":         self._equity,
            "trade_return":   float(trade_return),
        }

        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self._step:5d}  pos={'LONG' if self._position else 'FLAT'}  "
            f"equity={self._equity:.4f}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """
        Build the flattened observation vector:
            [ window_rows.flatten(),  current_normalised_position ]
        Shape: (window * n_features + 1,)
        """
        idx   = self._step
        start = max(0, idx - self.window + 1)
        chunk = self.X[start: idx + 1]                 # (≤window, F)

        if len(chunk) < self.window:
            pad   = np.zeros((self.window - len(chunk), self.X.shape[1]), dtype=np.float32)
            chunk = np.vstack([pad, chunk])

        obs = np.append(chunk.flatten(), float(self._position)).astype(np.float32)
        return obs


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    T, F = 2000, 44
    dummy_feats = np.random.randn(T, F).astype(np.float32)
    dummy_rets  = np.random.randn(T).astype(np.float32) * 0.001

    env = BTCTradingEnv(dummy_feats, dummy_rets, window=64)
    obs, _ = env.reset()
    print(f"obs shape   : {obs.shape}")
    print(f"action space: {env.action_space}")

    total_reward = 0.0
    for _ in range(200):
        action      = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        if terminated or truncated:
            break

    print(f"total reward: {total_reward:.4f}")
    print(f"final equity: {info['equity']:.4f}")
    print("✅ BTCTradingEnv OK")
