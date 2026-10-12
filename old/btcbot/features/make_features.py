"""
BTC Feature Engineering — Multi-Timeframe + Macro + Fear & Greed

Primary timeframe: 1h  (WINDOW=64 → 64-step observation)
Additional timeframes: 5m, 15m, 4h, 1d  (each resampled to align with 1h rows)

Feature groups
--------------
Per-timeframe (computed for each TF then merged onto 1h index):
    ret       log return (1 step)
    vol       rolling std of returns (20 bars)
    mom       momentum / pct-change (20 bars)
    rsi       RSI-14 normalized 0-1
    macd_diff MACD histogram / close
    bb_pos    Bollinger position  (close − lower) / (upper − lower)
    atr_pct   ATR-14 / close
    ma_diff   (EMA20 − EMA50) / close

Global (same value repeated across all rows):
    fng_norm        Fear & Greed → [-1, +1]
    vix_close       VIX daily close (normalized in z-score)
    dxy_ret         DXY log-return
    spx_ret         S&P 500 log-return
    corr_btc_spx    24h rolling correlation of BTC and SPX returns

Total features per timeframe: 8  × 5 timeframes = 40 + 4 global = 44 features
The observation for DreamerV3 is the FLATTENED last WINDOW rows: 44 × WINDOW.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Timeframes to include (must be 1h or coarser for merge_asof alignment)
# We add features from shorter TFs by resampling them into 1h rows.
FEATURE_TFS = ["5m", "15m", "1h", "4h", "1d"]
ANCHOR_TF   = "1h"    # the base grid all others are merged onto

# ── Low-level technical indicators ───────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).fillna(0)
    loss  = (-delta.clip(upper=0)).fillna(0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0) / 100.0


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _bollinger_position(series: pd.Series, period: int = 20) -> pd.Series:
    ma     = series.rolling(period).mean()
    sigma  = series.rolling(period).std()
    upper  = ma + 2 * sigma
    lower  = ma - 2 * sigma
    width  = (upper - lower).replace(0, np.nan)
    pos    = (series - lower) / width
    return pos.fillna(0.5).clip(0, 1)


def _compute_tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute 8 technical features for one timeframe.
    Returns DataFrame indexed by 'time' with prefixed column names.
    """
    d = df[["time", "open", "high", "low", "close", "volume"]].copy()
    d = d.sort_values("time").reset_index(drop=True)
    c = d["close"]

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()

    feat = pd.DataFrame({"time": d["time"]})
    feat[f"{prefix}_ret"]      = np.log(c).diff().fillna(0.0)
    feat[f"{prefix}_vol"]      = feat[f"{prefix}_ret"].rolling(20).std().fillna(0.0)
    feat[f"{prefix}_mom"]      = c.pct_change(20).fillna(0.0)
    feat[f"{prefix}_rsi"]      = _rsi(c, 14)
    feat[f"{prefix}_macd"]     = ((macd - sig) / c).fillna(0.0)
    feat[f"{prefix}_bb_pos"]   = _bollinger_position(c, 20)
    feat[f"{prefix}_atr_pct"]  = (_atr(d, 14) / c).fillna(0.0)
    feat[f"{prefix}_ma_diff"]  = ((ema20 - ema50) / c).fillna(0.0)

    return feat


# ── Multi-timeframe merge ─────────────────────────────────────────────────────

def _resample_tf_features(df_source: pd.DataFrame, df_anchor: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute features on df_source, then backward-merge onto the anchor 1h grid.
    This means: for each 1h candle row, get the most-recently-closed candle
    from the shorter/longer timeframe (no look-ahead).
    """
    feat = _compute_tf_features(df_source, prefix)

    # Ensure timezone-aware times
    if feat["time"].dt.tz is None:
        feat["time"] = feat["time"].dt.tz_localize("UTC")
    anchor_time = df_anchor["time"].copy()
    if anchor_time.dt.tz is None:
        anchor_time = anchor_time.dt.tz_localize("UTC")

    # Normalise both sides to the same unit before merge_asof
    anchor_time = anchor_time.astype("datetime64[us, UTC]")
    feat["time"] = feat["time"].astype("datetime64[us, UTC]")

    anchor_df = pd.DataFrame({"time": anchor_time})
    merged = pd.merge_asof(
        anchor_df.sort_values("time"),
        feat.sort_values("time"),
        on="time",
        direction="backward",
    )
    return merged


# ── Main feature pipeline ─────────────────────────────────────────────────────

def compute_features(
    data: Dict[str, pd.DataFrame],
    df_macro: pd.DataFrame | None = None,
    df_fng:   pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the full feature matrix for DreamerV3 training/inference.

    Args:
        data:     {tf_label: ohlcv_df}  must contain at least ANCHOR_TF = '1h'
        df_macro: output of fetch_macro()   (optional, zeros if None)
        df_fng:   output of fetch_fear_greed()  (optional, zeros if None)

    Returns:
        df_anchor – 1h OHLCV DataFrame used as the time grid
        feats     – (T, F) float32 feature matrix (normalized)
        rets      – (T,)   float32 1h log-returns (reward signal)
    """
    if ANCHOR_TF not in data:
        raise ValueError(f"data dict must contain key '{ANCHOR_TF}'")

    df_anchor = data[ANCHOR_TF].copy()
    if df_anchor["time"].dt.tz is None:
        df_anchor["time"] = df_anchor["time"].dt.tz_localize("UTC")
    df_anchor = df_anchor.sort_values("time").reset_index(drop=True)

    # ── Step 1: per-timeframe technical features ──────────────────────────────
    all_feat_dfs = []

    for tf_label in FEATURE_TFS:
        if tf_label not in data:
            logger.warning(f"Timeframe '{tf_label}' missing from data dict; skipping")
            continue
        prefix = tf_label.replace("m", "m").replace("h", "h").replace("d", "d")
        if tf_label == ANCHOR_TF:
            feat_df = _compute_tf_features(data[tf_label], prefix)
            if feat_df["time"].dt.tz is None:
                feat_df["time"] = feat_df["time"].dt.tz_localize("UTC")
        else:
            feat_df = _resample_tf_features(data[tf_label], df_anchor, prefix)

        # Only keep feature columns (drop 'time')
        feat_cols = [c for c in feat_df.columns if c != "time"]
        all_feat_dfs.append(feat_df[feat_cols].reset_index(drop=True))

    combined = pd.concat(all_feat_dfs, axis=1)

    # ── Step 2: Macro features ────────────────────────────────────────────────
    if df_macro is not None and not df_macro.empty:
        from old.btcbot.data.fetch_macro import align_macro
        df_anchor_with_macro = align_macro(df_anchor, df_macro)
        for col in ["vix_close", "dxy_ret", "spx_ret"]:
            if col in df_anchor_with_macro.columns:
                combined[col] = df_anchor_with_macro[col].values
            else:
                combined[col] = 0.0
        # BTC vs SPX rolling correlation (24h / 1h candles = 24 bars)
        btc_ret = combined.get("1h_ret", pd.Series(dtype=float))
        if "spx_ret" in combined.columns and len(btc_ret) > 0:
            combined["corr_btc_spx"] = (
                btc_ret.rolling(24).corr(combined["spx_ret"]).fillna(0.0)
            )
        else:
            combined["corr_btc_spx"] = 0.0
    else:
        for col in ["vix_close", "dxy_ret", "spx_ret", "corr_btc_spx"]:
            combined[col] = 0.0

    # ── Step 3: Fear & Greed ──────────────────────────────────────────────────
    if df_fng is not None and not df_fng.empty:
        from old.btcbot.data.fetch_fear_greed import align_fear_greed
        df_anchor_with_fng = align_fear_greed(df_anchor, df_fng)
        combined["fng_norm"] = df_anchor_with_fng["fng_norm"].values
    else:
        combined["fng_norm"] = 0.0

    # ── Step 4: Burn warmup rows, clean NaN ──────────────────────────────────
    WARMUP = 120   # drop first 120 rows (allow rolling windows to stabilize)
    combined = combined.iloc[WARMUP:].reset_index(drop=True)
    df_anchor = df_anchor.iloc[WARMUP:].reset_index(drop=True)

    feats = combined.to_numpy(dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # Reward = 1h log-return
    rets = df_anchor["close"].pipe(np.log).diff().fillna(0.0).to_numpy(dtype=np.float32)
    rets = np.nan_to_num(rets)

    # ── Step 5: Z-score normalisation ────────────────────────────────────────
    mu  = feats.mean(axis=0, keepdims=True)
    sig = feats.std(axis=0,  keepdims=True) + 1e-8
    feats = (feats - mu) / sig

    logger.info(
        f"Features ready: {feats.shape}  "
        f"{df_anchor['time'].iloc[0]}  →  {df_anchor['time'].iloc[-1]}"
    )
    return df_anchor, feats, rets


# ── Observation builder for live/backtest ─────────────────────────────────────

def build_observation(feats: np.ndarray, step: int, window: int) -> np.ndarray:
    """
    Return the flattened feature window ending at `step`.
    Shape: (window * n_features,)  — the DreamerV3 obs_dim.

    Pads with zeros if step < window.
    """
    start = max(0, step - window + 1)
    window_feats = feats[start: step + 1]   # (<=window, F)
    if len(window_feats) < window:
        pad = np.zeros((window - len(window_feats), feats.shape[1]), dtype=np.float32)
        window_feats = np.vstack([pad, window_feats])
    return window_feats.flatten().astype(np.float32)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from old.btcbot.data.fetch_upbit import fetch_all_timeframes
    from old.btcbot.data.fetch_macro import fetch_macro
    from old.btcbot.data.fetch_fear_greed import fetch_fear_greed

    print("Fetching candles …")
    data = fetch_all_timeframes(n_candles=500)
    print("Fetching macro …")
    df_macro = fetch_macro(years_back=2)
    print("Fetching Fear & Greed …")
    df_fng = fetch_fear_greed(n_days=500)

    df_anchor, feats, rets = compute_features(data, df_macro, df_fng)
    print(f"\nfeats shape : {feats.shape}")
    print(f"rets  shape : {rets.shape}")
    print(f"obs_dim     : {feats.shape[1] * 64}  (WINDOW=64)")
