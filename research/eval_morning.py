"""
Walk-forward check of the morning Kronos signal against simple baselines.

For each trading day at 09:20 KST, forecast every stock's 5m path to the
close with several context lengths, then score each signal by its daily
cross-sectional Spearman correlation (RankIC) with realized returns at
two horizons: 1h (10:20) and the close.

Universe is fixed (the symbols of a given morning report), so it is chosen
with hindsight; the bias hits Kronos and the baselines alike.

    python research/eval_morning.py fetch --start 2026-07-15
    python research/eval_morning.py run --since 2026-08-01
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coinbot import morning, toss  # noqa: E402

logger = logging.getLogger(__name__)

KST = morning.KST
OUT = Path(__file__).resolve().parent / "morning_eval.csv"
REPORT = toss.ROOT / "data" / "reports" / "morning_20261001_0920.md"
CONTEXTS = (80, 160, 300)
HOUR_STEPS = 12          # 09:20 + 12 x 5m = 10:20
ASOF = pd.Timedelta(hours=9, minutes=20)


def symbols() -> list[str]:
    return sorted(set(re.findall(r"\((\w{6})\) \|", REPORT.read_text())))


def fetch(start: pd.Timestamp) -> None:
    def one(sym: str) -> None:
        df = toss.load(sym, "1m", refresh=True, start=start)
        logger.info(f"{sym}: {len(df)} bars from {df['open_time'].iloc[0]}")
    toss.token()   # issue once up front so threads share it
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, symbols()))


def day_inputs(bars: dict[str, pd.DataFrame], day: pd.Timestamp) -> pd.DataFrame:
    """Per-stock context and realized prices for one day; stocks lacking data are dropped."""
    asof = day + ASOF
    rows = []
    for sym, s in bars.items():
        seen = s[s["close_time"] <= asof]
        today = s[(s["open_time"] >= day) & (s["open_time"] <= day + morning.CLOSE)]
        before = seen[seen["open_time"] < day]
        future = today[today["open_time"] >= asof]
        if len(before) < max(CONTEXTS) or len(future) != len(morning.horizon(asof)):
            continue
        price = seen["close"].iloc[-1]
        rows.append({
            "symbol": sym, "context": seen.tail(max(CONTEXTS)).reset_index(drop=True), "price": price,
            "sig_morning": price / before["close"].iloc[-1] - 1,
            "sig_open": price / today["open"].iloc[0] - 1,
            "ret_1h": future["close"].iloc[HOUR_STEPS - 1] / price - 1,
            "ret_close": future["close"].iloc[-1] / price - 1,
        })
    return pd.DataFrame(rows)


def paths(predictor, contexts: list[pd.DataFrame], future: pd.Series, samples: int) -> np.ndarray:
    """Sampled close paths, shape (n_stocks, samples, steps)."""
    naive = lambda t: pd.Series(t.dt.tz_localize(None).values)  # noqa: E731
    n = len(contexts)
    preds = predictor.predict_batch([c[morning.COLS] for c in contexts] * samples,
                                    [naive(c["open_time"]) for c in contexts] * samples,
                                    [naive(future)] * (n * samples), pred_len=len(future),
                                    sample_count=1, verbose=False)
    out = np.array([p["close"].to_numpy() for p in preds])
    return out.reshape(samples, n, -1).transpose(1, 0, 2)


def score(day: pd.Timestamp, df: pd.DataFrame, signal: str) -> list[dict]:
    rows = []
    for horizon in ("1h", "close"):
        ret = df[f"ret_{horizon}"]
        sig = df[signal if signal.startswith("sig_") else f"{signal}_{horizon}"]
        order = sig.sort_values().index
        rows.append({"day": day.date(), "signal": signal, "horizon": horizon, "n": len(df),
                     "ic": sig.corr(ret, method="spearman"),
                     "top_bottom": ret[order[-10:]].mean() - ret[order[:10]].mean()})
    return rows


def run(since: pd.Timestamp, samples: int) -> None:
    import torch
    sys.path.insert(0, str(morning.KRONOS_DIR))
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    predictor = KronosPredictor(Kronos.from_pretrained("NeoQuasar/Kronos-small"),
                                KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"), max_context=512)

    bars = {s: morning.session_bars(pd.read_parquet(toss.CACHE_DIR / f"{s}_1m.parquet")) for s in symbols()}
    days = sorted({t.normalize() for s in bars.values() for t in s["open_time"] if t.normalize() >= since})
    done = set(pd.read_csv(OUT)["day"]) if OUT.exists() else set()
    for day in days:
        if str(day.date()) in done:
            continue
        df = day_inputs(bars, day)
        if len(df) < 20:
            logger.info(f"{day.date()}: only {len(df)} stocks, skipped")
            continue
        future = morning.horizon(day + ASOF)
        torch.manual_seed(0)
        rows = []
        for sig in ("sig_morning", "sig_open"):
            rows += score(day, df, sig)
        for L in CONTEXTS:
            p = paths(predictor, [c.tail(L) for c in df["context"]], future, samples).mean(axis=1)
            df[f"kronos{L}_1h"] = p[:, HOUR_STEPS - 1] / df["price"] - 1
            df[f"kronos{L}_close"] = p[:, -1] / df["price"] - 1
            rows += score(day, df, f"kronos{L}")
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        logger.info(f"{day.date()}: {len(df)} stocks scored")
    summarize()


PATHS = toss.ROOT / "data" / "research" / "morning_paths_ctx{L}.parquet"


def dump_paths(since: pd.Timestamp, samples: int, L: int) -> None:
    """Save every sampled 5m path (OHLC) and the realized path, one row per stock x sample x step."""
    import torch
    sys.path.insert(0, str(morning.KRONOS_DIR))
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    predictor = KronosPredictor(Kronos.from_pretrained("NeoQuasar/Kronos-small"),
                                KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"), max_context=512)
    naive = lambda t: pd.Series(t.dt.tz_localize(None).values)  # noqa: E731

    bars = {s: morning.session_bars(pd.read_parquet(toss.CACHE_DIR / f"{s}_1m.parquet")) for s in symbols()}
    days = sorted({t.normalize() for s in bars.values() for t in s["open_time"] if t.normalize() >= since})
    frames = []
    for day in days:
        asof = day + ASOF
        future = morning.horizon(asof)
        stocks = []
        for sym, s in bars.items():
            seen = s[s["close_time"] <= asof]
            before = seen[seen["open_time"] < day]
            real = s[(s["open_time"] >= asof) & (s["open_time"] <= day + morning.CLOSE)]
            if len(before) >= L and len(real) == len(future):
                stocks.append((sym, seen.tail(L), before["close"].iloc[-1], real))
        if len(stocks) < 20:
            continue
        torch.manual_seed(0)
        preds = predictor.predict_batch([c[morning.COLS] for _, c, _, _ in stocks] * samples,
                                        [naive(c["open_time"]) for _, c, _, _ in stocks] * samples,
                                        [naive(future)] * (len(stocks) * samples), pred_len=len(future),
                                        sample_count=1, verbose=False)
        n = len(stocks)
        for k, p in enumerate(preds):
            sym, c, prev_close, real = stocks[k % n]
            frames.append(pd.DataFrame({
                "day": day.date(), "symbol": sym, "sample": k // n, "step": np.arange(1, len(future) + 1),
                "price": c["close"].iloc[-1], "prev_close": prev_close,
                "pred_open": p["open"].to_numpy(), "pred_high": p["high"].to_numpy(),
                "pred_low": p["low"].to_numpy(), "pred_close": p["close"].to_numpy(),
                "open": real["open"].to_numpy(), "high": real["high"].to_numpy(),
                "low": real["low"].to_numpy(), "close": real["close"].to_numpy(),
            }))
        logger.info(f"{day.date()}: {n} stocks")
    path = Path(str(PATHS).format(L=L))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(path, index=False)
    print(path)


def by_horizon(L: int) -> None:
    """Forecast quality per step: RankIC, direction hit rate, and MAE relative to a no-change forecast."""
    df = pd.read_parquet(Path(str(PATHS).format(L=L)))
    m = df.groupby(["day", "symbol", "step"], as_index=False).agg(
        price=("price", "first"), prev_close=("prev_close", "first"),
        pred=("pred_close", "mean"), close=("close", "first"))
    m["pred_ret"] = m["pred"] / m["price"] - 1
    m["ret"] = m["close"] / m["price"] - 1
    m["reversal"] = -(m["price"] / m["prev_close"] - 1)

    def daily(g: pd.DataFrame) -> pd.Series:
        moved = g["ret"] != 0
        return pd.Series({
            "ic": g["pred_ret"].corr(g["ret"], method="spearman"),
            "ic_reversal": g["reversal"].corr(g["ret"], method="spearman"),
            "hit": (np.sign(g["pred_ret"][moved]) == np.sign(g["ret"][moved])).mean(),
            "mae": (g["pred_ret"] - g["ret"]).abs().mean(),
            "mae_flat": g["ret"].abs().mean(),
        })

    d = m.groupby(["step", "day"]).apply(daily)
    g = d.groupby("step")
    out = pd.DataFrame({
        "minutes": g["ic"].count().index * 5,
        "ic": g["ic"].mean(),
        "t": g["ic"].mean() / g["ic"].std() * np.sqrt(g["ic"].count()),
        "ic_reversal": g["ic_reversal"].mean(),
        "hit": g["hit"].mean(),
        "mae_vs_flat": g["mae"].mean() / g["mae_flat"].mean(),
    })
    print(f"{d.index.get_level_values('day').nunique()} days, ctx {L}")
    print(out.loc[[1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 60, 75]].round(3).to_string())


def summarize() -> None:
    res = pd.read_csv(OUT)
    g = res.groupby(["horizon", "signal"])
    out = pd.DataFrame({
        "days": g["ic"].count(),
        "mean_ic": g["ic"].mean(),
        "t": g["ic"].mean() / g["ic"].std() * np.sqrt(g["ic"].count()),
        "ic>0": g["ic"].apply(lambda x: (x > 0).mean()),
        "top_bottom": g["top_bottom"].mean(),
    })
    print(out.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", default="2026-07-15")
    r = sub.add_parser("run")
    r.add_argument("--since", default="2026-08-01")
    r.add_argument("--samples", type=int, default=5)
    sub.add_parser("summary")
    d = sub.add_parser("paths")
    d.add_argument("--since", default="2026-08-01")
    d.add_argument("--samples", type=int, default=5)
    d.add_argument("--context", type=int, default=160)
    h = sub.add_parser("horizons")
    h.add_argument("--context", type=int, default=160)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cmd == "fetch":
        fetch(pd.Timestamp(args.start, tz=KST))
    elif args.cmd == "run":
        run(pd.Timestamp(args.since, tz=KST), args.samples)
    elif args.cmd == "paths":
        dump_paths(pd.Timestamp(args.since, tz=KST), args.samples, args.context)
    elif args.cmd == "horizons":
        by_horizon(args.context)
    else:
        summarize()


if __name__ == "__main__":
    main()
