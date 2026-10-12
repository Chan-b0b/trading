"""
Morning report: shortly after the KRX open, forecast each liquid stock's
path to the close with Kronos and rank stocks by expected return.

Universe: realtime top stocks by market trading amount, common shares only,
investment-caution stocks excluded. Bars: regular-session 1m bars (09:00 open
through the 15:30 closing-auction bar) aggregated to 5m; the forecast target
is the close of the 15:30 bin, i.e. the official closing price. Kronos sees only bars that closed at or before
`asof`; later bars are used only for the "actual" column when the report is
generated after the fact.

Kronos averages its sample paths internally, so each stock is repeated
`samples` times in the batch with sample_count=1 to get the distribution.

The report is for the account holder's own use: the Toss API terms forbid
distributing its data to third parties.

    python -m coinbot.morning                         # live, at ~09:20 KST
    python -m coinbot.morning --asof "2026-10-01 09:20"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coinbot import toss

logger = logging.getLogger(__name__)

KST = "Asia/Seoul"
OPEN, CLOSE = pd.Timedelta(hours=9), pd.Timedelta(hours=15, minutes=30)
BAR = "5min"
CONTEXT = 512            # Kronos-small/base max context
BARS_PER_DAY = 79        # 09:00..15:25 plus the 15:30 closing-auction bin
HISTORY = pd.Timedelta(days=14)
COLS = ["open", "high", "low", "close", "volume"]
KRONOS_DIR = toss.ROOT / "repo" / "Kronos"
REPORT_DIR = toss.ROOT / "data" / "reports"


def universe(n: int = 100) -> pd.DataFrame:
    """Top `n` common stocks by realtime market trading amount, with names."""
    names = {}
    for market in ("KOSPI", "KOSDAQ"):
        for s in toss.get("/api/v1/stocks/all", market=market, securityType="STOCK", commonShare="true"):
            names[s["symbol"]] = s["name"]
    ranked = toss.get("/api/v1/rankings", type="MARKET_TRADING_AMOUNT", marketCountry="KR",
                      duration="realtime", excludeInvestmentCaution="true", count=100)
    rows = [{"symbol": r["symbol"], "name": names[r["symbol"]]}
            for r in ranked["rankings"] if r["symbol"] in names]
    return pd.DataFrame(rows).head(n)


def session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Regular-session 1m bars (incl. the 15:30 closing auction) aggregated to 5m bars, KST times."""
    t = bars["open_time"].dt.tz_convert(KST)
    tod = t - t.dt.normalize()
    reg = bars.assign(open_time=t)[(tod >= OPEN) & (tod <= CLOSE)]
    g = reg.groupby(reg["open_time"].dt.floor(BAR))
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).rename_axis("open_time").reset_index()
    out["close_time"] = out["open_time"] + pd.Timedelta(BAR)
    return out


def split(bars: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    """
    Model context and report facts for one stock at `asof` (5m-aligned, KST).
    None if the stock has not traded today or has too little history.
    """
    seen = session_bars(bars[bars["close_time"] <= asof])
    seen = seen[seen["close_time"] <= asof]
    day = asof.normalize()
    today, before = seen[seen["open_time"] >= day], seen[seen["open_time"] < day]
    if today.empty or before.empty or len(seen) < CONTEXT:
        return None
    later = session_bars(bars)
    later = later[(later["open_time"] >= asof) & (later["open_time"] <= day + CLOSE)]
    return {
        "context": seen.tail(CONTEXT).reset_index(drop=True),
        "price": today["close"].iloc[-1],
        "prev_close": before["close"].iloc[-1],
        "actual": later["close"].iloc[-1] if len(later) else np.nan,
        "actual_final": len(later) > 0 and later["open_time"].iloc[-1] == day + CLOSE,
    }


def horizon(asof: pd.Timestamp) -> pd.Series:
    """Open times of the 5m bars from `asof` through the closing-auction bin."""
    return pd.Series(pd.date_range(asof, asof.normalize() + CLOSE, freq=BAR))


def forecast(contexts: list[pd.DataFrame], future: pd.Series, samples: int, model_name: str,
             weights: Path | None = None, seed: int = 0, chunk: int = 256) -> np.ndarray:
    """Sampled closing prices, shape (n_stocks, samples)."""
    sys.path.insert(0, str(KRONOS_DIR))
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    src = (lambda name: str(weights / name)) if weights else (lambda name: f"NeoQuasar/{name}")
    predictor = KronosPredictor(Kronos.from_pretrained(src(f"Kronos-{model_name}")),
                                KronosTokenizer.from_pretrained(src("Kronos-Tokenizer-base")),
                                max_context=CONTEXT)
    torch.backends.cuda.matmul.allow_tf32 = True   # ~30% faster on RTX GPUs, negligible accuracy cost
    torch.manual_seed(seed)

    def naive(times: pd.Series) -> pd.Series:   # Kronos time features expect local wall-clock time
        return pd.Series(times.dt.tz_localize(None).values)

    xs = [c[COLS] for c in contexts]
    x_times = [naive(c["open_time"]) for c in contexts]
    y_times = naive(future)
    jobs = np.repeat(np.arange(len(contexts)), samples)
    closes = np.empty(len(jobs))
    for start in range(0, len(jobs), chunk):
        idx = jobs[start:start + chunk]
        preds = predictor.predict_batch([xs[i] for i in idx], [x_times[i] for i in idx],
                                        [y_times] * len(idx), pred_len=len(future),
                                        sample_count=1, verbose=False)
        closes[start:start + len(idx)] = [p["close"].iloc[-1] for p in preds]
        logger.info(f"forecast {min(start + chunk, len(jobs))}/{len(jobs)} paths")
    return closes.reshape(len(contexts), samples)


def summarize(stocks: pd.DataFrame, closes: np.ndarray) -> pd.DataFrame:
    """Per-stock return distribution to the close, sorted by expected return."""
    ret = closes / stocks["price"].to_numpy()[:, None] - 1
    out = stocks.assign(
        morning=stocks["price"] / stocks["prev_close"] - 1,
        exp_ret=ret.mean(axis=1),
        median=np.median(ret, axis=1),
        p_up=(ret > 0).mean(axis=1),
        q10=np.quantile(ret, 0.1, axis=1),
        q90=np.quantile(ret, 0.9, axis=1),
        actual_ret=stocks["actual"] / stocks["price"] - 1,
    )
    return out.sort_values("exp_ret", ascending=False).reset_index(drop=True)


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.2%}"


def _table(df: pd.DataFrame, with_actual: bool) -> str:
    head = "| 종목 | 현재가 | 아침 등락 | 기대수익률 | 중앙값 | 상승확률 | 10~90% 범위 |"
    rule = "|---|---:|---:|---:|---:|---:|---|"
    if with_actual:
        head, rule = head + " 실제 |", rule + "---:|"
    lines = [head, rule]
    for r in df.itertuples():
        line = (f"| {r.name} ({r.symbol}) | {r.price:,.0f} | {_pct(r.morning)} | {_pct(r.exp_ret)} | "
                f"{_pct(r.median)} | {r.p_up:.0%} | {_pct(r.q10)} ~ {_pct(r.q90)} |")
        lines.append(line + (f" {_pct(r.actual_ret)} |" if with_actual else ""))
    return "\n".join(lines)


def render(table: pd.DataFrame, asof: pd.Timestamp, generated: pd.Timestamp, model_name: str,
           samples: int, top: int) -> str:
    with_actual = table["actual_ret"].notna().any()
    final = with_actual and table["actual_final"].all()
    lines = [
        f"# 장 초반 Kronos 리포트 — {asof:%Y-%m-%d %H:%M} KST",
        "",
        f"- 예측: {asof:%H:%M} 가격 → 오늘 종가 수익률 분포 (종목당 샘플 {samples}개)",
        f"- 모델: Kronos-{model_name}, 정규장 5분봉 {CONTEXT}개 컨텍스트 (약 {CONTEXT / BARS_PER_DAY:.1f}거래일)",
        f"- 대상: 실시간 거래대금 상위 보통주 {len(table)}종목 (상위 {top} 요청, 투자유의 제외)",
        "- **검증되지 않은 신호입니다.** 단순 기준(아침 등락 순위 등) 대비 성과를 아직 확인하지 않았습니다.",
    ]
    if generated - asof > pd.Timedelta(minutes=15):
        lines.append(f"- 사후 생성({generated:%H:%M}): 종목 선정이 생성 시점 랭킹 기준이라 {asof:%H:%M} 이후 정보가 섞여 있습니다.")
    if with_actual:
        lines.append(f"- 실제: {asof:%H:%M} → {'종가' if final else f'{generated:%H:%M} 현재가'} 수익률")
        valid = table.dropna(subset=["actual_ret"])
        if len(valid) > 2:
            ic = valid["exp_ret"].corr(valid["actual_ret"], method="spearman")
            lines.append(f"- 기대수익률과 실제 수익률의 순위상관(RankIC): {ic:+.3f}")
    lines += [
        "", "## 상승 기대 상위 10", "", _table(table.head(10), with_actual),
        "", "## 하락 기대 상위 10", "", _table(table.tail(10).iloc[::-1], with_actual),
        "", "## 전체", "", _table(table, with_actual),
        "", "> 본인 참고용입니다. 토스증권 API 약관상 제3자 배포가 금지되어 있습니다.", "",
    ]
    return "\n".join(lines)


def run(asof: pd.Timestamp, top: int, samples: int, model_name: str, weights: Path | None) -> Path:
    generated = pd.Timestamp.now(tz=KST)
    asof = asof.floor(BAR)
    uni = universe(top)
    rows = []
    for s in uni.itertuples():
        info = split(toss.load(s.symbol, "1m", start=asof - HISTORY), asof)
        if info is None:
            logger.info(f"skip {s.name} ({s.symbol}): no trades today or too little history")
            continue
        rows.append({"symbol": s.symbol, "name": s.name, **info})
    stocks = pd.DataFrame(rows)
    closes = forecast(list(stocks.pop("context")), horizon(asof), samples, model_name, weights)
    table = summarize(stocks, closes)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"morning_{asof:%Y%m%d_%H%M}.md"
    path.write_text(render(table, asof, generated, model_name, samples, top))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asof", help="KST time to forecast from (default: now)")
    parser.add_argument("--top", type=int, default=100, help="universe size before filtering")
    parser.add_argument("--samples", type=int, default=20, help="sample paths per stock")
    parser.add_argument("--model", choices=["small", "base"], default="small")
    parser.add_argument("--weights", type=Path,
                        help="local dir with Kronos-<model>/ and Kronos-Tokenizer-base/ (if Hugging Face is unreachable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asof = pd.Timestamp(args.asof, tz=KST) if args.asof else pd.Timestamp.now(tz=KST)
    print(run(asof, args.top, args.samples, args.model, args.weights))


if __name__ == "__main__":
    main()
