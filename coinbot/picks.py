"""
Daily buy picks, sent to my KakaoTalk before each market opens.

Before the KR open (~08:20 KST) and the US open (40 min before 09:30 New York
time, which moves with US daylight saving), `scripts/picks.sh` runs:

  due      exit 0 only on a trading day inside the send window, not yet sent
  prepare  build a candidate brief from Toss daily candles and print the
           prompt for Claude Code, which reads the brief (and news) and
           writes its picks as JSON
  send     check the picks against the brief, send them, log them
  settle   score logged picks over HOLD trading days, from the session open
           to the HOLD-th close, net of costs, against the equal-weight
           candidate pool and an index ETF

Candidates: PER_SOURCE common stocks from each of three sources, filled in
order and without repeats, so a stock already taken goes to the next in line:
  ① previous session's top by market trading amount
  ② past month's top by market trading amount
  ③ a random draw (seeded by the session date) from the top CAP_UNIVERSE by
    market cap, a stand-in for KOSPI200 / S&P500
Investment-caution stocks are excluded throughout. Each candidate keeps its
source, so the record can show which source's stocks, and picks, did better.
The brief uses only closed daily bars, so nothing from the session being
picked for is visible.

Reports are for the account holder only: the Toss API terms forbid
distributing its data to third parties.

    python -m coinbot.picks due KR
    python -m coinbot.picks prepare KR
    python -m coinbot.picks send KR
    python -m coinbot.picks settle [--send]
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from coinbot import kakao, toss

logger = logging.getLogger(__name__)

PICKS_DIR = toss.ROOT / "data" / "picks"
LOG_FILE = PICKS_DIR / "log.jsonl"
PROMPT_FILE = Path(__file__).with_name("picks_prompt.md")

TZ = {"KR": "Asia/Seoul", "US": "America/New_York"}
EXCHANGES = {"KR": ("KOSPI", "KOSDAQ"), "US": ("NYSE", "NASDAQ", "AMEX")}
BENCHMARK = {"KR": ("069500", "KODEX 200"), "US": ("SPY", "SPDR S&P 500")}
LABEL = {"KR": "국내", "US": "미국"}
# Send window before the regular open. KR is wider so a late open (수능 day, 10:00)
# is still caught by the 08:20 run; US is narrower so only one of the two cron
# runs (21:50 / 22:50 KST) falls inside it, whichever matches daylight saving.
WINDOW = {"KR": (pd.Timedelta(minutes=120), pd.Timedelta(minutes=20)),
          "US": (pd.Timedelta(minutes=60), pd.Timedelta(minutes=20))}
# Round-trip cost. KR: 0.015% commission each way + 0.20% sell-side tax.
# US: 0.1% commission each way; FX spread not included.
COST = {"KR": 0.0023, "US": 0.002}
SOURCES = {"①": "1일 거래대금 상위", "②": "1개월 거래대금 상위", "③": "시총 상위 무작위"}
PER_SOURCE = 15
CAP_UNIVERSE = {"KR": 200, "US": 500}
N_PICKS = 5             # ranked picks per message, always this many
CONVICTION = ("상", "중", "하")
HOLD = 5                # trading days, session day included
HISTORY = pd.Timedelta(days=200)


# ── schedule ──────────────────────────────────────────────────────────────────

def session_today(market: str) -> tuple[str, pd.Timestamp] | None:
    """(session date, regular-session open) for today, or None on a holiday."""
    today = toss.get(f"/api/v1/market-calendar/{market}")["today"] or {}
    sessions = today.get("integrated") if market == "KR" else today
    regular = (sessions or {}).get("regularMarket")
    if not regular:
        return None
    return today["date"], pd.Timestamp(regular["startTime"])


def in_window(market: str, start: pd.Timestamp, now: pd.Timestamp) -> bool:
    early, late = WINDOW[market]
    return start - early <= now <= start - late


def already_sent(market: str, session: str) -> bool:
    return any(e["market"] == market and e["session"] == session for e in read_log())


def due(market: str, now: pd.Timestamp | None = None) -> bool:
    today = session_today(market)
    if today is None:
        return False
    session, start = today
    return in_window(market, start, now or pd.Timestamp.now(tz="UTC")) and not already_sent(market, session)


# ── brief ─────────────────────────────────────────────────────────────────────

def listed(market: str) -> dict[str, str]:
    """Symbol → name of every active common stock on the market's exchanges."""
    return {st["symbol"]: st["name"]
            for exchange in EXCHANGES[market]
            for st in toss.get("/api/v1/stocks/all", market=exchange, securityType="STOCK", commonShare="true")}


def ranked(market: str, duration: str) -> list[str]:
    """Top 100 by market trading amount over `duration`, investment-caution stocks excluded."""
    result = toss.get("/api/v1/rankings", type="MARKET_TRADING_AMOUNT", marketCountry=market,
                      duration=duration, excludeInvestmentCaution="true", count=100)
    return [r["symbol"] for r in result["rankings"]]


def market_caps(symbols: list[str]) -> pd.Series:
    """Shares outstanding × last price, by symbol."""
    caps = {}
    for i in range(0, len(symbols), 200):
        batch = ",".join(symbols[i:i + 200])
        shares = {st["symbol"]: float(st["sharesOutstanding"] or 0) for st in toss.get("/api/v1/stocks", symbols=batch)}
        for pr in toss.get("/api/v1/prices", symbols=batch):
            caps[pr["symbol"]] = shares.get(pr["symbol"], 0.0) * float(pr["lastPrice"])
    return pd.Series(caps, dtype=float)


def cautioned(symbol: str) -> bool:
    """Whether a warning other than an intraday VI is active (overheated, warning, risk, liquidation...)."""
    return any(not w["warningType"].startswith("VI_") for w in toss.get(f"/api/v1/stocks/{symbol}/warnings"))


def fill(sources: dict[str, list[str]], names: dict[str, str], n: int,
         ok=lambda symbol, source: True) -> pd.DataFrame:
    """Take `n` listed, not-yet-taken symbols from each source in turn."""
    taken: dict[str, str] = {}
    for source, symbols in sources.items():
        added = 0
        for symbol in symbols:
            if added == n:
                break
            if symbol in names and symbol not in taken and ok(symbol, source):
                taken[symbol] = source
                added += 1
    return pd.DataFrame([{"symbol": s, "name": names[s], "source": src} for s, src in taken.items()],
                        columns=["symbol", "name", "source"])


def candidates(market: str, session: str, n: int = PER_SOURCE) -> pd.DataFrame:
    names = listed(market)
    caps = market_caps(list(names))
    big = list(caps.nlargest(CAP_UNIVERSE[market]).index)
    draw = list(np.random.default_rng(int(session.replace("-", ""))).permutation(big))
    # Rankings already exclude caution stocks; only the cap draw needs checking.
    return fill({"①": ranked(market, "1d"), "②": ranked(market, "1mo"), "③": draw}, names, n,
                ok=lambda symbol, source: source != "③" or not cautioned(symbol))


def daily(symbol: str, now: pd.Timestamp) -> pd.DataFrame:
    """Daily bars closed by `now`."""
    bars = toss.fetch(symbol, "1d", since=now - HISTORY)
    return bars[bars["close_time"] <= now].reset_index(drop=True)


def features(bars: pd.DataFrame) -> dict:
    """Price/volume summary of the last closed daily bar and the bars before it."""
    c, h, lo, v = bars["close"], bars["high"], bars["low"], bars["volume"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)

    def ret(n: int) -> float:
        return c.iloc[-1] / c.iloc[-1 - n] - 1 if len(c) > n else np.nan

    return {
        "last_day": bars["open_time"].iloc[-1],
        "close": c.iloc[-1],
        "ret_1d": ret(1), "ret_5d": ret(5), "ret_20d": ret(20), "ret_60d": ret(60),
        "vs_ma20": c.iloc[-1] / c.tail(20).mean() - 1,
        "vs_ma60": c.iloc[-1] / c.tail(60).mean() - 1,
        "from_high_60d": c.iloc[-1] / h.tail(60).max() - 1,
        "vol_x20": v.iloc[-1] / v.iloc[-21:-1].mean() if len(v) > 20 else np.nan,
        "atr14": tr.tail(14).mean() / c.iloc[-1],
    }


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.1%}"


def render_brief(market: str, session: str, start: pd.Timestamp, table: pd.DataFrame, bench: dict) -> str:
    money = "{:,.0f}" if market == "KR" else "{:,.2f}"
    rows = ["| 종목 | 출처 | 종가 | 1일 | 5일 | 20일 | 60일 | MA20 대비 | MA60 대비 | 60일 고점 대비 | 거래량/20일평균 | ATR14 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in [bench, *table.to_dict("records")]:
        rows.append(f"| {r['name']} ({r['symbol']}) | {r.get('source', '-')} | {money.format(r['close'])} | {_pct(r['ret_1d'])} | "
                    f"{_pct(r['ret_5d'])} | {_pct(r['ret_20d'])} | {_pct(r['ret_60d'])} | {_pct(r['vs_ma20'])} | "
                    f"{_pct(r['vs_ma60'])} | {_pct(r['from_high_60d'])} | {r['vol_x20']:.1f}x | {r['atr14']:.1%} |")
    last = pd.Timestamp(table["last_day"].max()).tz_convert(TZ[market]).date()
    return "\n".join([
        f"# {LABEL[market]} 매수 후보 브리핑 — {session} 세션",
        "",
        f"- 정규장 시작: {start.tz_convert('Asia/Seoul'):%Y-%m-%d %H:%M} KST",
        f"- 데이터: 토스증권 일봉, 마지막 완료 봉 {last} (이 세션의 가격은 포함되지 않음)",
        f"- 후보: 보통주 {len(table)}종목, 투자유의 제외. 첫 행은 지수 ETF(비교용, 후보 아님)",
        "- 출처: " + ", ".join(f"{k} {v}" for k, v in SOURCES.items())
        + f" (시총 상위 {CAP_UNIVERSE[market]}종목에서 무작위, 앞 출처와 겹치면 다음 종목으로)",
        "- 수익률은 종가 기준, ATR14는 종가 대비 비율",
        "",
        *rows,
        "",
        "> 본인 참고용입니다. 토스증권 API 약관상 제3자 배포가 금지되어 있습니다.",
        "",
    ])


def paths(market: str, session: str) -> tuple[Path, Path]:
    stem = f"{market}_{session.replace('-', '')}"
    return PICKS_DIR / f"{stem}_brief.md", PICKS_DIR / f"{stem}_picks.json"


def prepare(market: str) -> str:
    """Write the brief and the candidate list; return the prompt for Claude Code."""
    today = session_today(market)
    if today is None:
        raise SystemExit(f"{market}: no session today")
    session, start = today
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for s in candidates(market, session).itertuples():
        bars = daily(s.symbol, now)
        if len(bars) < 61:
            logger.info(f"skip {s.name} ({s.symbol}): only {len(bars)} daily bars")
            continue
        rows.append({"symbol": s.symbol, "name": s.name, "source": s.source, **features(bars)})
    table = pd.DataFrame(rows)
    symbol, name = BENCHMARK[market]
    bench = {"symbol": symbol, "name": name, **features(daily(symbol, now))}

    brief, out = paths(market, session)
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    brief.write_text(render_brief(market, session, start, table, bench))
    brief.with_suffix(".json").write_text(json.dumps(
        {"market": market, "session": session, "start": start.isoformat(),
         "bench": {k: bench[k] for k in ("symbol", "name", "close", "ret_1d", "ret_5d")},
         "candidates": table[["symbol", "name", "source", "close", "ret_1d"]].to_dict("records")}, ensure_ascii=False))
    out.unlink(missing_ok=True)
    return string.Template(PROMPT_FILE.read_text()).substitute(
        market=LABEL[market], session=session, brief=brief, out=out, n_picks=N_PICKS,
        open_kst=f"{start.tz_convert('Asia/Seoul'):%H:%M}")


# ── send ──────────────────────────────────────────────────────────────────────

def validate(picks: dict, candidate_symbols: set[str]) -> list[dict]:
    """The ranked picks, each tagged with its rank, if there are N_PICKS distinct valid candidates."""
    items = picks["picks"]
    if len(items) != N_PICKS:
        raise ValueError(f"{len(items)} picks, exactly {N_PICKS} required")
    if len({p["symbol"] for p in items}) != len(items):
        raise ValueError("a stock is picked twice")
    for p in items:
        if p["symbol"] not in candidate_symbols:
            raise ValueError(f"{p['symbol']} is not in the candidate brief")
        if not str(p.get("reason", "")).strip():
            raise ValueError(f"{p['symbol']} has no reason")
        if p.get("conviction") not in CONVICTION:
            raise ValueError(f"{p['symbol']} conviction must be one of {CONVICTION}, got {p.get('conviction')!r}")
    return [{**p, "rank": rank} for rank, p in enumerate(items, 1)]


def stock_link(market: str, symbol: str) -> str:
    """The stock's Naver Securities mobile page, or its front page if the stock is not found there.

    Naver keys US stocks by Reuters code: NASDAQ adds `.O`, NYSE adds nothing,
    AMEX adds `.K` or nothing, so the code is looked up rather than derived.
    """
    if market == "KR":
        return f"{kakao.LINK}/domestic/stock/{symbol}/total"
    for code in (symbol, f"{symbol}.O", f"{symbol}.K"):
        try:
            resp = requests.get(f"https://api.stock.naver.com/stock/{code}/basic", timeout=10)
        except requests.RequestException:
            break
        if resp.ok and resp.json().get("reutersCode") == code:
            return f"{kakao.LINK}/worldstock/stock/{code}/total"
    return kakao.LINK


def track_record(market: str, sessions: int = 20) -> str:
    """One line on the latest settled sessions, from the scores `settle` last wrote."""
    path = PICKS_DIR / "scores.csv"
    scores = pd.read_csv(path, dtype={"symbol": str}) if path.exists() else pd.DataFrame()
    if scores.empty or not scores[(scores["market"] == market) & scores["picked"]].size:
        return "성과: 채점 전 (검증 전 신호)"
    g = scores[scores["market"] == market]
    picked = g[g["picked"]].groupby("session")["net"].mean().sort_index().tail(sessions)
    pool = g.groupby("session")["net"].mean().loc[picked.index]
    return f"최근 {len(picked)}세션 추천 {picked.mean():+.1%} / 후보 {pool.mean():+.1%} ({HOLD}일, 비용 차감)"


def render_messages(market: str, items: list[dict], comment: str, bench: dict | None,
                    record: str) -> list[tuple[str, str | None]]:
    """A header message, then one message per pick linking to its stock page: (text, link)."""
    money = "{:,.0f}" if market == "KR" else "{:,.2f}"
    header = [f"[{LABEL[market]} 매수 후보 {len(items)}]"]
    if comment:
        header.append(f"시장: {comment}")
    if bench:
        header.append(f"지수 {bench['name']} {money.format(bench['close'])} "
                      f"(1일 {_pct(bench['ret_1d'])}, 5일 {_pct(bench['ret_5d'])})")
    counts = {c: sum(p["conviction"] == c for p in items) for c in CONVICTION}
    header.append("확신 " + " · ".join(f"{c} {n}" for c, n in counts.items()))
    header.append(record)
    messages: list[tuple[str, str | None]] = [("\n".join(header), None)]
    for p in items:
        price = f"종가 {money.format(p['close'])} ({_pct(p['ret_1d'])})" if p.get("close") else ""
        stop = (f"손절 {money.format(p['stop'])} ({_pct(p['stop'] / p['close'] - 1)})"
                if p.get("stop") and p.get("close") else "")
        text = (f"{p['rank']}/{len(items)} {p['name']}({p['symbol']})\n"
                f"확신 {p['conviction']}\n"
                + (" · ".join(x for x in (price, stop) if x) + "\n" if price or stop else "")
                + p["reason"])
        messages.append((text, stock_link(market, p["symbol"])))
    return messages


def read_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]


def send(market: str) -> None:
    """Send Claude's picks (or a failure notice) and log them with the candidate pool."""
    today = session_today(market)
    if today is None:
        raise SystemExit(f"{market}: no session today")
    session, _ = today
    brief, out = paths(market, session)
    meta = json.loads(brief.with_suffix(".json").read_text())
    try:
        picks = json.loads(out.read_text())
        items = validate(picks, {c["symbol"] for c in meta["candidates"]})
        candidate = {c["symbol"]: c for c in meta["candidates"]}
        items = [{**p, "source": candidate[p["symbol"]]["source"], "close": candidate[p["symbol"]]["close"],
                  "ret_1d": candidate[p["symbol"]].get("ret_1d")} for p in items]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as exc:
        kakao.send(f"[{LABEL[market]} 매수 후보] {session} 생성 실패: {exc}")
        raise
    messages = render_messages(market, items, picks.get("comment", ""), meta.get("bench"), track_record(market))
    days = kakao.refresh_days_left()
    if days < 10:
        header, link = messages[0]
        messages[0] = (header + f"\n⚠ 카카오 로그인 {days:.0f}일 후 만료: python -m coinbot.kakao login", link)
    for text, link in messages:
        if link:
            kakao.send(text, link=link, button="네이버 증권에서 보기")
        else:
            kakao.send(text)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({"market": market, "session": session, "picks": items,
                            "pool": [{"symbol": c["symbol"], "source": c["source"]} for c in meta["candidates"]],
                            "sent_at": pd.Timestamp.now(tz="UTC").isoformat()}, ensure_ascii=False) + "\n")


# ── settle ────────────────────────────────────────────────────────────────────

def holding_return(bars: pd.DataFrame, session: str, market: str) -> float | None:
    """Session open to the HOLD-th close, or None until that bar has closed."""
    days = bars["open_time"].dt.tz_convert(TZ[market]).dt.strftime("%Y-%m-%d")
    after = bars[days >= session]
    if len(after) < HOLD or days[after.index[0]] != session:
        return None
    return after["close"].iloc[HOLD - 1] / after["open"].iloc[0] - 1


def settle(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per candidate per logged session whose holding window has closed."""
    now = now or pd.Timestamp.now(tz="UTC")
    log = read_log()
    since = min((pd.Timestamp(e["session"], tz="UTC") for e in log), default=now) - pd.Timedelta(days=7)
    cache: dict[str, pd.DataFrame] = {}

    def bars(symbol: str) -> pd.DataFrame:
        if symbol not in cache:
            fetched = toss.fetch(symbol, "1d", since=since)
            cache[symbol] = fetched[fetched["close_time"] <= now].reset_index(drop=True)
        return cache[symbol]

    rows = []
    for e in log:
        market, session = e["market"], e["session"]
        bench = holding_return(bars(BENCHMARK[market][0]), session, market)
        if bench is None:
            continue
        picked = {p["symbol"]: p for p in e["picks"]}
        for c in e["pool"]:
            r = holding_return(bars(c["symbol"]), session, market)
            if r is None:       # no bar on the session day (halted) or delisted
                continue
            pick = picked.get(c["symbol"], {})
            rows.append({"market": market, "session": session, "symbol": c["symbol"], "source": c["source"],
                         "picked": bool(pick), "rank": pick.get("rank"), "conviction": pick.get("conviction"),
                         "net": r - COST[market], "bench": bench})
    scores = pd.DataFrame(rows, columns=["market", "session", "symbol", "source", "picked", "rank",
                                         "conviction", "net", "bench"])
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(PICKS_DIR / "scores.csv", index=False)
    return scores


def summarize(scores: pd.DataFrame) -> str:
    lines = [f"[매수 후보 성과] {HOLD}거래일 보유, 비용 차감 (지수는 미차감)"]
    for market, g in scores.groupby("market"):
        by_session = g.groupby("session")
        per = pd.DataFrame({"pool": by_session["net"].mean(), "bench": by_session["bench"].first(),
                            "picks": g[g["picked"]].groupby("session")["net"].mean()})
        traded = per.dropna(subset=["picks"])
        line = f"{LABEL[market]} {len(per)}세션(추천 {len(traded)}): "
        if len(traded):
            line += (f"추천 {traded['picks'].mean():+.2%}, 같은 날 후보전체 {traded['pool'].mean():+.2%}, "
                     f"우위 {(traded['picks'] > traded['pool']).mean():.0%} / ")
        line += f"전체 세션 후보 {per['pool'].mean():+.2%}, 지수 {per['bench'].mean():+.2%}"
        lines.append(line)
        lines.append("출처별 후보 " + ", ".join(f"{src} {m:+.2%}" for src, m in g.groupby("source")["net"].mean().items())
                     + " / 추천 출처 " + (", ".join(f"{src} {n}" for src, n in g[g["picked"]]["source"].value_counts().items()) or "없음"))
        by_conviction = g.dropna(subset=["conviction"]).groupby("conviction")["net"].agg(["mean", "size"])
        if len(by_conviction):
            lines.append("확신도별 추천 " + ", ".join(f"{c} {by_conviction.loc[c, 'mean']:+.2%}(n={by_conviction.loc[c, 'size']})"
                                                for c in CONVICTION if c in by_conviction.index))
    if len(lines) == 1:
        lines.append("아직 채점된 세션 없음")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["due", "prepare", "send", "settle"])
    parser.add_argument("market", nargs="?", choices=list(TZ))
    parser.add_argument("--send", action="store_true", help="settle: also send the summary to KakaoTalk")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    if args.command != "settle" and args.market is None:
        parser.error(f"{args.command} needs a market")
    if args.command == "due":
        sys.exit(0 if due(args.market) else 1)
    if args.command == "prepare":
        print(prepare(args.market))
    elif args.command == "send":
        send(args.market)
    else:
        text = summarize(settle())
        print(text)
        if args.send:
            kakao.send(text)


if __name__ == "__main__":
    main()
