import numpy as np
import pandas as pd
import pytest

from coinbot import kakao, picks


def daily_bars(days: list[str], market: str, opens=None, closes=None) -> pd.DataFrame:
    """Daily bars stamped at local midnight, like Toss."""
    stamps = pd.DatetimeIndex([pd.Timestamp(d, tz=picks.TZ[market]) for d in days]).tz_convert("UTC")
    n = len(days)
    opens = np.arange(n, dtype=float) + 100 if opens is None else np.asarray(opens, float)
    closes = opens + 1 if closes is None else np.asarray(closes, float)
    return pd.DataFrame({"open_time": stamps, "close_time": stamps + pd.Timedelta(days=1),
                         "open": opens, "high": np.maximum(opens, closes) + 1,
                         "low": np.minimum(opens, closes) - 1, "close": closes, "volume": 1000.0})


def test_us_window_matches_exactly_one_cron_run_in_each_dst_regime():
    kst = "Asia/Seoul"
    runs = [pd.Timestamp("2026-10-02 21:50", tz=kst), pd.Timestamp("2026-10-02 22:50", tz=kst)]
    summer = pd.Timestamp("2026-10-02 22:30", tz=kst)
    winter = pd.Timestamp("2026-10-02 23:30", tz=kst)
    assert [picks.in_window("US", summer, t) for t in runs] == [True, False]
    assert [picks.in_window("US", winter, t) for t in runs] == [False, True]


def test_kr_window_catches_normal_and_late_open():
    run = pd.Timestamp("2026-10-02 08:20", tz="Asia/Seoul")
    assert picks.in_window("KR", pd.Timestamp("2026-10-02 09:00", tz="Asia/Seoul"), run)
    assert picks.in_window("KR", pd.Timestamp("2026-10-02 10:00", tz="Asia/Seoul"), run)
    assert not picks.in_window("KR", pd.Timestamp("2026-10-02 09:00", tz="Asia/Seoul"),
                               pd.Timestamp("2026-10-02 08:50", tz="Asia/Seoul"))


def test_features_use_only_given_bars():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-06-01", periods=70)]
    bars = daily_bars(days, "KR")
    f = picks.features(bars)
    assert f["close"] == bars["close"].iloc[-1]
    assert f["ret_1d"] == pytest.approx(bars["close"].iloc[-1] / bars["close"].iloc[-2] - 1)
    assert f["ret_60d"] == pytest.approx(bars["close"].iloc[-1] / bars["close"].iloc[-61] - 1)
    assert f["vol_x20"] == pytest.approx(1.0)
    # adding a later bar must not change features computed before it
    assert picks.features(bars.iloc[:-1])["close"] == bars["close"].iloc[-2]


def ranked_picks(symbols, conviction="중"):
    return {"picks": [{"symbol": sym, "name": sym, "reason": "x", "conviction": conviction} for sym in symbols]}


def test_validate_requires_exactly_n_distinct_candidates_and_tags_rank():
    syms = [str(i) for i in range(picks.N_PICKS)]
    items = picks.validate(ranked_picks(syms), set(syms))
    assert [p["rank"] for p in items] == list(range(1, picks.N_PICKS + 1))
    assert [p["symbol"] for p in items] == syms
    with pytest.raises(ValueError, match="exactly"):
        picks.validate(ranked_picks(syms[:-1]), set(syms))
    with pytest.raises(ValueError, match="twice"):
        picks.validate(ranked_picks([syms[0]] * picks.N_PICKS), set(syms))
    with pytest.raises(ValueError, match="not in the candidate"):
        picks.validate(ranked_picks(syms), set(syms[1:]))
    with pytest.raises(ValueError, match="conviction"):
        picks.validate(ranked_picks(syms, conviction="high"), set(syms))
    bad = ranked_picks(syms)
    bad["picks"][2]["reason"] = " "
    with pytest.raises(ValueError, match="no reason"):
        picks.validate(bad, set(syms))


def test_holding_return_from_session_open_to_hold_th_close():
    days = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]
    bars = daily_bars(days, "US", opens=[10, 20, 30, 40, 50, 60, 70], closes=[11, 21, 31, 41, 51, 61, 71])
    # session 09-29: open 20 → 5th bar (10-05) close 61
    assert picks.holding_return(bars, "2026-09-29", "US") == pytest.approx(61 / 20 - 1)
    # not enough closed bars yet
    assert picks.holding_return(bars, "2026-10-01", "US") is None
    # session day itself missing (e.g. halted): not scored
    assert picks.holding_return(bars.drop(index=1), "2026-09-29", "US") is None


def test_kakao_chunks_respect_limit_and_keep_lines():
    text = "\n".join(f"{i}. " + "가" * 50 for i in range(10))
    parts = kakao.chunks(text)
    assert all(len(p) <= kakao.MAX_TEXT for p in parts)
    assert "\n".join(parts) == text
    assert kakao.chunks("x" * 450) == ["x" * 200, "x" * 200, "x" * 50]


def test_one_message_per_pick_each_fitting_one_kakao_message_with_its_link():
    items = [{"symbol": "005930", "name": "삼성전자", "reason": "가" * 100, "stop": 1_100_000, "close": 1_234_000,
              "ret_1d": 0.139, "conviction": "상" if r == 1 else "하", "source": "①", "rank": r}
             for r in range(1, picks.N_PICKS + 1)]
    bench = {"name": "KODEX 200", "close": 111520, "ret_1d": 0.018, "ret_5d": -0.003}
    messages = picks.render_messages("KR", items, "나" * 60, bench, "성과: 채점 전 (검증 전 신호)")
    assert len(messages) == 1 + picks.N_PICKS
    header, link = messages[0]
    assert link is None and "2026" not in header
    assert "지수 KODEX 200 111,520 (1일 +1.8%, 5일 -0.3%)" in header and "확신 상 1 · 중 0 · 하 4" in header
    text, link = messages[1]
    assert text.startswith("1/5 삼성전자(005930)\n확신 상\n종가 1,234,000 (+13.9%) · 손절 1,100,000 (-10.9%)\n")
    assert link == "https://m.stock.naver.com/domestic/stock/005930/total"
    assert all(len(kakao.chunks(t)) == 1 for t, _ in messages)


def test_track_record_reads_latest_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(picks, "PICKS_DIR", tmp_path)
    assert picks.track_record("KR") == "성과: 채점 전 (검증 전 신호)"
    pd.DataFrame([
        {"market": "KR", "session": "2026-10-06", "symbol": "A", "source": "①", "picked": True, "net": 0.02},
        {"market": "KR", "session": "2026-10-06", "symbol": "B", "source": "①", "picked": False, "net": 0.0},
        {"market": "US", "session": "2026-10-06", "symbol": "C", "source": "①", "picked": True, "net": 0.5},
    ]).to_csv(tmp_path / "scores.csv", index=False)
    assert picks.track_record("KR") == "최근 1세션 추천 +2.0% / 후보 +1.0% (5일, 비용 차감)"


def test_us_link_uses_the_reuters_code_naver_knows(monkeypatch):
    class Resp:
        def __init__(self, code):
            self.code, self.ok = code, True

        def json(self):
            return {"reutersCode": self.code} if self.code in known else {"code": "StockConflict"}

    monkeypatch.setattr(picks.requests, "get", lambda url, timeout: Resp(url.split("/")[-2]))
    known = {"MU.O", "KO"}
    assert picks.stock_link("US", "MU") == "https://m.stock.naver.com/worldstock/stock/MU.O/total"
    assert picks.stock_link("US", "KO") == "https://m.stock.naver.com/worldstock/stock/KO/total"
    assert picks.stock_link("US", "ZZZZ") == "https://m.stock.naver.com"


def test_fill_takes_n_unique_per_source_in_order():
    names = {s: s for s in "ABCDEFGH"}
    out = picks.fill({"①": ["A", "B", "X"], "②": ["A", "B", "C", "D"], "③": ["D", "E", "F", "G"]}, names, 2)
    assert out.set_index("symbol")["source"].to_dict() == {"A": "①", "B": "①", "C": "②", "D": "②",
                                                           "E": "③", "F": "③"}


def test_fill_skips_rejected_symbols_only_where_checked():
    names = {s: s for s in "ABCD"}
    out = picks.fill({"①": ["A"], "③": ["B", "C", "D"]}, names, 1,
                     ok=lambda symbol, source: source != "③" or symbol != "B")
    assert list(out["symbol"]) == ["A", "C"]


def test_settle_scores_each_candidate_and_summarizes_by_source(monkeypatch, tmp_path):
    days = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    up = daily_bars(days, "KR", opens=[100] * 6, closes=[110] * 6)      # +10% over the window
    flat = daily_bars(days, "KR", opens=[100] * 6, closes=[100] * 6)
    monkeypatch.setattr(picks.toss, "fetch", lambda symbol, interval, since: up if symbol == "AAA" else flat)
    monkeypatch.setattr(picks, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(picks, "read_log", lambda: [{
        "market": "KR", "session": "2026-09-28",
        "picks": [{"symbol": "AAA", "source": "③", "rank": 1, "conviction": "상"}],
        "pool": [{"symbol": "AAA", "source": "③"}, {"symbol": "BBB", "source": "①"}]}])
    scores = picks.settle(now=pd.Timestamp("2026-10-06", tz="UTC"))
    assert len(scores) == 2
    row = scores.set_index("symbol").loc["AAA"]
    assert row["picked"] and row["net"] == pytest.approx(0.10 - picks.COST["KR"])
    text = picks.summarize(scores)
    assert "추천 +9.77%" in text and "③ +9.77%" in text and "추천 출처 ③ 1" in text
    assert "확신도별 추천 상 +9.77%(n=1)" in text
