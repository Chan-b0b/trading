import json

import pandas as pd
import pytest

from coinbot import toss


def raw_candle(stamp: pd.Timestamp, price: float, volume: float = 1.0) -> dict:
    return {
        "timestamp": stamp.isoformat(),
        "openPrice": str(price),
        "highPrice": str(price + 1),
        "lowPrice": str(price - 1),
        "closePrice": str(price),
        "volume": str(volume),
        "currency": "KRW",
    }


def minute_candles(n: int, start: str = "2026-03-25 09:01") -> list[dict]:
    stamps = pd.date_range(start, periods=n, freq="1min", tz="Asia/Seoul")
    return [raw_candle(t, float(i)) for i, t in enumerate(stamps)]


def fake_pages(candles: list[dict], calls: list):
    """Serve `candles` like the API: newest first, `before` inclusive, nextBefore = oldest returned."""
    def page(symbol, interval, before):
        calls.append(before)
        older = [c for c in candles
                 if before is None or pd.Timestamp(c["timestamp"]) <= pd.Timestamp(before)]
        chunk = list(reversed(older))[: toss.MAX_COUNT]
        more = len(older) > len(chunk)
        return {"candles": chunk, "nextBefore": chunk[-1]["timestamp"] if more else None}
    return page


class FakeResponse:
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def token_file(tmp_path, monkeypatch):
    path = tmp_path / "token.json"
    monkeypatch.setattr(toss, "TOKEN_FILE", path)
    monkeypatch.setattr(toss, "credentials", lambda: ("id", "secret"))
    return path


def test_minute_timestamp_is_close_time():
    df = toss.to_frame(minute_candles(1), "1m")
    assert df["close_time"].iloc[0] == pd.Timestamp("2026-03-25 09:01", tz="Asia/Seoul")
    assert df["open_time"].iloc[0] == pd.Timestamp("2026-03-25 09:00", tz="Asia/Seoul")


def test_daily_bar_closes_at_next_local_midnight():
    df = toss.to_frame([raw_candle(pd.Timestamp("2026-03-25", tz="Asia/Seoul"), 100)], "1d")
    assert df["close_time"].iloc[0] == pd.Timestamp("2026-03-26", tz="Asia/Seoul")


def test_fetch_paginates_and_stops_at_since(monkeypatch):
    candles = minute_candles(450)
    calls = []
    monkeypatch.setattr(toss, "_get_page", fake_pages(candles, calls))
    monkeypatch.setattr(toss, "REQUEST_DELAY", 0)

    full = toss.fetch("005930", "1m")
    assert len(full) == 450 and full["open_time"].is_monotonic_increasing
    assert len(calls) == 3

    calls.clear()
    since = toss.to_frame(candles[300:301], "1m")["open_time"].iloc[0]
    recent = toss.fetch("005930", "1m", since=since)
    assert recent["open_time"].iloc[0] == since and len(recent) == 150
    assert len(calls) == 1


def test_load_refetches_when_history_was_adjusted(tmp_path, monkeypatch):
    monkeypatch.setattr(toss, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(toss, "REQUEST_DELAY", 0)
    candles = minute_candles(10)
    monkeypatch.setattr(toss, "_get_page", fake_pages(candles, []))
    toss.load("005930", "1m")

    split = [dict(c, closePrice=str(float(c["closePrice"]) / 2)) for c in candles] + minute_candles(2, "2026-03-25 09:11")
    monkeypatch.setattr(toss, "_get_page", fake_pages(split, []))
    out = toss.load("005930", "1m")

    assert len(out) == 12
    assert out["close"].iloc[4] == 2.0   # old bars replaced by adjusted prices, not mixed


def test_token_is_reused_from_cache(token_file, monkeypatch):
    posts = []
    monkeypatch.setattr(toss.requests, "post", lambda *a, **k: posts.append(1) or
                        FakeResponse(200, {"access_token": "t1", "expires_in": 86400}))
    assert toss.token() == "t1"
    assert toss.token() == "t1"
    assert len(posts) == 1


def test_rejected_token_prefers_newer_cached_one(token_file, monkeypatch):
    token_file.write_text(json.dumps({"access_token": "from-other-process",
                                      "expires_at": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).isoformat()}))
    monkeypatch.setattr(toss.requests, "post", lambda *a, **k: pytest.fail("should not reissue"))
    assert toss.token(rejected="old") == "from-other-process"


def test_get_retries_after_401_and_429(token_file, monkeypatch):
    monkeypatch.setattr(toss.requests, "post", lambda *a, **k:
                        FakeResponse(200, {"access_token": "fresh", "expires_in": 86400}))
    token_file.write_text(json.dumps({"access_token": "stale",
                                      "expires_at": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).isoformat()}))
    monkeypatch.setattr(toss.time, "sleep", lambda s: None)
    replies = iter([FakeResponse(401, {"error": {"code": "token-revoked"}}),
                    FakeResponse(429, {"error": {"code": "rate-limit-exceeded"}}, {"Retry-After": "1"}),
                    FakeResponse(200, {"result": {"ok": True}})])
    used = []

    def fake_get(url, params, timeout, headers):
        used.append(headers["Authorization"])
        return next(replies)

    monkeypatch.setattr(toss.requests, "get", fake_get)
    assert toss.get("/api/v1/prices", symbols="005930") == {"ok": True}
    assert used == ["Bearer stale", "Bearer fresh", "Bearer fresh"]
