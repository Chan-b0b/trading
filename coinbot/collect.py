"""
Record Upbit orderbook and trade streams to parquet for short-horizon research.

    nohup .venv/bin/python -m coinbot.collect >> data/ws/collect.log 2>&1 &

Files: data/ws/{orderbook,trade}/{market}/{YYYY-MM-DD}/{HHMMSS}.parquet, one per
FLUSH_SEC per market (UTC). Connects/disconnects go to data/ws/events.jsonl so
gaps in the record are known. Trades may repeat across reconnects (SNAPSHOT
messages); dedupe on `seq` when reading.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
import websockets

logger = logging.getLogger(__name__)

WS_URL = "wss://api.upbit.com/websocket/v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "ws"
MARKETS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
FLUSH_SEC = 600
PING_SEC = 60      # Upbit expects a text "PING"; it answers {"status": "UP"}
STALL_SEC = 30     # ~25 msg/s normally, so 30s of silence means a dead connection
LEVELS = 30


def orderbook_row(msg: dict, recv_ms: int) -> dict:
    row = {"ts": msg["timestamp"], "recv_ts": recv_ms,
           "total_ask_size": msg["total_ask_size"], "total_bid_size": msg["total_bid_size"]}
    units = msg["orderbook_units"]
    for i in range(LEVELS):
        u = units[i] if i < len(units) else {}
        for k in ("ask_price", "ask_size", "bid_price", "bid_size"):
            row[f"{k}_{i}"] = u.get(k, float("nan"))
    return row


def trade_row(msg: dict, recv_ms: int) -> dict:
    # ask_bid is the taker side: BID = aggressive buy, ASK = aggressive sell
    return {"ts": msg["trade_timestamp"], "recv_ts": recv_ms, "price": msg["trade_price"],
            "volume": msg["trade_volume"], "side": msg["ask_bid"], "seq": msg["sequential_id"],
            "stream_type": msg.get("stream_type", "")}


PARSERS = {"orderbook": orderbook_row, "trade": trade_row}


class Buffer:
    """Rows per (type, market); `flush` writes each group to its own parquet file."""

    def __init__(self, out_dir: Path | None = None):
        self.out_dir = out_dir or OUT_DIR
        self.rows: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def add(self, msg: dict, recv_ms: int) -> None:
        kind = msg.get("type")
        if kind in PARSERS:
            self.rows[(kind, msg["code"])].append(PARSERS[kind](msg, recv_ms))

    def flush(self) -> list[Path]:
        written = []
        for (kind, market), rows in self.rows.items():
            if not rows:
                continue
            start = pd.Timestamp(rows[0]["recv_ts"], unit="ms", tz="UTC")
            path = (self.out_dir / kind / market / start.strftime("%Y-%m-%d")
                    / f"{start.strftime('%H%M%S')}.parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
            written.append(path)
        self.rows.clear()
        return written


def _event(kind: str, **info) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "events.jsonl", "a") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now(tz="UTC").isoformat(), "event": kind, **info}) + "\n")


async def run(markets: list[str]) -> None:
    buf = Buffer()
    sub = [{"ticket": str(uuid.uuid4())},
           {"type": "orderbook", "codes": markets},
           {"type": "trade", "codes": markets}]
    next_flush = time.time() + FLUSH_SEC
    backoff = 1.0
    try:
        while True:
            try:
                async with websockets.connect(WS_URL, ping_interval=None, max_size=2 ** 22) as ws:
                    await ws.send(json.dumps(sub))
                    _event("connected", markets=markets)
                    backoff = 1.0
                    next_ping = time.time() + PING_SEC
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=STALL_SEC)
                        now = time.time()
                        buf.add(json.loads(raw), int(now * 1000))
                        if now >= next_ping:
                            await ws.send("PING")
                            next_ping = now + PING_SEC
                        if now >= next_flush:
                            logger.info(f"flushed {len(buf.flush())} files")
                            next_flush = now + FLUSH_SEC
            except Exception as exc:          # network errors, stalls, malformed frames
                _event("disconnected", error=f"{type(exc).__name__}: {exc}")
                logger.warning(f"websocket error: {exc!r}; reconnecting in {backoff:.0f}s")
                buf.flush()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
    finally:
        buf.flush()
        _event("stopped")


def _terminate(*_) -> None:
    raise SystemExit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=MARKETS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _terminate)     # flush the buffer on kill
    asyncio.run(run(args.markets))


if __name__ == "__main__":
    main()
