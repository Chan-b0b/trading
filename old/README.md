# Crypto Price Live (minimal)

This small project provides a minimal FastAPI backend that polls Binance for the current `BTCUSDT` price and serves a static frontend that displays live price updates via WebSocket (with a polling fallback).

Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open http://localhost:8000/ in your browser — the page will show the current price and update in near-real-time.

Notes
- The polling interval is controlled by `POLL_INTERVAL` in `backend/main.py`.
- Change the symbol by editing `SYMBOL` in `backend/main.py` (e.g., `ETHUSDT`).

Short-timeframe BTC strategy backtest

This project also includes a rule-based KRW-BTC 5-30 minute strategy backtest.
It uses 5-minute candles for execution, 15-minute candles for momentum/trend,
and 1-hour candles as a broader trend filter.

```bash
python -m btcbot.backtest.short_timeframe_strategy --candles 5000 --plot
```

To search a small parameter grid and then run the best result:

```bash
python -m btcbot.backtest.short_timeframe_strategy --candles 8000 --optimize --plot
```

CSV results and charts are saved under `btcbot/logs/short_timeframe_backtests/`.
