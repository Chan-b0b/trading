import asyncio
import time
import json
import gzip
import os
import uuid
import re
from typing import List

from old.backend.token_utils import _create_jwt
import websockets
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve the frontend static files from the frontend/ directory under /static
app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

SYMBOL = "KRW-BTC"
POLL_INTERVAL = 2.0
MAX_HISTORY = 1000
 
USE_POLLING = True

current_price = {"symbol": SYMBOL, "price": None, "ts": None}
clients: List[WebSocket] = []
price_history: List[dict] = []


async def record_price(price_val: float, ts: int, symbol: str):
    """Append a snapshot to history and broadcast to clients."""
    price_history.append({'ts': ts, 'price': price_val, 'symbol': symbol})
    if len(price_history) > MAX_HISTORY:
        price_history.pop(0)
    await broadcast_price()


async def upbit_ws_loop(symbols: List[str]):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_message = [{"ticket": "trade"}, {"type": "ticker", "codes": symbols}]
    while True:
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps(subscribe_message))
                async for message in ws:
                    try:
                        if isinstance(message, (bytes, bytearray)):
                            try:
                                msg = gzip.decompress(message).decode('utf-8')
                            except Exception:
                                msg = message.decode('utf-8', errors='ignore')
                        else:
                            msg = message
                        data = json.loads(msg)
                        # Upbit ticker fields: 'code', 'trade_price', 'timestamp'(ms)
                        price_val = float(data.get('trade_price'))
                        ts = int(data.get('timestamp') / 1000)
                        symbol = data.get('code')
                        current_price.update({'symbol': symbol, 'price': price_val, 'ts': ts})
                        # only record websocket updates when not polling
                        if not USE_POLLING:
                            await record_price(price_val, ts, symbol)
                    except Exception as exc:
                        print('upbit message handling error:', exc)
        except Exception as e:
            print('upbit ws error:', e)
        await asyncio.sleep(2)


async def polling_loop():
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(f"https://api.upbit.com/v1/ticker?markets={SYMBOL}")
                resp.raise_for_status()
                arr = resp.json()
                if arr and isinstance(arr, list):
                    d = arr[0]
                    price_val = float(d.get('trade_price'))
                    ts = int(d.get('timestamp') / 1000) if d.get('timestamp') else int(time.time())
                    symbol = d.get('market') or SYMBOL
                    current_price.update({'symbol': symbol, 'price': price_val, 'ts': ts})
                    price_history.append({'ts': ts, 'price': price_val, 'symbol': symbol})
                    if len(price_history) > MAX_HISTORY:
                        price_history.pop(0)
                    await broadcast_price()
            except Exception as exc:
                print('polling error:', exc)
            await asyncio.sleep(POLL_INTERVAL)


async def upbit_private_ws_loop():
    # requires UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY in env
    access_key = os.environ.get('UPBIT_ACCESS_KEY')
    secret_key = os.environ.get('UPBIT_SECRET_KEY')
    if not access_key or not secret_key:
        print('private keys not set; skipping private websocket')
        return

    try:
        import websocket as ws_client  # websocket-client
        import threading
    except Exception as e:
        print('websocket-client not available:', e)
        return

    uri = 'wss://api.upbit.com/websocket/v1/private'
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }
    # create JWT using shared helper (ensures HS512 and string result)
    token = _create_jwt(access_key, secret_key)
    authorization_token = f'Bearer {token}'
    # websocket-client expects header lines; include Origin to match Upbit expectations
    header = [f'Authorization: {authorization_token}', 'Origin: https://api.upbit.com']
    subscribe_message = [{"ticket": "private"}, {"type": "myAsset"}]

    loop = asyncio.get_running_loop()

    def run_ws():
        def on_message(ws, message):
            try:
                msg = None
                if isinstance(message, (bytes, bytearray)):
                    try:
                        msg = gzip.decompress(message).decode('utf-8')
                    except Exception:
                        msg = message.decode('utf-8', errors='ignore')
                else:
                    msg = message
                data = json.loads(msg)
                print('private msg:', data)
                price_val = data.get('price') or data.get('trade_price')
                ts_ms = data.get('timestamp') or data.get('created_at')
                if price_val and ts_ms:
                    try:
                        price_val = float(price_val)
                        ts = int(ts_ms / 1000) if isinstance(ts_ms, (int, float)) else int(time.time())
                        symbol = data.get('code', SYMBOL)
                        current_price.update({'symbol': symbol, 'price': price_val, 'ts': ts})
                        # only record private websocket updates when not polling
                        if not USE_POLLING:
                            asyncio.run_coroutine_threadsafe(record_price(price_val, ts, symbol), loop)
                    except Exception:
                        pass
            except Exception as exc:
                print('private message handling error (thread):', exc)

        def on_open(ws):
            try:
                ws.send(json.dumps(subscribe_message))
            except Exception as e:
                print('private ws send error on open:', e)

        def on_error(ws, err):
            print('private ws error (thread):', err)

        def on_close(ws, code, reason):
            print('private ws closed', code, reason)

        while True:
            try:
                ws_app = ws_client.WebSocketApp(uri, header=header, on_message=on_message, on_open=on_open, on_error=on_error, on_close=on_close)
                ws_app.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print('private ws run_forever error:', e)
            time.sleep(2)

    t = threading.Thread(target=run_ws, daemon=True)
    t.start()


async def broadcast_price() -> None:
    dead = []
    for ws in clients.copy():
        try:
            await ws.send_json(current_price)
        except Exception:
            dead.append(ws)
    for d in dead:
        try:
            clients.remove(d)
        except ValueError:
            pass


@app.on_event("startup")
async def on_startup() -> None:
    # startup
    # attempt to load API keys from a local file (api_keys.txt by default)
    def load_keys_from_file(path=None):
        p = path or os.environ.get('API_KEYS_FILE') or os.path.join(os.getcwd(), 'api_keys.txt')
        if not os.path.exists(p):
            return False
        try:
            text = open(p, 'r', encoding='utf-8').read()
            akm = re.search(r'Access Key\s*\n\s*([^\n\r]+)', text, re.IGNORECASE)
            skm = re.search(r'Secret Key\s*\n\s*([^\n\r]+)', text, re.IGNORECASE)
            if akm and skm:
                ak = akm.group(1).strip()
                sk = skm.group(1).strip()
                os.environ['UPBIT_ACCESS_KEY'] = ak
                os.environ['UPBIT_SECRET_KEY'] = sk
                print(f'Loaded API keys from {p}')
                return True
        except Exception as e:
            print('error loading key file:', e)
        return False

    loaded = load_keys_from_file()
    # start either polling or websocket feed depending on configuration
    symbols = [SYMBOL]
    if USE_POLLING:
        asyncio.create_task(polling_loop())
    else:
        asyncio.create_task(upbit_ws_loop(symbols))
    # if keys were loaded (or present in env), start private websocket loop
    if os.environ.get('UPBIT_ACCESS_KEY') and os.environ.get('UPBIT_SECRET_KEY'):
        asyncio.create_task(upbit_private_ws_loop())


@app.get("/price")
async def get_price():
    return JSONResponse(current_price)


@app.get("/history")
async def get_history():
    return JSONResponse(price_history)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        # send current price immediately
        await websocket.send_json(current_price)
        while True:
            # keep connection alive; we don't expect client messages
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        try:
            clients.remove(websocket)
        except ValueError:
            pass
