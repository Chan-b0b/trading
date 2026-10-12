"""
KakaoTalk "send to me" (나에게 보내기): text messages to the account holder's
own KakaoTalk, so reports built from Toss data never reach a third party.

One-time setup:
  1. developers.kakao.com → 내 애플리케이션 → create an app, then in its 앱 관리 페이지:
     - [카카오 로그인] > [사용 설정]: 상태 ON
     - [카카오 로그인] > [리다이렉트 URI]: http://localhost:8080
     - [카카오 로그인] > [동의항목]: "카카오톡 메시지 전송" (talk_message)
     - [앱] > [제품 링크 관리] > [웹 도메인]: https://m.stock.naver.com (message link target)
     - [앱] > [플랫폼 키] > [REST API 키]: the key, and its 클라이언트 시크릿 if enabled
  2. kakao_api.txt in the repo root (label line, value line, like toss_api.txt):
         rest_api_key
         <REST API key>
         client_secret
         <client secret, only if enabled in the app's security settings>
  3. python -m coinbot.kakao login         # prints the consent URL; open it and approve
     python -m coinbot.kakao login CODE    # CODE = the `code` param of the localhost URL you land on
  4. python -m coinbot.kakao test

The access token lives ~6 hours and is refreshed automatically. The refresh
token lives ~2 months; Kakao renews it only during its last month, on refresh.
If it lapses, repeat step 3. `refresh_days_left()` lets callers warn early.
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd
import requests

from coinbot.toss import ROOT

logger = logging.getLogger(__name__)

AUTH_URL = "https://kauth.kakao.com"
API_URL = "https://kapi.kakao.com"
REDIRECT_URI = "http://localhost:8080"
CRED_FILE = ROOT / "kakao_api.txt"
TOKEN_FILE = ROOT / "data" / "kakao" / "token.json"
LINK = "https://m.stock.naver.com"      # opens on phones; Toss's web does not
MAX_TEXT = 200          # text template limit


class KakaoError(RuntimeError):
    pass


def credentials() -> dict[str, str]:
    lines = [ln.strip() for ln in CRED_FILE.read_text().splitlines() if ln.strip()]
    return {lines[i].lower(): lines[i + 1] for i in range(0, len(lines) - 1, 2)}


def _client() -> dict[str, str]:
    cred = credentials()
    params = {"client_id": cred["rest_api_key"]}
    if cred.get("client_secret"):
        params["client_secret"] = cred["client_secret"]
    return params


def authorize_url() -> str:
    return (f"{AUTH_URL}/oauth/authorize?client_id={credentials()['rest_api_key']}"
            f"&redirect_uri={REDIRECT_URI}&response_type=code&scope=talk_message")


def _token_request(**data) -> dict:
    resp = requests.post(f"{AUTH_URL}/oauth/token", timeout=15, data={**_client(), **data})
    if resp.status_code != 200:
        raise KakaoError(f"token request failed: {resp.status_code} {resp.text}")
    return resp.json()


def _save(body: dict, old: dict | None = None) -> dict:
    """Store a token response; a refresh response omits the refresh token unless it was renewed."""
    now = pd.Timestamp.now(tz="UTC")
    saved = dict(old or {})
    saved["access_token"] = body["access_token"]
    saved["expires_at"] = (now + pd.Timedelta(seconds=body["expires_in"])).isoformat()
    if "refresh_token" in body:
        saved["refresh_token"] = body["refresh_token"]
        saved["refresh_expires_at"] = (now + pd.Timedelta(seconds=body["refresh_token_expires_in"])).isoformat()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(saved))
    TOKEN_FILE.chmod(0o600)
    return saved


def _load() -> dict:
    try:
        return json.loads(TOKEN_FILE.read_text())
    except FileNotFoundError:
        raise KakaoError("not logged in: run `python -m coinbot.kakao login`") from None


def login(code: str) -> None:
    _save(_token_request(grant_type="authorization_code", redirect_uri=REDIRECT_URI, code=code))


def access_token() -> str:
    saved = _load()
    if pd.Timestamp(saved["expires_at"]) - pd.Timedelta(minutes=5) > pd.Timestamp.now(tz="UTC"):
        return saved["access_token"]
    body = _token_request(grant_type="refresh_token", refresh_token=saved["refresh_token"])
    logger.info("refreshed Kakao access token")
    return _save(body, saved)["access_token"]


def refresh_days_left() -> float:
    saved = _load()
    return (pd.Timestamp(saved["refresh_expires_at"]) - pd.Timestamp.now(tz="UTC")) / pd.Timedelta(days=1)


def chunks(text: str, limit: int = MAX_TEXT) -> list[str]:
    """Split on line boundaries into messages of at most `limit` chars; overlong lines are cut."""
    out, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        out.append(cur)
    return out


def send(text: str, link: str = LINK, button: str | None = None) -> None:
    """Send `text` to my own KakaoTalk, as several messages if it is long.

    Every message carries a link (the template requires one); its domain must be
    registered under [앱] > [제품 링크 관리] > [웹 도메인]. `button` relabels the
    default "자세히 보기" button.
    """
    tok = access_token()
    for part in chunks(text):
        template = {"object_type": "text", "text": part,
                    "link": {"web_url": link, "mobile_web_url": link}}
        if button:
            template["button_title"] = button
        resp = requests.post(f"{API_URL}/v2/api/talk/memo/default/send", timeout=15,
                             headers={"Authorization": f"Bearer {tok}"},
                             data={"template_object": json.dumps(template, ensure_ascii=False)})
        if resp.status_code != 200 or resp.json().get("result_code") != 0:
            raise KakaoError(f"send failed: {resp.status_code} {resp.text}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "login" and not rest:
        print(authorize_url())
    elif cmd == "login":
        login(rest[0])
        print(f"logged in; refresh token valid {refresh_days_left():.0f} more days")
    elif cmd == "test":
        send("coinbot 카카오톡 연결 테스트입니다.")
        print("sent")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
