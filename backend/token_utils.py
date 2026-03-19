from urllib.parse import unquote, urlencode
from typing import Any
from collections.abc import Mapping
import hashlib
import uuid
import jwt # PyJWT
import requests

def _create_jwt(access_key: str, secret_key: str, query_string: str = None) -> str:
    """
    JWT 토큰 생성
    """
    payload = {"access_key": access_key, "nonce": str(uuid.uuid4())}

    if query_string is not None:
        query_hash = hashlib.sha512(query_string.encode("utf-8")).hexdigest()
        payload["query_hash"] = query_hash
        payload["query_hash_alg"] = "SHA512"

    token = jwt.encode(payload, secret_key, algorithm="HS512")
    return token if isinstance(token, str) else token.decode('utf-8')

def _build_query_string(params: Mapping) -> str:
    """
    Dictionary 파라미터를 쿼리 문자열 형식으로 변환
    """
    return unquote(urlencode(params, doseq=True))

if __name__ == "__main__":
    base_url = "https://api.upbit.com"
    access_key = "<YOUR_ACCESS_KEY>"
    secret_key = "<YOUR_SECRET_KEY>" # 실제로는 안전한 방식으로 로드하거나 주입하세요.
    
    # 파라미터가 없는 요청 예시
    jwt_token = _create_jwt(access_key, secret_key)
    headers = {"Authorization": f"Bearer {jwt_token}"}
                
    response = requests.get(f"{base_url}/v1/accounts", headers=headers).json()
        
    print(response)
    
    # 파라미터가 있는 GET 요청 예시
    params = {
        "market": "KRW-BTC",
        "states[]": ["wait", "watch"],
        "limit": 10
    }
    query_string = _build_query_string(params)
    jwt_token = _create_jwt(access_key, secret_key, query_string)
    headers = {"Authorization": f"Bearer {jwt_token}"}
        
    response = requests.get(f"{base_url}/v1/orders/open?{query_string}", headers=headers).json()
    print(response)    
    
    # POST 요청 예시
    order_data = {
        "market": "KRW-BTC",
        "side": "bid",
        "volume": "0.001",
        "price": "50000000",
        "ord_type": "limit"
    }
        
    query_string = _build_query_string(order_data)
    jwt_token = _create_jwt(access_key, secret_key, query_string)
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    # 아래 주석처리된 부분 실행시 실제 주문이 발생하므로 실행 전 반드시 확인하세요.
    # response = requests.post(f"{base_url}/v1/orders", json=order_data, headers=headers).json()
    # print(response)   
