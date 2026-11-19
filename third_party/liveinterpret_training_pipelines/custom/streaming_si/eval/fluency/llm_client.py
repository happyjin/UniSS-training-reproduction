"""LiteLLM 网关 HTTP 调用封装（与本项目既有调用方式一致）。

只依赖 requests。密钥来自环境变量：
    LITELLM_API_KEY（或本项目原有的 SPEECH_LLM_API_KEY）
    LITELLM_API_BASE（默认 https://litellm.ai.levelinfinite.com/v1）
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import LITELLM_API_BASE, LITELLM_API_KEY, LITELLM_MODEL, LITELLM_TIMEOUT

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# 限流 / 服务端错误退避重试
MAX_RETRIES = int(os.getenv("LITELLM_MAX_RETRIES", "5"))
RETRY_BASE_S = float(os.getenv("LITELLM_RETRY_BASE_S", "3"))
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _rate_limit() -> None:
    path = os.getenv("LITELLM_RATE_FILE")
    rpm = float(os.getenv("LITELLM_RPM", "90"))
    if not path or rpm <= 0:
        return
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        last = float(f.read().strip() or 0)
        now = time.time()
        target = max(now, last + 60.0 / rpm)
        f.seek(0)
        f.truncate()
        f.write(str(target))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    time.sleep(max(0.0, target - time.time()))


def _api_key() -> str:
    return os.getenv("LITELLM_API_KEY") or os.getenv("SPEECH_LLM_API_KEY") or LITELLM_API_KEY


def available() -> bool:
    """是否已配置可用的网关密钥。"""
    return bool(_api_key())


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    no_cache: bool = True,
) -> Tuple[str, str]:
    """调用 /chat/completions，返回 (content, request_id)。"""
    key = api_key or _api_key()
    if not key:
        raise LLMError("未设置 LITELLM_API_KEY / SPEECH_LLM_API_KEY，无法调用网关。")

    base = (api_base or LITELLM_API_BASE).rstrip("/")
    payload: Dict[str, Any] = {
        "model": model or LITELLM_MODEL,
        "messages": messages,
    }
    if no_cache:
        # 网关默认开启 Redis 缓存，关掉以保证同样输入得到新鲜结果
        payload["cache"] = {"no-cache": True}

    for attempt in range(MAX_RETRIES + 1):
        _rate_limit()
        try:
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout or LITELLM_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise LLMError(f"请求网关失败：{type(exc).__name__}") from exc
        else:
            if resp.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                break
        time.sleep(min(60, RETRY_BASE_S * 2 ** attempt) + random.uniform(0, 1))

    try:
        body = resp.json()
    except ValueError:
        raise LLMError(f"网关返回非 JSON（HTTP {resp.status_code}）：{resp.text[:300]}")

    if resp.status_code != 200 or "choices" not in body:
        err = body.get("error") if isinstance(body, dict) else None
        msg = err.get("message") if isinstance(err, dict) else str(body)[:300]
        raise LLMError(f"网关返回错误（HTTP {resp.status_code}）：{msg}")

    choice = body["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    return content, body.get("id", "")


def chat_json(
    messages: List[Dict[str, str]],
    **kwargs,
) -> Tuple[Any, str]:
    """同上，但把返回内容解析为 JSON（容忍 ```json 代码块与前后多余文字）。"""
    content, request_id = chat(messages, **kwargs)
    text = content.strip()

    candidates = [text]
    m = _FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())

    # 截取第一个 [ ... ] 或 { ... }
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        s, e = text.find(open_ch), text.rfind(close_ch)
        if s != -1 and e > s:
            candidates.append(text[s:e + 1])

    for cand in candidates:
        try:
            return json.loads(cand), request_id
        except (ValueError, TypeError):
            continue
    raise LLMError(f"无法把模型返回解析为 JSON：{text[:300]}")
