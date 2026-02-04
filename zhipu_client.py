import os
from typing import Any, Dict, List, Optional

import requests


class ZhipuClient:
    """
    智谱（BigModel）OpenAI 兼容风格客户端：
    - Chat:      POST {base_url}/chat/completions
    - Embedding: POST {base_url}/embeddings

    环境变量：
    - ZHIPU_API_KEY（必填）
    - ZHIPU_BASE_URL（默认 https://open.bigmodel.cn/api/paas/v4）
    - ZHIPU_CHAT_MODEL（默认 glm-4）
    - ZHIPU_EMBED_MODEL（默认 embedding-3）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 ZHIPU_API_KEY（可通过环境变量或构造参数传入）")

        self.base_url = self._normalize_base_url(base_url or os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4")
        self.chat_model = chat_model or os.getenv("ZHIPU_CHAT_MODEL") or "glm-4"
        self.embed_model = embed_model or os.getenv("ZHIPU_EMBED_MODEL") or "embedding-3"
        self.timeout_s = timeout_s

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        return (url or "").strip().rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise RuntimeError(f"智谱请求失败（url={url}）：{e}") from e

        text = r.text or ""
        if not r.ok:
            err_detail: Any
            try:
                err_detail = r.json()
            except ValueError:
                err_detail = (text[:800].replace("\r", " ").replace("\n", " ").strip()) or r.reason or "unknown error"
            raise RuntimeError(f"智谱 HTTP {r.status_code}（url={url}）：{err_detail}")

        try:
            return r.json()
        except ValueError as e:
            preview = text[:800].replace("\r", " ").replace("\n", " ").strip()
            raise RuntimeError(f"智谱返回非 JSON（url={url}）：{preview or 'empty body'}") from e

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post_json("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"智谱 chat 响应结构异常：{data}") from e

    def embeddings(self, inputs: List[str], model: Optional[str] = None) -> List[List[float]]:
        payload: Dict[str, Any] = {"model": model or self.embed_model, "input": inputs}
        data = self._post_json("/embeddings", payload)
        try:
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            raise RuntimeError(f"智谱 embedding 响应结构异常：{data}") from e

