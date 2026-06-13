"""
loop-ollama Ollama REST API 客户端。

封装 Ollama 的四个核心端点：/api/chat（唯一推理端点）、
/api/show（模型详情）、/api/ps（运行中模型）、/api/tags（已下载模型）。

关键设计决策：
    - 硬编码 /api/chat —— 绝不使用 /api/generate（不支持 tool_calls）。
    - keep_alive 为顶层请求参数（不在 options 内）。
    - stream=False（非流式响应，简化 Tier-1 正则修复）。

Classes:
    OllamaClient: Ollama API 客户端。
    ChatResponse: /api/chat 响应数据类。
    OllamaConnectionError: 连接错误。
    OllamaTimeoutError: 请求超时。
    OllamaModelNotLoaded: 模型未加载。
    OllamaAPIError: API 返回错误。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from requests.exceptions import (
    ConnectionError as _RequestsConnectionError,
    Timeout as _RequestsTimeout,
    RequestException as _RequestsException,
)


# ── 数据类 ─────────────────────────────────────────────────────────


@dataclass
class ChatResponse:
    """Ollama /api/chat 响应封装。

    Attributes:
        model: 模型名称。
        content: assistant 文本回复内容。
        tool_calls: tool_calls 数组（可能为 None）。
        eval_count: evaluation token 数。
        eval_duration_ns: evaluation 耗时（纳秒）。
        prompt_eval_count: prompt evaluation token 数。
        total_duration_ns: 总耗时（纳秒）。
        raw: 原始响应 JSON（调试用）。
    """

    model: str = ""
    content: str = ""
    tool_calls: Optional[list[dict[str, Any]]] = None
    eval_count: int = 0
    eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    total_duration_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


# ── 自定义异常 ────────────────────────────────────────────────────


class OllamaConnectionError(Exception):
    """无法连接到 Ollama 服务。"""


class OllamaTimeoutError(Exception):
    """Ollama 请求超时。"""


class OllamaModelNotLoaded(Exception):
    """目标模型未加载到内存。"""


class OllamaAPIError(Exception):
    """Ollama API 返回非 200 状态码。"""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"Ollama API 错误 [{status_code}]: {message}")


# ── 客户端 ─────────────────────────────────────────────────────────


class OllamaClient:
    """Ollama REST API 客户端。

    Attributes:
        base_url: Ollama 服务基础 URL。
        default_timeout_ms: 默认请求超时（毫秒）。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_timeout_ms: int = 60000,
    ) -> None:
        """初始化 OllamaClient。

        Args:
            base_url: Ollama 服务地址。
            default_timeout_ms: 默认请求超时（毫秒），用于无显式参数的请求。
        """
        self.base_url: str = base_url.rstrip("/")
        self.default_timeout_ms: int = default_timeout_ms

    # ── /api/chat —— 核心推理端点 ────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        keep_alive: int = -1,
        stream: bool = False,
        options: Optional[dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
    ) -> ChatResponse:
        """发送 /api/chat 推理请求。

        Args:
            model: 模型名称。
            messages: 消息历史列表。
            tools: 工具定义列表（Ollama tool_calls 格式）。
            keep_alive: 模型驻留策略。-1=永驻，0=立即释放。
            stream: 是否流式响应（loop-ollama 固定为 False）。
            options: 推理参数（temperature, top_p, num_predict 等）。
            timeout_ms: 请求超时（毫秒）。None 则使用默认值。

        Returns:
            ChatResponse 对象。

        Raises:
            OllamaConnectionError: 连接失败。
            OllamaTimeoutError: 请求超时。
            OllamaAPIError: API 返回错误。
        """
        url = f"{self.base_url}/api/chat"

        # 构造请求体 —— keep_alive 为顶层参数！
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        if tools is not None:
            body["tools"] = tools
        if options is not None:
            body["options"] = options
        else:
            body["options"] = {
                "temperature": 0.1,
                "num_predict": 2048,
            }

        to_ms = (
            timeout_ms
            if timeout_ms is not None
            else self.default_timeout_ms
        )
        timeout_sec = to_ms / 1000.0

        try:
            resp = requests.post(
                url,
                json=body,
                timeout=timeout_sec,
                headers={"Content-Type": "application/json"},
            )
        except _RequestsConnectionError as e:
            raise OllamaConnectionError(
                f"无法连接 Ollama ({self.base_url}): {e}"
            ) from e
        except _RequestsTimeout as e:
            raise OllamaTimeoutError(
                f"Ollama 请求超时 ({to_ms}ms): {e}"
            ) from e
        except _RequestsException as e:
            raise OllamaConnectionError(
                f"Ollama 请求异常: {e}"
            ) from e

        if resp.status_code != 200:
            raise OllamaAPIError(resp.status_code, resp.text[:500])

        data = resp.json()
        message = data.get("message", {})

        return ChatResponse(
            model=data.get("model", model),
            content=message.get("content", "") or "",
            tool_calls=message.get("tool_calls"),
            eval_count=data.get("eval_count", 0),
            eval_duration_ns=data.get("eval_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            total_duration_ns=data.get("total_duration", 0),
            raw=data,
        )

    # ── /api/show —— 模型信息 ─────────────────────────────────

    def show_model(self, model_name: str) -> dict[str, Any]:
        """获取模型详细信息。

        通过 POST /api/show 获取模型的参数量、量化级别、上下文长度。

        Args:
            model_name: 模型名称（如 "qwen2.5-coder:7b"）。

        Returns:
            原始响应字典，含 details 和 model_info。

        Raises:
            OllamaAPIError: API 返回错误。
            OllamaConnectionError: 连接失败。
        """
        url = f"{self.base_url}/api/show"
        try:
            resp = requests.post(
                url,
                json={"name": model_name},
                timeout=self.default_timeout_ms / 1000.0,
            )
            if resp.status_code != 200:
                raise OllamaAPIError(
                    resp.status_code, f"{model_name}: {resp.text[:200]}"
                )
            return resp.json()
        except _RequestsConnectionError as e:
            raise OllamaConnectionError(
                f"无法连接 Ollama: {e}"
            ) from e

    # ── /api/ps —— 运行中模型 ─────────────────────────────────

    def list_running_models(self) -> list[dict[str, Any]]:
        """获取当前已加载到内存的模型列表。

        Returns:
            运行中的模型列表（可能为空）。
        """
        url = f"{self.base_url}/api/ps"
        try:
            resp = requests.get(
                url,
                timeout=self.default_timeout_ms / 1000.0,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("models", [])
        except _RequestsException:
            return []

    # ── /api/tags —— 已下载模型 ───────────────────────────────

    def list_available_models(self) -> list[dict[str, Any]]:
        """获取已下载到本地的所有模型。

        Returns:
            已下载模型列表。
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(
                url,
                timeout=self.default_timeout_ms / 1000.0,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("models", [])
        except _RequestsException:
            return []

    # ── 模型生命周期 ──────────────────────────────────────────

    def is_model_loaded(self, model_name: str) -> bool:
        """检查指定模型是否已在内存中。

        Args:
            model_name: 模型名称。

        Returns:
            True 如果模型正在运行。
        """
        running = self.list_running_models()
        for m in running:
            if m.get("name") == model_name:
                return True
        return False

    def ensure_model_loaded(self, model_name: str) -> bool:
        """确保模型已加载到内存。

        如果模型未在运行，发送 warmup 请求触发加载。
        最多重试 3 次（Ollama 首次加载 7B 约需 3-8 秒）。

        Args:
            model_name: 模型名称。

        Returns:
            True 如果模型成功加载。
        """
        if self.is_model_loaded(model_name):
            return True

        for attempt in range(1, 4):
            try:
                self.chat(
                    model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    keep_alive=-1,
                    options={"num_predict": 1},
                    timeout_ms=30000,
                )
                if self.is_model_loaded(model_name):
                    return True
            except (OllamaTimeoutError, OllamaAPIError, OllamaConnectionError):
                time.sleep(min(attempt * 2, 10.0))

        return self.is_model_loaded(model_name)

    def release_model(self, model_name: str) -> None:
        """释放模型（设置 keep_alive=0）。

        Args:
            model_name: 要释放的模型名称。
        """
        try:
            self.chat(
                model_name,
                messages=[{"role": "user", "content": "bye"}],
                keep_alive=0,
                options={"num_predict": 1},
                timeout_ms=10000,
            )
        except Exception:
            pass  # 释放失败不抛异常

    # ── 健康检查 ─────────────────────────────────────────────

    def health_check(self) -> bool:
        """检测 Ollama 服务是否可达。

        Returns:
            True 如果服务正常运行。
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/ps",
                timeout=5.0,
            )
            return resp.status_code == 200
        except _RequestsException:
            return False
