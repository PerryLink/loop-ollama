"""
loop-ollama OllamaClient 单元测试。

测试 /api/chat, /api/show, /api/ps, /api/tags 端点、
连接错误处理、超时处理、keep_alive 参数、ChatResponse 构造。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ollama_client import (
    OllamaClient,
    ChatResponse,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaModelNotLoaded,
    OllamaAPIError,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:11434")


@pytest.fixture
def mock_requests():
    with patch("src.ollama_client.requests") as mock_req:
        yield mock_req


# ── Tests: ChatResponse ─────────────────────────────────────────


def test_chat_response_defaults():
    """ChatResponse 默认值。"""
    resp = ChatResponse()
    assert resp.model == ""
    assert resp.content == ""
    assert resp.tool_calls is None
    assert resp.eval_count == 0


def test_chat_response_full():
    """ChatResponse 完整构造。"""
    resp = ChatResponse(
        model="llama3",
        content="Hello",
        tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "/tmp"}}}],
        eval_count=100,
        eval_duration_ns=500_000_000,
        prompt_eval_count=300,
        total_duration_ns=900_000_000,
        raw={"key": "value"},
    )
    assert resp.model == "llama3"
    assert resp.content == "Hello"
    assert len(resp.tool_calls) == 1
    assert resp.eval_count == 100
    assert resp.raw == {"key": "value"}


# ── Tests: 初始化 ───────────────────────────────────────────────


def test_client_default_base_url():
    """默认 base_url 为 localhost:11434。"""
    c = OllamaClient()
    assert c.base_url == "http://localhost:11434"


def test_client_custom_base_url():
    """自定义 base_url。"""
    c = OllamaClient(base_url="http://192.168.1.100:11434")
    assert c.base_url == "http://192.168.1.100:11434"


def test_client_strips_trailing_slash():
    """去除末尾斜杠。"""
    c = OllamaClient(base_url="http://localhost:11434/")
    assert c.base_url == "http://localhost:11434"


def test_client_default_timeout():
    """默认超时为 60000ms。"""
    c = OllamaClient()
    assert c.default_timeout_ms == 60000


def test_client_custom_timeout():
    """自定义默认超时。"""
    c = OllamaClient(default_timeout_ms=120000)
    assert c.default_timeout_ms == 120000


# ── Tests: chat() 成功 ──────────────────────────────────────────


def test_chat_success_no_tools(client, mock_requests):
    """chat() 发送正确请求并解析响应。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Hello!"},
            "eval_count": 100,
            "eval_duration": 500000000,
            "prompt_eval_count": 200,
            "total_duration": 800000000,
        },
    )

    resp = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(resp, ChatResponse)
    assert resp.content == "Hello!"
    assert resp.model == "llama3"


def test_chat_with_tools(client, mock_requests):
    """chat() 携带 tools 参数。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "read_file", "arguments": {"path": "/tmp/test"}}
                }],
            },
            "eval_count": 80,
            "eval_duration": 400000000,
            "prompt_eval_count": 150,
            "total_duration": 600000000,
        },
    )

    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    resp = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Read test"}],
        tools=tools,
    )

    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "read_file"


def test_chat_with_options(client, mock_requests):
    """chat() 携带 options 参数。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3",
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 50,
            "eval_duration": 200000000,
            "prompt_eval_count": 100,
            "total_duration": 400000000,
        },
    )

    resp = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Test"}],
        options={"temperature": 0.7, "num_predict": 512},
    )
    assert resp.content == "ok"


def test_chat_keep_alive_default(client, mock_requests):
    """默认 keep_alive=-1（永驻）。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3",
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 30,
            "eval_duration": 100000000,
            "prompt_eval_count": 50,
            "total_duration": 200000000,
        },
    )

    client.chat(model="llama3", messages=[{"role": "user", "content": "Test"}])
    _, kwargs = mock_requests.post.call_args
    assert kwargs["json"]["keep_alive"] == -1


# ── Tests: chat() 错误处理 ──────────────────────────────────────


def test_chat_connection_error(client, mock_requests):
    """连接失败抛出 OllamaConnectionError。"""
    mock_requests.post.side_effect = requests.exceptions.ConnectionError("refused")
    with pytest.raises(OllamaConnectionError):
        client.chat(model="llama3", messages=[{"role": "user", "content": "Test"}])


def test_chat_timeout_error(client, mock_requests):
    """超时抛出 OllamaTimeoutError。"""
    mock_requests.post.side_effect = requests.exceptions.Timeout("timeout")
    with pytest.raises(OllamaTimeoutError):
        client.chat(model="llama3", messages=[{"role": "user", "content": "Test"}])


def test_chat_api_error(client, mock_requests):
    """API 返回非 200 抛出 OllamaAPIError。"""
    mock_requests.post.return_value = MagicMock(
        status_code=500,
        json=lambda: {"error": "server error"},
    )
    with pytest.raises(OllamaAPIError) as excinfo:
        client.chat(model="llama3", messages=[{"role": "user", "content": "Test"}])
    assert excinfo.value.status_code == 500


# ── Tests: health_check() ───────────────────────────────────────


def test_health_check_ok(client, mock_requests):
    """health_check() 正常时返回 True。"""
    mock_requests.get.return_value = MagicMock(status_code=200)
    assert client.health_check() is True


def test_health_check_fail(client, mock_requests):
    """health_check() 失败时返回 False。"""
    mock_requests.get.side_effect = requests.exceptions.ConnectionError("refused")
    assert client.health_check() is False


# ── Tests: is_model_loaded() ────────────────────────────────────


def test_is_model_loaded_true(client, mock_requests):
    """模型已加载时返回 True。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3"}]},
    )
    assert client.is_model_loaded("llama3") is True


def test_is_model_loaded_false(client, mock_requests):
    """模型未加载时返回 False。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": []},
    )
    assert client.is_model_loaded("llama3") is False


def test_is_model_loaded_api_error(client, mock_requests):
    """API 错误时返回 False。"""
    mock_requests.get.side_effect = requests.exceptions.ConnectionError("refused")
    assert client.is_model_loaded("llama3") is False


# ── Tests: list_running_models() ────────────────────────────────


def test_list_running_models(client, mock_requests):
    """list_running_models() 返回运行中模型列表。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3"}, {"name": "qwen2.5-coder:7b"}]},
    )
    models = client.list_running_models()
    assert len(models) == 2
    assert "llama3" in [m["name"] for m in models]


def test_list_running_models_empty(client, mock_requests):
    """无运行中模型时返回空列表。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200, json=lambda: {"models": []}
    )
    models = client.list_running_models()
    assert models == []


# ── Tests: show_model() ─────────────────────────────────────────


def test_show_model(client, mock_requests):
    """show_model() 返回模型详情。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "details": {"parameter_size": "7B", "quantization_level": "Q4_K_M"},
        },
    )
    info = client.show_model("qwen2.5-coder:7b")
    assert "details" in info
    assert info["details"]["parameter_size"] == "7B"


def test_show_model_not_found(client, mock_requests):
    """模型不存在时抛出错误。"""
    mock_requests.post.return_value = MagicMock(
        status_code=404, json=lambda: {"error": "model not found"}
    )
    with pytest.raises(OllamaAPIError):
        client.show_model("nonexistent")


# ── Tests: list_available_models() ────────────────────────────────────────


def test_list_available_models(client, mock_requests):
    """list_available_models() 返回已下载模型列表。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3:latest"}]},
    )
    models = client.list_available_models()
    assert len(models) > 0
    assert models[0]["name"] == "llama3:latest"


# ── Tests: ensure_model_loaded() ────────────────────────────────────


def test_ensure_model_already_loaded(client, mock_requests):
    """模型已加载时直接返回 True。"""
    mock_requests.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3"}]},
    )
    assert client.ensure_model_loaded("llama3") is True


# ── Tests: release_model() ──────────────────────────────────────────


def test_release_model_calls_chat_with_keep_alive_0(client, mock_requests):
    """release_model 以 keep_alive=0 释放模型。"""
    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3",
            "message": {"role": "assistant", "content": "bye"},
            "eval_count": 1,
            "eval_duration": 100000,
        },
    )
    client.release_model("llama3")
    _, kwargs = mock_requests.post.call_args
    assert kwargs["json"]["keep_alive"] == 0


# ── Tests: edge cases for list methods ──────────────────────────────


def test_list_running_models_non_200(client, mock_requests):
    """非 200 状态码时返回空列表。"""
    mock_requests.get.return_value = MagicMock(
        status_code=500, json=lambda: {"error": "server error"}
    )
    assert client.list_running_models() == []


def test_list_available_models_non_200(client, mock_requests):
    """/api/tags 非 200 返回空列表。"""
    mock_requests.get.return_value = MagicMock(
        status_code=500, json=lambda: {"error": "internal error"}
    )
    assert client.list_available_models() == []


# ── Tests: generic request exception ────────────────────────────────


def test_chat_generic_request_exception(client, mock_requests):
    """通用 requests 异常抛出 OllamaConnectionError。"""
    mock_requests.post.side_effect = requests.exceptions.RequestException("generic error")
    with pytest.raises(OllamaConnectionError):
        client.chat(model="llama3", messages=[{"role": "user", "content": "Test"}])


def test_show_model_connection_error(client, mock_requests):
    """show_model 连接失败抛出异常。"""
    mock_requests.post.side_effect = requests.exceptions.ConnectionError("refused")
    with pytest.raises(OllamaConnectionError):
        client.show_model("llama3")
