"""
loop-ollama 全局测试 Fixture 配置文件。

提供 Phase 0 所有 150 个测试所需的共享 mock 对象、fixture 数据
和辅助方法。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest


# ═══════════════════════════════════════════════════════════════════
# 路径 Fixture
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir() -> str:
    """创建临时目录，测试后自动清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def fixture_dir() -> str:
    """返回测试 fixture 数据目录。"""
    return str(
        Path(__file__).parent / "fixtures"
    )


# ═══════════════════════════════════════════════════════════════════
# 状态 Fixture
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_state_dict() -> dict[str, Any]:
    """完整的示例 state.json 字典。"""
    return {
        "session_id": "test_session_001",
        "version": "0.1.0",
        "created_at": "2026-06-13T00:00:00Z",
        "updated_at": "2026-06-13T00:00:00Z",
        "phase": "init",
        "task": "Write a Hello World Python script",
        "model": {
            "name": "qwen2.5-coder:7b",
            "grade": "A",
            "capability_score": 0.45,
            "upgrade_history": [],
            "upgrade_occurred_this_cycle": False,
        },
        "hardware": {
            "gpu_name": "NVIDIA RTX 3070",
            "vram_gb": 12.0,
            "ram_gb": 32.0,
            "cpu_cores": 8,
            "platform": "Linux",
        },
        "config": {},
        "convergence": {
            "convergence_counter": 0,
            "convergence_rounds_required": 2,
            "convergence_rounds_achieved": 0,
            "last_substantive_change_turn": 0,
            "convergence_reset_reason": None,
            "degraded_convergence_penalty": 0,
        },
        "fault_tolerance": {
            "tier1_total_repairs": 0,
            "tier2_total_retries": 0,
            "tier3_total_degradations": 0,
            "tier3_consecutive_count": 0,
            "current_tier": 1,
            "degraded_mode_active": False,
            "degraded_since_turn": None,
        },
        "housekeeping": {
            "turn_count": 0,
            "invocation_count": 0,
            "tokens_prompt_total": 0,
            "tokens_completion_total": 0,
            "total_duration_ms": 0,
        },
        "termination": {
            "status": None,
            "exit_reason": None,
            "summary": None,
            "verification_command": None,
        },
        "issues": {
            "active": {"p0": [], "p1": [], "p2": []},
            "resolved": [],
            "total_p0_triggered": 0,
            "total_p1_triggered": 0,
        },
        "artifacts": [],
        "modified_files_summary": [],
        "message_history_summary": [],
        "tool_stats": {},
        "_transient_is_substantive": False,
    }


# ═══════════════════════════════════════════════════════════════════
# 配置 Fixture
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def test_config(temp_dir: str) -> Any:
    """创建一个临时目录中的 Config 实例。"""
    from src.config import Config
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    return cfg


# ═══════════════════════════════════════════════════════════════════
# StateManager Fixture
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def test_state_manager(temp_dir: str) -> Any:
    """创建一个临时目录中的 StateManager 实例。"""
    from src.state_manager import StateManager
    state_dir = os.path.join(temp_dir, "state")
    return StateManager(state_dir=state_dir)


# ═══════════════════════════════════════════════════════════════════
# OllamaClient Mock Fixture
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_ollama_client(mocker: Any) -> Any:
    """创建一个 mock 的 OllamaClient 实例。"""
    from src.ollama_client import OllamaClient, ChatResponse

    client = OllamaClient(base_url="http://mock:11434")

    # Mock chat()
    def _mock_chat_response(tool_calls: Optional[list] = None):
        return ChatResponse(
            model="qwen2.5-coder:7b",
            content="",
            tool_calls=tool_calls,
            eval_count=120,
            eval_duration_ns=850000000,
            prompt_eval_count=450,
            total_duration_ns=1250000000,
        )

    mocker.patch.object(
        client, "chat", return_value=_mock_chat_response()
    )
    mocker.patch.object(client, "health_check", return_value=True)
    mocker.patch.object(client, "is_model_loaded", return_value=True)
    mocker.patch.object(client, "list_running_models", return_value=[])

    return client


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════


def load_fixture_json(fixture_dir: str, filename: str) -> dict[str, Any]:
    """从 fixtures 目录加载 JSON 文件。

    Args:
        fixture_dir: fixtures 根目录。
        filename: JSON 文件名（可含子目录）。

    Returns:
        解析后的 JSON 字典。
    """
    path = os.path.join(fixture_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_tool_call(
    name: str, arguments: dict[str, Any], call_id: str = "call_001"
) -> dict[str, Any]:
    """构造一个 Ollama 格式的 tool_call 对象。

    Args:
        name: 工具名称。
        arguments: 工具参数。
        call_id: 调用 ID。

    Returns:
        Ollama tool_calls 格式字典。
    """
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }
