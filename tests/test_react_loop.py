"""
ReAct Loop 单元测试。

测试 think/act/observe 循环的各个阶段，包括：
    - 主循环 run() 基本流程
    - should_terminate() 终止条件
    - 动态超时计算
    - 工具执行安全包装
    - 模型恢复检测
    - 容错集成
    - Guard 集成
    - 收敛控制集成
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.react_loop import ReactLoop
from src.ollama_client import (
    ChatResponse,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaAPIError,
)
from src.state_manager import StateManager


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def mock_ollama():
    """创建 mock OllamaClient。"""
    client = MagicMock()
    client.chat.return_value = ChatResponse(
        model="test-model:7b",
        content="test response",
        tool_calls=None,
        eval_count=100,
        eval_duration_ns=500_000_000,
        prompt_eval_count=200,
        total_duration_ns=800_000_000,
    )
    client.is_model_loaded.return_value = True
    client.ensure_model_loaded.return_value = True
    client.health_check.return_value = True
    return client


@pytest.fixture
def mock_state_manager(tmp_path):
    """创建实际可用的 StateManager。"""
    state_dir = str(tmp_path / "state")
    return StateManager(state_dir=state_dir)


@pytest.fixture
def mock_config():
    """创建 mock Config。"""
    cfg = MagicMock()
    cfg.default_model = "test-model:7b"
    cfg.max_turns = 5
    cfg.tier3_max_consecutive = 3
    cfg.auto_model_upgrade = False
    cfg.convergence_rounds = 2
    return cfg


@pytest.fixture
def mock_guard():
    """创建 mock GuardLayer。"""
    guard = MagicMock()
    guard.check.return_value = {
        "allowed": True, "blocked_by": None, "reason": None, "layer": None
    }
    return guard


@pytest.fixture
def mock_ft():
    """创建 mock FaultToleranceEngine —— 透传 response.tool_calls。"""
    ft = MagicMock()
    def parse_response_side_effect(response, tool_registry, messages, retry_fn):
        """从 response 中提取 tool_calls 并返回解析结果。"""
        tool_calls = getattr(response, "tool_calls", None)
        content = getattr(response, "content", "") or ""
        return {
            "success": bool(tool_calls),
            "tool_calls": tool_calls,
            "tier_used": 1,
            "content": content,
            "needs_retry": False,
            "degraded_text": None,
            "confidence": 1.0 if tool_calls else 0.5,
            "tier": 1,
        }
    ft.parse_response.side_effect = parse_response_side_effect
    ft.get_ft_snapshot.return_value = {
        "tier1_total_repairs": 0,
        "tier2_total_retries": 0,
        "tier3_total_degradations": 0,
        "tier3_consecutive_count": 0,
        "current_tier": 1,
        "degraded_mode_active": False,
        "degraded_since_turn": None,
    }
    return ft


@pytest.fixture
def mock_convergence():
    """创建 mock ConvergenceController —— 保持 state 完整性。"""
    cv = MagicMock()
    def after_action_side_effect(state, sub=False, ft=None):
        """在原 state 上更新 convergence 字段后返回。"""
        conv = state.setdefault("convergence", {})
        if sub:
            conv["convergence_counter"] = 0
        else:
            conv.setdefault("convergence_counter", 0)
            conv["convergence_counter"] += 1
        return state
    cv.after_action.side_effect = after_action_side_effect
    cv.route.return_value = ("continue", "default")
    cv.should_terminate.return_value = False
    cv.check_termination_conditions.return_value = None
    return cv


@pytest.fixture
def react_loop(
    mock_ollama, mock_state_manager, mock_config,
    mock_guard, mock_ft, mock_convergence,
):
    """创建完整的 ReactLoop 实例。"""
    return ReactLoop(
        ollama_client=mock_ollama,
        state_manager=mock_state_manager,
        config=mock_config,
        convergence_controller=mock_convergence,
        fault_tolerance=mock_ft,
        guard_layer=mock_guard,
    )


# ── 基本循环测试 ────────────────────────────────────────────


class TestReactLoopBasic:
    """基本 ReAct 循环测试"""

    def test_initialization(self, react_loop):
        """测试 ReactLoop 初始化"""
        assert react_loop is not None
        assert react_loop.ollama is not None
        assert react_loop.state_mgr is not None
        assert react_loop.config is not None
        assert react_loop.guard is not None
        assert react_loop.fault_tolerance is not None
        assert react_loop.convergence is not None

    def test_run_simple_task_completes(
        self, react_loop, mock_ollama
    ):
        """测试简单任务完整运行"""
        # 模拟 task_complete 响应
        mock_ollama.chat.return_value = ChatResponse(
            model="test-model:7b",
            content="task done",
            tool_calls=[
                {
                    "function": {
                        "name": "task_complete",
                        "arguments": {"summary": "完成"},
                    }
                }
            ],
            eval_count=50,
            total_duration_ns=500_000_000,
        )

        state = react_loop.run("hello world 任务", "test-model:7b")

        assert state is not None
        assert "session_id" in state
        assert state.get("task_complete") is True

    def test_run_with_file_write_then_complete(
        self, react_loop, mock_ollama
    ):
        """测试写文件后完成的流程"""
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ChatResponse(
                    model="test-model:7b",
                    content="writing file",
                    tool_calls=[{
                        "function": {
                            "name": "write_file",
                            "arguments": {"file_path": "/tmp/test.txt", "content": "hello"},
                        }
                    }],
                    eval_count=80,
                    total_duration_ns=600_000_000,
                )
            else:
                return ChatResponse(
                    model="test-model:7b",
                    content="done",
                    tool_calls=[{
                        "function": {
                            "name": "task_complete",
                            "arguments": {"summary": "完成"},
                        }
                    }],
                    eval_count=60,
                    total_duration_ns=400_000_000,
                )

        mock_ollama.chat.side_effect = side_effect

        state = react_loop.run("创建 /tmp/test.txt", "test-model:7b")

        assert state is not None
        assert call_count[0] >= 2
        assert state.get("task_complete") is True

    def test_run_max_turns_limit(
        self, react_loop, mock_config
    ):
        """测试达到最大轮次限制"""
        mock_config.max_turns = 1
        state = react_loop.run("长任务", "test-model:7b")
        assert state is not None
        term = state.get("termination", {})
        assert term.get("status") in ("limit_reached", None)


# ── 终止条件测试 ────────────────────────────────────────────


class TestShouldTerminate:
    """终止条件测试"""

    def test_terminate_on_task_complete(self, react_loop, mock_state_manager):
        """task_complete 触发终止"""
        state = mock_state_manager.create_new(
            task="test", model_name="test-model"
        )
        state["task_complete"] = True
        assert react_loop.should_terminate(state) is True

    def test_terminate_on_max_turns(self, react_loop, mock_state_manager):
        """达到最大轮次触发终止"""
        state = mock_state_manager.create_new(
            task="test", model_name="test-model"
        )
        state["housekeeping"]["turn_count"] = 100
        assert react_loop.should_terminate(state) is True

    def test_terminate_on_degraded_limit(self, react_loop, mock_state_manager):
        """Tier-3 降级上限触发终止"""
        state = mock_state_manager.create_new(
            task="test", model_name="test-model"
        )
        state["fault_tolerance"]["tier3_consecutive_count"] = 10
        assert react_loop.should_terminate(state) is True

    def test_no_terminate_normal(self, react_loop, mock_state_manager):
        """正常状态不应终止"""
        state = mock_state_manager.create_new(
            task="test", model_name="test-model"
        )
        assert react_loop.should_terminate(state) is False

    def test_terminate_on_consecutive_bash_errors(
        self, react_loop, mock_state_manager
    ):
        """连续 bash 错误触发终止"""
        state = mock_state_manager.create_new(
            task="test", model_name="test-model"
        )
        state["consecutive_bash_errors"] = 5
        assert react_loop.should_terminate(state) is True


# ── 动态超时测试 ────────────────────────────────────────────


class TestDynamicTimeout:
    """动态超时计算测试"""

    def test_base_timeout_no_history(self, react_loop):
        """无历史数据时使用基础超时"""
        timeout = react_loop._calculate_dynamic_timeout()
        assert timeout == 60000  # base

    def test_increased_timeout_with_slow_responses(self, react_loop):
        """慢响应增加超时"""
        react_loop._recent_durations = [30.0, 35.0, 32.0]
        timeout = react_loop._calculate_dynamic_timeout()
        assert timeout > 60000  # should increase (avg 32.3s * 3000 = 96900ms > 60000)

    def test_timeout_capped(self, react_loop):
        """超时有上限"""
        react_loop._recent_durations = [200.0] * 5
        timeout = react_loop._calculate_dynamic_timeout()
        assert timeout <= 300_000

    def test_avg_recent_duration(self, react_loop):
        """平均耗时计算"""
        react_loop._recent_durations = [1.0, 2.0, 3.0]
        avg = react_loop._avg_recent_duration()
        assert avg == 2.0

    def test_empty_durations_avg(self, react_loop):
        """空历史平均为 0"""
        react_loop._recent_durations = []
        assert react_loop._avg_recent_duration() == 0.0


# ── 异常处理测试 ────────────────────────────────────────────


class TestExceptionHandling:
    """异常处理测试"""

    def test_connection_error_recovery(
        self, react_loop, mock_ollama
    ):
        """连接错误后恢复"""
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OllamaConnectionError("connection lost")
            return ChatResponse(
                model="test-model:7b",
                content="done",
                tool_calls=[{
                    "function": {
                        "name": "task_complete",
                        "arguments": {"summary": "完成"},
                    }
                }],
                eval_count=50,
                total_duration_ns=400_000_000,
            )

        mock_ollama.chat.side_effect = side_effect
        state = react_loop.run("恢复测试", "test-model:7b")
        assert state is not None

    def test_api_error_handling(self, react_loop, mock_ollama):
        """API 错误处理"""
        mock_ollama.chat.side_effect = OllamaAPIError(500, "internal error")
        state = react_loop.run("错误测试", "test-model:7b")
        assert state is not None
        term = state.get("termination", {})
        assert term.get("status") in ("error", "limit_reached", None)


# ── Guard 集成测试 ──────────────────────────────────────────


class TestGuardIntegration:
    """Guard 安全护栏集成测试"""

    def test_guard_blocks_dangerous_tool(
        self, react_loop, mock_ollama, mock_guard
    ):
        """Guard 拦截危险工具调用"""
        mock_ollama.chat.return_value = ChatResponse(
            model="test-model:7b",
            content="dangerous",
            tool_calls=[{
                "function": {
                    "name": "run_command",
                    "arguments": {"command": "rm -rf /"},
                }
            }],
            eval_count=50,
            total_duration_ns=400_000_000,
        )
        mock_guard.check.return_value = {
            "allowed": False,
            "blocked_by": "L0",
            "reason": "灾难操作",
            "layer": "L0",
        }

        state = react_loop.run("危险任务", "test-model:7b")
        assert state is not None

    def test_guard_allows_safe_tool(
        self, react_loop, mock_ollama, mock_guard
    ):
        """Guard 允许安全工具调用"""
        mock_ollama.chat.return_value = ChatResponse(
            model="test-model:7b",
            content="safe",
            tool_calls=[{
                "function": {
                    "name": "read_file",
                    "arguments": {"file_path": "/tmp/test.txt"},
                }
            }],
            eval_count=50,
            total_duration_ns=400_000_000,
        )
        mock_guard.check.return_value = {
            "allowed": True,
            "blocked_by": None,
            "reason": None,
            "layer": None,
        }

        state = react_loop.run("安全任务", "test-model:7b")
        assert state is not None


# ── 容错集成测试 ────────────────────────────────────────────


class TestFaultToleranceIntegration:
    """容错集成测试"""

    def test_ft_parses_response(self, react_loop, mock_ollama, mock_ft):
        """容错解析响应"""
        mock_ft.parse_response.return_value = {
            "success": True,
            "tool_calls": [{
                "function": {
                    "name": "task_complete",
                    "arguments": {"summary": "ok"},
                }
            }],
            "tier_used": 1,
            "content": "test",
            "needs_retry": False,
            "degraded_text": None,
            "confidence": 1.0,
            "tier": 1,
        }

        state = react_loop.run("容错测试", "test-model:7b")
        assert state is not None
        assert mock_ft.parse_response.called

    def test_ft_degraded_response(self, react_loop, mock_ollama, mock_ft):
        """容错降级响应处理"""
        mock_ft.parse_response.return_value = {
            "success": False,
            "tool_calls": None,
            "tier_used": 3,
            "content": "garbled text",
            "needs_retry": True,
            "degraded_text": json.dumps({"actions": []}),
            "confidence": 0.3,
            "tier": 3,
        }
        mock_ft.get_ft_snapshot.return_value = {
            "tier1_total_repairs": 0,
            "tier2_total_retries": 0,
            "tier3_total_degradations": 1,
            "tier3_consecutive_count": 1,
            "current_tier": 3,
            "degraded_mode_active": False,
        }

        state = react_loop.run("降级测试", "test-model:7b")
        assert state is not None


# ── 收敛集成测试 ────────────────────────────────────────────


class TestConvergenceIntegration:
    """收敛控制集成测试"""

    def test_convergence_continue(self, react_loop, mock_ollama, mock_convergence):
        """收敛路由继续"""
        mock_convergence.route.return_value = ("continue", "default")
        mock_ollama.chat.side_effect = [
            ChatResponse(
                model="test-model:7b",
                content="thinking",
                tool_calls=None,
                eval_count=50,
                total_duration_ns=500_000_000,
            ),
            ChatResponse(
                model="test-model:7b",
                content="done",
                tool_calls=[{
                    "function": {
                        "name": "task_complete",
                        "arguments": {"summary": "完成"},
                    }
                }],
                eval_count=50,
                total_duration_ns=500_000_000,
            ),
        ]

        state = react_loop.run("收敛测试", "test-model:7b")
        assert state is not None

    def test_convergence_pause(self, react_loop, mock_ollama, mock_convergence, mock_config):
        """收敛路由暂停——pause 在收敛路由返回 pause 时触发"""
        mock_config.max_turns = 20
        mock_convergence.route.return_value = ("pause", "p0_issues")
        state = react_loop.run("暂停测试", "test-model:7b")
        assert state is not None
        term = state.get("termination", {})
        assert term.get("status") == "paused" or state.get("phase") in ("paused", "terminated")

    def test_convergence_prompt_injection(
        self, react_loop, mock_ollama, mock_convergence, mock_config
    ):
        """收敛提示注入——当 route 返回 convergence_prompt 时触发"""
        mock_config.max_turns = 10
        # route 在多次调用中依次返回
        mock_convergence.route.side_effect = [
            ("convergence_prompt", "test"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
            ("continue", "default"),
        ]

        call_count = [0]
        def chat_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                return ChatResponse(
                    model="test-model:7b",
                    content="ok",
                    tool_calls=[{
                        "function": {
                            "name": "task_complete",
                            "arguments": {"summary": "ok"},
                        }
                    }],
                )
            return ChatResponse(model="test-model:7b", content="thinking")

        mock_ollama.chat.side_effect = chat_side_effect

        state = react_loop.run("收敛提示测试", "test-model:7b")
        assert state is not None


# ── 模型恢复测试 ────────────────────────────────────────────


class TestModelRecovery:
    """模型恢复测试"""

    def test_model_already_loaded(self, react_loop, mock_ollama, mock_state_manager):
        """模型已加载"""
        state = mock_state_manager.create_new(task="t", model_name="m")
        react_loop._ensure_model_loaded(state, "test-model")
        mock_ollama.is_model_loaded.assert_called()

    def test_model_needs_reload(self, react_loop, mock_ollama, mock_state_manager):
        """模型需重新加载"""
        mock_ollama.is_model_loaded.return_value = False
        state = mock_state_manager.create_new(task="t", model_name="m")
        react_loop._ensure_model_loaded(state, "test-model")
        mock_ollama.ensure_model_loaded.assert_called_once()

    def test_model_reload_fails(self, react_loop, mock_ollama, mock_state_manager):
        """模型加载失败"""
        mock_ollama.is_model_loaded.return_value = False
        mock_ollama.ensure_model_loaded.return_value = False
        state = mock_state_manager.create_new(task="t", model_name="m")
        react_loop._ensure_model_loaded(state, "test-model")
        # 应有 P1 issue
        p1 = (
            state.get("issues", {})
            .get("active", {})
            .get("p1", [])
        )
        assert len(p1) >= 1
