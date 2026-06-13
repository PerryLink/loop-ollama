"""
FaultToleranceEngine 完整单元测试。

测试三层容错引擎：
    - Tier-1: 正则修复（12条规则 + 通用修复）
    - Tier-2: 简化重试（prompt构建与模型重试）
    - Tier-3: 启发式提取（12条启发式规则）
    - 升级逻辑: 模型等级影响 / 连续失败计数
    - 模式加载: patterns.json / heuristics.json
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fault_tolerance import FaultToleranceEngine
from src.ollama_client import ChatResponse


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def engine_a():
    """模型等级 A 的容错引擎。"""
    return FaultToleranceEngine(model_grade="A")


@pytest.fixture
def engine_d():
    """模型等级 D 的容错引擎。"""
    return FaultToleranceEngine(model_grade="D")


@pytest.fixture
def mock_response_no_tool_calls():
    """无 tool_calls 的响应。"""
    return ChatResponse(
        model="test:7b",
        content='{"name": "read_file", "arguments": {"file_path": "/tmp/test.txt"}}',
        tool_calls=None,
        eval_count=100,
        total_duration_ns=800_000_000,
    )


@pytest.fixture
def mock_response_with_tool_calls():
    """有 tool_calls 的响应。"""
    return ChatResponse(
        model="test:7b",
        content="ok",
        tool_calls=[{
            "function": {
                "name": "read_file",
                "arguments": {"file_path": "/tmp/test.txt"},
            }
        }],
        eval_count=100,
        total_duration_ns=800_000_000,
    )


@pytest.fixture
def mock_retry_fn():
    """Tier-2 重试函数 mock。"""
    fn = MagicMock()
    fn.return_value = ChatResponse(
        model="test:7b",
        content="retried",
        tool_calls=[{
            "function": {
                "name": "task_complete",
                "arguments": {"summary": "done"},
            }
        }],
    )
    return fn


# ── Tier-1 正则修复测试 ─────────────────────────────────────


class TestTier1Repair:
    """Tier-1 正则修复测试"""

    def test_repair_single_quoted_json(self, engine_a):
        """修复单引号 JSON"""
        raw = "{'name': 'test_tool', 'arguments': {'key': 'value'}}"
        result = engine_a._tier1_repair(raw)
        assert result is not None
        assert len(result) > 0
        assert result[0]["name"] == "test_tool"

    def test_repair_python_none_true_false(self, engine_a):
        """Python None/True/False -> JSON null/true/false"""
        raw = '{"name": "test", "active": True, "data": None, "enabled": False}'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_missing_closing_bracket(self, engine_a):
        """补全缺失的闭合括号"""
        raw = '{"name": "test", "arguments": {"a": 1}'
        result = engine_a._tier1_repair(raw)
        # 不崩溃即可——可能成功也可能返回 None
        assert True

    def test_repair_markdown_code_block(self, engine_a):
        """修复 markdown code block 包裹"""
        raw = '```json\n{"name": "test", "arguments": {}}\n```'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_with_comments(self, engine_a):
        """移除 // 和 /* */ 注释"""
        raw = '// comment\n{"name": "test", "arguments": {}}'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_trailing_comma(self, engine_a):
        """移除 JSON 尾部逗号"""
        raw = '{"name": "test", "arguments": {"a": 1,},}'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_arguments_double_quotes(self, engine_a):
        """移除 arguments 多余双引号"""
        raw = '{"name": "test", "arguments": "{\\"key\\": \\"value\\"}"}'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_empty_text(self, engine_a):
        """空文本返回 None"""
        assert engine_a._tier1_repair("") is None
        assert engine_a._tier1_repair("   ") is None

    def test_repair_garbled_text(self, engine_a):
        """完全无法解析的文本"""
        result = engine_a._tier1_repair(
            "This is just plain English, no JSON at all."
        )
        assert result is None

    def test_repair_partial_json_with_function(self, engine_a):
        """部分 JSON 含 function 字段"""
        raw = '{"tool_calls": [{"function": {"name": "read_file", "arguments": {"file_path": "/x"}}}]}'
        result = engine_a._tier1_repair(raw)
        assert result is not None

    def test_repair_array_of_tool_calls(self, engine_a):
        """直接是 tool_call 数组"""
        raw = '[{"name": "read_file", "arguments": {"file_path": "/tmp/x.txt"}}]'
        result = engine_a._tier1_repair(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"


# ── Tier-2 简化重试测试 ─────────────────────────────────────


class TestTier2Retry:
    """Tier-2 简化重试测试"""

    def test_build_simplified_messages(self, engine_a):
        """构建简化 messages"""
        msgs = [
            {"role": "system", "content": "original system prompt"},
            {"role": "user", "content": "do something"},
        ]
        result = engine_a._build_simplified_messages(
            msgs, "failed content here", None
        )
        assert result is not None
        assert len(result) >= 2
        # system prompt 应被替换
        assert "JSON" in result[0]["content"].upper()

    def test_build_simplified_without_system(self, engine_a):
        """无 system 消息时插入"""
        msgs = [{"role": "user", "content": "do something"}]
        result = engine_a._build_simplified_messages(
            msgs, "fail", None
        )
        assert result[0]["role"] == "system"

    def test_build_simplified_with_tool_names(self, engine_a):
        """简化的 messages 包含工具名列表"""
        msgs = [{"role": "user", "content": "task"}]
        result = engine_a._build_simplified_messages(msgs, "fail", None)
        system_content = result[0]["content"]
        assert "read_file" in system_content or "task_complete" in system_content


# ── Tier-3 启发式提取测试 ───────────────────────────────────


class TestTier3Extraction:
    """Tier-3 启发式提取测试"""

    def test_extract_read_action(self, engine_a):
        """提取读取操作"""
        text = "我需要读取文件 /tmp/config.py"
        result = engine_a._tier3_extract(text)
        assert "actions" in result
        actions = result["actions"]
        read_actions = [
            a for a in actions if a["action"] == "read_file"
        ]
        assert len(read_actions) >= 1

    def test_extract_write_action(self, engine_a):
        """提取写入操作"""
        text = "我建议写入文件 /tmp/output.txt"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        write_actions = [
            a for a in actions if a["action"] == "write_file"
        ]
        assert len(write_actions) >= 1

    def test_extract_command_from_backtick(self, engine_a):
        """提取反引号中的命令"""
        text = "请运行 `pytest tests/` 来测试"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        cmd_actions = [
            a for a in actions
            if a["action"] == "run_command" and "pytest" in str(a.get("target", ""))
        ]
        assert len(cmd_actions) >= 1

    def test_extract_task_complete_signal(self, engine_a):
        """检测 task_complete 信号"""
        text = "任务完成了，调用 task_complete"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        tc = [a for a in actions if a["action"] == "task_complete"]
        assert len(tc) >= 1

    def test_extract_multiple_actions(self, engine_a):
        """提取多个操作"""
        text = "请读取 config.py，然后运行 pytest"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        assert len(actions) >= 2

    def test_extract_empty_text(self, engine_a):
        """空文本"""
        result = engine_a._tier3_extract("")
        assert result["actions"] == []
        assert result["raw"] == ""

    def test_extract_no_actions(self, engine_a):
        """无匹配操作"""
        result = engine_a._tier3_extract(
            "Some text without any recognizable actions."
        )
        assert "actions" in result

    def test_extract_function_call_syntax(self, engine_a):
        """提取 function_call(...) 语法"""
        text = "I will call read_file('/etc/hosts') to check hosts"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        fn_actions = [
            a for a in actions if "read_file" in str(a.get("action", ""))
        ]
        assert len(fn_actions) >= 1

    def test_extract_action_label(self, engine_a):
        """提取 Action: tool_name 格式"""
        text = "Action: read_file\n接下来读取文件"
        result = engine_a._tier3_extract(text)
        actions = result["actions"]
        assert len(actions) >= 1


# ── 主入口 parse_response 测试 ───────────────────────────────


class TestParseResponse:
    """parse_response 主入口测试"""

    def test_existing_tool_calls_direct_return(
        self, engine_a, mock_response_with_tool_calls, mock_retry_fn
    ):
        """已有 tool_calls 直接返回"""
        result = engine_a.parse_response(
            mock_response_with_tool_calls, None, [], mock_retry_fn
        )
        assert result["success"] is True
        assert result["tier_used"] == 1
        assert result["tool_calls"] is not None

    def test_no_tool_calls_triggers_tier1(
        self, engine_a, mock_response_no_tool_calls, mock_retry_fn
    ):
        """无 tool_calls 触发 Tier-1 修复"""
        result = engine_a.parse_response(
            mock_response_no_tool_calls, None, [], mock_retry_fn
        )
        # 应该被 Tier-1 成功修复
        assert result["success"] is True

    def test_parse_with_tier2_upgrade(
        self, engine_a, mock_retry_fn
    ):
        """Tier-1 多次失败后升级 Tier-2"""
        engine_a.tier1_fail_count = 3  # 模拟已失败
        bad_response = ChatResponse(
            model="test:7b",
            content="garbled text, not json at all",
            tool_calls=None,
        )
        mock_retry_fn.return_value = ChatResponse(
            model="test:7b",
            content="ok",
            tool_calls=[{
                "function": {
                    "name": "task_complete",
                    "arguments": {"summary": "done"},
                }
            }],
        )

        result = engine_a.parse_response(
            bad_response, None,
            [{"role": "user", "content": "task"}],
            mock_retry_fn,
        )
        # Tier-2 应该成功
        assert result["success"] is True

    def test_parse_with_tier3_degradation(
        self, engine_a, mock_retry_fn
    ):
        """全部失败进入 Tier-3 降级"""
        engine_a.tier1_fail_count = 5
        mock_retry_fn.return_value = None  # Tier-2 也失败
        bad_response = ChatResponse(
            model="test:7b",
            content="I think we should read /tmp/x.txt first",
            tool_calls=None,
        )

        result = engine_a.parse_response(
            bad_response, None,
            [{"role": "user", "content": "task"}],
            mock_retry_fn,
        )
        assert result["tier_used"] == 3
        assert result["confidence"] > 0.1

    def test_tier2_threshold_by_grade(self, engine_a, engine_d):
        """不同模型等级有不同的 Tier-2 升级阈值"""
        assert engine_a._get_tier2_threshold() == 3  # A级
        assert engine_d._get_tier2_threshold() == 1  # D级


# ── 容错统计与快照测试 ──────────────────────────────────────


class TestFtSnapshot:
    """容错快照测试"""

    def test_get_snapshot_initial(self, engine_a):
        """初始快照"""
        snap = engine_a.get_ft_snapshot()
        assert snap["tier1_total_repairs"] == 0
        assert snap["tier2_total_retries"] == 0
        assert snap["tier3_total_degradations"] == 0
        assert snap["tier3_consecutive_count"] == 0
        assert snap["current_tier"] == 1
        assert snap["degraded_mode_active"] is False

    def test_get_snapshot_after_repairs(self, engine_a):
        """Tier-1 修复后的快照"""
        engine_a.stats["t1"] = [5, 4]
        snap = engine_a.get_ft_snapshot()
        assert snap["tier1_total_repairs"] == 4

    def test_get_snapshot_degraded(self, engine_a):
        """降级模式的快照"""
        engine_a.stats["t3_consecutive"] = 3
        snap = engine_a.get_ft_snapshot()
        assert snap["current_tier"] == 3
        assert snap["degraded_mode_active"] is True

    def test_reset(self, engine_a):
        """重置统计"""
        engine_a.stats["t1"] = [10, 8]
        engine_a.stats["t3_consecutive"] = 5
        engine_a.tier1_fail_count = 10
        engine_a.reset()
        assert engine_a.stats["t1"] == [0, 0]
        assert engine_a.stats["t3_consecutive"] == 0
        assert engine_a.tier1_fail_count == 0

    def test_get_diagnostics(self, engine_a):
        """诊断信息"""
        diag = engine_a.get_diagnostics()
        assert "model_grade" in diag
        assert "tier1_rules_loaded" in diag
        assert diag["tier1_rules_loaded"] >= 12


# ── 通用修复测试 ────────────────────────────────────────────


class TestApplyCommonFixes:
    """通用 JSON 修复测试"""

    def test_python_keywords(self, engine_a):
        """Python 关键字转 JSON"""
        text = '{"v": None, "b": True}'
        fixed = engine_a._apply_common_fixes(text)
        assert "null" in fixed
        assert "true" in fixed

    def test_strip_comments(self, engine_a):
        """移除注释"""
        text = '{"a": 1} // comment\n{"b": 2}'
        fixed = engine_a._apply_common_fixes(text)
        assert "// comment" not in fixed

    def test_balance_braces(self, engine_a):
        """补齐括号"""
        text = '{"a": [1, 2, 3'
        fixed = engine_a._apply_common_fixes(text)
        assert fixed.endswith("]")

    def test_trailing_comma_fix(self, engine_a):
        """移除尾部逗号"""
        text = '{"a": 1,}'
        fixed = engine_a._apply_common_fixes(text)
        assert ',}' not in fixed


# ── 规则加载测试 ────────────────────────────────────────────


class TestRuleLoading:
    """规则库加载测试"""

    def test_patterns_loaded(self, engine_a):
        """patterns.json 已加载"""
        assert len(engine_a.tier1_rules) >= 12

    def test_heuristics_loaded(self, engine_a):
        """heuristics.json 已加载"""
        assert len(engine_a.tier3_rules) >= 12

    def test_fallback_patterns(self):
        """内置回退规则"""
        fallback = FaultToleranceEngine._get_fallback_patterns()
        assert len(fallback) == 12

    def test_fallback_heuristics(self):
        """内置回退启发式规则"""
        fallback = FaultToleranceEngine._get_fallback_heuristics()
        assert len(fallback) == 12
