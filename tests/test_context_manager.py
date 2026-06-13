"""
context_manager 单元测试

测试消息历史的 token 估算、阈值检查和智能裁剪。
"""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.context_manager import ContextManager


class TestTokenEstimation:
    """Token 估算测试"""

    def test_empty_messages_returns_minimal_tokens(self):
        cm = ContextManager(max_context_tokens=4096)
        tokens = cm.estimate_tokens([])
        assert tokens >= 0
        assert tokens < 10

    def test_single_message_estimation(self):
        cm = ContextManager(max_context_tokens=4096)
        msgs = [{"role": "user", "content": "hello world"}]
        tokens = cm.estimate_tokens(msgs)
        # hello world = 11 chars / 3.5 ≈ 3 + 4 overhead = 7
        assert tokens > 0
        assert tokens < 20


class TestTrimLogic:
    """智能裁剪测试"""

    def test_should_not_trim_short_history(self):
        cm = ContextManager(max_context_tokens=4096)
        msgs = [{"role": "system", "content": "You are a helper."}]
        assert not cm.should_trim(msgs)

    def test_should_trim_large_history(self):
        cm = ContextManager(max_context_tokens=100)  # very small window
        # 创建超长消息强制触发裁剪
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
        ]
        assert cm.should_trim(msgs)

    def test_trim_preserves_system_message(self):
        cm = ContextManager(max_context_tokens=100)
        msgs = [
            {"role": "system", "content": "IMPORTANT SYSTEM PROMPT"},
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
            {"role": "user", "content": "z" * 500},
        ]
        result = cm.trim_messages(msgs)
        assert len(result) > 0
        assert result[0]["role"] == "system"
        assert "IMPORTANT" in result[0]["content"]

    def test_trim_maintains_recent_messages(self):
        cm = ContextManager(max_context_tokens=200)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old" * 100},
            {"role": "assistant", "content": "old_resp" * 50},
            {"role": "user", "content": "recent" * 5},
            {"role": "assistant", "content": "recent_resp" * 5},
        ]
        result = cm.trim_messages(msgs)
        # 最近的消息应该被保留
        assert any("recent" in str(m.get("content", "")) for m in result)


class TestSummarizeObservation:
    """Observation 摘要测试"""

    def test_short_observation_not_summarized(self):
        cm = ContextManager(max_context_tokens=4096)
        short = "short message"
        result = cm.summarize_observation(short)
        assert result == short

    def test_long_observation_summarized(self):
        cm = ContextManager(max_context_tokens=4096)
        long_msg = "x" * 500
        result = cm.summarize_observation(long_msg, max_chars=200)
        assert len(result) <= 203  # 200 + "[truncated]" overhead
        assert "truncated" in result
