"""LLM对话上下文管理器 - 管理消息历史的token窗口限制。

智能裁剪、关键消息保留、token 估算、多级摘要、紧急截断。
支持 S/A/B/C/D 五级模型的差异化策略。

Classes:
    ContextManager: 主管理器。
    TrimStats: 裁剪统计数据类。
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ===== 数据类 =====


@dataclass
class TrimStats:
    """单次裁剪操作的统计数据。

    Attributes:
        original_tokens: 裁剪前 token 数。
        final_tokens: 裁剪后 token 数。
        removed_messages: 移除的消息条数。
        summarized_messages: 摘要压缩的消息条数。
        truncated_observations: 截断的 observation 数。
        strategy_used: 使用的裁剪策略。
    """
    original_tokens: int = 0
    final_tokens: int = 0
    removed_messages: int = 0
    summarized_messages: int = 0
    truncated_observations: int = 0
    strategy_used: str = "none"


# ===== 配置常量 =====

# 不同等级模型的分层策略参数
_GRADE_PARAMS: dict[str, dict[str, Any]] = {
    "S": {  # 大模型(>=20B): 宽窗口，少裁剪
        "headroom_ratio": 0.85,
        "min_history_turns": 5,
        "max_history_turns": 20,
        "summarize_threshold_chars": 3000,
        "observation_truncate_chars": 800,
        "reserved_system_ratio": 0.05,
        "reserved_tools_ratio": 0.10,
        "emergency_truncate_ratio": 0.95,
    },
    "A": {  # >=7B 模型: 标准窗口
        "headroom_ratio": 0.80,
        "min_history_turns": 3,
        "max_history_turns": 12,
        "summarize_threshold_chars": 2000,
        "observation_truncate_chars": 500,
        "reserved_system_ratio": 0.05,
        "reserved_tools_ratio": 0.10,
        "emergency_truncate_ratio": 0.90,
    },
    "B": {  # 3-7B: 较窄窗口
        "headroom_ratio": 0.75,
        "min_history_turns": 2,
        "max_history_turns": 8,
        "summarize_threshold_chars": 1000,
        "observation_truncate_chars": 300,
        "reserved_system_ratio": 0.05,
        "reserved_tools_ratio": 0.08,
        "emergency_truncate_ratio": 0.88,
    },
    "C": {  # 1-3B: 窄窗口
        "headroom_ratio": 0.70,
        "min_history_turns": 1,
        "max_history_turns": 5,
        "summarize_threshold_chars": 500,
        "observation_truncate_chars": 150,
        "reserved_system_ratio": 0.05,
        "reserved_tools_ratio": 0.05,
        "emergency_truncate_ratio": 0.85,
    },
    "D": {  # <1B: 极窄窗口
        "headroom_ratio": 0.65,
        "min_history_turns": 1,
        "max_history_turns": 3,
        "summarize_threshold_chars": 300,
        "observation_truncate_chars": 100,
        "reserved_system_ratio": 0.05,
        "reserved_tools_ratio": 0.05,
        "emergency_truncate_ratio": 0.80,
    },
}


class ContextManager:
    """管理对话历史，避免超出模型的 context window 上限。

    核心策略（按优先级）：
        1. 始终保留 system 消息（messages[0]）
        2. 保留最近 N 个完整轮次（user+assistant+tool）
        3. 最少保留 min_history_turns 个历史轮次
        4. 仍超限: 摘要历史轮中的 tool 输出
        5. 绝不移除当前未完成轮次的消息
        6. 紧急截断: 单条消息超 limit 时截断内容

    消息结构约定: [sys, usr, ast, tool, usr, ast, tool, ...]
    最后一个 user 可能是当前未完成轮次的起点。
    """

    OVERHEAD = 4       # 每条消息角色标记开销 (tokens)
    CHAR_RATIO = 3.5   # 中英文混合文本每 token 字符数

    def __init__(
        self,
        max_context_tokens: int = 4096,
        model_grade: str = "A",
    ):
        """初始化上下文管理器。

        Args:
            max_context_tokens: 模型上下文窗口大小 (tokens)。
            model_grade: 模型等级 (S/A/B/C/D)，决定裁剪策略参数。
        """
        self.max_tokens = max_context_tokens
        self.model_grade = model_grade.upper() if model_grade else "A"
        self._params = _GRADE_PARAMS.get(
            self.model_grade, _GRADE_PARAMS["A"]
        )
        self.limit = int(max_context_tokens * self._params["headroom_ratio"])
        self.last_stats: Optional[TrimStats] = None

    # ===== Token 估算 =====

    def estimate_tokens(self, messages: list[dict]) -> int:
        """估算消息列表的 token 数。

        每字符按 CHAR_RATIO 个 token 估算，每条消息加 OVERHEAD。

        对于 tool 角色的大段输出，估算值会偏大——这正是我们想要的，
        因为大段输出通常是 token 消耗的主要来源。

        Args:
            messages: 消息列表。

        Returns:
            估算的 token 总数。
        """
        t = 0.0
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            t += len(content) / self.CHAR_RATIO + self.OVERHEAD
            # tool_calls 也消耗 token
            if "tool_calls" in m and m["tool_calls"]:
                tc_str = str(m["tool_calls"])
                t += len(tc_str) / self.CHAR_RATIO
        return int(t)

    def estimate_message_tokens(self, message: dict) -> int:
        """估算单条消息的 token 数。

        Args:
            message: 单条消息。

        Returns:
            估算 token 数。
        """
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        t = len(content) / self.CHAR_RATIO + self.OVERHEAD
        if "tool_calls" in message and message["tool_calls"]:
            t += len(str(message["tool_calls"])) / self.CHAR_RATIO
        return int(t)

    # ===== 窗口检查 =====

    def should_trim(self, messages: list[dict]) -> bool:
        """估算 token 数 >= headroom 窗口上限时返回 True。

        Args:
            messages: 消息列表。

        Returns:
            是否需要裁剪。
        """
        return self.estimate_tokens(messages) >= self.limit

    def get_context_usage_ratio(self, messages: list[dict]) -> float:
        """返回 token 用量比率 (0.0~1.0)。

        Args:
            messages: 消息列表。

        Returns:
            用量比率，1.0 表示已达上限。
        """
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.estimate_tokens(messages) / self.max_tokens)

    def get_budget_allocation(self, messages: list[dict]) -> dict[str, int]:
        """计算 token 预算分配。

        将 context window 按比例分配给:
            - system: 系统提示
            - history_turns: 历史对话
            - current_turn: 当前轮次
            - tools: 工具输出
            - free: 剩余可用

        Args:
            messages: 消息列表。

        Returns:
            各分区的预算 token 数。
        """
        total = self.max_tokens
        sys_ratio = self._params["reserved_system_ratio"]
        tools_ratio = self._params["reserved_tools_ratio"]

        # 实际消耗
        actual = self.estimate_tokens(messages)

        return {
            "total_budget": total,
            "system_reserved": int(total * sys_ratio),
            "tools_reserved": int(total * tools_ratio),
            "actual_used": actual,
            "free": max(0, total - actual),
            "headroom_limit": self.limit,
            "usage_ratio": round(actual / total, 3) if total > 0 else 0.0,
        }

    # ===== 消息分类 =====

    def _classify_messages(
        self, messages: list[dict]
    ) -> dict[str, Any]:
        """分类消息并识别轮次边界。

        返回:
            {
                "system": 系统消息列表,
                "turns": [[msgs_in_turn], ...],   # 每轮的消息组
                "current_turn_idx": int | None,    # 当前轮索引
                "turn_starts": [idx_in_body, ...], # 每轮在 body 中的起始索引
                "error_tool_indices": {idx, ...},  # 包含错误的 tool 消息索引
            }
        """
        if len(messages) <= 1:
            return {
                "system": messages[:1] if messages else [],
                "turns": [],
                "current_turn_idx": None,
                "turn_starts": [],
                "error_tool_indices": set(),
            }

        sys_part = messages[:1]
        body = messages[1:]

        # 找每个 user 消息的索引 (轮次起始)
        turn_starts = [
            i for i, m in enumerate(body)
            if m.get("role") == "user"
        ]

        # 分组为轮次
        turns = []
        for idx, start in enumerate(turn_starts):
            end = turn_starts[idx + 1] if idx + 1 < len(turn_starts) else len(body)
            turns.append(body[start:end])

        # 当前轮 = 最后一轮
        current_turn_idx = len(turns) - 1 if turns else None

        # 找出包含错误的 tool 消息
        error_indices: set[int] = set()
        for i, m in enumerate(messages):
            if m.get("role") == "tool":
                content = m.get("content", "")
                if self._is_error_output(content):
                    error_indices.add(i)

        return {
            "system": sys_part,
            "turns": turns,
            "current_turn_idx": current_turn_idx,
            "turn_starts": turn_starts,
            "error_tool_indices": error_indices,
        }

    def _is_error_output(self, content: str) -> bool:
        """判断 tool 输出是否为错误。

        Args:
            content: tool 输出文本。

        Returns:
            True 如果看起来像错误输出。
        """
        if not content:
            return False
        lower = content.lower()
        error_markers = [
            "error:", "traceback", "exception",
            "command not found", "permission denied",
            "no such file", "cannot find", "failed",
            "syntax error", "typeerror", "valueerror",
            "connection refused", "timeout",
        ]
        return any(marker in lower for marker in error_markers)

    # ===== 摘要生成 =====

    def summarize_observation(
        self, observation: str, max_chars: int = 200
    ) -> str:
        """智能截断长文本。

        策略:
            1. 短文本 (< max_chars): 直接返回。
            2. 结构化输出 (JSON/代码): 保留头部和尾部。
            3. 普通文本: 保留前 60% + 省略标记 + 后 20%。

        Args:
            observation: 原始文本。
            max_chars: 最大字符数。

        Returns:
            截断后的文本。
        """
        s = observation if isinstance(observation, str) else str(observation)
        if len(s) <= max_chars:
            return s

        # 结构化输出: 保留首尾
        if s.strip().startswith("{") or s.strip().startswith("["):
            head_size = int(max_chars * 0.7)
            tail_size = max_chars - head_size - 20
            return (
                s[:head_size]
                + "\n...[JSON truncated]...\n"
                + s[-tail_size:]
            )

        # 普通文本: 保留前 60% + 后 20%
        head_size = int(max_chars * 0.6)
        tail_size = int(max_chars * 0.2)
        summary_mid = f"...[truncated {len(s) - head_size - tail_size} chars]..."
        return s[:head_size] + summary_mid + s[-tail_size:]

    def _summarize_turn(
        self, turn_messages: list[dict], max_chars_per_msg: int
    ) -> list[dict]:
        """摘要压缩一轮对话。

        保留 user/assistant 原文，压缩 tool 输出。

        Args:
            turn_messages: 一轮中的所有消息。
            max_chars_per_msg: 每条 tool 消息的最大字符。

        Returns:
            压缩后的消息列表。
        """
        result = []
        for m in turn_messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if content:
                    summary = self.summarize_observation(
                        str(content), max_chars_per_msg
                    )
                    result.append({**m, "content": summary})
                else:
                    result.append(m)
            else:
                result.append(m)
        return result

    # ===== 主裁剪逻辑 =====

    def trim_messages(self, messages: list[dict]) -> list[dict]:
        """裁剪消息历史适配上下文窗口。

        策略（逐级降级）:
            1. 始终保留 system 消息 (messages[0])。
            2. 保留最近 N 个完整轮次 (user+assistant+tool)。
            3. 至少保留 min_history_turns 个历史轮次。
            4. 仍超限: 摘要历史轮中的 tool 输出。
            5. 绝不移除当前未完成轮次的消息。
            6. 紧急截断: 单条超大 tool 输出强行截断。

        消息结构: [sys, usr, ast, tool, usr, ast, tool, ...]
        最后一个 user 可能是当前未完成轮次的起点。

        Args:
            messages: 完整的消息列表。

        Returns:
            裁剪后的消息列表。
        """
        stats = TrimStats(
            original_tokens=self.estimate_tokens(messages),
        )

        if len(messages) <= 1:
            stats.strategy_used = "noop_too_short"
            self.last_stats = stats
            return messages[:]

        if not self.should_trim(messages):
            stats.final_tokens = stats.original_tokens
            stats.strategy_used = "noop_within_limit"
            self.last_stats = stats
            return messages[:]

        classification = self._classify_messages(messages)
        turns = classification["turns"]
        sys_part = classification["system"]
        body = messages[1:]
        turn_starts = classification["turn_starts"]
        current_turn_idx = classification["current_turn_idx"]
        error_indices = classification["error_tool_indices"]

        if not turns:
            return messages[:]

        min_keep = min(
            self._params["min_history_turns"],
            len(turns),
        )
        max_keep = self._params["max_history_turns"]
        summarise_threshold = self._params["summarize_threshold_chars"]

        # === 策略 1: 从最新轮次往回保留 ===
        best_keep = len(turns)
        for k in range(min(len(turns), max_keep), min_keep - 1, -1):
            # 保留最后 k 轮
            start_turn_idx = len(turns) - k
            if start_turn_idx < 0:
                start_turn_idx = 0
            # 找到该轮在 body 中的起始位置
            body_start = turn_starts[start_turn_idx] if start_turn_idx < len(turn_starts) else 0
            candidate = sys_part + body[body_start:]
            if self.estimate_tokens(candidate) <= self.limit:
                best_keep = k
                break
        else:
            best_keep = max(min_keep, 1)

        # 构建裁剪后的结果
        if best_keep >= len(turns):
            result = messages[:]
        else:
            start_turn_idx = len(turns) - best_keep
            body_start = (
                turn_starts[start_turn_idx]
                if start_turn_idx < len(turn_starts)
                else 0
            )
            removed_count = body_start
            result = sys_part + body[body_start:]
            stats.removed_messages = removed_count
            stats.strategy_used = f"trim_turns_kept_{best_keep}"

        # === 策略 2: 仍超限，摘要 tool 输出 ===
        if self.estimate_tokens(result) > self.limit:
            # 当前轮的消息（不可移除，但可截断）
            if current_turn_idx is not None and current_turn_idx < len(turns):
                cur_turn_start_in_body = turn_starts[current_turn_idx]
                cur_ids = {
                    id(m) for m in body[cur_turn_start_in_body:]
                }
            else:
                cur_ids = set()

            for i, m in enumerate(result):
                if m.get("role") == "tool":
                    content = m.get("content", "")
                    if not content:
                        continue
                    if self._params["summarize_threshold_chars"] < len(str(content)):
                        stats.truncated_observations += 1
                    # 当前轮的 tool + 包含错误的 tool: 轻度截断
                    if id(m) in cur_ids or i in error_indices:
                        new_content = self.summarize_observation(
                            str(content), summarise_threshold * 2
                        )
                    else:
                        # 历史轮的普通 tool: 重度截断
                        new_content = self.summarize_observation(
                            str(content), summarise_threshold
                        )
                    result[i] = {**result[i], "content": new_content}
                    stats.summarized_messages += 1

            stats.strategy_used = (
                stats.strategy_used + "+summarized"
                if stats.strategy_used
                else "summarized"
            )

        # === 策略 3: 紧急截断单条超大消息 ===
        if self.estimate_tokens(result) > self.limit:
            for i, m in enumerate(result):
                if m.get("role") == "tool":
                    content = m.get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    if len(content) > self._params["summarize_threshold_chars"]:
                        result[i] = {
                            **result[i],
                            "content": self.summarize_observation(
                                content,
                                self._params["observation_truncate_chars"],
                            ),
                        }
                        stats.truncated_observations += 1
            stats.strategy_used = (
                stats.strategy_used + "+emergency"
                if stats.strategy_used
                else "emergency"
            )

        stats.final_tokens = self.estimate_tokens(result)
        self.last_stats = stats
        return result

    # ===== 预裁剪检查 =====

    def preflight_check(self, messages: list[dict]) -> dict[str, Any]:
        """在发送请求前评估上下文状态。

        返回包含预算、警告和建议的完整报告。

        Args:
            messages: 待发送的消息列表。

        Returns:
            评估报告字典。
        """
        tokens = self.estimate_tokens(messages)
        ratio = tokens / max(self.max_tokens, 1)
        budget = self.get_budget_allocation(messages)

        warnings = []
        suggestions = []

        if ratio >= 0.95:
            warnings.append("critical: context nearly exhausted")
            suggestions.append(
                "考虑减少本轮工具调用次数，或手动清理历史"
            )
        elif ratio >= 0.85:
            warnings.append("warning: context usage high")
            suggestions.append("下一轮可能触发裁剪")
        elif ratio >= self._params["headroom_ratio"] - 0.05:
            warnings.append("info: approaching headroom limit")

        # 检查是否有超大单条消息
        max_single = max(
            (self.estimate_message_tokens(m) for m in messages),
            default=0,
        )
        if max_single > self.limit * 0.5:
            warnings.append("warning: single message exceeds 50% of limit")
            suggestions.append("该消息将被紧急截断")

        return {
            "estimated_tokens": tokens,
            "max_tokens": self.max_tokens,
            "limit_tokens": self.limit,
            "usage_ratio": round(ratio, 3),
            "budget": budget,
            "warnings": warnings,
            "suggestions": suggestions,
            "model_grade": self.model_grade,
            "message_count": len(messages),
            "needs_trim": ratio >= self._params["headroom_ratio"],
        }

    # ===== 统计 =====

    def get_last_stats(self) -> Optional[TrimStats]:
        """返回最近一次裁剪的统计数据。

        Returns:
            TrimStats 或 None（若从未裁剪）。
        """
        return self.last_stats

    def reset_stats(self) -> None:
        """重置裁剪统计。"""
        self.last_stats = None
