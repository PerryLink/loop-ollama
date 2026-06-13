"""
三层容错引擎 —— Tier-1 格式修复 → Tier-2 简化重试 → Tier-3 启发式提取。

当 Ollama 模型（尤其低参数量级模型）输出的 JSON tool_calls 格式不正确时，
本模块逐层尝试修复，确保 Agent 循环不会因单次解析失败而中断。

层级结构：
    Tier-1 (格式修复):
        从 regex_lib/patterns.json 加载 12 条正则规则，逐条应用到原始文本，
        尝试修复常见 JSON 格式错误（单引号、未闭合括号、markdown 包裹等）。
        连续 3 次失败后升级到 Tier-2。

    Tier-2 (简化重试):
        构建简化 prompt（移除 tool_calls 格式要求，只请求纯 JSON 文本），
        以更低的 temperature (0.0) 重新调用模型。最多重试 tier2_max_retries 次。
        仍然失败则升级到 Tier-3。

    Tier-3 (启发式提取):
        从 regex_lib/heuristics.json 加载 12 条启发式规则，
        从纯文本响应中识别操作意图（read/write/execute 等）。
        结果置信度较低（0.2-0.5），仅作为最后的降级兜底。

降级升级逻辑：
    - Tier-1 fail_count >= 3 → 升级到 Tier-2
    - Tier-2 所有重试失败   → 升级到 Tier-3
    - Tier-3 连续 >= tier3_max_consecutive 次 → 请求终止循环

Classes:
    FaultToleranceEngine: 三层容错引擎。
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .logger import Logger

# ── 正则库路径 ──────────────────────────────────────────────────

_REGEX_LIB_DIR: Path = Path(__file__).parent / "regex_lib"
_PATTERNS_FILE: str = str(_REGEX_LIB_DIR / "patterns.json")
_HEURISTICS_FILE: str = str(_REGEX_LIB_DIR / "heuristics.json")


class FaultToleranceEngine:
    """三层容错引擎。

    负责解析可能格式不正确的 Ollama 响应，逐层升级修复策略。

    Attributes:
        model_grade: 模型等级 (S/A/B/C/D)，影响容错策略激进程度。
        tier2_max_retries: Tier-2 最大重试次数。
        tier3_max_consecutive: Tier-3 连续降级上限。
        log: Logger 实例。
        stats: 各层级修复/失败统计。
        tier1_rules: 从 patterns.json 加载的 12 条 Tier-1 规则。
        tier3_rules: 从 heuristics.json 加载的 12 条 Tier-3 规则。
        tier1_fail_count: Tier-1 连续失败计数（用于升级判断）。
    """

    def __init__(
        self,
        model_grade: str = "A",
        tier2_max_retries: int = 3,
        tier3_max_consecutive: int = 5,
    ) -> None:
        """初始化 FaultToleranceEngine。

        Args:
            model_grade: 模型等级，决定容错策略激进程度。
            tier2_max_retries: Tier-2 简化重试的最大尝试次数。
            tier3_max_consecutive: Tier-3 连续降级上限，超出则建议终止。
        """
        self.model_grade: str = model_grade.upper()
        self.tier2_max_retries: int = tier2_max_retries
        self.tier3_max_consecutive: int = tier3_max_consecutive
        self.log: Logger = Logger()

        # 统计：每个 tier 的 [attempts, successes]
        self.stats: dict[str, list[int]] = {
            "t1": [0, 0],   # [尝试次数, 成功次数]
            "t2": [0, 0],
            "t3": [0, 0],
            "t3_consecutive": 0,  # Tier-3 连续次数
        }
        self.tier1_fail_count: int = 0  # Tier-1 连续失败计数

        # 加载规则库
        self.tier1_rules: list[dict[str, str]] = self._load_patterns()
        self.tier3_rules: list[dict[str, str]] = self._load_heuristics()

        # 已注册的 Tier-1 修复规则名
        self._tier1_method_map: dict[str, str] = {
            "R01": "_修复未闭合tool_call",
            "R02": "_移除arguments多余双引号",
            "R03": "_修复单引号JSON",
            "R04": "_补全外层方括号",
            "R05": "_移除markdown代码块",
            "R06": "_引号转义规范化",
            "R07": "_补全缺失右花括号",
            "R08": "_移除function嵌套",
            "R09": "_修复tool_call间逗号",
            "R10": "_arguments_null转空对象",
            "R11": "_移除BOM零宽字符",
            "R12": "_修复content嵌套",
        }

    # ── 规则加载 ────────────────────────────────────────────────

    def _load_patterns(self) -> list[dict[str, str]]:
        """从 regex_lib/patterns.json 加载 Tier-1 修复规则。

        Returns:
            规则列表，每项含 id/pattern/replacement/description。
            加载失败返回空列表。
        """
        patterns_path = _PATTERNS_FILE
        try:
            if not os.path.exists(patterns_path):
                self.log.warn(f"patterns.json 不存在: {patterns_path}")
                return self._get_fallback_patterns()
            with open(patterns_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", [])
            if not rules:
                self.log.warn("patterns.json 中无规则，使用内置回退")
                return self._get_fallback_patterns()
            # 将 $1/$2 转换为 Python re.sub 的 \1/\2 语法
            rules = self._normalize_backreferences(rules)
            self.log.debug(f"已加载 {len(rules)} 条 Tier-1 正则规则")
            return rules
        except (json.JSONDecodeError, OSError) as e:
            self.log.error(f"加载 patterns.json 失败: {e}，使用内置回退")
            return self._get_fallback_patterns()

    def _load_heuristics(self) -> list[dict[str, str]]:
        """从 regex_lib/heuristics.json 加载 Tier-3 启发式规则。

        Returns:
            规则列表，每项含 id/pattern/description/extract。
            加载失败返回空列表。
        """
        heuristics_path = _HEURISTICS_FILE
        try:
            if not os.path.exists(heuristics_path):
                self.log.warn(f"heuristics.json 不存在: {heuristics_path}")
                return self._get_fallback_heuristics()
            with open(heuristics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", [])
            if not rules:
                return self._get_fallback_heuristics()
            self.log.debug(f"已加载 {len(rules)} 条 Tier-3 启发式规则")
            return rules
        except (json.JSONDecodeError, OSError) as e:
            self.log.error(f"加载 heuristics.json 失败: {e}，使用内置回退")
            return self._get_fallback_heuristics()

    @staticmethod
    def _get_fallback_patterns() -> list[dict[str, str]]:
        """内置回退 Tier-1 规则（当 patterns.json 不可用时）。"""
        return [
            {"id": "F01", "pattern": r"```(?:json)?\s*\n?(.*?)\n?```", "replacement": r"\1",
             "description": "移除 markdown code block 包裹"},
            {"id": "F02", "pattern": "'(name|arguments|content|role)':\\s*'([^']*)'", "replacement": r'"\1": "\2"',
             "description": "修复单引号 JSON"},
            {"id": "F03", "pattern": r"\bNone\b", "replacement": "null",
             "description": "Python None → JSON null"},
            {"id": "F04", "pattern": r"\bTrue\b", "replacement": "true",
             "description": "Python True → JSON true"},
            {"id": "F05", "pattern": r"\bFalse\b", "replacement": "false",
             "description": "Python False → JSON false"},
            {"id": "F06", "pattern": r',\s*([\]\}])', "replacement": r"\1",
             "description": "移除尾部多余逗号"},
            {"id": "F07", "pattern": r'"arguments":\s*"(\{[^\"]*\})"', "replacement": r'"arguments": \1',
             "description": "移除 arguments 多余双引号"},
            {"id": "F08", "pattern": r'"arguments":\s*null', "replacement": r'"arguments": {}',
             "description": "arguments null → {}"},
            {"id": "F09", "pattern": r"^\\s*\\{\"name\":", "replacement": r'[{"name":',
             "description": "补全 tool_calls 外层 []"},
            {"id": "F10", "pattern": r"//[^\n]*", "replacement": "",
             "description": "移除 // 注释"},
            {"id": "F11", "pattern": r"/\\*.*?\\*/", "replacement": "",
             "description": "移除 /* */ 注释"},
            {"id": "F12", "pattern": r"[﻿​‌‍‎‏]", "replacement": "",
             "description": "移除 BOM 和零宽字符"},
        ]

    @staticmethod
    def _get_fallback_heuristics() -> list[dict[str, str]]:
        """内置回退 Tier-3 规则（当 heuristics.json 不可用时）。"""
        return [
            {"id": "FH01", "pattern": r"(\w+)\(([^)]*)\)", "extract": "tool_call_with_args",
             "description": "提取 function_call(...) 语法"},
            {"id": "FH02", "pattern": r'(?:Action|调用|执行)[：:]\s*(\w+)', "extract": "action_label",
             "description": "提取 Action: tool_name 格式"},
            {"id": "FH03", "pattern": r'"function"\s*:\s*"(\w+)"', "extract": "partial_json_fn",
             "description": "从不完整 JSON 提取 function"},
            {"id": "FH04", "pattern": r'(?:read|读取)\s+"?([^\s\"\'\\n,，]+)"?', "extract": "read_file_path",
             "description": "提取 read_file 文件名"},
            {"id": "FH05", "pattern": r'(?:write|写入|创建)\s+"?([^\s\"\'\\n,，]+)"?', "extract": "write_file_path",
             "description": "提取 write_file 文件名"},
            {"id": "FH06", "pattern": r"`([^`]+)`", "extract": "backtick_command",
             "description": "提取反引号中的命令"},
            {"id": "FH07", "pattern": r"(?:task_complete|任务完成|完成|done|DONE)", "extract": "completion_signal",
             "description": "检测完成声明"},
            {"id": "FH08", "pattern": r'(?:Thought|思考|分析)[：:]\s*(.+?)(?:\n|$)', "extract": "thought_text",
             "description": "提取思考内容"},
            {"id": "FH09", "pattern": r'(?:replace|替换|old_string)\s+"([^"]+)"', "extract": "edit_old_string",
             "description": "提取编辑原字符串"},
            {"id": "FH10", "pattern": r'(?:glob|搜索文件)\s+"?([^\s\"\'\\n]+)"?', "extract": "glob_pattern",
             "description": "提取 Glob 模式"},
            {"id": "FH11", "pattern": r'(?:grep|搜索内容)\s+"?([^\s\"\'\\n]+)"?', "extract": "grep_pattern",
             "description": "提取 Grep 正则"},
            {"id": "FH12", "pattern": r"^([A-Z][a-z_]+)", "extract": "first_capitalized",
             "description": "首行大写词作为工具名"},
        ]

    @staticmethod
    def _normalize_backreferences(
        rules: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """将规则 replacement 中的 $N 语法转换为 Python re.sub 的 \\N 语法。

        patterns.json 使用 JavaScript 风格的 $1/$2 反向引用，
        但 Python 的 re.sub() 需要 \\1/\\2 格式。

        Args:
            rules: 规则列表。

        Returns:
            转换后的规则列表。
        """
        import re as _re_mod
        for rule in rules:
            repl = rule.get("replacement", "")
            if "$" in repl:
                # $1 → \\g<1>, $2 → \\g<2>
                fixed = _re_mod.sub(
                    r'\$(\d+)', r'\\\1', repl
                )
                rule["replacement"] = fixed
        return rules

    # ── 快照 ────────────────────────────────────────────────────

    def get_ft_snapshot(self) -> dict[str, Any]:
        """获取当前容错状态的快照字典。

        供 StateManager.update_fault_tolerance() 使用，
        将运行时的容错状态同步到持久化的 state 字典中。

        Returns:
            容错状态快照，包含各 tier 统计和当前层级。
        """
        current_tier = 1
        if self.stats["t3_consecutive"] > 0:
            current_tier = 3
        elif self.stats["t2"][0] > 0:
            current_tier = 2

        return {
            "tier1_total_repairs": self.stats["t1"][1],
            "tier2_total_retries": self.stats["t2"][1],
            "tier3_total_degradations": self.stats["t3"][0],
            "tier3_consecutive_count": self.stats["t3_consecutive"],
            "current_tier": current_tier,
            "degraded_mode_active": self.stats["t3_consecutive"] >= 2,
            "degraded_since_turn": None,
        }

    # ── 主入口：解析响应 ────────────────────────────────────────

    def parse_response(
        self,
        response: Any,
        tool_registry: Any,
        messages: list[dict[str, Any]],
        retry_fn: Callable[[list[dict[str, Any]]], Any],
    ) -> dict[str, Any]:
        """主入口——逐层解析模型响应。

        层级决策逻辑：
            1. 如果 response 已有有效 tool_calls → 直接返回 (Tier-1 成功)
            2. 调用 _tier1_repair() 尝试正则修复
            3. 若 Tier-1 连续失败 >= 3 → 调用 _tier2_retry() 简化重试
            4. 若 Tier-2 也失败 → 调用 _tier3_extract() 启发式提取
            5. 根据模型等级调整 Tier-2 升级阈值（低等级模型更早升级）

        Args:
            response: ChatResponse 对象或兼容对象（含 content 和 tool_calls）。
            tool_registry: 工具注册表（用于 Tier-2 注入有效工具名）。
            messages: 消息历史（用于 Tier-2 构建简化 prompt）。
            retry_fn: Tier-2 重试函数，接收 messages 返回 ChatResponse。

        Returns:
            解析结果字典：
                {
                    "success": bool,        # 是否成功解析
                    "tool_calls": list|None, # 解析出的 tool_calls
                    "tier_used": int,        # 使用的容错层级 (1/2/3)
                    "content": str,          # 原始响应文本
                    "needs_retry": bool,     # 是否需要下一轮重试
                    "degraded_text": str|None, # Tier-3 降级文本
                    "confidence": float,     # 置信度 (0.0-1.0)
                    "tier": int,             # 同 tier_used
                }
        """
        raw_content: str = getattr(response, "content", "") or ""

        # 情况 1: 响应已包含有效 tool_calls
        existing_tool_calls = getattr(response, "tool_calls", None)
        if existing_tool_calls:
            self.stats["t1"][0] += 1
            self.stats["t1"][1] += 1
            self.stats["t3_consecutive"] = 0
            self.tier1_fail_count = 0
            self.log.debug("Tier-1: 响应已含有效 tool_calls，直接返回")
            return {
                "success": True,
                "tool_calls": existing_tool_calls,
                "tier_used": 1,
                "content": raw_content,
                "needs_retry": False,
                "degraded_text": None,
                "confidence": 1.0,
                "tier": 1,
            }

        # ── Tier-1: 正则修复 ──
        repaired = self._tier1_repair(raw_content)
        if repaired is not None:
            self.stats["t1"][0] += 1
            self.stats["t1"][1] += 1
            self.stats["t3_consecutive"] = 0
            self.tier1_fail_count = 0
            self.log.info(
                f"Tier-1 修复成功，t1_success={self.stats['t1'][1]}"
            )
            return {
                "success": True,
                "tool_calls": repaired,
                "tier_used": 1,
                "content": raw_content,
                "needs_retry": False,
                "degraded_text": None,
                "confidence": 0.9,
                "tier": 1,
            }

        # Tier-1 失败
        self.stats["t1"][0] += 1
        self.tier1_fail_count += 1
        self.log.warn(
            f"Tier-1 修复失败 (连续 {self.tier1_fail_count} 次)"
        )

        # 根据模型等级决定 Tier-2 升级阈值
        tier2_threshold: int = self._get_tier2_threshold()
        if self.tier1_fail_count < tier2_threshold:
            # 尚未达到升级阈值，返回失败但不触发下一层
            return {
                "success": False,
                "tool_calls": None,
                "tier_used": 1,
                "content": raw_content,
                "needs_retry": True,
                "degraded_text": None,
                "confidence": 0.3,
                "tier": 1,
            }

        # ── Tier-2: 简化重试 ──
        for attempt in range(1, self.tier2_max_retries + 1):
            simplified_msgs = self._build_simplified_messages(
                messages, raw_content, tool_registry
            )
            try:
                r2 = retry_fn(simplified_msgs)
            except Exception as e:
                self.log.warn(f"Tier-2 重试 #{attempt} 异常: {e}")
                continue

            if r2 is not None:
                r2_tool_calls = getattr(r2, "tool_calls", None)
                if r2_tool_calls:
                    self.stats["t2"][0] += 1
                    self.stats["t2"][1] += 1
                    self.stats["t3_consecutive"] = 0
                    self.tier1_fail_count = 0
                    r2_content = getattr(r2, "content", "") or ""
                    self.log.info(
                        f"Tier-2 重试 #{attempt} 成功"
                    )
                    return {
                        "success": True,
                        "tool_calls": r2_tool_calls,
                        "tier_used": 2,
                        "content": r2_content,
                        "needs_retry": False,
                        "degraded_text": None,
                        "confidence": 0.7,
                        "tier": 2,
                    }

        # Tier-2 全部失败
        self.stats["t2"][0] += 1
        self.log.warn(
            f"Tier-2 全部 {self.tier2_max_retries} 次重试失败"
        )

        # ── Tier-3: 启发式提取 ──
        self.stats["t3"][0] += 1
        self.stats["t3_consecutive"] += 1
        extracted = self._tier3_extract(raw_content)

        # 根据提取结果计算置信度
        actions = extracted.get("actions", [])
        if len(actions) >= 2:
            confidence = 0.5
        elif len(actions) == 1:
            confidence = 0.35
        else:
            confidence = 0.2

        # 检测到 task_complete 信号时提高置信度
        if any(a.get("action") == "task_complete" for a in actions):
            confidence = max(confidence, 0.65)

        degraded_text = json.dumps(extracted, ensure_ascii=False)

        self.log.log_tier3_extraction(
            0,  # turn 由调用方提供
            "H-COMBINED",
            confidence,
        )

        return {
            "success": False,
            "tool_calls": None,
            "tier_used": 3,
            "content": raw_content,
            "needs_retry": self.stats["t3_consecutive"] < self.tier3_max_consecutive,
            "degraded_text": degraded_text,
            "confidence": confidence,
            "tier": 3,
        }

    def _get_tier2_threshold(self) -> int:
        """根据模型等级返回 Tier-1 → Tier-2 升级阈值。

        低等级模型更容易产生格式错误，因此更早升级到 Tier-2。

        Returns:
            Tier-1 连续失败次数阈值。
        """
        thresholds = {"S": 4, "A": 3, "B": 2, "C": 2, "D": 1}
        return thresholds.get(self.model_grade, 3)

    # ── Tier-1 修复实现 ────────────────────────────────────────

    def _tier1_repair(self, text: str) -> Optional[list[dict[str, Any]]]:
        """Tier-1 正则修复——逐条应用 12 条规则修复 JSON。

        对原始文本依次应用所有加载的 Tier-1 规则（pattern → replacement），
        然后用 _apply_common_fixes 做通用后处理，最后尝试 JSON 解析。

        Args:
            text: 模型原始响应文本。

        Returns:
            解析出的 tool_calls 列表，或 None 表示修复失败。
        """
        if not text or not text.strip():
            return None

        working_text: str = text.strip()
        applied_rules: list[str] = []

        for rule in self.tier1_rules:
            rule_id: str = rule.get("id", "??")
            pattern: str = rule.get("pattern", "")
            replacement: str = rule.get("replacement", "")

            if not pattern:
                continue

            try:
                new_text = re.sub(
                    pattern, replacement, working_text, flags=re.DOTALL
                )
                if new_text != working_text:
                    applied_rules.append(rule_id)
                    working_text = new_text
                    self.log.log_tier1_repair(0, rule_id, text[:80])
            except re.error as e:
                self.log.warn(f"Tier-1 规则 {rule_id} 正则错误: {e}")
                continue

        # 通用后处理：Python 字面量、注释、补齐括号
        working_text = self._apply_common_fixes(working_text)

        # 尝试直接解析
        result = self._try_parse_as_tool_calls(working_text)
        if result is not None:
            self.log.debug(
                f"Tier-1 解析成功，应用规则={', '.join(applied_rules) if applied_rules else '基础修复'}"
            )
            return result

        # 尝试从文本中提取 JSON 数组块再解析
        json_blocks = re.findall(r'(\[.*\])', working_text, re.DOTALL)
        for block in json_blocks:
            result = self._try_parse_as_tool_calls(block)
            if result is not None:
                self.log.debug("Tier-1 通过 JSON 数组块解析成功")
                return result

        # 尝试提取花括号包围的 JSON 对象
        try:
            brace_match = re.search(
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', working_text, re.DOTALL
            )
            if brace_match:
                obj = json.loads(brace_match.group(0))
                if isinstance(obj, dict):
                    tc = obj.get("tool_calls")
                    if tc is not None:
                        return tc if isinstance(tc, list) else [tc]
                    if "name" in obj:
                        return [obj]
        except (json.JSONDecodeError, re.error):
            pass

        return None

    @staticmethod
    def _apply_common_fixes(text: str) -> str:
        """应用通用的 JSON 格式修复——不依赖外部规则库。

        修复内容：
            - Python 关键字 → JSON 字面量（None/True/False）
            - 移除 // 和 /* */ 注释
            - 移除 BOM 和零宽字符
            - 移除 JSON 不允许的尾部逗号
            - 补齐缺失的 } 和 ]
            - 修复未闭合的字符串引号

        Args:
            text: 待修复文本。

        Returns:
            修复后的文本。
        """
        # Python 关键字
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)

        # 注释
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        # BOM 和零宽字符
        text = re.sub(r'[﻿​-‏]', '', text)

        # 尾部逗号
        text = re.sub(r',\s*([\]\}])', r'\1', text)

        # 单引号键名转换为双引号键名（处理 'key': 情况）
        text = re.sub(r"'([a-zA-Z_]\w*)'\s*:", r'"\1":', text)

        # 单引号字符串值转换为双引号字符串值
        # 匹配 '"key": 'value'' 模式 → '"key": "value"'
        text = re.sub(r':\s*\'([^\']*)\'', r': "\1"', text)

        # 补齐括号
        open_curly: int = text.count('{') - text.count('}')
        open_square: int = text.count('[') - text.count(']')
        text += '}' * max(0, open_curly)
        text += ']' * max(0, open_square)

        # 检查未闭合字符串
        in_string: bool = False
        escaped: bool = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            text += '"'

        return text

    @staticmethod
    def _try_parse_as_tool_calls(text: str) -> Optional[list[dict[str, Any]]]:
        """将文本解析为 tool_calls 列表，支持多种嵌套格式。

        解析顺序：
            1. 直接 json.loads —— 已是完整 JSON
            2. 如果是 list[dict] 且含 "name" → 直接作为 tool_calls
            3. 如果是 dict —— 尝试提取 tool_calls 或 function 字段
            4. 如果是 message 对象 —— 提取 function 字段

        Args:
            text: JSON 文本。

        Returns:
            tool_calls 列表，或 None 表示无法解析。
        """
        try:
            obj: Any = json.loads(text)
        except json.JSONDecodeError:
            return None

        # 情况 1: 已是 tool_call 数组
        if isinstance(obj, list):
            if all(isinstance(item, dict) for item in obj):
                # 检查是否每个元素都有 name 字段
                if obj and "name" in obj[0]:
                    return obj
                # 可能是嵌套 {tool_calls: [...]}
                if obj and isinstance(obj[0], dict):
                    tc = obj[0].get("tool_calls")
                    if isinstance(tc, list):
                        return tc
            return obj

        # 情况 2: 单个 dict
        if isinstance(obj, dict):
            # 直接是 tool_call（含 name 字段）
            if "name" in obj:
                return [obj]
            # 含 tool_calls 字段
            tc = obj.get("tool_calls")
            if tc is not None:
                return tc if isinstance(tc, list) else [tc]
            # 含 function 字段（Ollama 原生格式）
            fn = obj.get("function")
            if isinstance(fn, dict) and "name" in fn:
                return [fn]
            # message 格式
            msg = obj.get("message")
            if isinstance(msg, dict):
                fn2 = msg.get("function")
                if isinstance(fn2, dict) and "name" in fn2:
                    return [fn2]

        return None

    # ── Tier-2 简化重试 ────────────────────────────────────────

    def _build_simplified_messages(
        self,
        original_messages: list[dict[str, Any]],
        failed_content: str,
        tool_registry: Any,
    ) -> list[dict[str, Any]]:
        """构建 Tier-2 的简化 prompt。

        从原始 messages 中移除 system prompt 中复杂的 tool_calls 格式要求，
        改为要求模型输出纯 JSON 数组。降低 temperature 以减少随机性。

        Args:
            original_messages: 原始消息历史。
            failed_content: 上次失败响应内容（用于给模型上下文）。
            tool_registry: 工具注册表（获取有效工具名列表）。

        Returns:
            简化后的 messages 列表。
        """
        msgs: list[dict[str, Any]] = copy.deepcopy(original_messages)

        # 获取有效工具名
        try:
            if hasattr(tool_registry, "get_names"):
                tool_names = tool_registry.get_names()
            else:
                tool_names = sorted(ToolRegistry.get_names()) if hasattr(
                    self, "_static_names"
                ) else [
                    "read_file", "write_file", "edit_file",
                    "run_command", "glob_search", "grep_search",
                    "task_complete"
                ]
        except Exception:
            tool_names = [
                "read_file", "write_file", "edit_file",
                "run_command", "glob_search", "grep_search",
                "task_complete"
            ]

        tool_names_str: str = ", ".join(tool_names)

        # 构建简化的 system prompt
        simplified_system: str = (
            f"你是一个编程助手。请仅回复一个 JSON 数组，"
            f"数组中每个元素是一个工具调用对象。\n"
            f"格式: [{{\"name\": \"工具名\", \"arguments\": {{...}}}}]\n"
            f"可用工具: {tool_names_str}\n"
            f"不要添加任何解释文字，只输出 JSON 数组。\n"
            f"如果任务完成，输出: "
            f'[{{"name": "task_complete", "arguments": {{"summary": "..."}}}}]'
        )

        # 替换或添加 system 消息
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = simplified_system
        else:
            msgs.insert(0, {"role": "system", "content": simplified_system})

        # 截取最近的上次失败内容作为上下文
        failed_snippet = failed_content[-500:] if len(failed_content) > 500 else failed_content
        msgs.append({
            "role": "user",
            "content": (
                f"上一次你的输出格式不正确。请严格按 JSON 格式重新输出。\n"
                f"之前的输出片段: {failed_snippet[:200]}\n"
                f"请现在仅输出 JSON 数组，不要其他文字。"
            ),
        })

        return msgs

    # ── Tier-3 启发式提取 ──────────────────────────────────────

    def _tier3_extract(self, text: str) -> dict[str, Any]:
        """Tier-3 启发式提取——从纯文本中提取操作意图。

        对文本应用 12 条启发式规则（从 heuristics.json 加载），
        识别模型的操作意图并构造降级的 actions 列表。

        识别能力：
            - read/write/execute/search 等操作的意图
            - 文件路径提取
            - task_complete 完成信号
            - 命令文本（反引号包裹）
            - 思考内容（Thought 段落）
            - 编辑替换字符串

        Args:
            text: 纯文本响应。

        Returns:
            提取结果字典: {"actions": [...], "raw": "原始文本"}
        """
        actions: list[dict[str, Any]] = []
        detected_signals: set[str] = set()

        # 基础关键词匹配（补充规则库）
        keyword_map: list[tuple[str, str, str]] = [
            ("read", "read_file", r'(?:read|读取|打开|查看)\s*[：:]*\s*["\']?([^"\'\n，,。]{3,200})["\']?'),
            ("write", "write_file", r'(?:write|写入|保存|创建文件)\s*[：:]*\s*["\']?([^"\'\n，,。]{3,200})["\']?'),
            ("delete", "run_command", r'(?:delete|删除|移除|rm)\s*["\']?([^"\'\n，,。]{2,200})["\']?'),
            ("search", "grep_search", r'(?:search|搜索|查找|find)\s*["\']?([^"\'\n，,。]{2,200})["\']?'),
            ("execute", "run_command", r'(?:execute|运行|执行|run)\s*["\']?([^"\'\n，,。]{2,200})["\']?'),
        ]

        for _act_label, tool_name, pattern in keyword_map:
            for match in re.findall(pattern, text, re.IGNORECASE):
                target = match.strip()
                if target and target not in detected_signals:
                    detected_signals.add(target)
                    actions.append({
                        "action": tool_name,
                        "target": target,
                        "confidence": 0.4,
                    })

        # 启发式规则库匹配
        for rule in self.tier3_rules:
            rule_id = rule.get("id", "??")
            pattern = rule.get("pattern", "")
            extract_type = rule.get("extract", "")

            if not pattern:
                continue

            try:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    groups = match.groups()

                    if extract_type == "tool_name_and_args_from_parentheses":
                        if groups:
                            fn_name = groups[0].strip()
                            fn_args = groups[1].strip() if len(groups) > 1 else ""
                            if fn_name and fn_name not in detected_signals:
                                detected_signals.add(fn_name)
                                actions.append({
                                    "action": fn_name,
                                    "target": fn_args,
                                    "confidence": 0.35,
                                })

                    elif extract_type in ("tool_name_after_action_label", "action_label"):
                        if groups:
                            tool_nm = groups[0].strip()
                            if tool_nm and tool_nm not in detected_signals:
                                detected_signals.add(tool_nm)
                                actions.append({
                                    "action": tool_nm,
                                    "target": "",
                                    "confidence": 0.3,
                                })

                    elif extract_type in ("file_path_from_read_action", "read_file_path"):
                        if groups:
                            fp = groups[0].strip()
                            if fp and fp not in detected_signals:
                                detected_signals.add(fp)
                                actions.append({
                                    "action": "read_file",
                                    "target": fp,
                                    "confidence": 0.45,
                                })

                    elif extract_type in ("file_path_from_write_action", "write_file_path"):
                        if groups:
                            fp = groups[0].strip()
                            if fp and fp not in detected_signals:
                                detected_signals.add(fp)
                                actions.append({
                                    "action": "write_file",
                                    "target": fp,
                                    "confidence": 0.4,
                                })

                    elif extract_type in ("command_from_backtick", "backtick_command"):
                        if groups:
                            cmd = groups[0].strip()
                            if cmd and cmd not in detected_signals:
                                detected_signals.add(cmd)
                                actions.append({
                                    "action": "run_command",
                                    "target": cmd,
                                    "confidence": 0.3,
                                })

                    elif extract_type in ("completion_signal", "task_complete_signal"):
                        if "task_complete" not in detected_signals:
                            detected_signals.add("task_complete")
                            actions.append({
                                "action": "task_complete",
                                "target": "",
                                "confidence": 0.55,
                            })

                    elif extract_type == "old_string_from_edit_action":
                        if groups:
                            os_str = groups[0].strip()
                            if os_str and os_str not in detected_signals:
                                detected_signals.add(os_str)
                                actions.append({
                                    "action": "edit_file",
                                    "target": os_str,
                                    "confidence": 0.3,
                                })

                    elif extract_type == "glob_pattern_from_search":
                        if groups:
                            gp = groups[0].strip()
                            if gp and gp not in detected_signals:
                                detected_signals.add(gp)
                                actions.append({
                                    "action": "glob_search",
                                    "target": gp,
                                    "confidence": 0.3,
                                })

                    elif extract_type == "grep_pattern_from_search":
                        if groups:
                            gp = groups[0].strip()
                            if gp and gp not in detected_signals:
                                detected_signals.add(gp)
                                actions.append({
                                    "action": "grep_search",
                                    "target": gp,
                                    "confidence": 0.3,
                                })

                    # H08/H12: 思考内容和首行大写词——仅记录，不转为 action
                    elif extract_type in ("thought_content", "thought_text", "first_capitalized"):
                        pass  # 不转换为操作

            except re.error as e:
                self.log.warn(f"Tier-3 规则 {rule_id} 正则错误: {e}")
                continue

        return {
            "actions": actions,
            "raw": text,
        }

    # ── 重置与诊断 ─────────────────────────────────────────────

    def reset(self) -> None:
        """重置所有容错统计和降级状态。

        通常在新的 session 开始时调用。
        """
        self.stats = {
            "t1": [0, 0],
            "t2": [0, 0],
            "t3": [0, 0],
            "t3_consecutive": 0,
        }
        self.tier1_fail_count = 0

    def get_diagnostics(self) -> dict[str, Any]:
        """获取容错引擎的诊断信息。

        Returns:
            诊断字典，包含统计、规则数量和模型等级。
        """
        return {
            "model_grade": self.model_grade,
            "tier1_rules_loaded": len(self.tier1_rules),
            "tier3_rules_loaded": len(self.tier3_rules),
            "tier1_fail_count": self.tier1_fail_count,
            "tier2_threshold": self._get_tier2_threshold(),
            "stats": {
                "t1_attempts": self.stats["t1"][0],
                "t1_successes": self.stats["t1"][1],
                "t2_attempts": self.stats["t2"][0],
                "t2_successes": self.stats["t2"][1],
                "t3_attempts": self.stats["t3"][0],
                "t3_consecutive": self.stats["t3_consecutive"],
            },
            "tier3_max_consecutive": self.tier3_max_consecutive,
            "tier2_max_retries": self.tier2_max_retries,
        }