"""
ReAct Loop: think / act / observe / repeat —— 通过 Ollama /api/chat 驱动。

完整的 ReAct (Reasoning + Acting) 循环引擎，负责编排整个 Agent 生命周期：
    1. think  阶段 —— 构建 messages，调用 OllamaClient.chat() 获取推理结果
    2. act    阶段 —— 解析 tool_calls，通过 ToolRegistry.execute() 执行工具
    3. observe 阶段 —— 将工具执行结果追加到 messages 历史
    4. 循环控制 —— 收敛控制器、容错引擎、安全护栏在每轮前后协同工作

集成的子系统：
    - FaultToleranceEngine —— 每轮 chat() 后用于解析可能损坏的响应
    - GuardLayer            —— 每次工具调用前做安全性检查
    - ConvergenceController —— 每轮结束后更新收敛计数器并路由决策
    - OllamaClient          —— 负责与 Ollama 服务的 HTTP 通信
    - StateManager          —— 状态持久化（原子写入 state.json）

设计原则：
    - 所有异常都在循环内捕获，不因单次失败而终止整个 session
    - 模型 unload 自动检测并恢复（Ollama 默认 5 分钟空闲释放）
    - 动态超时调整：根据历史耗时自适应增减超时预算
    - 支持被 ConvergenceController 路由为 pause / terminate

Classes:
    ReactLoop: ReAct 循环主控制器。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .ollama_client import (
    ChatResponse,
    OllamaClient,
    OllamaAPIError,
    OllamaConnectionError,
    OllamaModelNotLoaded,
    OllamaTimeoutError,
)
from .state_manager import StateManager, StateWriteError
from .tool_registry import ToolRegistry, ToolResult
from .logger import Logger
from .config import Config


class ReactLoop:
    """ReAct 循环主控制器 —— think / act / observe 循环。

    协调 Ollama 推理、工具执行、状态管理和三个安全/质量子系统
    （FaultToleranceEngine、GuardLayer、ConvergenceController）。

    Attributes:
        ollama: OllamaClient 实例。
        state_mgr: StateManager 实例。
        config: Config 配置实例。
        tool_registry: ToolRegistry 执行引擎。
        convergence: ConvergenceController 收敛控制器（可选）。
        fault_tolerance: FaultToleranceEngine 容错引擎（可选）。
        guard: GuardLayer 安全护栏实例。
        log: Logger 日志记录器。
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        state_manager: StateManager,
        config: Config,
        tool_registry: Optional[ToolRegistry] = None,
        convergence_controller: Any = None,
        fault_tolerance: Any = None,
        guard_layer: Any = None,
        prompt_builder: Any = None,
        context_manager: Any = None,
    ) -> None:
        """初始化 ReactLoop。

        Args:
            ollama_client: Ollama HTTP 客户端。
            state_manager: 状态管理器。
            config: 配置对象。
            tool_registry: 工具注册表（可选，默认使用 ToolRegistry 静态方法）。
            convergence_controller: 收敛控制器（可选）。
            fault_tolerance: 容错引擎（可选）。
            guard_layer: 安全护栏（可选）。
            prompt_builder: Prompt 构建器（保留接口，暂未使用）。
            context_manager: 上下文管理器（保留接口，暂未使用）。
        """
        self.ollama: OllamaClient = ollama_client
        self.state_mgr: StateManager = state_manager
        self.config: Config = config
        self.tool_registry: Any = tool_registry or ToolRegistry
        self.convergence: Any = convergence_controller
        self.fault_tolerance: Any = fault_tolerance
        self.guard: Any = guard_layer
        self.prompt_builder: Any = prompt_builder
        self.context_mgr: Any = context_manager
        self.log: Logger = Logger()

        # 动态超时状态
        self._base_timeout_ms: int = 60000
        self._current_timeout_ms: int = 60000
        self._recent_durations: list[float] = []  # 最近 N 轮耗时（秒）
        self._timeout_window_size: int = 5

        # 初始化工具定义列表
        self._tool_definitions: list[dict[str, Any]] = (
            ToolRegistry.get_definitions()
        )

    # ── 主循环 ───────────────────────────────────────────────────

    def run(
        self,
        task: str,
        model_name: str = "",
        tool_definitions: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """启动 ReAct 主循环，执行给定的编程任务。

        完整流程：
            1. 创建/加载 state
            2. 构建初始 messages 和 tools
            3. while not should_terminate():
                a. think  —— 构建本轮 messages → chat() 获取响应
                b. fault_tolerance.parse_response() 解析/修复响应
                c. act    —— guard.check() 每个 tool_call → execute()
                d. observe —— 将 tool results 追加到 messages
                e. convergence.after_action() 更新收敛状态
                f. convergence.route() 决定下一步（continue/pause/terminate）
            4. 返回终止后的 state

        Args:
            task: 用户任务描述。
            model_name: 模型名称，空字符串则使用配置默认值。
            tool_definitions: 自定义工具定义列表，None 则使用内置 7 工具。

        Returns:
            终止后的完整 state 字典。
        """
        # ── 初始化 ──
        model_name = model_name or self.config.default_model
        state: dict[str, Any] = self.state_mgr.create_new(
            task=task, model_name=model_name
        )
        self.state_mgr.update_phase(state, "analyzing")
        self.log.info(
            f"ReAct 循环启动 sid={state['session_id']} task={task[:80]} model={model_name}"
        )

        tools: list[dict[str, Any]] = (
            tool_definitions if tool_definitions is not None
            else self._tool_definitions
        )

        # ── 构建初始 messages ──
        system_content: str = (
            f"你是一个本地编程Agent。请在本地文件系统中完成任务。\n"
            f"任务: {task}\n"
            f"规则:\n"
            f"1. 调用 tool_calls 来执行文件操作和命令\n"
            f"2. 完成任务后调用 task_complete 工具\n"
            f"3. 每次只调用一个工具\n"
            f"4. 所有文件路径使用绝对路径\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"请开始执行任务: {task}"},
        ]

        max_turns: int = self.config.max_turns
        turn: int = 0
        consecutive_errors: int = 0
        self._recent_durations.clear()

        # ── 主循环 ──
        while not self.should_terminate(state):
            turn += 1

            # 检查模型是否在线，必要时恢复
            self._ensure_model_loaded(state, model_name)

            # 检查容错降级是否已达上限
            if self._check_degraded_limit(state):
                break

            # 构建本轮上下文并更新 messages
            self._build_turn_context(state, turn, messages)

            # ── think 阶段 ──
            if state.get("phase") != "analyzing":
                self.state_mgr.update_phase(state, "analyzing")
            try:
                response: ChatResponse = self._chat_with_dynamic_timeout(
                    model_name, messages, tools, turn
                )
            except (OllamaConnectionError, OllamaTimeoutError) as e:
                consecutive_errors += 1
                self.log.error(f"T{turn} 连接/超时: {e}")
                if consecutive_errors >= 3:
                    self.state_mgr.update_termination(
                        state, "error", f"连接连续失败 {consecutive_errors} 次"
                    )
                    break
                time.sleep(2.0)
                continue
            except OllamaAPIError as e:
                consecutive_errors += 1
                self.log.error(f"T{turn} API错误 [{e.status_code}]: {e}")
                if consecutive_errors >= 3:
                    self.state_mgr.update_termination(
                        state, "error", f"API 连续错误 {consecutive_errors} 次"
                    )
                    break
                time.sleep(3.0)
                continue
            except Exception as e:
                consecutive_errors += 1
                self.log.error(f"T{turn} 未知异常: {type(e).__name__}: {e}")
                if consecutive_errors >= 5:
                    self.state_mgr.update_termination(
                        state, "error", f"连续未知异常 {consecutive_errors} 次"
                    )
                    break
                time.sleep(2.0)
                continue

            consecutive_errors = 0  # 成功调用后重置错误计数

            # 更新 housekeeping 统计
            hk = state.setdefault("housekeeping", {})
            hk["tokens_prompt_total"] = (
                hk.get("tokens_prompt_total", 0) + response.prompt_eval_count
            )
            hk["tokens_completion_total"] = (
                hk.get("tokens_completion_total", 0) + response.eval_count
            )
            hk["total_duration_ms"] = (
                hk.get("total_duration_ms", 0)
                + response.total_duration_ns // 1_000_000
            )
            hk["turn_count"] = turn

            # 记录耗时用于动态超时
            duration_s = response.total_duration_ns / 1e9
            self._recent_durations.append(duration_s)
            if len(self._recent_durations) > self._timeout_window_size:
                self._recent_durations.pop(0)

            # ── 容错解析响应 ──
            parse_result: Optional[dict[str, Any]] = None
            if self.fault_tolerance is not None:
                parse_result = self._parse_with_fault_tolerance(
                    response, messages, model_name, tools, turn
                )
                # 更新容错状态
                ft_snap = self.fault_tolerance.get_ft_snapshot()
                self.state_mgr.update_fault_tolerance(state, ft_snap)
            else:
                parse_result = {
                    "success": True,
                    "tool_calls": response.tool_calls,
                    "content": response.content,
                    "tier_used": 1,
                }

            # 将 assistant 消息加入 history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
            }
            if parse_result and parse_result.get("tool_calls"):
                assistant_msg["tool_calls"] = parse_result["tool_calls"]
            messages.append(assistant_msg)

            tool_calls_list = parse_result.get("tool_calls") if parse_result else None

            # ── act 阶段 ──
            if state.get("phase") != "executing":
                self.state_mgr.update_phase(state, "executing")

            # 没有 tool_calls 时，检查是否有实质性内容
            no_tool_calls = not tool_calls_list

            if not no_tool_calls:
                substantive = False
                for tc in tool_calls_list:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    # ── GuardLayer 安全检查 ──
                    guard_result = None
                    if self.guard is not None:
                        guard_result = self.guard.check(tool_name, tool_args)
                        if not guard_result.get("allowed", True):
                            self.log.warn(
                                f"T{turn} Guard 拦截 tool={tool_name} "
                                f"layer={guard_result.get('layer')} "
                                f"reason={guard_result.get('reason')}"
                            )
                            messages.append({
                                "role": "tool",
                                "content": (
                                    f"[BLOCKED by Guard {guard_result.get('layer')}] "
                                    f"{guard_result.get('reason')}"
                                ),
                                "name": tool_name,
                            })
                            continue

                    # ── 执行工具 ──
                    tool_result = self._execute_tool_safe(
                        tool_name, tool_args, state, turn
                    )

                    # 记录工具统计
                    ts = state.setdefault("tool_stats", {}).setdefault(
                        tool_name, {"calls": 0, "successes": 0}
                    )
                    ts["calls"] += 1
                    if tool_result and (
                        tool_result.error is None
                        or tool_name == "task_complete"
                    ):
                        ts["successes"] += 1

                    # ── observe: 追加 tool result 到 messages ──
                    tool_content = tool_result.content if tool_result else ""
                    if tool_result and tool_result.error and tool_name != "task_complete":
                        tool_content = f"[ERROR] {tool_result.error}\n{tool_content}"

                    messages.append({
                        "role": "tool",
                        "content": tool_content,
                        "name": tool_name,
                    })

                    # 标记实质性变更
                    if tool_name in ("write_file", "edit_file", "run_command"):
                        if tool_result and tool_result.error is None:
                            substantive = True

                    # 记录 artifact
                    if tool_name in ("write_file", "edit_file") and tool_result and tool_result.error is None:
                        file_path = tool_args.get("file_path", "")
                        if file_path:
                            self.state_mgr.add_artifact(
                                state, file_path, turn,
                                artifact_type="modified",
                            )

                    # task_complete 特殊处理
                    if tool_name == "task_complete":
                        state.setdefault("termination", {})["status"] = "complete"
                        state["termination"]["exit_reason"] = (
                            f"task_complete: {tool_content[:120]}"
                        )
                        state["task_complete"] = True

                # 设置实质性变更标记
                state["_transient_is_substantive"] = substantive
            else:
                # 无 tool_calls 时视为无实质性变更
                state["_transient_is_substantive"] = False

                # 如果容错降级，将提取的文本也加入
                if parse_result and parse_result.get("degraded_text"):
                    messages.append({
                        "role": "tool",
                        "content": f"[DEGRADED] {parse_result['degraded_text']}",
                        "name": "fault_tolerance_degraded",
                    })

            # ── 收敛控制 ──
            if self.convergence is not None:
                ft_tier = parse_result.get("tier_used", 0) if parse_result else 0
                ft_info = {
                    "tier": ft_tier,
                    "confidence": parse_result.get("confidence", 1.0) if parse_result else 1.0,
                }
                state = self.convergence.after_action(
                    state,
                    sub=state.get("_transient_is_substantive", False),
                    ft=ft_info,
                )

                # 路由决策
                route_decision = self.convergence.route(state)
                action, reason = route_decision

                if action == "terminate":
                    self.state_mgr.update_termination(state, reason, reason)
                    self.log.info(f"T{turn} 收敛路由终止: {reason}")
                    break
                elif action == "pause":
                    self.state_mgr.update_phase(state, "paused")
                    self.state_mgr.update_termination(
                        state, "paused", f"收敛路由暂停: {reason}"
                    )
                    self.log.warn(f"T{turn} 收敛路由暂停: {reason}")
                    break
                elif action == "model_upgrade":
                    self._handle_model_upgrade(state, reason, model_name)
                    # 升级后继续循环
                elif action == "convergence_prompt":
                    # 注入收敛提示
                    messages.append({
                        "role": "user",
                        "content": (
                            "[收敛检测] 任务似乎已达到收敛状态。"
                            "如果任务确实已完成，请调用 task_complete。"
                            "如果需要继续，请说明还需要执行的操作。"
                        ),
                    })
                    self.log.info(f"T{turn} 收敛提示注入")

            # ── 持久化 ──
            try:
                self.state_mgr.save(state)
            except StateWriteError:
                self.log.warn(f"T{turn} 状态保存失败，将在下一轮重试")

        # ── 循环结束 ──
        self.state_mgr.update_phase(state, "terminated")
        # 确保终止状态已设置
        term = state.setdefault("termination", {})
        if not term.get("status"):
            term["status"] = "limit_reached"
            term["exit_reason"] = f"达到最大轮次 {max_turns}"
        # 计算运行统计
        hk = state.get("housekeeping", {})
        tokens_total = hk.get("tokens_prompt_total", 0) + hk.get(
            "tokens_completion_total", 0
        )
        duration_total_s = hk.get("total_duration_ms", 0) / 1000.0
        self.log.info(
            f"ReAct 循环结束 sid={state['session_id']} "
            f"turns={turn}/{max_turns} "
            f"tokens={tokens_total} "
            f"duration={duration_total_s:.1f}s "
            f"exit={term['status']}:{term.get('exit_reason','')[:60]}"
        )

        try:
            self.state_mgr.save(state)
        except StateWriteError:
            self.log.warn("最终状态保存失败")

        return state

    # ── 辅助方法 ───────────────────────────────────────────────

    def should_terminate(self, state: dict[str, Any]) -> bool:
        """检查是否应终止循环。

        条件（4+1）：
            1. task_complete 标志已设置
            2. 达到最大轮次
            3. termination.status 已非空
            4. Tier-3 连续降级超出上限
            +1: 连续 bash 错误 >= 5

        Args:
            state: 当前状态字典。

        Returns:
            True 如果应终止循环。
        """
        term = state.get("termination", {})

        # 条件 1: 明确的任务完成
        if state.get("task_complete"):
            return True

        # 条件 2: 达到最大轮次
        hk = state.get("housekeeping", {})
        if hk.get("turn_count", 0) >= self.config.max_turns:
            term["status"] = "limit_reached"
            term["exit_reason"] = f'达到最大轮次 {self.config.max_turns}'
            return True

        # 条件 3: 已设置终止状态
        if term.get("status"):
            return True

        # 条件 4: Tier-3 连续降级
        ft = state.get("fault_tolerance", {})
        if ft.get("tier3_consecutive_count", 0) >= self.config.tier3_max_consecutive:
            term["status"] = "degraded_limit"
            term["exit_reason"] = (
                f'Tier-3 连续降级达到上限 {self.config.tier3_max_consecutive}'
            )
            return True

        # 条件 +1: 连续 bash 错误
        if state.get("consecutive_bash_errors", 0) >= 5:
            term["status"] = "error_loop"
            term["exit_reason"] = f'连续 bash 错误 {state["consecutive_bash_errors"]} 次'
            return True

        return False

    def _build_turn_context(
        self,
        state: dict[str, Any],
        turn: int,
        messages: list[dict[str, Any]],
    ) -> None:
        """构建本轮上下文并追加到 messages。

        Args:
            state: 当前状态字典。
            turn: 当前轮次编号。
            messages: 消息历史列表（原地修改）。
        """
        hk = state.get("housekeeping", {})
        current_tokens = (
            hk.get("tokens_prompt_total", 0) + hk.get("tokens_completion_total", 0)
        )

        ctx_parts: list[str] = [
            f"--- 轮次 {turn}/{self.config.max_turns} ---",
            f"已消耗 tokens: {current_tokens}",
            f"模型: {state['model']['name']} (等级: {state['model']['grade']})",
        ]

        # 添加工具使用统计
        tool_stats = state.get("tool_stats", {})
        if tool_stats:
            stats_lines = []
            for tname, tdata in sorted(tool_stats.items()):
                c = tdata.get("calls", 0)
                s = tdata.get("successes", 0)
                stats_lines.append(f"  {tname}: {s}/{c} 成功")
            if stats_lines:
                ctx_parts.append("工具统计:\n" + "\n".join(stats_lines))

        # 添加容错状态
        ft = state.get("fault_tolerance", {})
        if ft.get("degraded_mode_active"):
            ctx_parts.append(
                f"[容错降级中] 当前层级: Tier-{ft.get('current_tier', 1)} "
                f"连续降级: {ft.get('tier3_consecutive_count', 0)}"
            )

        # 添加收敛状态
        conv = state.get("convergence", {})
        if conv.get("convergence_rounds_achieved", 0) > 0:
            ctx_parts.append(f"[收敛检测] 已达成 {conv['convergence_rounds_achieved']} 轮收敛")

        # 添加已修改文件列表
        mfs = state.get("modified_files_summary", [])
        if mfs:
            ctx_parts.append(f"已修改文件: {', '.join(mfs[:15])}")

        ctx_parts.append("请继续执行任务，必要时调用 tool_calls。")
        context_msg = "\n".join(ctx_parts)

        # 如果上一条是 user 消息，替换；否则追加
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = context_msg
        else:
            messages.append({"role": "user", "content": context_msg})

    def _chat_with_dynamic_timeout(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        turn: int,
    ) -> ChatResponse:
        """使用动态超时调用 chat API。

        根据最近 N 轮的历史耗时计算自适应超时：
            timeout = max(base_timeout, avg_duration * 3)

        Args:
            model_name: 模型名称。
            messages: 消息历史。
            tools: 工具定义列表。
            turn: 当前轮次。

        Returns:
            ChatResponse 对象。
        """
        timeout_ms = self._calculate_dynamic_timeout()
        self.log.debug(
            f"T{turn} chat() timeout={timeout_ms}ms "
            f"(base={self._base_timeout_ms}ms, "
            f"avg_duration={self._avg_recent_duration():.1f}s)"
        )
        return self.ollama.chat(
            model=model_name,
            messages=messages,
            tools=tools,
            keep_alive=-1,
            stream=False,
            options={
                "temperature": 0.1,
                "num_predict": 2048,
            },
            timeout_ms=timeout_ms,
        )

    def _calculate_dynamic_timeout(self) -> int:
        """根据最近耗时计算动态超时。

        Returns:
            超时毫秒数。
        """
        if not self._recent_durations:
            return self._base_timeout_ms
        avg_s = sum(self._recent_durations) / len(self._recent_durations)
        # 超时 = max(base, avg * 3)，最小 30 秒，最大 300 秒
        dynamic = max(self._base_timeout_ms, int(avg_s * 3000))
        return min(dynamic, 300_000)

    def _avg_recent_duration(self) -> float:
        """计算最近 N 轮平均耗时（秒）。

        Returns:
            平均耗时秒数，无数据返回 0.0。
        """
        if not self._recent_durations:
            return 0.0
        return sum(self._recent_durations) / len(self._recent_durations)

    def _ensure_model_loaded(
        self, state: dict[str, Any], model_name: str
    ) -> None:
        """确保模型已加载到内存。

        检测模型是否在线，如果不在则尝试重新加载。
        记录模型事件到 state。

        Args:
            state: 当前状态字典。
            model_name: 模型名称。
        """
        try:
            if self.ollama.is_model_loaded(model_name):
                return
        except Exception:
            # 无法检测模型状态时跳过
            return

        self.log.warn(f"模型 {model_name} 未加载，尝试恢复...")
        self.log.log_model_event("unload_detected", model_name)

        loaded = self.ollama.ensure_model_loaded(model_name)
        if loaded:
            self.log.log_model_event("recovered", model_name)
        else:
            self.log.error(f"模型 {model_name} 加载恢复失败")
            state.setdefault("issues", {}).setdefault("active", {}).setdefault(
                "p1", []
            ).append({
                "id": f"model_unload_{int(time.time())}",
                "severity": "P1",
                "message": f"模型 {model_name} 自动恢复失败",
            })

    def _check_degraded_limit(self, state: dict[str, Any]) -> bool:
        """检查容错降级是否已超出上限。

        Args:
            state: 当前状态字典。

        Returns:
            True 如果应终止。
        """
        ft = state.get("fault_tolerance", {})
        t3c = ft.get("tier3_consecutive_count", 0)
        if t3c >= self.config.tier3_max_consecutive:
            self.state_mgr.update_termination(
                state,
                "degraded_limit",
                f"Tier-3 连续降级 {t3c} >= {self.config.tier3_max_consecutive}",
            )
            return True
        return False

    def _parse_with_fault_tolerance(
        self,
        response: ChatResponse,
        messages: list[dict[str, Any]],
        model_name: str,
        tools: list[dict[str, Any]],
        turn: int,
    ) -> dict[str, Any]:
        """使用容错引擎解析模型响应。

        调用 FaultToleranceEngine.parse_response() 尝试逐层修复。

        Args:
            response: ChatResponse 对象。
            messages: 消息历史（用于 Tier-2 重试）。
            model_name: 模型名称。
            tools: 工具定义列表。
            turn: 当前轮次。

        Returns:
            解析结果字典，含 tool_calls/tier_used/content 等字段。
        """
        def retry_fn(msgs: list[dict[str, Any]]) -> Optional[ChatResponse]:
            """Tier-2 重试函数：用简化 messages 重新调用 chat。"""
            try:
                return self.ollama.chat(
                    model=model_name,
                    messages=msgs,
                    tools=tools,
                    keep_alive=-1,
                    stream=False,
                    options={
                        "temperature": 0.0,
                        "num_predict": 1024,
                    },
                    timeout_ms=self._calculate_dynamic_timeout(),
                )
            except Exception:
                return None

        result = self.fault_tolerance.parse_response(
            response, self.tool_registry, messages, retry_fn
        )

        # 记录容错事件
        tier = result.get("tier_used", 1)
        if tier >= 2:
            self.log.warn(
                f"T{turn} 容错升级到 Tier-{tier}, "
                f"confidence={result.get('confidence', 0):.2f}"
            )
        if tier >= 3:
            self.log.log_tier3_extraction(
                turn, "T3-auto", result.get("confidence", 0.3)
            )

        return result

    def _execute_tool_safe(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        state: dict[str, Any],
        turn: int,
    ) -> Optional[ToolResult]:
        """安全执行工具，捕获所有异常。

        Args:
            tool_name: 工具名称。
            tool_args: 工具参数。
            state: 状态字典（用于更新错误计数）。
            turn: 当前轮次。

        Returns:
            ToolResult 或 None（工具不存在时）。
        """
        try:
            if isinstance(self.tool_registry, type):
                # 静态类调用
                result = ToolRegistry.execute(tool_name, tool_args)
            else:
                result = self.tool_registry.execute(tool_name, tool_args)

            # 检查 bash 命令错误
            if tool_name == "run_command" and result.error:
                cbe = state.setdefault("consecutive_bash_errors", 0) + 1
                state["consecutive_bash_errors"] = cbe
            elif tool_name == "run_command":
                state["consecutive_bash_errors"] = 0

            return result
        except KeyError:
            self.log.warn(f"T{turn} 未知工具: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                content=f"[ERROR] 未知工具: {tool_name}",
                error=f"未知工具: {tool_name}",
            )
        except Exception as e:
            self.log.error(f"T{turn} 工具 {tool_name} 执行异常: {e}")
            return ToolResult(
                tool_name=tool_name,
                content=f"[ERROR] {type(e).__name__}: {e}",
                error=str(e),
            )

    def _handle_model_upgrade(
        self,
        state: dict[str, Any],
        reason: str,
        current_model_name: str,
    ) -> None:
        """处理模型升级决策。

        添加新的 P0 issue 通知用户，并尝试获取更好模型。

        Args:
            state: 当前状态字典。
            reason: 升级原因。
            current_model_name: 当前模型名称。
        """
        model_info = state.setdefault("model", {})
        current_grade = model_info.get("grade", "B")
        uh = model_info.setdefault("upgrade_history", [])
        uh.append({
            "from_grade": current_grade,
            "from_model": current_model_name,
            "reason": reason,
            "timestamp": int(time.time()),
        })
        model_info["upgrade_occurred_this_cycle"] = True

        self.state_mgr.add_issue(state, {
            "id": f"model_upgrade_{len(uh)}",
            "severity": "P0",
            "message": (
                f"模型等级 [{current_grade}] 持续无法收敛，"
                f"建议手动升级模型。当前: {current_model_name}。"
                f"原因: {reason}"
            ),
        })
        self.log.log_model_event(
            "upgrade_recommended",
            current_model_name,
            grade=current_grade,
            reason=reason,
        )

# Chunk 1 complete
