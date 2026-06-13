"""
收敛控制器 —— 检测 ReAct 循环收敛状态并路由决策。

收敛控制器监控 Agent 循环的执行状态，通过 convergence_counter
和一系列条件判断 Agent 是否已接近任务完成（收敛）或已陷入无效执行。
基于分析结果做出决策：继续、提示收敛、升级模型、暂停或终止。

核心机制：
    convergence_counter:
        一个累积计数器，记录连续 "无实质性变更" 的轮次。
        当 counter >= convergence_rounds_required 时触发收敛检测。

    after_action() — 每轮后更新 counter:
        +1: 无实质性变更且容错正常
        reset: 有实质性变更 / 模型升级 / Tier-3 严重降级 / P0 issue

    route() — 基于状态做出决策:
        terminate:  4+1 终止条件满足
        pause:      P0 issues 需要用户输入
        model_upgrade: 连续 P2 / 低等级模型无法收敛
        convergence_prompt: 已收敛 → 注入提示
        continue:   继续执行

    should_terminate() — 4+1 终止条件:
        1. task_complete 标志
        2. 达到最大轮次
        3. 连续 bash 错误 >= 5
        4. Tier-3 连续降级 >= 上限
        +1. convergence_counter 清零次数 >= 3 (收敛检测失败)

收敛计数器操作表:
    ┌──────────────────────────────────┬─────────────────────┐
    │ 事件                            │ convergence_counter │
    ├──────────────────────────────────┼─────────────────────┤
    │ 无实质性变更 (sub=False)         │ +1                  │
    │ 有实质性变更 (sub=True)          │ → 0                 │
    │ 模型升级                         │ → 0                 │
    │ Tier-3 连续 >= 3                │ → 0, penalty +1      │
    │ Tier-3 置信度 < 0.70             │ → 0                 │
    │ Tier-1/Tier-2 修复成功           │ 不变                 │
    │ convergence_prompt 注入          │ → 0                 │
    └──────────────────────────────────┴─────────────────────┘

Classes:
    ConvergenceController: 收敛状态监控与路由决策。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .logger import Logger


class ConvergenceController:
    """收敛状态监控与路由决策控制器。

    在每轮 ReAct 循环后调用 after_action() 更新收敛状态，
    然后调用 route() 获取下一步决策。

    Attributes:
        state_mgr: StateManager 实例（用于持久化更新）。
        convergence_rounds_default: 默认收敛所需连续轮次。
        model_grade: 当前模型等级。
        auto_upgrade_enabled: 是否启用自动模型升级。
        tier3_max_consecutive: Tier-3 连续降级上限。
        model_detector: 模型探测器（用于获取更好模型）。
        max_turns: 最大轮次限制。
        log: Logger 实例。
        _convergence_failure_count: 收敛检测连续失败次数。
    """

    def __init__(
        self,
        state_manager: Any = None,
        convergence_rounds_default: int = 2,
        model_grade: str = "A",
        auto_upgrade_enabled: bool = True,
        tier3_max_consecutive: int = 5,
        model_detector: Any = None,
        max_turns: int = 30,
    ) -> None:
        """初始化 ConvergenceController。

        Args:
            state_manager: StateManager 实例。
            convergence_rounds_default: 默认收敛所需轮次（连续无变更）。
            model_grade: 模型等级 (S/A/B/C/D)。
            auto_upgrade_enabled: 是否启用自动模型升级。
            tier3_max_consecutive: Tier-3 连续降级上限。
            model_detector: 模型探测器实例。
            max_turns: 最大轮次限制。
        """
        self.state_mgr: Any = state_manager
        self.convergence_rounds_default: int = convergence_rounds_default
        self.model_grade: str = model_grade.upper()
        self.auto_upgrade_enabled: bool = auto_upgrade_enabled
        self.tier3_max_consecutive: int = tier3_max_consecutive
        self.model_detector: Any = model_detector
        self.max_turns: int = max_turns
        self.log: Logger = Logger()

        # 收敛失败计数器（连续清空 convergence_counter 的次数）
        self._convergence_failure_count: int = 0
        self._max_convergence_failures: int = 3

    # ── 收敛计数器操作表实现 ──────────────────────────────────

    def after_action(
        self,
        state: dict[str, Any],
        sub: bool = False,
        ft: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """每轮结束后更新收敛计数器（收敛操作表实现）。

        根据本轮执行结果（实质性变更、容错状态）更新 convergence_counter。
        严格按照收敛计数器操作表执行。

        Args:
            state: 当前状态字典（原地修改）。
            sub: 本轮是否有实质性变更（write_file/edit_file/run_command 成功）。
            ft: 容错解析结果，含 tier 和 confidence 字段。

        Returns:
            更新后的 state 字典。
        """
        if ft is None:
            ft = {}

        conv = state.setdefault("convergence", {})

        # 确保必需字段存在
        defaults: dict[str, Any] = {
            "convergence_counter": 0,
            "convergence_rounds_required": self.convergence_rounds_default,
            "convergence_rounds_achieved": 0,
            "last_substantive_change_turn": 0,
            "convergence_reset_reason": None,
            "degraded_convergence_penalty": 0,
        }
        for k, v in defaults.items():
            conv.setdefault(k, v)

        hk = state.get("housekeeping", {})
        turn = hk.get("turn_count", 0)

        # ── 操作 1: 模型升级 → 清空 counter ──
        if state.get("model", {}).get("upgrade_occurred_this_cycle"):
            self._reset_counter(
                conv, "model_upgrade", turn
            )
            self.log.log_convergence(turn, 0, "模型升级, counter清零")
            return state

        # ── 操作 2: Tier-3 连续 >= 3 → 清空 counter + 惩罚 ──
        ft_tier = ft.get("tier", 0)
        t3c = state.get("fault_tolerance", {}).get(
            "tier3_consecutive_count", 0
        )
        if ft_tier == 3 and t3c >= 3:
            self._reset_counter(
                conv, "tier3_consecutive_penalty", turn
            )
            conv["convergence_rounds_required"] += 1
            conv["degraded_convergence_penalty"] += 1
            self._convergence_failure_count += 1
            self.log.log_convergence(
                turn, 0,
                f"Tier-3连续降级{t3c}次, 惩罚+1, req={conv['convergence_rounds_required']}"
            )
            return state

        # ── 操作 3: Tier-3 低置信度 → 清空 counter ──
        if ft_tier == 3 and ft.get("confidence", 0) < 0.70:
            self._reset_counter(
                conv, "tier3_low_confidence", turn
            )
            self._convergence_failure_count += 1
            self.log.log_convergence(
                turn, 0,
                f"Tier-3置信度{ft['confidence']:.2f}<0.70, counter清零"
            )
            return state

        # ── 操作 4: Tier-1/Tier-2 修复 → 不变 ──
        if ft_tier in (1, 2):
            # counter 不变，不记录任何事件
            self.log.log_convergence(
                turn, conv["convergence_counter"],
                f"Tier-{ft_tier}修复, counter不变"
            )
            return state

        # ── 操作 5: 有实质性变更 → 清空 counter ──
        if sub:
            self._reset_counter(
                conv, "substantive_change", turn
            )
            conv["last_substantive_change_turn"] = turn
            self._convergence_failure_count = max(
                0, self._convergence_failure_count - 1
            )
            self.log.log_convergence(
                turn, 0, "实质性变更, counter清零"
            )
            return state

        # ── 操作 6: 无实质性变更且无异常 → counter +1 ──
        conv["convergence_counter"] += 1
        conv.setdefault("convergence_reset_reason", None)
        conv["convergence_reset_reason"] = None

        self.log.log_convergence(
            turn, conv["convergence_counter"],
            f"无变更, counter={conv['convergence_counter']}"
            f"/{conv['convergence_rounds_required']}"
        )

        # ── 检查收敛达成 ──
        if conv["convergence_counter"] >= conv["convergence_rounds_required"]:
            conv["convergence_rounds_achieved"] += 1
            conv["convergence_counter"] = 0
            conv["convergence_rounds_required"] = self.convergence_rounds_default
            conv["convergence_reset_reason"] = None
            self._convergence_failure_count = 0
            self.log.log_convergence(
                turn, 0,
                f"收敛达成! achieved={conv['convergence_rounds_achieved']}"
            )

        return state

    @staticmethod
    def _reset_counter(
        conv: dict[str, Any], reason: str, turn: int
    ) -> None:
        """重置收敛计数器。

        Args:
            conv: convergence 子字典。
            reason: 重置原因。
            turn: 当前轮次。
        """
        conv["convergence_counter"] = 0
        conv["convergence_reset_reason"] = reason

    # ── 路由决策 ──────────────────────────────────────────────

    def route(self, state: dict[str, Any]) -> tuple[str, str]:
        """根据当前状态做出路由决策。

        决策优先级（从高到低）:
            1. terminate —— 满足终止条件
            2. pause     —— P0 issues 需用户输入
            3. model_upgrade —— 模型等级不足需升级
            4. convergence_prompt —— 检测到收敛状态
            5. continue  —— 继续执行

        Args:
            state: 当前状态字典。

        Returns:
            (action, reason) 元组。
            action 取值: terminate / pause / model_upgrade / convergence_prompt / continue
            reason: 决策原因描述。
        """
        # 优先级 1: 检查终止条件
        term_result = self.check_termination_conditions(state)
        if term_result:
            return ("terminate", term_result)

        # 优先级 2: P0 issues → 暂停等待用户
        issues = state.get("issues", {})
        active_issues = issues.get("active", {})
        p0_list = active_issues.get("p0", [])
        if p0_list:
            latest_p0 = p0_list[-1]
            return (
                "pause",
                f"p0_issue: {latest_p0.get('message', '需用户输入')[:120]}",
            )

        # 优先级 3: 评估模型升级
        upgrade = self.evaluate_model_upgrade(state)
        if upgrade:
            return upgrade

        # 优先级 4: 收敛提示
        conv = state.get("convergence", {})
        if conv.get("convergence_rounds_achieved", 0) > 0:
            conv["convergence_rounds_achieved"] -= 1
            return (
                "convergence_prompt",
                f"convergence_achieved (剩余触发次数: "
                f"{conv['convergence_rounds_achieved']})",
            )

        # 优先级 5: 检查收敛失败次数
        if self._convergence_failure_count >= self._max_convergence_failures:
            return (
                "pause",
                f"收敛检测连续失败 {self._convergence_failure_count} 次，"
                f"建议检查任务是否正确。",
            )

        # 默认: 继续
        return ("continue", "default")

    # ── 模型升级评估 ──────────────────────────────────────────

    def evaluate_model_upgrade(
        self, state: dict[str, Any]
    ) -> Optional[tuple[str, str]]:
        """评估是否需要升级模型。

        决策逻辑:
            1. auto_upgrade_enabled 关闭 → 不升级
            2. 模型等级 S/A → 不升级（硬件足够）
            3. 模型等级 B/C/D + 存在 P1 issues →
                - 升级历史 >= 2 → 新增 P0，不升级
                - 无更好模型 → 新增 P1，不升级
                - 否则 → 建议升级

        Args:
            state: 当前状态字典。

        Returns:
            ("model_upgrade", reason) 或 None。
        """
        if not self.auto_upgrade_enabled:
            return None

        model_info = state.get("model", {})
        current_grade = model_info.get("grade", self.model_grade)

        # S/A 等级无需升级
        if current_grade in ("S", "A"):
            return None

        # B/C/D 检查是否有 P1 issues
        issues = state.get("issues", {})
        active = issues.get("active", {})
        p1_list = active.get("p1", [])
        has_p1_pending = bool(p1_list)

        # 也可检查 convergence 是否反复重置
        conv = state.get("convergence", {})
        degraded_penalty = conv.get("degraded_convergence_penalty", 0)
        has_degraded = degraded_penalty >= 2

        if not has_p1_pending and not has_degraded:
            return None

        # 检查升级历史
        upgrade_history = model_info.get("upgrade_history", [])
        if len(upgrade_history) >= 2:
            old_id = f"upgrade_exhausted_{int(time.time())}"
            state.setdefault("issues", {}).setdefault("active", {}).setdefault(
                "p0", []
            ).append({
                "id": old_id,
                "severity": "P0",
                "message": (
                    f"模型升级历史已达 {len(upgrade_history)} 次，无法继续自动升级。"
                    f"当前模型: {model_info.get('name', '?')} ({current_grade})。"
                    f"请手动更换模型或调整任务。"
                ),
            })
            self.log.log_convergence(
                0, 0,
                f"升级历史 >= 2, 新增P0: 无法继续自动升级"
            )
            return None

        # 检查是否有更好的模型可用
        if self.model_detector:
            try:
                better_model = self.model_detector.get_better_model(current_grade)
                if not better_model:
                    state.setdefault("issues", {}).setdefault(
                        "active", {}
                    ).setdefault("p1", []).append({
                        "id": f"no_better_model_{int(time.time())}",
                        "severity": "P1",
                        "message": (
                            f"无更好模型可升级。当前: {current_grade}，"
                            f"已下载模型无法满足升级需求。"
                        ),
                    })
                    return None
            except Exception as e:
                self.log.warn(f"model_detector 查询失败: {e}")
                # 降级：不依赖 model_detector，直接建议升级
                pass

        return (
            "model_upgrade",
            f"upgrade_from_{current_grade}_due_to_"
            f"{'p1_issues' if has_p1_pending else 'degraded_convergence'}",
        )

    # ── 终止条件检查 (4+1) ─────────────────────────────────────

    def check_termination_conditions(
        self, state: dict[str, Any]
    ) -> Optional[str]:
        """检查是否满足终止条件（4+1）。

        4 个主要条件 + 1 个辅助条件:
            1. task_complete → "task_complete"
            2. 达到最大轮次 → "limit_reached"
            3. 连续 bash 错误 >= 5 → "error_loop"
            4. Tier-3 连续降级 >= 上限 → "degraded_unrecoverable"
            +1. 收敛失败计数器 >= 3 → "convergence_stuck"

        Args:
            state: 当前状态字典。

        Returns:
            终止原因字符串，或 None 表示不应终止。
        """
        # 条件 1: task_complete
        if state.get("task_complete"):
            return "task_complete"

        # 条件 2: 最大轮次
        hk = state.get("housekeeping", {})
        turn_count = hk.get("turn_count", 0)
        if turn_count >= self.max_turns:
            return f"limit_reached: {turn_count}/{self.max_turns}"

        # 条件 3: 连续 bash 错误
        consecutive_bash_errors = state.get("consecutive_bash_errors", 0)
        if consecutive_bash_errors >= 5:
            return f"error_loop: 连续 bash 错误 {consecutive_bash_errors} 次"

        # 条件 4: Tier-3 连续降级
        ft = state.get("fault_tolerance", {})
        t3c = ft.get("tier3_consecutive_count", 0)
        if t3c >= self.tier3_max_consecutive:
            return (
                f"degraded_unrecoverable: "
                f"Tier-3 连续降级 {t3c} >= {self.tier3_max_consecutive}"
            )

        # 条件 +1: 收敛检测持续失败
        if self._convergence_failure_count >= self._max_convergence_failures:
            return (
                f"convergence_stuck: "
                f"收敛检测连续失败 {self._convergence_failure_count} 次"
            )

        return None

    def should_terminate(self, state: dict[str, Any]) -> bool:
        """判断是否应终止循环（便捷方法）。

        Args:
            state: 当前状态字典。

        Returns:
            True 如果应终止。
        """
        return self.check_termination_conditions(state) is not None

    # ── 诊断与重置 ────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        """获取收敛控制器的诊断信息。

        Returns:
            诊断字典。
        """
        return {
            "convergence_rounds_default": self.convergence_rounds_default,
            "model_grade": self.model_grade,
            "auto_upgrade_enabled": self.auto_upgrade_enabled,
            "tier3_max_consecutive": self.tier3_max_consecutive,
            "max_turns": self.max_turns,
            "convergence_failure_count": self._convergence_failure_count,
            "max_convergence_failures": self._max_convergence_failures,
        }

    def reset(self) -> None:
        """重置收敛控制器状态（新 session 前调用）。"""
        self._convergence_failure_count = 0