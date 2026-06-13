"""
ConvergenceController 完整单元测试。

测试收敛控制器：
    - convergence_counter 操作表
    - after_action() 各种事件对 counter 的影响
    - route() P0/P1/P2 决策优先级
    - evaluate_model_upgrade() 模型升级评估
    - should_terminate() 4+1 终止条件
    - check_termination_conditions() 详细条件检查
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convergence_controller import ConvergenceController


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def controller():
    """创建基本的收敛控制器。"""
    return ConvergenceController(
        convergence_rounds_default=2,
        model_grade="A",
        auto_upgrade_enabled=False,
        tier3_max_consecutive=5,
        max_turns=30,
    )


@pytest.fixture
def controller_with_upgrade():
    """启用自动升级的收敛控制器。"""
    mock_detector = MagicMock()
    mock_detector.get_better_model.return_value = "better-model:14b"
    return ConvergenceController(
        convergence_rounds_default=2,
        model_grade="C",
        auto_upgrade_enabled=True,
        tier3_max_consecutive=5,
        model_detector=mock_detector,
        max_turns=30,
    )


@pytest.fixture
def base_state():
    """基础 state 字典。"""
    return {
        "convergence": {
            "convergence_counter": 0,
            "convergence_rounds_required": 2,
            "convergence_rounds_achieved": 0,
            "last_substantive_change_turn": 0,
            "convergence_reset_reason": None,
            "degraded_convergence_penalty": 0,
        },
        "fault_tolerance": {
            "tier3_consecutive_count": 0,
            "current_tier": 1,
        },
        "housekeeping": {
            "turn_count": 0,
        },
        "model": {
            "name": "test-model:7b",
            "grade": "A",
            "upgrade_history": [],
            "upgrade_occurred_this_cycle": False,
        },
        "issues": {
            "active": {"p0": [], "p1": [], "p2": []},
            "resolved": [],
        },
        "_transient_is_substantive": False,
    }


# ── after_action() 测试 ──────────────────────────────────────


class TestAfterAction:
    """after_action() 收敛计数器操作表测试"""

    def test_counter_increments_no_change(self, controller, base_state):
        """无变更: counter +1"""
        base_state["convergence"]["convergence_counter"] = 0
        state = controller.after_action(base_state, sub=False)
        c = state["convergence"]
        assert c["convergence_counter"] == 1

    def test_counter_resets_on_substantive(self, controller, base_state):
        """有实质性变更: counter → 0"""
        base_state["convergence"]["convergence_counter"] = 3
        state = controller.after_action(base_state, sub=True)
        c = state["convergence"]
        assert c["convergence_counter"] == 0
        assert c["convergence_reset_reason"] == "substantive_change"

    def test_counter_resets_on_model_upgrade(self, controller, base_state):
        """模型升级: counter → 0"""
        base_state["convergence"]["convergence_counter"] = 5
        base_state["model"]["upgrade_occurred_this_cycle"] = True
        state = controller.after_action(base_state, sub=False)
        c = state["convergence"]
        assert c["convergence_counter"] == 0
        assert c["convergence_reset_reason"] == "model_upgrade"

    def test_counter_resets_on_tier3_consecutive(self, controller, base_state):
        """Tier-3 连续 >= 3: counter → 0 + penalty"""
        base_state["convergence"]["convergence_counter"] = 3
        base_state["fault_tolerance"]["tier3_consecutive_count"] = 3
        state = controller.after_action(
            base_state, sub=False, ft={"tier": 3, "confidence": 0.8}
        )
        c = state["convergence"]
        assert c["convergence_counter"] == 0
        assert c["degraded_convergence_penalty"] >= 1

    def test_counter_resets_on_tier3_low_confidence(self, controller, base_state):
        """Tier-3 置信度 < 0.70: counter → 0"""
        base_state["convergence"]["convergence_counter"] = 5
        state = controller.after_action(
            base_state, sub=False, ft={"tier": 3, "confidence": 0.5}
        )
        c = state["convergence"]
        assert c["convergence_counter"] == 0
        assert c["convergence_reset_reason"] == "tier3_low_confidence"

    def test_counter_unchanged_on_tier1_tier2(self, controller, base_state):
        """Tier-1/Tier-2 修复: counter 不变"""
        base_state["convergence"]["convergence_counter"] = 3
        state = controller.after_action(
            base_state, sub=False, ft={"tier": 1, "confidence": 0.9}
        )
        c = state["convergence"]
        assert c["convergence_counter"] == 3

    def test_convergence_achieved(self, controller, base_state):
        """收敛达成: achieved +1, counter 清零, required 重置"""
        base_state["convergence"]["convergence_counter"] = 1
        base_state["convergence"]["convergence_rounds_required"] = 1
        state = controller.after_action(base_state, sub=False)
        c = state["convergence"]
        assert c["convergence_rounds_achieved"] >= 1
        assert c["convergence_counter"] == 0

    def test_convergence_not_achieved_below_required(self, controller, base_state):
        """不足所需轮次: 不触发收敛"""
        base_state["convergence"]["convergence_counter"] = 0
        base_state["convergence"]["convergence_rounds_required"] = 2
        state = controller.after_action(base_state, sub=False)
        c = state["convergence"]
        # counter 增加了但 achieved 未增加
        assert c["convergence_counter"] == 1
        assert c["convergence_rounds_achieved"] == 0

    def test_last_substantive_change_turn_updated(self, controller, base_state):
        """实质性变更时更新 last_substantive_change_turn"""
        base_state["housekeeping"]["turn_count"] = 5
        state = controller.after_action(base_state, sub=True)
        c = state["convergence"]
        assert c["last_substantive_change_turn"] == 5


# ── route() 测试 ─────────────────────────────────────────────


class TestRoute:
    """route() 路由决策测试"""

    def test_route_continue_default(self, controller, base_state):
        """默认继续"""
        action, reason = controller.route(base_state)
        assert action == "continue"
        assert reason == "default"

    def test_route_terminate_task_complete(self, controller, base_state):
        """task_complete 触发终止"""
        base_state["task_complete"] = True
        action, reason = controller.route(base_state)
        assert action == "terminate"
        assert reason == "task_complete"

    def test_route_terminate_limit_reached(self, controller, base_state):
        """达到最大轮次触发终止"""
        controller.max_turns = 1
        base_state["housekeeping"]["turn_count"] = 1
        action, reason = controller.route(base_state)
        assert action == "terminate"

    def test_route_terminate_error_loop(self, controller, base_state):
        """连续 bash 错误触发终止"""
        base_state["consecutive_bash_errors"] = 5
        action, reason = controller.route(base_state)
        assert action == "terminate"

    def test_route_terminate_degraded(self, controller, base_state):
        """Tier-3 降级上限触发终止"""
        base_state["fault_tolerance"]["tier3_consecutive_count"] = 5
        action, reason = controller.route(base_state)
        assert action == "terminate"

    def test_route_pause_p0_issues(self, controller, base_state):
        """P0 issues 触发暂停"""
        base_state["issues"]["active"]["p0"] = [
            {"id": "p0_1", "severity": "P0", "message": "严重问题"}
        ]
        action, reason = controller.route(base_state)
        assert action == "pause"

    def test_route_convergence_prompt(self, controller, base_state):
        """收敛检测触发收敛提示"""
        base_state["convergence"]["convergence_rounds_achieved"] = 1
        action, reason = controller.route(base_state)
        assert action == "convergence_prompt"

    def test_route_terminate_convergence_stuck(self, controller, base_state):
        """收敛检测连续失败触发终止（+1 条件）"""
        controller._convergence_failure_count = 3
        action, reason = controller.route(base_state)
        assert action == "terminate"


# ── evaluate_model_upgrade() 测试 ─────────────────────────────


class TestEvaluateModelUpgrade:
    """模型升级评估测试"""

    def test_no_upgrade_when_disabled(self, controller, base_state):
        """禁用状态不升级"""
        assert controller.evaluate_model_upgrade(base_state) is None

    def test_no_upgrade_for_s_grade(self, controller_with_upgrade, base_state):
        """S 级不升级"""
        base_state["model"]["grade"] = "S"
        assert controller_with_upgrade.evaluate_model_upgrade(base_state) is None

    def test_no_upgrade_for_a_grade(self, controller_with_upgrade, base_state):
        """A 级不升级"""
        base_state["model"]["grade"] = "A"
        assert controller_with_upgrade.evaluate_model_upgrade(base_state) is None

    def test_upgrade_for_c_grade_with_p1(self, controller_with_upgrade, base_state):
        """C 级 + P1 issues 触发升级"""
        base_state["model"]["grade"] = "C"
        base_state["issues"]["active"]["p1"] = [
            {"id": "p1_1", "severity": "P1", "message": "问题"}
        ]
        result = controller_with_upgrade.evaluate_model_upgrade(base_state)
        assert result is not None
        assert result[0] == "model_upgrade"

    def test_upgrade_exhausted_after_two(self, controller_with_upgrade, base_state):
        """升级历史 >= 2: 不再升级"""
        base_state["model"]["grade"] = "C"
        base_state["model"]["upgrade_history"] = [
            {"from_grade": "D", "reason": "test"},
            {"from_grade": "C", "reason": "test"},
        ]
        base_state["issues"]["active"]["p1"] = [
            {"id": "p1_1", "severity": "P1", "message": "问题"}
        ]
        result = controller_with_upgrade.evaluate_model_upgrade(base_state)
        assert result is None
        # 应有 P0
        p0 = base_state["issues"]["active"].get("p0", [])
        assert len(p0) >= 1

    def test_no_better_model(self, controller_with_upgrade, base_state):
        """无更好模型: 不升级并标记 P1"""
        controller_with_upgrade.model_detector.get_better_model.return_value = None
        base_state["model"]["grade"] = "C"
        base_state["issues"]["active"]["p1"] = [
            {"id": "p1_1", "severity": "P1", "message": "问题"}
        ]
        result = controller_with_upgrade.evaluate_model_upgrade(base_state)
        assert result is None
        p1 = base_state["issues"]["active"].get("p1", [])
        assert len(p1) >= 2  # 原有 + 新增


# ── should_terminate() 测试 ──────────────────────────────────


class TestShouldTerminate:
    """should_terminate() 4+1 条件测试"""

    def test_no_terminate_normal(self, controller, base_state):
        """正常状态不终止"""
        assert controller.should_terminate(base_state) is False

    def test_terminate_task_complete(self, controller, base_state):
        """task_complete 终止"""
        base_state["task_complete"] = True
        assert controller.should_terminate(base_state) is True

    def test_terminate_max_turns(self, controller, base_state):
        """max_turns 终止"""
        controller.max_turns = 5
        base_state["housekeeping"]["turn_count"] = 5
        assert controller.should_terminate(base_state) is True

    def test_terminate_bash_errors(self, controller, base_state):
        """连续 bash 错误终止"""
        base_state["consecutive_bash_errors"] = 5
        assert controller.should_terminate(base_state) is True

    def test_terminate_degraded(self, controller, base_state):
        """Tier-3 降级终止"""
        base_state["fault_tolerance"]["tier3_consecutive_count"] = 6
        assert controller.should_terminate(base_state) is True

    def test_terminate_convergence_stuck(self, controller, base_state):
        """收敛卡死终止"""
        controller._convergence_failure_count = 3
        assert controller.should_terminate(base_state) is True

    def test_not_terminate_bash_errors_below_threshold(self, controller, base_state):
        """bash 错误未达阈值不终止"""
        base_state["consecutive_bash_errors"] = 3
        assert controller.should_terminate(base_state) is False


# ── check_termination_conditions() 详细检查 ──────────────────


class TestCheckTerminationConditions:
    """终止条件详细检查"""

    def test_all_conditions_normal(self, controller, base_state):
        """所有条件正常——不终止"""
        result = controller.check_termination_conditions(base_state)
        assert result is None

    def test_condition_task_complete(self, controller, base_state):
        """条件 1: task_complete"""
        base_state["task_complete"] = True
        result = controller.check_termination_conditions(base_state)
        assert result == "task_complete"

    def test_condition_limit_reached(self, controller, base_state):
        """条件 2: limit_reached"""
        controller.max_turns = 3
        base_state["housekeeping"]["turn_count"] = 3
        result = controller.check_termination_conditions(base_state)
        assert "limit_reached" in result

    def test_condition_error_loop(self, controller, base_state):
        """条件 3: error_loop"""
        base_state["consecutive_bash_errors"] = 6
        result = controller.check_termination_conditions(base_state)
        assert "error_loop" in result

    def test_condition_degraded(self, controller, base_state):
        """条件 4: degraded_unrecoverable"""
        base_state["fault_tolerance"]["tier3_consecutive_count"] = 5
        result = controller.check_termination_conditions(base_state)
        assert "degraded_unrecoverable" in result

    def test_plus_one_convergence_stuck(self, controller, base_state):
        """条件 +1: convergence_stuck"""
        controller._convergence_failure_count = 3
        result = controller.check_termination_conditions(base_state)
        assert "convergence_stuck" in result


# ── 诊断与重置 ──────────────────────────────────────────────


class TestDiagnostics:
    """诊断和重置测试"""

    def test_get_diagnostics(self, controller):
        """获取诊断信息"""
        diag = controller.get_diagnostics()
        assert diag["convergence_rounds_default"] == 2
        assert diag["model_grade"] == "A"
        assert "convergence_failure_count" in diag

    def test_reset(self, controller, base_state):
        """重置状态"""
        controller._convergence_failure_count = 5
        controller.reset()
        assert controller._convergence_failure_count == 0

    def test_state_preserves_fields_after_action(self, controller, base_state):
        """after_action 保留 state 原有字段"""
        state = controller.after_action(base_state, sub=False)
        assert "convergence" in state
        assert "fault_tolerance" in state
        assert "housekeeping" in state
        assert "model" in state


# ── 降级收敛惩罚测试 ────────────────────────────────────────


class TestDegradedPenalty:
    """降级收敛惩罚测试"""

    def test_penalty_increases_with_tier3(self, controller, base_state):
        """Tier-3 降级增加惩罚"""
        base_state["fault_tolerance"]["tier3_consecutive_count"] = 3
        state = controller.after_action(
            base_state, sub=False, ft={"tier": 3, "confidence": 0.8}
        )
        c = state["convergence"]
        assert c["degraded_convergence_penalty"] >= 1
        # required 也应增加
        assert c["convergence_rounds_required"] > controller.convergence_rounds_default

    def test_penalty_triggers_upgrade_evaluation(self, controller_with_upgrade, base_state):
        """惩罚 >= 2 触发模型升级评估"""
        base_state["model"]["grade"] = "B"
        base_state["convergence"]["degraded_convergence_penalty"] = 2
        result = controller_with_upgrade.evaluate_model_upgrade(base_state)
        assert result is not None
        assert result[0] == "model_upgrade"
