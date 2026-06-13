"""
loop-ollama StateManager 单元测试。

测试 state.json 原子写入（4 步协议）、加载/保存、
Schema 校验、阶段转换、异常处理。
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state_manager import (
    StateManager,
    StateFileNotFoundError,
    StateSchemaValidationError,
    StateCorruptedError,
    StateWriteError,
    StateLockError,
    InvalidPhaseTransitionError,
    _now_iso,
    _VALID_PHASE_TRANSITIONS,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def state_mgr(temp_dir):
    """在临时目录中创建 StateManager。"""
    state_dir = os.path.join(temp_dir, "state")
    return StateManager(state_dir=state_dir)


@pytest.fixture
def populated_state(state_mgr):
    """已初始化状态的 StateManager。"""
    state = state_mgr.create_new(task="Test task", model_name="test-model")
    return state_mgr, state


# ── Tests: _now_iso ─────────────────────────────────────────────


def test_now_iso_returns_string():
    """_now_iso() 返回 ISO 8601 字符串。"""
    ts = _now_iso()
    assert isinstance(ts, str)
    assert "T" in ts
    assert len(ts) > 10


# ── Tests: 创建新状态 ───────────────────────────────────────────


def test_create_new_returns_state(state_mgr):
    """create_new() 返回完整状态字典。"""
    state = state_mgr.create_new(task="Hello World", model_name="llama3")
    assert state["session_id"] is not None
    assert len(state["session_id"]) > 0
    assert state["phase"] == "init"
    assert state["task"] == "Hello World"
    assert state["model"]["name"] == "llama3"


def test_create_new_writes_to_disk(state_mgr):
    """create_new() 应写入 state.json。"""
    state_mgr.create_new(task="Write test", model_name="test:7b")
    assert state_mgr.state_file.exists()


def test_create_new_custom_session_id(state_mgr):
    """可指定自定义 session_id。"""
    state = state_mgr.create_new(
        task="Test", model_name="test", session_id="my-session-001"
    )
    assert state["session_id"] == "my-session-001"


def test_create_new_defaults(state_mgr):
    """create_new() 使用默认值填充缺失字段。"""
    state = state_mgr.create_new(task="", model_name="")
    assert state["convergence"]["convergence_counter"] == 0
    assert state["fault_tolerance"]["current_tier"] == 1
    assert state["termination"]["status"] is None
    assert state["issues"]["active"]["p0"] == []


def test_create_new_with_hardware(state_mgr):
    """create_new() 接受硬件信息。"""
    hw = {"gpu_name": "RTX 4090", "vram_gb": 24.0}
    state = state_mgr.create_new(task="Test", model_name="test", hardware=hw)
    assert state["hardware"]["gpu_name"] == "RTX 4090"


def test_create_new_with_config(state_mgr):
    """create_new() 接受自定义配置。"""
    cfg = {"max_turns": 50, "safe_mode": True}
    state = state_mgr.create_new(task="Test", model_name="test", config=cfg)
    assert state["config"]["max_turns"] == 50


# ── Tests: 加载状态 ─────────────────────────────────────────────


def test_load_existing_state(populated_state):
    """load() 加载已有状态。"""
    state_mgr, state = populated_state
    loaded = state_mgr.load()
    assert loaded["phase"] == "init"
    assert loaded["task"] == "Test task"


def test_load_not_found(state_mgr):
    """load() 文件不存在时抛出 StateFileNotFoundError。"""
    with pytest.raises(StateFileNotFoundError):
        state_mgr.load()


def test_load_corrupted_json(state_mgr):
    """load() 损坏的 JSON 时抛出 StateCorruptedError。"""
    state_mgr.state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_mgr.state_file, "w") as f:
        f.write("{ this is not valid json")
    with pytest.raises(StateCorruptedError):
        state_mgr.load()


# ── Tests: 保存状态 ─────────────────────────────────────────────


def test_save_new_state(state_mgr):
    """save() 写入新状态。"""
    state = {
        "session_id": "test-001",
        "version": "0.1.0",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "phase": "init",
        "task": "Test",
    }
    state_mgr.save(state)
    assert state_mgr.state_file.exists()
    loaded = json.loads(state_mgr.state_file.read_text())
    assert loaded["session_id"] == "test-001"


def test_save_overwrites_existing(populated_state):
    """save() 覆盖已有状态。"""
    state_mgr, state = populated_state
    state["phase"] = "executing"
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["phase"] == "executing"


# ── Tests: 阶段转换 ─────────────────────────────────────────────


@pytest.mark.parametrize("from_phase,to_phase,expected", [
    ("init", "building_tools", True),
    ("init", "analyzing", True),
    ("init", "terminated", False),
    ("building_tools", "executing", True),
    ("analyzing", "converging", True),
    ("executing", "paused", True),
    ("converging", "terminated", True),
    ("paused", "analyzing", True),
    ("paused", "executing", True),
    ("terminated", "init", False),
    ("terminated", "analyzing", False),
    ("unknown", "init", False),
    ("init", "unknown", False),
])
def test_transition_validity(from_phase, to_phase, expected):
    """阶段转换合法性检查。"""
    allowed = _VALID_PHASE_TRANSITIONS.get(from_phase, set())
    result = to_phase in allowed
    assert result == expected


def test_transition_executes(state_mgr):
    """update_phase() 成功执行合法转换。"""
    state = state_mgr.create_new(task="Test", model_name="test")
    new_state = state_mgr.update_phase(state, "analyzing")
    assert new_state["phase"] == "analyzing"


def test_transition_invalid_raises(state_mgr):
    """非法转换抛出 InvalidPhaseTransitionError。"""
    state = state_mgr.create_new(task="Test", model_name="test")
    with pytest.raises(InvalidPhaseTransitionError):
        state_mgr.update_phase(state, "terminated")


# ── Tests: 更新字段 ─────────────────────────────────────────────


def test_update_termination_status(populated_state):
    """更新终止状态。"""
    state_mgr, state = populated_state
    state_mgr.update_termination(state, "completed", exit_reason="converged")
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["termination"]["status"] == "completed"
    assert loaded["termination"]["exit_reason"] == "converged"


def test_update_add_artifact(populated_state):
    """添加 artifact 条目。"""
    state_mgr, state = populated_state
    state_mgr.add_artifact(state, "/tmp/test.py", turn=1)
    state_mgr.save(state)
    after = state_mgr.load()
    assert len(after["artifacts"]) >= 1
    assert after["artifacts"][0]["path"] == "/tmp/test.py"


# ── Tests: 问题追踪 ─────────────────────────────────────────────


def test_add_p0_issue(populated_state):
    """添加 P0 问题。"""
    state_mgr, state = populated_state
    state_mgr.add_issue(state, {
        "id": "p0-001",
        "title": "Critical bug",
        "severity": "P0",
        "description": "app crash",
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert len(loaded["issues"]["active"]["p0"]) == 1
    assert loaded["issues"]["active"]["p0"][0]["id"] == "p0-001"


def test_add_p1_issue(populated_state):
    """添加 P1 问题。"""
    state_mgr, state = populated_state
    state_mgr.add_issue(state, {
        "id": "p1-001",
        "title": "Warning",
        "severity": "P1",
        "description": "minor issue",
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert len(loaded["issues"]["active"]["p1"]) == 1


def test_resolve_issue(populated_state):
    """解决一个问题。"""
    state_mgr, state = populated_state
    state_mgr.add_issue(state, {
        "id": "p0-001",
        "title": "Bug",
        "severity": "P0",
        "description": "description",
    })
    # 手动 resolve: 从 active 移到 resolved
    state["issues"]["resolved"].append(
        state["issues"]["active"]["p0"].pop(0)
    )
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert len(loaded["issues"]["active"]["p0"]) == 0
    assert len(loaded["issues"]["resolved"]) == 1


def test_total_p0_count(populated_state):
    """P0 总计数正确。"""
    state_mgr, state = populated_state
    state_mgr.add_issue(state, {
        "id": "p0-1", "title": "Bug 1", "severity": "P0", "description": "d1",
    })
    state_mgr.add_issue(state, {
        "id": "p0-2", "title": "Bug 2", "severity": "P0", "description": "d2",
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["issues"]["total_p0_triggered"] == 2


# ── Tests: 容错追踪 ─────────────────────────────────────────────


def test_record_fault_tier1(state_mgr):
    """记录 Tier-1 修复。"""
    state = state_mgr.create_new(task="Test", model_name="test")
    state_mgr.update_fault_tolerance(state, {
        "tier1_total_repairs": 1,
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["fault_tolerance"]["tier1_total_repairs"] == 1


def test_record_fault_tier2(state_mgr):
    """记录 Tier-2 重试。"""
    state = state_mgr.create_new(task="Test", model_name="test")
    state_mgr.update_fault_tolerance(state, {
        "tier2_total_retries": 1,
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["fault_tolerance"]["tier2_total_retries"] == 1


def test_record_fault_tier3(state_mgr):
    """记录 Tier-3 退化。"""
    state = state_mgr.create_new(task="Test", model_name="test")
    state_mgr.update_fault_tolerance(state, {
        "tier3_total_degradations": 1,
        "tier3_consecutive_count": 1,
        "degraded_mode_active": True,
        "current_tier": 3,
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["fault_tolerance"]["tier3_total_degradations"] == 1
    assert loaded["fault_tolerance"]["tier3_consecutive_count"] == 1
    assert loaded["fault_tolerance"]["degraded_mode_active"] is True


# ── Tests: Session 统计 ─────────────────────────────────────────


def test_increment_turn(populated_state):
    """递增 turn 计数。"""
    state_mgr, state = populated_state
    state_mgr.update_turn_counters(state, turn=1)
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["housekeeping"]["turn_count"] == 1


def test_add_tokens(populated_state):
    """添加 token 统计。"""
    state_mgr, state = populated_state
    state_mgr.update_turn_counters(state, turn=1, tokens_prompt=500, tokens_completion=200)
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["housekeeping"]["tokens_prompt_total"] == 500
    assert loaded["housekeeping"]["tokens_completion_total"] == 200


def test_add_duration(populated_state):
    """添加耗时统计。"""
    state_mgr, state = populated_state
    state_mgr.update_turn_counters(state, turn=1, duration_ms=1500)
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["housekeeping"]["total_duration_ms"] == 1500


# ── Tests: 原子写入（4 步协议） ─────────────────────────────────


def test_atomic_write_creates_tmp_then_renames(state_mgr):
    """原子写入：tmp 文件 -> rename -> 最终文件。"""
    state = {
        "session_id": "atomic-test",
        "version": "0.1.0",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    state_mgr.save(state)
    # 最终文件应存在
    assert state_mgr.state_file.exists()
    # tmp 文件应已被清理
    tmp_files = list(state_mgr.state_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


# ── Tests: 收敛状态更新 ─────────────────────────────────────────


def test_update_convergence(populated_state):
    """更新收敛状态。"""
    state_mgr, state = populated_state
    state_mgr.update_convergence(state, {
        "convergence_counter": 3,
        "convergence_rounds_required": 2,
    })
    state_mgr.save(state)
    loaded = state_mgr.load()
    assert loaded["convergence"]["convergence_counter"] == 3


# ── Tests: exists() ─────────────────────────────────────────────


def test_exists_false_when_no_file(state_mgr):
    """state.json 不存在时 exists() 返回 False。"""
    assert state_mgr.exists() is False


def test_exists_true_after_create(state_mgr):
    """创建后 exists() 返回 True。"""
    state_mgr.create_new(task="Test", model_name="test")
    assert state_mgr.exists() is True


# ── Tests: update_termination 含 summary/verify_cmd ────────────


class TestUpdateTermination:
    """update_termination 完整参数测试"""

    def test_update_termination_with_summary(self, state_mgr):
        """含 summary 参数。"""
        state = state_mgr.create_new(task="Test", model_name="test")
        state_mgr.update_termination(state, "completed",
                                     exit_reason="done", summary="All tests pass")
        assert state["termination"]["summary"] == "All tests pass"

    def test_update_termination_with_verify_cmd(self, state_mgr):
        """含 verify_cmd 参数。"""
        state = state_mgr.create_new(task="Test", model_name="test")
        state_mgr.update_termination(state, "completed",
                                     verify_cmd="pytest tests/")
        assert state["termination"]["verification_command"] == "pytest tests/"


# ── Tests: add_issue 去重 ─────────────────────────────────────


class TestIssueDedup:
    """add_issue 去重逻辑测试"""

    def test_duplicate_issue_not_added(self, state_mgr):
        """同 id issue 不重复添加。"""
        state = state_mgr.create_new(task="Test", model_name="test")
        issue = {"id": "dup-001", "severity": "P1", "description": "dup"}
        state_mgr.add_issue(state, issue)
        state_mgr.add_issue(state, issue)
        state_mgr.save(state)
        loaded = state_mgr.load()
        # 只应有一条 P1
        p1_dup = [i for i in loaded["issues"]["active"]["p1"] if i["id"] == "dup-001"]
        assert len(p1_dup) == 1


# ── Tests: add_artifact 更新已有记录 ───────────────────────────


class TestArtifactUpdate:
    """add_artifact 更新已有记录测试"""

    def test_artifact_update_existing(self, state_mgr):
        """同路径 artifact 会更新而非追加。"""
        state = state_mgr.create_new(task="Test", model_name="test")
        state_mgr.add_artifact(state, "/tmp/test.py", turn=1, artifact_type="created")
        state_mgr.save(state)
        state_mgr.add_artifact(state, "/tmp/test.py", turn=2, artifact_type="modified")
        state_mgr.save(state)
        loaded = state_mgr.load()
        test_arts = [a for a in loaded["artifacts"] if a["path"] == "/tmp/test.py"]
        assert len(test_arts) == 1
        assert test_arts[0]["turn"] == 2


# ── Tests: load 缺失必需字段 ────────────────────────────────────


class TestLoadValidation:
    """load() 字段校验测试"""

    def test_load_missing_required_field_raises(self, state_mgr):
        """缺失 phase 字段抛出 SchemaValidationError。"""
        state_mgr.state_dir.mkdir(parents=True, exist_ok=True)
        invalid_state = {"session_id": "test", "convergence": {}, "termination": {}}
        import json
        state_mgr.state_file.write_text(json.dumps(invalid_state))
        with pytest.raises(StateSchemaValidationError):
            state_mgr.load()
