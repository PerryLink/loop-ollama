"""
loop-ollama 状态管理器。

以 state.json 为核心的状态持久化模块。提供原子写入（4 步协议：
tmp -> fsync -> rename -> fsync dir）、Schema 验证、阶段转换校验。

Classes:
    StateManager: 状态读写与生命周期管理。
    StateFileNotFoundError: 状态文件不存在。
    StateSchemaValidationError: Schema 校验失败。
    StateCorruptedError: 状态文件损坏。
    StateWriteError: 写入失败。
    StateLockError: 并发锁冲突。
    InvalidPhaseTransitionError: 非法阶段转换。
"""

import copy
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── 自定义异常 ────────────────────────────────────────────────────


class StateFileNotFoundError(Exception):
    """状态文件未找到。"""


class StateSchemaValidationError(Exception):
    """状态 JSON Schema 校验失败。"""


class StateCorruptedError(Exception):
    """状态文件已损坏且无法恢复。"""


class StateWriteError(Exception):
    """状态文件写入失败。"""


class StateLockError(Exception):
    """状态文件并发锁冲突。"""


class InvalidPhaseTransitionError(Exception):
    """非法的阶段转换。"""


# ── 合法阶段转换表 ────────────────────────────────────────────────

_VALID_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "init": {"building_tools", "analyzing"},
    "building_tools": {"analyzing", "executing"},
    "analyzing": {"executing", "converging", "terminated", "paused"},
    "executing": {"analyzing", "converging", "terminated", "paused"},
    "converging": {"executing", "analyzing", "terminated", "paused"},
    "terminated": set(),  # 终态
    "paused": {"analyzing", "executing", "converging", "terminated"},
}


def _now_iso() -> str:
    """返回当前 UTC 时间 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class StateManager:
    """状态管理器 —— state.json 原子读写与生命周期控制。

    Attributes:
        state_dir: 状态文件存放目录。
        state_file: state.json 完整路径。
        lock_file: 锁文件 (.lock) 路径。
        schema_path: JSON Schema 文件路径（可选）。
    """

    def __init__(
        self,
        state_dir: Optional[str] = None,
        schema_path: Optional[str] = None,
    ) -> None:
        """初始化 StateManager。

        Args:
            state_dir: 状态目录。默认 ~/.loop-ollama/state。
            schema_path: JSON Schema 文件路径。None 则使用内置校验。
        """
        if state_dir is None:
            self.state_dir = Path(
                os.path.expanduser("~/.loop-ollama/state")
            )
        else:
            self.state_dir = Path(state_dir).resolve()

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        self.lock_file = self.state_dir / "state.json.lock"
        self.schema_path = (
            Path(schema_path).resolve() if schema_path else None
        )

    # ── 创建 ─────────────────────────────────────────────────

    def create_new(
        self,
        session_id: Optional[str] = None,
        task: str = "",
        model_name: str = "",
        model_grade: str = "A",
        capability_score: float = 0.5,
        hardware: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """创建全新状态字典并原子写入磁盘。

        Args:
            session_id: 会话 ID，None 则自动生成 UUID。
            task: 用户任务描述。
            model_name: 使用的模型名称。
            model_grade: 模型等级 (S/A/B/C/D)。
            capability_score: 模型能力分数 (0.0-1.0)。
            hardware: 硬件检测结果。
            config: 配置字典（用于衍生 session 级参数）。

        Returns:
            完整的 state 字典。
        """
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]

        now = _now_iso()

        state: dict[str, Any] = {
            "session_id": session_id,
            "version": "0.1.0",
            "created_at": now,
            "updated_at": now,
            "phase": "init",
            "task": task,
            "model": {
                "name": model_name,
                "grade": model_grade,
                "capability_score": capability_score,
                "upgrade_history": [],
                "upgrade_occurred_this_cycle": False,
            },
            "hardware": hardware or {},
            "config": config or {},
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

        self.save(state)
        return state

    # ── 加载 ─────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """从磁盘加载状态文件。

        Returns:
            完整的 state 字典。

        Raises:
            StateFileNotFoundError: 文件不存在。
            StateCorruptedError: 文件损坏。
        """
        if not self.state_file.exists():
            raise StateFileNotFoundError(
                f"状态文件不存在: {self.state_file}"
            )

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise StateCorruptedError(
                f"状态文件 JSON 解析失败: {e}"
            ) from e
        except (OSError, PermissionError) as e:
            raise StateCorruptedError(
                f"状态文件读取失败: {e}"
            ) from e

        # 基本字段校验
        required = ["session_id", "phase", "convergence", "termination"]
        for key in required:
            if key not in data:
                raise StateSchemaValidationError(
                    f"状态缺少必需字段: {key}"
                )

        return data

    def exists(self) -> bool:
        """检查状态文件是否存在。

        Returns:
            True 如果 state.json 存在。
        """
        return self.state_file.exists()

    # ── 原子保存（4 步协议） ─────────────────────────────────

    def save(self, state: dict[str, Any]) -> None:
        """原子写入状态文件。

        4 步原子写入协议：
        1. 写入临时文件 state.json.tmp
        2. fsync 临时文件
        3. os.rename 原子替换
        4. fsync 目录（确保元数据持久化）

        Args:
            state: 要写入的状态字典。

        Raises:
            StateWriteError: 写入失败。
        """
        state["updated_at"] = _now_iso()

        tmp_path = self.state_file.with_suffix(".json.tmp")

        try:
            # Step 1: 写入临时文件
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                # Step 2: fsync
                os.fsync(f.fileno())

            # Step 3: 原子 rename
            os.replace(tmp_path, self.state_file)

            # Step 4: fsync 目录
            try:
                dir_fd = os.open(
                    str(self.state_dir), os.O_RDONLY
                )
                os.fsync(dir_fd)
                os.close(dir_fd)
            except (OSError, PermissionError):
                pass  # 部分文件系统不支持目录 fsync

        except (OSError, PermissionError, json.JSONEncodeError) as e:
            raise StateWriteError(
                f"状态文件写入失败: {e}"
            ) from e

    # ── 阶段转换 ──────────────────────────────────────────────

    def update_phase(
        self, state: dict[str, Any], new_phase: str
    ) -> dict[str, Any]:
        """更新状态阶段，校验转换合法性。

        Args:
            state: 当前状态字典（原地修改）。
            new_phase: 目标阶段。

        Returns:
            更新后的 state 字典。

        Raises:
            InvalidPhaseTransitionError: 非法阶段转换。
        """
        current = state.get("phase", "init")

        allowed = _VALID_PHASE_TRANSITIONS.get(current, set())
        if new_phase not in allowed:
            raise InvalidPhaseTransitionError(
                f"非法阶段转换: {current} -> {new_phase}"
            )

        state["phase"] = new_phase
        state["updated_at"] = _now_iso()
        return state

    # ── Turn 计数器更新 ──────────────────────────────────────

    def update_turn_counters(
        self,
        state: dict[str, Any],
        turn: int,
        invocations: int = 0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """更新 Turn 相关的计数统计。

        Args:
            state: 状态字典。
            turn: 当前 turn 编号。
            invocations: 本次工具调用次数。
            tokens_prompt: 本次 prompt tokens。
            tokens_completion: 本次 completion tokens。
            duration_ms: 本次耗时。

        Returns:
            更新后的 state。
        """
        hk = state.setdefault("housekeeping", {})
        hk["turn_count"] = turn
        hk["invocation_count"] = hk.get("invocation_count", 0) + invocations
        hk["tokens_prompt_total"] = (
            hk.get("tokens_prompt_total", 0) + tokens_prompt
        )
        hk["tokens_completion_total"] = (
            hk.get("tokens_completion_total", 0) + tokens_completion
        )
        hk["total_duration_ms"] = (
            hk.get("total_duration_ms", 0) + duration_ms
        )
        state["updated_at"] = _now_iso()
        return state

    # ── 容错状态更新 ─────────────────────────────────────────

    def update_fault_tolerance(
        self, state: dict[str, Any], ft_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """更新容错状态快照。

        Args:
            state: 状态字典。
            ft_snapshot: 容错状态字典，字段覆盖现有值。

        Returns:
            更新后的 state。
        """
        ft = state.setdefault("fault_tolerance", {})
        for key, value in ft_snapshot.items():
            ft[key] = value
        state["updated_at"] = _now_iso()
        return state

    # ── 收敛状态更新 ─────────────────────────────────────────

    def update_convergence(
        self, state: dict[str, Any], conv_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """更新收敛状态快照。

        Args:
            state: 状态字典。
            conv_snapshot: 收敛字段字典。

        Returns:
            更新后的 state。
        """
        conv = state.setdefault("convergence", {})
        for key, value in conv_snapshot.items():
            conv[key] = value
        state["updated_at"] = _now_iso()
        return state

    # ── 终止状态更新 ─────────────────────────────────────────

    def update_termination(
        self,
        state: dict[str, Any],
        status: str,
        exit_reason: str = "",
        summary: Optional[str] = None,
        verify_cmd: Optional[str] = None,
    ) -> dict[str, Any]:
        """更新终止状态。

        Args:
            state: 状态字典。
            status: 终止状态 (complete/limit_reached/error_loop/...)。
            exit_reason: 退出原因描述。
            summary: 任务完成摘要。
            verify_cmd: 验证命令。

        Returns:
            更新后的 state。
        """
        term = state.setdefault("termination", {})
        term["status"] = status
        term["exit_reason"] = exit_reason
        if summary is not None:
            term["summary"] = summary
        if verify_cmd is not None:
            term["verification_command"] = verify_cmd
        state["updated_at"] = _now_iso()
        return state

    # ── Issue 管理 ───────────────────────────────────────────

    def add_issue(
        self, state: dict[str, Any], issue: dict[str, Any]
    ) -> dict[str, Any]:
        """添加一条 issue (P0/P1/P2)，自动去重。

        Args:
            state: 状态字典。
            issue: issue 字典，需含 id/severity/description。

        Returns:
            更新后的 state。
        """
        issues = state.setdefault("issues", {})
        active = issues.setdefault("active", {"p0": [], "p1": [], "p2": []})

        severity = issue.get("severity", "P2").lower()
        sev_key = f"p{severity[-1]}" if severity.startswith("p") else "p2"

        # 去重：同 id 不重复添加
        existing_ids = {i.get("id") for i in active.get(sev_key, [])}
        if issue.get("id") in existing_ids:
            return state

        active.setdefault(sev_key, []).append(issue)
        issues[f"total_{sev_key}_triggered"] = (
            issues.get(f"total_{sev_key}_triggered", 0) + 1
        )
        state["updated_at"] = _now_iso()
        return state

    def add_artifact(
        self,
        state: dict[str, Any],
        path: str,
        turn: int,
        artifact_type: str = "created",
        status: str = "normal",
        size_bytes: int = 0,
        sha256: str = "",
    ) -> dict[str, Any]:
        """添加一条 artifact 记录。

        Args:
            state: 状态字典。
            path: 文件路径。
            turn: 产生该 artifact 的 turn 编号。
            artifact_type: 类型 (created/modified)。
            status: 状态 (normal/degraded)。
            size_bytes: 文件大小。
            sha256: 文件 SHA256 摘要。

        Returns:
            更新后的 state。
        """
        artifacts = state.setdefault("artifacts", [])
        record = {
            "path": path,
            "turn": turn,
            "type": artifact_type,
            "status": status,
            "size_bytes": size_bytes,
            "sha256": sha256 or self._compute_sha256(path),
            "timestamp": _now_iso(),
        }

        # 更新已有记录（同文件）
        for i, existing in enumerate(artifacts):
            if existing.get("path") == path:
                artifacts[i] = record
                state["updated_at"] = _now_iso()
                return state

        artifacts.append(record)
        # 更新 modified_files_summary
        mfs = state.setdefault("modified_files_summary", [])
        if path not in mfs:
            mfs.append(path)

        state["updated_at"] = _now_iso()
        return state

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        """计算文件的 SHA256 摘要。

        Args:
            file_path: 文件路径。

        Returns:
            十六进制 SHA256 字符串，失败返回空字符串。
        """
        try:
            sha = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except (OSError, FileNotFoundError):
            return ""
