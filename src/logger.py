"""
loop-ollama 日志模块。

提供结构化日志记录能力，输出到 runs.log 文件。
支持分级日志、turn 生命周期事件、容错事件追踪。

Classes:
    Logger: 日志管理器，单例模式。
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class Logger:
    """结构化日志管理器。

    Attributes:
        log_dir: 日志文件存放目录。
        log_file: runs.log 文件完整路径。
        level: 日志级别 ("DEBUG" / "INFO" / "WARN" / "ERROR")。
    """

    _instance: Optional["Logger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        """单例模式 —— 全局唯一 Logger 实例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: Optional[str] = None,
        level: str = "INFO",
    ) -> None:
        """初始化 Logger。

        Args:
            log_dir: 日志目录路径。默认 ~/.loop-ollama/logs 。
            level: 最低记录级别。DEBUG / INFO / WARN / ERROR。
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        if log_dir is None:
            self.log_dir = Path(
                os.path.expanduser("~/.loop-ollama/logs")
            )
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "runs.log"
        self.level = level.upper()
        self._levels = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
        self._write_lock = threading.Lock()

    # ── 核心日志方法 ──────────────────────────────────────────

    def log_event(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """记录一条结构化日志事件。

        Args:
            level: 日志级别 (DEBUG/INFO/WARN/ERROR)。
            message: 日志消息。
            **kwargs: 附加的结构化数据字段。
        """
        if self._levels.get(level.upper(), 0) < self._levels.get(
            self.level, 20
        ):
            return

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "message": message,
        }
        entry.update(kwargs)

        with self._write_lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except (OSError, PermissionError):
                pass  # 日志写入失败不应中断主流程

    def debug(self, message: str, **kwargs: Any) -> None:
        """记录 DEBUG 级别日志。"""
        self.log_event("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """记录 INFO 级别日志。"""
        self.log_event("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        """记录 WARN 级别日志。"""
        self.log_event("WARN", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """记录 ERROR 级别日志。"""
        self.log_event("ERROR", message, **kwargs)

    # ── 专用事件记录 ─────────────────────────────────────────

    def log_turn(
        self,
        turn: int,
        tool: str,
        result: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> None:
        """记录一个 ReAct Turn 事件。

        Args:
            turn: 当前 turn 编号。
            tool: 执行的工具名称。
            result: 工具执行结果摘要。
            duration_ms: 本轮耗时（毫秒）。
        """
        self.log_event(
            "INFO",
            f"Turn {turn}: {tool}",
            turn=turn,
            tool=tool,
            result=result,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_tier1_repair(
        self, turn: int, pattern_id: str, snippet: str
    ) -> None:
        """记录 Tier-1 格式修复事件。

        Args:
            turn: 当前 turn 编号。
            pattern_id: 匹配的修复规则 ID (如 T1-001)。
            snippet: 被修复文本的截取片段。
        """
        self.log_event(
            "INFO",
            f"TIER1 turn={turn} pattern_id={pattern_id} "
            f"original_snippet={snippet[:80]}",
            turn=turn,
            pattern_id=pattern_id,
            snippet=snippet[:80],
        )

    def log_tier3_extraction(
        self, turn: int, rule_id: str, confidence: float
    ) -> None:
        """记录 Tier-3 启发式提取事件。

        Args:
            turn: 当前 turn 编号。
            rule_id: 匹配的启发式规则 ID (如 T3-001)。
            confidence: 提取置信度。
        """
        self.log_event(
            "WARN" if confidence < 0.70 else "INFO",
            f"TIER3 turn={turn} rule_id={rule_id} confidence={confidence:.2f}",
            turn=turn,
            rule_id=rule_id,
            confidence=confidence,
        )

    def log_error(
        self,
        exception: Exception,
        context: str = "",
        **kwargs: Any,
    ) -> None:
        """记录异常事件。

        Args:
            exception: 异常对象。
            context: 异常发生时的上下文描述。
        """
        self.log_event(
            "ERROR",
            f"{context}: {type(exception).__name__}: {exception}",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            context=context,
            **kwargs,
        )

    def log_model_event(
        self, event: str, model_name: str, **kwargs: Any
    ) -> None:
        """记录模型相关事件（加载/切换/unload/升级）。

        Args:
            event: 事件类型描述。
            model_name: 模型名称。
        """
        self.log_event(
            "INFO",
            f"MODEL {event}: {model_name}",
            model_event=event,
            model_name=model_name,
            **kwargs,
        )

    def log_convergence(
        self, turn: int, counter: int, event: str, **kwargs: Any
    ) -> None:
        """记录收敛控制事件。

        Args:
            turn: 当前 turn 编号。
            counter: 当前 convergence_counter 值。
            event: 收敛事件描述。
        """
        self.log_event(
            "INFO",
            f"CONVERGENCE turn={turn} counter={counter}: {event}",
            turn=turn,
            convergence_counter=counter,
            convergence_event=event,
            **kwargs,
        )
