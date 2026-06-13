"""
loop-ollama 工具注册表——7 个内置工具注册 + 执行调度引擎。

工具清单（Ollama tool_calls 兼容格式）：
    read_file, write_file, edit_file, run_command,
    glob_search, grep_search, task_complete

设计原则：
    - 工具定义与执行分离（定义发送给 Ollama，执行在本模块调度）
    - 每个工具独立实现函数，通过注册表映射调用
    - 所有工具输出结构化 ToolResult

Classes:
    ToolRegistry: 工具注册表与执行引擎。
    ToolResult: 工具执行结果数据类。
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ── 数据类 ─────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        tool_name: 工具名称。
        content: 输出内容（给模型阅读的文本）。
        error: 错误信息（None 表示成功）。
        exit_code: 子进程退出码（仅 run_command）。
        stdout: 标准输出（仅 run_command）。
        stderr: 标准错误（仅 run_command）。
    """

    tool_name: str = ""
    content: str = ""
    error: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


# ── 工具定义（Ollama tool_calls 格式） ─────────────────────────


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，支持行范围限制",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"},
                    "offset": {"type": "integer", "description": "起始行号"},
                    "limit": {"type": "integer", "description": "读取行数"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建或覆写文件，自动创建父目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "精确字符串替换——old_string 必须在文件中唯一匹配",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"},
                    "old_string": {"type": "string", "description": "要替换的原字符串"},
                    "new_string": {"type": "string", "description": "替换后的字符串"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行系统命令，返回 stdout/stderr/exit_code",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "timeout_ms": {"type": "integer", "description": "超时毫秒数"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "按 glob 模式匹配文件列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob 模式"},
                    "path": {"type": "string", "description": "搜索起始目录"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "使用 ripgrep 搜索文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式"},
                    "path": {"type": "string", "description": "搜索路径"},
                    "glob": {"type": "string", "description": "文件名过滤（如 *.py）"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "标记任务完成——需提供完成摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "任务完成摘要"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ── 默认超时与限制 ──────────────────────────────────────────────

DEFAULT_BASH_TIMEOUT_MS = 60000
DEFAULT_READ_MAX_LINES = 2000
DEFAULT_GLOB_MAX_RESULTS = 500
DEFAULT_GREP_MAX_RESULTS = 250
BASH_MAX_OUTPUT_BYTES = 51200


# ── 工具实现函数 ─────────────────────────────────────────────────


def _read_file(file_path: str, offset: int = 0, limit: int = 0) -> ToolResult:
    """读取文件内容。"""
    result = ToolResult(tool_name="read_file")
    try:
        p = Path(file_path).resolve()
        if not p.exists():
            result.error = f"文件不存在: {p}"
            result.content = f"[ERROR] 文件不存在: {p}"
            return result
        if p.is_dir():
            ents = sorted(p.iterdir(), key=lambda e: e.name)
            result.content = "\n".join(
                f"{i+1:4d}\t{e.name}" for i, e in enumerate(ents)
            ) or "(空目录)"
            return result
        # 二进制检测
        raw = p.read_bytes()
        null_count = raw[:1024].count(0)
        if null_count > len(raw[:1024]) * 0.1 and len(raw) > 512:
            result.content = f"[二进制文件: {len(raw)} bytes]"
            return result
        lines = raw.decode("utf-8", errors="replace").split("\n")
        eff_limit = min(limit, DEFAULT_READ_MAX_LINES) if limit > 0 else DEFAULT_READ_MAX_LINES
        start = max(0, offset)
        end = min(len(lines), start + eff_limit)
        result.content = "\n".join(
            f"{i+1:6d}\t{line}" for i, line in enumerate(lines[start:end], start)
        )
        if end < len(lines):
            result.content += f"\n\n[已截断: 共{len(lines)}行, 显示{start+1}-{end}]"
    except Exception as e:
        result.error = str(e)
        result.content = f"[ERROR] {e}"
    return result


def _write_file(file_path: str, content: str) -> ToolResult:
    """覆写文件。"""
    result = ToolResult(tool_name="write_file")
    if not file_path:
        result.error = "file_path 不能为空"
        result.content = "[ERROR] file_path 不能为空"
        return result
    try:
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        p.write_bytes(data)
        result.content = f"[OK] 已写入 {p}（{len(data)} bytes）"
    except Exception as e:
        result.error = str(e)
        result.content = f"[ERROR] {e}"
    return result


def _edit_file(file_path: str, old_str: str, new_str: str) -> ToolResult:
    """精确字符串替换。"""
    result = ToolResult(tool_name="edit_file")
    if not file_path:
        result.error = "file_path 不能为空"
        result.content = "[ERROR] file_path 不能为空"
        return result
    if not old_str:
        result.error = "old_string 不能为空"
        result.content = "[ERROR] old_string 不能为空"
        return result
    try:
        p = Path(file_path).resolve()
        if not p.exists():
            result.error = f"文件不存在: {p}"
            result.content = f"[ERROR] 文件不存在: {p}"
            return result
        text = p.read_text("utf-8")
        count = text.count(old_str)
        if count == 0:
            # 尝试去缩进匹配
            trimmed = old_str.lstrip()
            if trimmed and text.count(trimmed) == 1:
                new_text = text.replace(trimmed, new_str)
                p.write_text(new_text, "utf-8")
                result.content = f"[OK] 在 {p} 中替换 1 处（去缩进匹配）"
                return result
            result.error = f'未找到匹配 "{old_str[:60]}"'
            result.content = f"[ERROR] {result.error}"
            return result
        if count > 1:
            result.error = f"找到 {count} 处匹配，old_string 必须唯一"
            result.content = f"[ERROR] {result.error}"
            return result
        new_text = text.replace(old_str, new_str)
        p.write_text(new_text, "utf-8")
        result.content = f"[OK] 在 {p} 中替换 1 处"
    except Exception as e:
        result.error = str(e)
        result.content = f"[ERROR] {e}"
    return result


def _run_command(command: str, timeout_ms: int = DEFAULT_BASH_TIMEOUT_MS) -> ToolResult:
    """执行系统命令。"""
    result = ToolResult(tool_name="run_command")
    if not command:
        result.error = "command 不能为空"; result.content = "[ERROR] command 不能为空"
        return result
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True,
            timeout=timeout_ms / 1000.0, text=True,
        )
        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        parts = []
        if proc.stdout:
            parts.append(proc.stdout[-5000:])
        if proc.stderr:
            parts.append(f"\n[stderr]:\n{proc.stderr[-2000:]}")
        result.content = "".join(parts) or f"(exit: {proc.returncode})"
    except subprocess.TimeoutExpired:
        result.error = f"命令超时 ({timeout_ms}ms)"
        result.content = f"[ERROR] {result.error}"
    except Exception as e:
        result.error = str(e)
        result.content = f"[ERROR] {e}"
    return result


def _glob_search(pattern: str, path_str: str = ".") -> ToolResult:
    """Glob 文件匹配。"""
    result = ToolResult(tool_name="glob_search")
    if not pattern:
        result.error = "pattern 不能为空"; result.content = "[ERROR] pattern 不能为空"
        return result
    try:
        base = Path(path_str).resolve()
        matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        sliced = matches[:DEFAULT_GLOB_MAX_RESULTS]
        lines = [str(m.relative_to(base)) for m in sliced]
        if len(sliced) < len(matches):
            lines.append(f"\n[已截断: {len(matches)} 结果, 显示前 {DEFAULT_GLOB_MAX_RESULTS}]")
        result.content = "\n".join(lines) or "(无匹配)"
    except Exception as e:
        result.error = str(e); result.content = f"[ERROR] {e}"
    return result


def _grep_search(pattern: str, path_str: str = ".", glob_filter: str = "") -> ToolResult:
    """ripgrep 搜索。"""
    result = ToolResult(tool_name="grep_search")
    if not pattern:
        result.error = "pattern 不能为空"; result.content = "[ERROR] pattern 不能为空"
        return result
    try:
        cmd = ["rg", "--files-with-matches", "--no-heading"]
        if glob_filter:
            cmd.extend(["--glob", glob_filter])
        cmd.extend(["--", pattern, path_str])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            lines = proc.stdout.strip().split("\n")
            if len(lines) > DEFAULT_GREP_MAX_RESULTS:
                result.content = "\n".join(lines[:DEFAULT_GREP_MAX_RESULTS]) + \
                    f"\n\n[已截断: {len(lines)} 结果, 显示前 {DEFAULT_GREP_MAX_RESULTS}]"
            else:
                result.content = proc.stdout.strip()
        elif proc.returncode == 1:
            result.content = "(无匹配)"
        elif "not found" in proc.stderr.lower():
            result.error = "ripgrep (rg) 未安装"; result.content = f"[ERROR] {result.error}"
        else:
            result.error = proc.stderr[:500]; result.content = f"[ERROR] {proc.stderr[:500]}"
    except FileNotFoundError:
        result.error = "ripgrep (rg) 未安装"; result.content = f"[ERROR] {result.error}"
    except Exception as e:
        result.error = str(e); result.content = f"[ERROR] {e}"
    return result


def _task_complete(summary: str = "") -> ToolResult:
    """标记任务完成。"""
    result = ToolResult(tool_name="task_complete")
    result.content = f"[TASK_COMPLETE] {summary}" if summary else "[TASK_COMPLETE]"
    return result


# ── 工具执行映射表 ────────────────────────────────────────────────

_TOOL_MAP: dict[str, Callable[..., ToolResult]] = {
    "read_file": lambda **kw: _read_file(
        kw.get("file_path", ""), kw.get("offset", 0), kw.get("limit", 0)),
    "write_file": lambda **kw: _write_file(
        kw.get("file_path", ""), kw.get("content", "")),
    "edit_file": lambda **kw: _edit_file(
        kw.get("file_path", ""), kw.get("old_string", ""), kw.get("new_string", "")),
    "run_command": lambda **kw: _run_command(
        kw.get("command", ""), kw.get("timeout_ms", DEFAULT_BASH_TIMEOUT_MS)),
    "glob_search": lambda **kw: _glob_search(
        kw.get("pattern", ""), kw.get("path", ".")),
    "grep_search": lambda **kw: _grep_search(
        kw.get("pattern", ""), kw.get("path", "."), kw.get("glob", "")),
    "task_complete": lambda **kw: _task_complete(kw.get("summary", "")),
}


# ── ToolRegistry 类 ───────────────────────────────────────────────


class ToolRegistry:
    """工具注册表与执行调度引擎。

    提供：
        - 工具定义导出（给 Ollama 的 tools 参数）
        - 工具执行调度（按名称查找实现并调用）
        - 工具列表查询
    """

    @staticmethod
    def get_definitions() -> list[dict[str, Any]]:
        """获取所有已注册工具的定义列表（Ollama tool_calls 格式）。"""
        return TOOL_DEFINITIONS

    @staticmethod
    def get_names() -> list[str]:
        """获取已注册工具名称列表。"""
        return list(_TOOL_MAP.keys())

    @staticmethod
    def execute(name: str, arguments: dict[str, Any]) -> ToolResult:
        """按名称执行工具。

        Args:
            name: 工具名称。
            arguments: 参数字典。

        Returns:
            ToolResult 执行结果。

        Raises:
            KeyError: 未知工具名。
        """
        fn = _TOOL_MAP.get(name)
        if fn is None:
            raise KeyError(
                f'未知工具: "{name}"（已注册: {", ".join(_TOOL_MAP.keys())}）'
            )
        return fn(**arguments)
