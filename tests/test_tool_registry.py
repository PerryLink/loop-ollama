"""
tool_registry 单元测试

测试 7 个工具的注册、定义导出和执行调度。
"""

import json
import pytest
from pathlib import Path

# 由于 linter 已重写 tool_registry，导入实际模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import ToolRegistry, ToolResult


class TestToolDefinitions:
    """工具定义测试"""

    def test_get_definitions_returns_7_tools(self):
        """验证注册了 7 个工具"""
        defs = ToolRegistry.get_definitions()
        assert len(defs) == 7

    def test_all_definitions_have_name(self):
        """每个工具定义都有 name"""
        for d in ToolRegistry.get_definitions():
            assert "function" in d
            assert "name" in d["function"]
            assert d["function"]["name"]

    def test_required_tools_present(self):
        """验证必须的工具都在注册表中"""
        names = ToolRegistry.get_names()
        required = {"read_file", "write_file", "edit_file",
                     "run_command", "glob_search", "grep_search", "task_complete"}
        assert required.issubset(set(names))


class TestToolExecution:
    """工具执行测试"""

    def test_read_file_nonexistent(self):
        """读取不存在的文件应返回错误"""
        result = ToolRegistry.execute("read_file", {"file_path": "/nonexistent/path_12345"})
        assert result.tool_name == "read_file"
        assert result.error is not None

    def test_read_file_directory(self, tmp_path):
        """读取目录应列出入条目"""
        d = tmp_path / "testdir"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        result = ToolRegistry.execute("read_file", {"file_path": str(d)})
        assert "a.txt" in result.content

    def test_write_file_creates_file(self, tmp_path):
        """write_file 应创建文件"""
        p = tmp_path / "new_file.txt"
        result = ToolRegistry.execute("write_file", {
            "file_path": str(p),
            "content": "test content",
        })
        assert "[OK]" in result.content or result.error is None
        assert p.exists()

    def test_edit_file_unique_match(self, tmp_path):
        """edit_file 应替换唯一匹配"""
        p = tmp_path / "edit_test.txt"
        p.write_text("hello world")
        result = ToolRegistry.execute("edit_file", {
            "file_path": str(p),
            "old_string": "hello",
            "new_string": "hi",
        })
        assert "[OK]" in result.content or result.error is None
        assert p.read_text() == "hi world"

    def test_edit_file_multiple_matches(self, tmp_path):
        """edit_file 多处匹配应报错"""
        p = tmp_path / "multi.txt"
        p.write_text("hello hello")
        result = ToolRegistry.execute("edit_file", {
            "file_path": str(p),
            "old_string": "hello",
            "new_string": "hi",
        })
        assert result.error is not None

    def test_run_command_echo(self):
        """run_command 应执行简单命令"""
        result = ToolRegistry.execute("run_command", {"command": "echo hello_test"})
        assert "hello_test" in result.content

    def test_task_complete(self):
        """task_complete 应返回完成标记"""
        result = ToolRegistry.execute("task_complete", {"summary": "done"})
        assert result.tool_name == "task_complete"

    def test_unknown_tool_raises(self):
        """未知工具应抛出 KeyError"""
        with pytest.raises(KeyError):
            ToolRegistry.execute("unknown_tool", {})
