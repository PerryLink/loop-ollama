"""
loop-ollama main.py 单元测试。

测试 _resolve_resource_path 路径解析（开发模式 / PyInstaller 模式）。
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import _resolve_resource_path


class TestResolveResourcePath:
    """_resolve_resource_path 路径解析测试"""

    def test_dev_mode_returns_absolute_path(self):
        """开发模式返回项目根目录下的绝对路径。"""
        result = _resolve_resource_path("regex_lib/patterns.json")
        assert os.path.isabs(result)
        assert "regex_lib" in result

    def test_dev_mode_handles_leading_slash(self):
        """处理前导斜杠。"""
        result = _resolve_resource_path("/regex_lib/patterns.json")
        assert os.path.isabs(result)
        assert "regex_lib" in result

    def test_frozen_mode_uses_meipass(self):
        """PyInstaller 单文件模式使用 _MEIPASS。"""
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "_MEIPASS", "/tmp/_meipass", create=True):
            result = _resolve_resource_path("data/file.json")
            assert result.startswith("/tmp/_meipass")
            assert "data/file.json" in result
