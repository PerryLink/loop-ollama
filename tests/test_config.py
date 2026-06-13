"""
loop-ollama Config 单元测试。

测试 Config 类的完整生命周期：默认值、加载、保存、校验、
路径展开、访问器方法。
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, _expand_path


# ── Tests: _expand_path ─────────────────────────────────────────


def test_expand_path_home():
    """展开 ~ 为用户主目录。"""
    result = _expand_path("~/test")
    assert str(result).endswith("test")
    assert "~" not in str(result)


def test_expand_path_absolute():
    """绝对路径直接返回。"""
    result = _expand_path("/tmp/test")
    assert result == Path("/tmp/test").resolve()


# ── Tests: Config 默认初始化 ────────────────────────────────────


def test_config_default_init(temp_dir):
    """使用临时目录加载默认配置。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data is not None
    assert "ollama" in cfg.data
    assert "agent" in cfg.data
    assert "safety" in cfg.data
    assert "paths" in cfg.data


def test_config_default_base_url(temp_dir):
    """默认 base_url 为 localhost:11434。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data["ollama"]["base_url"] == "http://localhost:11434"


def test_config_default_max_turns(temp_dir):
    """默认 max_turns 为 30。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data["agent"]["max_turns"] == 30


def test_config_default_auto_upgrade(temp_dir):
    """默认 auto_model_upgrade 为 True。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data["agent"]["auto_model_upgrade"] is True


def test_config_default_tier3_degraded_write(temp_dir):
    """默认 tier3_degraded_write 为 False（安全）。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data["safety"]["enable_tier3_degraded_write"] is False


def test_config_destructive_commands_blocked(temp_dir):
    """默认阻止危险命令列表。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    blocked = cfg.data["safety"]["destructive_commands_blocked"]
    assert "rm -rf" in blocked
    assert "DROP TABLE" in blocked
    assert "> /dev/sda" in blocked


def test_config_protected_paths(temp_dir):
    """默认保护敏感路径。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    paths = cfg.data["safety"]["protected_paths"]
    assert "~/.ssh" in paths
    assert "/etc/passwd" in paths


# ── Tests: Config 保存与重载 ────────────────────────────────────


def test_config_save_and_reload(temp_dir):
    """保存配置后重新加载，验证数据一致。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    cfg.data["ollama"]["base_url"] = "http://custom:8080"
    cfg.save()

    cfg2 = Config(config_path=config_path)
    cfg2.load()
    assert cfg2.data["ollama"]["base_url"] == "http://custom:8080"


def test_config_partial_override(temp_dir):
    """部分覆盖后缺失字段由默认值填充。"""
    config_path = os.path.join(temp_dir, "config.json")
    # 写入仅含 ollama.base_url 的配置文件
    partial = {"ollama": {"base_url": "http://test:9999"}}
    with open(config_path, "w") as f:
        json.dump(partial, f)

    cfg = Config(config_path=config_path)
    cfg.load()
    assert cfg.data["ollama"]["base_url"] == "http://test:9999"
    # 其余字段应为默认值
    assert cfg.data["agent"]["max_turns"] == 30
    assert "safety" in cfg.data


def test_config_default_path(temp_dir):
    """使用默认路径初始化。"""
    cfg = Config()
    # 默认路径应包含 ~/.loop-ollama/config.json
    assert "loop-ollama" in str(cfg.config_path)
    assert str(cfg.config_path).endswith("config.json")


# ── Tests: Config 校验 ──────────────────────────────────────────


def test_config_validate_known_keys(temp_dir):
    """已知必填字段通过校验。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    errors = cfg.validate()
    assert errors == []


def test_config_validate_missing_ollama(temp_dir):
    """缺少 ollama 段时返回错误。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    del cfg.data["ollama"]
    errors = cfg.validate()
    assert len(errors) > 0


def test_config_validate_invalid_max_turns(temp_dir):
    """max_turns 为负值返回错误。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    cfg.data["agent"]["max_turns"] = -5
    errors = cfg.validate()
    assert any("max_turns" in e.lower() for e in errors)


def test_config_get_method(temp_dir):
    """Config.get() 快捷访问。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    val = cfg.get("agent.max_turns")
    assert val == 30


def test_config_get_with_default(temp_dir):
    """Config.get() 未知键返回默认值。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    val = cfg.get("unknown.key", default=999)
    assert val == 999


def test_config_set_method(temp_dir):
    """Config.set() 写入并持久化。"""
    config_path = os.path.join(temp_dir, "config.json")
    cfg = Config(config_path=config_path)
    cfg.load()
    cfg.set("agent.max_turns", 50)
    assert cfg.data["agent"]["max_turns"] == 50

    # 重新加载验证
    cfg2 = Config(config_path=config_path)
    cfg2.load()
    assert cfg2.data["agent"]["max_turns"] == 50


# ── Tests: config 属性访问器 ────────────────────────────────────


class TestConfigProperties:
    """config 便捷属性测试"""

    def test_ollama_base_url(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_default_model(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert isinstance(cfg.default_model, str)

    def test_max_turns(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.max_turns == 30

    def test_state_dir(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.state_dir is not None

    def test_log_dir(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.log_dir is not None

    def test_artifacts_dir(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.artifacts_dir is not None

    def test_auto_model_upgrade(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.auto_model_upgrade is True

    def test_convergence_rounds(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.convergence_rounds >= 1

    def test_tier3_max_consecutive(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        assert cfg.tier3_max_consecutive >= 1


# ── Tests: config 校验边缘 ──────────────────────────────────────


class TestConfigValidationEdge:
    """config validate() 边缘情况"""

    def test_validate_invalid_tier3(self, temp_dir):
        """tier3_max_consecutive 为非整数时报错。"""
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        cfg.data["agent"]["tier3_max_consecutive"] = "abc"
        errors = cfg.validate()
        assert any("tier3" in e.lower() for e in errors)

    def test_validate_destructive_list_invalid(self, temp_dir):
        """destructive_commands_blocked 为非列表时报错。"""
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        cfg.data["safety"]["destructive_commands_blocked"] = "not_a_list"
        errors = cfg.validate()
        assert any("destructive" in e.lower() for e in errors)

    def test_set_creates_missing_keys(self, temp_dir):
        """set() 创建深层缺失键。"""
        config_path = os.path.join(temp_dir, "config.json")
        cfg = Config(config_path=config_path)
        cfg.load()
        cfg.set("new_section.sub_key", "value")
        assert cfg.data["new_section"]["sub_key"] == "value"
