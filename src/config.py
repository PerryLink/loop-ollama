"""
loop-ollama 配置管理模块。

负责加载、保存和验证 ~/.loop-ollama/config.json 配置文件。
提供默认值以支持首次运行无需手动配置。

Classes:
    Config: 配置管理器，提供完整的配置读写与校验能力。
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

# ── 默认配置 ──────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "default_model": "",
        "keep_alive": -1,
        "request_timeout_ms": None,
        "_comment_timeout": "null = 使用动态超时。设具体值(如 120000)则使用固定超时。",
    },
    "agent": {
        "max_turns": 30,
        "tier1_enabled": True,
        "tier2_max_retries": 3,
        "tier3_max_consecutive": 5,
        "auto_model_upgrade": True,
        "convergence_rounds": 2,
        "dynamic_timeout_enabled": True,
    },
    "safety": {
        "destructive_commands_blocked": [
            "rm -rf", "> /dev/sda", "mkfs.", "dd if=",
            ":(){ :|:& };:", "chmod 777 /", "DROP TABLE",
            "DROP DATABASE",
        ],
        "protected_paths": [
            "~/.ssh", "~/.gnupg", "/etc/passwd",
            "/etc/shadow", "~/.ollama",
        ],
        "enable_tier3_degraded_write": False,
    },
    "paths": {
        "state_dir": "~/.loop-ollama/state",
        "artifacts_dir": "~/.loop-ollama/state/artifacts",
        "log_dir": "~/.loop-ollama/logs",
    },
}


def _expand_path(path_str: str) -> Path:
    """展开 ~ 为用户主目录，返回绝对 Path 对象。"""
    return Path(os.path.expanduser(path_str)).resolve()


class Config:
    """loop-ollama 配置管理器。

    从 ~/.loop-ollama/config.json 加载用户配置，
    缺失字段用内置默认值填充，支持运行时保存与校验。

    Attributes:
        config_path: 配置文件完整路径。
        data: 加载后的配置字典（含默认值填充）。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """初始化 Config 实例。

        Args:
            config_path: 配置文件路径。
                默认 ~/.loop-ollama/config.json。
        """
        if config_path is None:
            self.config_path: Path = _expand_path(
                "~/.loop-ollama/config.json"
            )
        else:
            self.config_path = Path(config_path).resolve()
        self.data: dict[str, Any] = deepcopy(_DEFAULT_CONFIG)

    # ── 加载 / 保存 ──────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """从磁盘加载配置文件，缺失字段以默认值填充。

        Returns:
            合并默认值后的完整配置字典。

        Raises:
            FileNotFoundError: 配置文件不存在时，写入默认配置并返回。
            json.JSONDecodeError: 配置文件 JSON 解析失败。
        """
        if not self.config_path.exists():
            self.save()
            return self.data

        with open(self.config_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        self.data = self._deep_merge(deepcopy(_DEFAULT_CONFIG), user_data)
        return self.data

    def save(self) -> None:
        """将当前配置原子写入磁盘。

        确保配置文件目录存在后再写入。
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.config_path)
        # 目录 fsync (跨平台尽力)
        try:
            dir_fd = os.open(
                str(self.config_path.parent), os.O_RDONLY
            )
            os.fsync(dir_fd)
            os.close(dir_fd)
        except (OSError, PermissionError):
            pass

    # ── 校验 ─────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """验证当前配置的合法性。

        Returns:
            错误信息列表，空列表表示校验通过。
        """
        errors: list[str] = []

        # ollama 段
        o = self.data.get("ollama", {})
        if not o.get("base_url"):
            errors.append("ollama.base_url 不能为空")
        if not isinstance(o.get("keep_alive"), int):
            errors.append("ollama.keep_alive 必须为整数")

        # agent 段
        a = self.data.get("agent", {})
        if not isinstance(a.get("max_turns"), int) or a["max_turns"] < 1:
            errors.append("agent.max_turns 必须为 >=1 的整数")
        if not isinstance(a.get("tier3_max_consecutive"), int):
            errors.append("agent.tier3_max_consecutive 必须为整数")

        # safety 段
        s = self.data.get("safety", {})
        if not isinstance(s.get("destructive_commands_blocked"), list):
            errors.append("safety.destructive_commands_blocked 必须为列表")

        # paths 段
        p = self.data.get("paths", {})
        for key in ("state_dir", "artifacts_dir", "log_dir"):
            if key not in p:
                errors.append(f"paths.{key} 缺失")

        return errors

    # ── 便捷属性 ───────────────────────────────────────────────

    @property
    def ollama_base_url(self) -> str:
        """Ollama 服务基础 URL。"""
        return self.data["ollama"]["base_url"]

    @property
    def default_model(self) -> str:
        """默认模型名称（空字符串表示未设置）。"""
        return self.data["ollama"]["default_model"]

    @property
    def max_turns(self) -> int:
        """最大 ReAct 轮次。"""
        return self.data["agent"]["max_turns"]

    @property
    def state_dir(self) -> Path:
        """状态文件目录。"""
        return _expand_path(self.data["paths"]["state_dir"])

    @property
    def log_dir(self) -> Path:
        """日志文件目录。"""
        return _expand_path(self.data["paths"]["log_dir"])

    @property
    def artifacts_dir(self) -> Path:
        """产物文件目录。"""
        return _expand_path(self.data["paths"]["artifacts_dir"])

    @property
    def auto_model_upgrade(self) -> bool:
        """是否启用模型自动升级。"""
        return self.data["agent"]["auto_model_upgrade"]

    @property
    def convergence_rounds(self) -> int:
        """收敛判定所需轮次数。"""
        return self.data["agent"]["convergence_rounds"]

    @property
    def tier3_max_consecutive(self) -> int:
        """Tier-3 连续降级上限。"""
        return self.data["agent"]["tier3_max_consecutive"]

    # ── 辅助方法 ──────────────────────────────────────────────

    @staticmethod
    def _deep_merge(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """递归合并两个字典，override 的值覆盖 base。

        Args:
            base: 基础字典。
            override: 覆盖字典。

        Returns:
            合并后的新字典。
        """
        result = deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """按点号分隔的路径获取配置值。

        Args:
            key_path: 点号分隔的键路径，如 "ollama.base_url"。
            default: 找不到时的默认值。

        Returns:
            配置值或默认值。
        """
        keys = key_path.split(".")
        node: Any = self.data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, key_path: str, value: Any) -> None:
        """按点号分隔的路径设置配置值。

        Args:
            key_path: 点号分隔的键路径。
            value: 要设置的值。
        """
        keys = key_path.split(".")
        node = self.data
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        self.save()
