"""
五层安全护栏 —— 逐层检查工具调用中的危险操作。

每层护栏负责拦截不同类别的危险操作，从灾难级到路径级逐层升级。
L0 为最高优先级（拦截不可逆破坏），L4 为最低（路径访问控制）。

层级定义：
    L0 - 灾难操作 (CATASTROPHIC):
        不可逆系统破坏命令。直接拦截，绝不允许执行。
        黑名单: rm -rf /, dd if=/dev/zero, :(){:|:&};:, chmod 777 /,
                mkfs.*, >/dev/sda, rm -rf /*, DROP TABLE/DATABASE

    L1 - 系统危险操作 (SYSTEM_CRITICAL):
        影响系统运行状态的操作。在 safe_mode 下拦截。
        黑名单: sudo, poweroff, reboot, shutdown, init 0/6,
                systemctl stop/disable, killall, pkill

    L2 - 文件系统危险操作 (FILESYSTEM_DANGER):
        可能导致数据丢失或权限失控的文件操作。
        黑名单: chmod -R 777 /, chown -R, rm -rf /etc /var /usr /boot,
                chmod 777 /etc/*, chown root:root

    L3 - 网络安全操作 (NETWORK_DANGER):
        通过管道执行远程代码的危险网络操作。
        黑名单: curl ... | bash, wget ... | sh, /dev/tcp 反向 shell,
                nc -e /bin/bash, python -c 远程执行

    L4 - 路径保护 (PATH_PROTECTION):
        保护系统和安全敏感路径。在 safe_mode 下激活。
        保护路径: /etc/, /boot/, /sys/, /proc/, /dev/,
                  ~/.ssh, ~/.gnupg, ~/.ollama, /root,
                  C:/Windows/, C:/System32/, C:/Program Files/

安全等级 (safe_mode):
    - safe_mode=True: 启用全部 5 层 (L0-L4)
    - safe_mode=False + auto_mode=True: 取决于模型等级
        S/A: L0-L4, B: L0-L2, C/D: L0-L1
    - safe_mode=False + auto_mode=False: 仅 L0

Classes:
    GuardLayer: 五层安全护栏。
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any, Optional

from .logger import Logger


# ═══════════════════════════════════════════════════════════════════
# L0: 灾难操作 —— 不可逆系统破坏
# ═══════════════════════════════════════════════════════════════════

_L0_CATASTROPHIC_PATTERNS: list[str] = [
    # UNIX 灾难命令
    r'\brm\s+-rf\s+/(?:\s|$|\*)',          # rm -rf / 或 rm -rf /*
    r'\brm\s+-rf\s+/\*',                     # rm -rf /*
    r'\brm\s+-rf\s+--no-preserve-root\s+/', # rm -rf --no-preserve-root /
    r'\bdd\s+if=/dev/zero\s+of=/dev/',       # dd 磁盘清零
    r'\bdd\s+if=/dev/urandom\s+of=/dev/',    # dd 随机数据覆盖
    r'\bmkfs\.\w*\s+/dev/',                  # mkfs 格式化
    r'\b>[\s]*/dev/sd[a-z]',                 # 输出重定向覆盖磁盘
    r'\b>[\s]*/dev/nvme',                    # 输出重定向覆盖 NVMe
    r':\(\)\s*\{[^}]*:\|',                   # Fork 炸弹
    r':\(\)\s*\{[^}]*\|[^}]*&\}',            # Fork 炸弹变体
    r'\bchmod\s+777\s+/',                     # chmod 777 根目录
    r'\bchmod\s+-R\s+000\s+/',               # chmod 000 根目录
    r'\bchmod\s+777\s+/\*',                   # chmod 777 根目录下所有
    # SQL 危险操作
    r'\bDROP\s+TABLES?\b',                    # DROP TABLE
    r'\bDROP\s+DATABASES?\b',                 # DROP DATABASE
    r'\bTRUNCATE\s+TABLES?\b',               # TRUNCATE TABLE
    # Windows 灾难命令
    r'\bformat\s+[A-Z]:\s*/',                # format C: /
    r'\bdel\s+/f\s+/s\s+/q\s+C:\\',          # del C:\ 递归删除
    r'\brmdir\s+/s\s+/q\s+C:\\',             # rmdir C:\ 递归删除
]


# ═══════════════════════════════════════════════════════════════════
# L1: 系统危险操作 —— 影响运行状态
# ═══════════════════════════════════════════════════════════════════

_L1_SYSTEM_PATTERNS: list[str] = [
    r'\bsudo\s+rm\b',                        # sudo rm
    r'\bsudo\s+chmod\b',                     # sudo chmod
    r'\bsudo\s+chown\b',                     # sudo chown
    r'\bsudo\s+-i\b',                         # sudo -i root shell
    r'\bsudo\s+su\b',                         # sudo su
    r'\bpoweroff\b',                          # 关机
    r'\breboot\b',                            # 重启
    r'\bshutdown\b',                          # shutdown 命令
    r'\binit\s+[06]\b',                       # init 0/6
    r'\bsystemctl\s+(stop|disable|mask)\b',   # systemctl 停止/禁用服务
    r'\bkillall\s+-9\b',                      # killall -9
    r'\bpkill\s+-9\b',                        # pkill -9
    r'\bkill\s+-9\s+1\b',                     # kill -9 1 (PID 1)
    r'\bkill\s+-9\s+-1\b',                    # kill -9 -1 (所有进程)
    r'\bhalt\b',                              # halt
    r'\btelinit\s+[06]\b',                    # telinit 0/6
]


# ═══════════════════════════════════════════════════════════════════
# L2: 文件系统危险操作 —— 数据丢失/权限失控
# ═══════════════════════════════════════════════════════════════════

_L2_FILESYSTEM_PATTERNS: list[str] = [
    r'\bchmod\s+-R\s+777\b',                 # chmod -R 777 递归
    r'\bchown\s+-R\b',                        # chown -R 递归改所有者
    r'\bchown\s+root:root\b',                 # chown root:root
    r'\brm\s+-rf\s+/(?:etc|var|usr|boot|opt|srv|sys|proc|dev|root|tmp|home)(?:/|\s|$|\*)',
    r'\brm\s+-rf\s+~',                        # rm -rf ~ (危险操作，可能删除整个 home)
    r'\brm\s+-rf\s+\$HOME',                   # rm -rf $HOME
    r'\brm\s+-rf\s+\.(\s|$)',                # rm -rf . (危险但有时合法)
    r'\bchmod\s+777\s+/(?:etc|var|usr|boot)', # chmod 777 系统目录
    r'\bchmod\s+-R\s+o\+w\s+/',              # chmod o+w 递归根目录
    r'\bmv\s+/\S+\s+/dev/null',               # mv 文件到 /dev/null
]


# ═══════════════════════════════════════════════════════════════════
# L3: 网络危险操作 —— 远程代码执行
# ═══════════════════════════════════════════════════════════════════

_L3_NETWORK_PATTERNS: list[str] = [
    r'\bcurl\b.*\|\s*(ba)?sh\b',             # curl | bash
    r'\bcurl\b.*\|\s*sudo\s*(ba)?sh\b',      # curl | sudo bash
    r'\bwget\b.*-O\s*-\s*\|\s*(ba)?sh\b',    # wget -O - | sh
    r'\bwget\b.*\|\s*(ba)?sh\b',             # wget | sh
    r'\bcurl\b.*\|\s*(python|perl|ruby|node)\b', # curl | python/perl/ruby/node
    r'\b/dev/tcp/',                           # bash /dev/tcp 反向 shell
    r'\bnc\s+.*-e\s+/bin/',                   # nc -e /bin/bash
    r'\bnc\s+.*-c\s+(ba)?sh\b',               # nc -c bash
    r'\bncat\s+.*-e\s+/bin/',                 # ncat -e
    r'\bpyth?on\d?\s+-c\s+.*socket',          # python -c socket 反向 shell
    r'\bexec\s+\d+<>/dev/tcp/',               # exec fd<>/dev/tcp
    r'\bbash\s+-i\s+>&\s+/dev/tcp/',          # bash -i >& /dev/tcp
    r'\bscp\s+.*\S+@\S+:\S+\s+/etc/',         # scp 覆盖系统文件
]


# ═══════════════════════════════════════════════════════════════════
# L4: 路径保护 —— 敏感文件/目录
# ═══════════════════════════════════════════════════════════════════

_L4_PROTECTED_PATHS: list[str] = [
    # Unix/Linux 系统关键路径
    '/etc/passwd', '/etc/shadow', '/etc/sudoers',
    '/etc/ssh/', '/etc/ssl/', '/etc/crontab',
    '/etc/fstab', '/etc/hosts', '/etc/resolv.conf',
    '/etc/group', '/etc/gshadow', '/etc/security/',
    '/boot/', '/boot/efi/', '/boot/grub/',
    '/sys/', '/proc/', '/dev/',
    # 用户安全数据
    '~/.ssh/', '~/.gnupg/', '~/.ollama/',
    '~/.aws/', '~/.config/gcloud/', '~/.azure/',
    '~/.git-credentials', '~/.netrc',
    # 系统二进制目录
    '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
    '/usr/local/bin/', '/usr/local/sbin/',
    # 系统库
    '/lib/', '/lib64/', '/usr/lib/', '/usr/lib64/',
    '/usr/lib/systemd/',
    # root 目录
    '/root/',
    # Windows 路径
    'C:\\Windows\\', 'C:\\Windows\\System32\\',
    'C:\\Windows\\SysWOW64\\',
    'C:\\Program Files\\', 'C:\\Program Files (x86)\\',
    'C:\\ProgramData\\',
    # Windows 注册表
    'HKEY_LOCAL_MACHINE', 'HKLM\\',
    'HKEY_CLASSES_ROOT', 'HKCR\\',
]


# ═══════════════════════════════════════════════════════════════════
# 工具名映射 —— 哪些工具的哪些参数应触发命令检查
# ═══════════════════════════════════════════════════════════════════

_COMMAND_TOOL_NAMES: set[str] = {
    'bash', 'shell', 'execute_command', 'run', 'exec',
    'cmd', 'terminal', 'sh', 'run_command', 'execute',
}

_COMMAND_PARAM_NAMES: tuple[str, ...] = (
    'command', 'cmd', 'script', 'code', 'input',
    'bash_command', 'shell_command', 'expression',
)

_FILE_PATH_PARAM_NAMES: tuple[str, ...] = (
    'file_path', 'path', 'filename', 'file', 'target', 'dest',
    'source', 'src', 'dst', 'output', 'input_file', 'output_file',
    'directory', 'dir', 'folder',
)

# 各层的可读名称和描述
_LAYER_DESCRIPTIONS: dict[str, str] = {
    'L0': '灾难操作: 系统不可逆损坏 (rm -rf /, dd, fork bomb, mkfs)',
    'L1': '系统危险操作: sudo/关机/重启/kill/killall',
    'L2': '文件系统危险操作: 递归改权(chmod -R)/chown/删除系统目录',
    'L3': '网络管道危险: curl|bash, wget|sh, /dev/tcp 反向shell',
    'L4': '受保护路径访问: /etc, /boot, /sys, /proc, ~/.ssh, C:\\Windows',
}


class GuardLayer:
    """五层安全护栏 —— 拦截工具调用中的危险操作。

    每层有独立的黑名单/白名单规则，支持 safe_mode 和 auto_mode。
    在 safe_mode 下启用全部 5 层；auto_mode 下根据模型等级动态启用层数；
    最低保障 L0 始终启用。

    Attributes:
        model_grade: 模型等级 (S/A/B/C/D)。
        safe_mode: 安全模式（True 启用全部 5 层）。
        auto_mode: 自动模式（根据模型等级决定启用层数）。
        l0_patterns: L0 编译后的正则列表。
        l1_patterns: L1 编译后的正则列表。
        l2_patterns: L2 编译后的正则列表。
        l3_patterns: L3 编译后的正则列表。
        l4_protected: L4 保护路径列表。
        log: Logger 实例。
        stats: 拦截统计。
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """初始化 GuardLayer。

        Args:
            config: 配置字典，支持:
                - model_grade: 模型等级 (默认 'A')
                - safe_mode: 安全模式 (默认 True)
                - auto_mode: 自动模式 (默认 False)
                - custom_blocked_commands: 自定义 L0 黑名单
                - custom_protected_paths: 自定义 L4 保护路径
                - allowlist_commands: 白名单命令（优先级高于黑名单）
                - allowlist_paths: 白名单路径
        """
        c: dict[str, Any] = config or {}
        self.model_grade: str = c.get('model_grade', 'A').upper()
        self.safe_mode: bool = c.get('safe_mode', True)
        self.auto_mode: bool = c.get('auto_mode', False)
        self.log: Logger = Logger()

        # 编译各层正则
        custom_commands: list[str] = c.get('custom_blocked_commands', [])
        self.l0_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in _L0_CATASTROPHIC_PATTERNS + custom_commands
        ]
        self.l1_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in _L1_SYSTEM_PATTERNS
        ]
        self.l2_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in _L2_FILESYSTEM_PATTERNS
        ]
        self.l3_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in _L3_NETWORK_PATTERNS
        ]

        # L4 保护路径
        custom_paths: list[str] = c.get('custom_protected_paths', [])
        self.l4_protected: list[str] = [
            p.lower().replace('\\', '/')
            for p in _L4_PROTECTED_PATHS + custom_paths
        ]

        # 白名单
        self.allowlist_commands: set[str] = set(
            c.get('allowlist_commands', [])
        )
        self.allowlist_paths: set[str] = set(
            p.lower().replace('\\', '/')
            for p in c.get('allowlist_paths', [])
        )

        # 拦截统计
        self.stats: dict[str, int] = {
            'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0,
            'allowed': 0,
        }

    # ── 确定启用的层级 ────────────────────────────────────────

    def _active_layers(self) -> list[str]:
        """根据配置确定当前启用的安全层级。

        Returns:
            启用的层级 ID 列表（按优先级排序）。
        """
        if self.safe_mode:
            return ['L0', 'L1', 'L2', 'L3', 'L4']

        # auto_mode 根据模型等级调整
        if self.auto_mode:
            grade_layers: dict[str, int] = {
                'S': 5, 'A': 5, 'B': 3, 'C': 2, 'D': 2,
            }
            depth = grade_layers.get(self.model_grade, 5)
            return ['L0', 'L1', 'L2', 'L3', 'L4'][:depth]

        # 最低保障: 仅 L0
        return ['L0']

    def _is_layer_active(self, layer: str) -> bool:
        """检查指定层级是否启用。

        Args:
            layer: 层级 ID (L0-L4)。

        Returns:
            True 如果该层启用。
        """
        return layer in self._active_layers()

    # ── 主检查入口 ────────────────────────────────────────────

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """主检查入口 —— 检查一个工具调用的安全性。

        检查顺序：
            1. 如果工具名在黑名单中 → 检查所有字符串参数
            2. 提取 command 类参数 → check_command()
            3. 提取 file_path 类参数 → check_file_path()
            4. 对其他长字符串参数也执行命令检查

        Args:
            tool_name: 工具名称。
            arguments: 工具参数字典。

        Returns:
            检查结果字典:
                {"allowed": bool, "blocked_by": str|None,
                 "reason": str|None, "layer": str|None}
        """
        if not tool_name:
            return self._allow_result()

        tool_lower = tool_name.lower()

        # 检查命令参数
        if tool_lower in _COMMAND_TOOL_NAMES:
            cmd = self._extract_param(arguments, _COMMAND_PARAM_NAMES)
            if cmd:
                return self.check_command(cmd)

        # 检查文件路径参数
        path_val = self._extract_param(arguments, _FILE_PATH_PARAM_NAMES)
        if path_val:
            result = self.check_file_path(path_val)
            if not result['allowed']:
                return result

        # 对其他长字符串值也执行命令检查
        for value in arguments.values():
            if isinstance(value, str) and len(value) > 10:
                cmd_result = self.check_command(value)
                if not cmd_result['allowed']:
                    return cmd_result

        return self._allow_result()

    def check_command(self, command: str) -> dict[str, Any]:
        """检查命令字符串的安全性。

        按优先级从 L0 到当前启用的最高层逐层匹配。

        Args:
            command: 命令字符串。

        Returns:
            检查结果字典。
        """
        if not isinstance(command, str) or not command.strip():
            return self._allow_result()

        # 白名单检查
        cmd_normalized = command.strip()
        for allowed in self.allowlist_commands:
            if allowed in cmd_normalized:
                self.stats['allowed'] += 1
                return self._allow_result()

        active = self._active_layers()
        layer_patterns: dict[str, list[re.Pattern]] = {
            'L0': self.l0_patterns,
            'L1': self.l1_patterns,
            'L2': self.l2_patterns,
            'L3': self.l3_patterns,
        }

        for layer in active:
            if layer == 'L4':
                # L4 检查路径而非命令
                cmd_lower = command.lower().replace('\\', '/')
                for protected in self.l4_protected:
                    expanded = os.path.expanduser(
                        protected.replace('\\', '/')
                    )
                    if protected in cmd_lower or expanded in cmd_lower:
                        self.stats['L4'] += 1
                        return self._block_result(
                            'L4',
                            f'L4 路径保护: 禁止访问 {protected}'
                            f' | 匹配位置: {command[:100]}',
                        )
                continue

            patterns = layer_patterns.get(layer, [])
            for pattern in patterns:
                if pattern.search(command):
                    self.stats[layer] += 1
                    self.log.warn(
                        f"Guard {layer} 拦截命令: "
                        f"pattern={pattern.pattern[:60]} "
                        f"command_snippet={command[:100]}"
                    )
                    return self._block_result(
                        layer,
                        f'{layer} {_LAYER_DESCRIPTIONS.get(layer, layer)}'
                        f' | 匹配: {pattern.pattern[:60]}'
                        f' | 命令片段: {command[:100]}',
                    )

            # 额外的检查逻辑
            block_reason = self._check_command_extra(layer, command)
            if block_reason:
                self.stats[layer] += 1
                return self._block_result(layer, block_reason)

        self.stats['allowed'] += 1
        return self._allow_result()

    def _check_command_extra(
        self, layer: str, command: str
    ) -> Optional[str]:
        """各层的额外安全逻辑（超越正则匹配的上下文感知检查）。

        Args:
            layer: 层级 ID。
            command: 命令字符串。

        Returns:
            拦截原因字符串，或 None 表示通过。
        """
        cmd_lower = command.lower()

        if layer == 'L1':
            # 额外: sudo 配合危险参数
            if 'sudo' in cmd_lower:
                dangerous_with_sudo = [
                    'passwd', 'visudo', 'crontab',
                    'iptables', 'ufw', 'firewall-cmd',
                ]
                for bad_cmd in dangerous_with_sudo:
                    if bad_cmd in cmd_lower:
                        return (
                            f'L1: sudo + {bad_cmd} 禁止执行'
                            f' | 命令: {command[:100]}'
                        )

        elif layer == 'L2':
            # 额外: find 配合 -exec
            if 'find' in cmd_lower and ('-exec' in cmd_lower or '-delete' in cmd_lower):
                if 'rm' in cmd_lower or 'chmod' in cmd_lower or 'chown' in cmd_lower:
                    return (
                        f'L2: find -exec 危险操作禁止'
                        f' | 命令: {command[:100]}'
                    )

            # 额外: xargs 配合 rm/chmod
            if 'xargs' in cmd_lower:
                if 'rm' in cmd_lower or 'chmod' in cmd_lower:
                    return (
                        f'L2: xargs 管道危险操作禁止'
                        f' | 命令: {command[:100]}'
                    )

        elif layer == 'L3':
            # 额外: python/node 一行远程执行
            if re.search(
                r'(python|node|perl|ruby)\s+.*(urllib|requests|http\.|fetch|socket)',
                cmd_lower
            ):
                return (
                    f'L3: 脚本远程执行禁止'
                    f' | 命令: {command[:100]}'
                )

            # 额外: ssh 隧道/端口转发
            if re.search(r'ssh\s+.*-R\s+\d+:', cmd_lower):
                return (
                    f'L3: SSH 反向隧道禁止'
                    f' | 命令: {command[:100]}'
                )

        return None

    def check_file_path(self, file_path: str) -> dict[str, Any]:
        """检查文件路径是否在受保护目录下。

        通过 Path.resolve() 解析真实路径后与 L4 保护列表比对。
        同时支持 Windows 和 Unix 路径格式。

        Args:
            file_path: 文件路径字符串。

        Returns:
            检查结果字典。
        """
        if not isinstance(file_path, str) or not file_path.strip():
            return self._allow_result()

        if not self._is_layer_active('L4'):
            return self._allow_result()

        # 白名单检查
        path_lower = file_path.lower().replace('\\', '/')
        for allowed in self.allowlist_paths:
            if allowed in path_lower:
                return self._allow_result()

        # 解析真实路径
        try:
            resolved = str(Path(file_path).resolve())
            resolved_norm = resolved.replace('\\', '/').lower()
        except (OSError, ValueError):
            resolved_norm = path_lower

        expanded = os.path.expanduser(path_lower)
        expanded_resolved = ""
        try:
            expanded_resolved = str(
                Path(os.path.expanduser(file_path)).resolve()
            ).replace('\\', '/').lower()
        except (OSError, ValueError):
            expanded_resolved = expanded

        # 三层对比: 原始路径、解析后路径、展开后路径
        for protected in self.l4_protected:
            prot_expanded = os.path.expanduser(
                protected.replace('\\', '/')
            ).lower()
            prot_resolved = ""
            try:
                prot_resolved = str(
                    Path(os.path.expanduser(protected))
                ).replace('\\', '/').lower()
            except (OSError, ValueError):
                prot_resolved = prot_expanded

            # 前缀匹配和前向匹配
            checks = [
                (resolved_norm, prot_expanded),
                (expanded, prot_expanded),
                (path_lower, protected),
                (expanded_resolved, prot_resolved),
            ]
            for check_path, check_prot in checks:
                if not check_path:
                    continue
                if check_path == check_prot or check_path.startswith(
                    check_prot + '/'
                ):
                    self.stats['L4'] += 1
                    return self._block_result(
                        'L4',
                        f'L4 路径保护: {protected}'
                        f' | 访问路径: {file_path[:120]}',
                    )

            # 反向: 检查受保护路径是否在访问路径内
            if prot_expanded in path_lower or prot_expanded in resolved_norm:
                self.stats['L4'] += 1
                return self._block_result(
                    'L4',
                    f'L4 路径保护(包含): {protected}'
                    f' | 访问路径: {file_path[:120]}',
                )

        return self._allow_result()

    # ── 辅助方法 ───────────────────────────────────────────────

    @staticmethod
    def _extract_param(
        args: dict[str, Any], keys: tuple[str, ...]
    ) -> str:
        """从参数字典中提取匹配键的第一个字符串值。

        Args:
            args: 参数字典。
            keys: 优先匹配的键名列表。

        Returns:
            找到的字符串值，或空字符串。
        """
        if not isinstance(args, dict):
            return ''
        for key in keys:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ''

    @staticmethod
    def _allow_result() -> dict[str, Any]:
        """生成允许通过的结果字典。"""
        return {
            "allowed": True,
            "blocked_by": None,
            "reason": None,
            "layer": None,
        }

    @staticmethod
    def _block_result(layer: str, reason: str) -> dict[str, Any]:
        """生成拦截结果字典。

        Args:
            layer: 拦截层级。
            reason: 拦截原因。

        Returns:
            拦截结果字典。
        """
        return {
            "allowed": False,
            "blocked_by": layer,
            "reason": reason,
            "layer": layer,
        }

    # ── 白名单管理 ────────────────────────────────────────────

    def add_command_allowlist(self, command: str) -> None:
        """添加一条命令白名单。

        Args:
            command: 命令字符串。
        """
        self.allowlist_commands.add(command.strip())

    def add_path_allowlist(self, path: str) -> None:
        """添加一条路径白名单。

        Args:
            path: 路径字符串。
        """
        self.allowlist_paths.add(path.lower().replace('\\', '/'))

    def remove_command_allowlist(self, command: str) -> None:
        """移除一条命令白名单。

        Args:
            command: 命令字符串。
        """
        self.allowlist_commands.discard(command.strip())

    # ── 诊断 ──────────────────────────────────────────────────

    def get_active_layers_detail(self) -> dict[str, Any]:
        """获取激活层级详情。

        Returns:
            包含每层状态和统计的诊断字典。
        """
        active = self._active_layers()
        return {
            "safe_mode": self.safe_mode,
            "auto_mode": self.auto_mode,
            "model_grade": self.model_grade,
            "active_layers": active,
            "l0_patterns_count": len(self.l0_patterns),
            "l1_patterns_count": len(self.l1_patterns),
            "l2_patterns_count": len(self.l2_patterns),
            "l3_patterns_count": len(self.l3_patterns),
            "l4_paths_count": len(self.l4_protected),
            "allowlist_commands_count": len(self.allowlist_commands),
            "allowlist_paths_count": len(self.allowlist_paths),
            "stats": dict(self.stats),
        }

    def reset_stats(self) -> None:
        """重置拦截统计。"""
        self.stats = {
            'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0,
            'allowed': 0,
        }

    def audit_command(self, command: str) -> dict[str, Any]:
        """审计检查命令（详细输出，不做拦截）。

        返回每层的匹配情况，便于调试和分析。

        Args:
            command: 要审计的命令。

        Returns:
            审计结果，包含每层匹配信息。
        """
        all_layers = ['L0', 'L1', 'L2', 'L3', 'L4']
        layer_patterns: dict[str, list[re.Pattern]] = {
            'L0': self.l0_patterns,
            'L1': self.l1_patterns,
            'L2': self.l2_patterns,
            'L3': self.l3_patterns,
        }

        audit: dict[str, Any] = {
            "command": command[:200],
            "active_layers": self._active_layers(),
            "findings": {},
        }

        for layer in all_layers:
            findings: list[str] = []
            if layer == 'L4':
                cmd_lower = command.lower().replace('\\', '/')
                for protected in self.l4_protected:
                    if protected in cmd_lower:
                        findings.append(f"L4 命中路径: {protected}")
            elif layer in layer_patterns:
                for pattern in layer_patterns[layer]:
                    if pattern.search(command):
                        findings.append(
                            f"命中: {pattern.pattern[:80]}"
                        )
            audit["findings"][layer] = findings

        result = self.check_command(command)
        audit["final_result"] = result

        return audit