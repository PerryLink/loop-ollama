"""
loop-ollama 主入口 — PyInstaller 引导点.

本文件是 PyInstaller 单文件二进制的入口脚本。
运行时直接委托给 `src.cli.main()` 处理所有 CLI 逻辑。

编译:
    pyinstaller build/loop-ollama.spec --clean --noconfirm
    等效: pyinstaller --onefile src/main.py --name loop-ollama

Usage:
    loop-ollama --version
    loop-ollama --check
    loop-ollama --init
    loop-ollama --task "用 Python 写一个 TODO CLI" --model qwen2.5-coder:7b
"""

import sys
import os

# ---- PyInstaller 资源路径解析 ----
# 当打包为单文件 exe 时，sys._MEIPASS 指向临时解压目录。
# 数据文件（如 regex_lib/*.json / prompts/*.j2）需要从此路径读取。


def _resolve_resource_path(relative_path: str) -> str:
    """解析资源文件路径，兼容开发模式与 PyInstaller 单文件模式。

    开发模式：
        relative_path 直接相对于 src/ 同级的项目根目录。

    PyInstaller 单文件模式：
        sys._MEIPASS 是临时解压目录，资源文件打包在其中。

    Args:
        relative_path: 相对于项目根目录的资源路径。

    Returns:
        资源的绝对路径。
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 单文件模式
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, relative_path.lstrip("/\\"))
    else:
        # 开发模式
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, relative_path.lstrip("/\\"))


def main() -> int:
    """主入口：委托给 CLI 模块。

    引导顺序：
        1. 确保项目根目录在 sys.path 中（PyInstaller 兼容）
        2. 调用 src.cli.main()

    Returns:
        CLI 退出码。
    """
    # 确保 src/ 包在导入路径中
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(src_dir))

    # 委托给 CLI
    from src.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
