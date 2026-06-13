"""
loop-ollama CLI 入口模块。

提供命令行接口，支持硬件检测（--check）、模型选择（--model）、
安全模式（--safe/--auto/--unsafe）、首次初始化（--init）等功能。

Usage:
    python -m src.cli --version
    python -m src.cli --check
    python -m src.cli --init
    python -m src.cli --model qwen2.5-coder:7b --auto
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import Config
from .hardware_detector import HardwareDetector
from .logger import Logger
from .model_detector import ModelDetector
from .ollama_client import OllamaClient


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="loop-ollama",
        description=(
            "本地 AI 编程 Agent —— 基于 Ollama 本地模型的"
            "自建 ReAct 自主编程 agent。零 API 费用、"
            "零数据外泄、三层容错对抗弱模型幻觉。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 核心参数
    parser.add_argument(
        "--version", action="store_true",
        help="输出版本号并退出",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="检测硬件环境与可用模型，输出完整报告",
    )
    parser.add_argument(
        "--init", action="store_true",
        help="执行首次运行设置向导",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="指定使用的模型名称（如 qwen2.5-coder:7b）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（默认 ~/.loop-ollama/config.json）",
    )

    # 安全模式
    safety = parser.add_mutually_exclusive_group()
    safety.add_argument(
        "--safe", action="store_true", default=False,
        help="安全模式：启用完整护栏，禁用破坏性操作",
    )
    safety.add_argument(
        "--auto", action="store_true", default=True,
        help="自动模式（默认）：按模型等级动态调整护栏",
    )
    safety.add_argument(
        "--unsafe", action="store_true", default=False,
        help="非安全模式：跳过所有护栏检查（调试用，需谨慎）",
    )

    parser.add_argument(
        "--task", type=str, default=None,
        help="要执行的编程任务描述文本",
    )

    return parser


def _print_header() -> None:
    """打印 loop-ollama 横幅。"""
    print(f"loop-ollama v{__version__} —— 本地 AI 编程 Agent")
    print("-" * 50)


def cmd_version() -> int:
    """执行 --version 命令。

    Returns:
        退出码 0。
    """
    print(f"loop-ollama {__version__}")
    return 0


def cmd_check() -> int:
    """执行 --check 命令：完整的硬件 + 模型检测报告。

    Returns:
        0 成功，1 部分失败。
    """
    _print_header()
    print()

    # 1. 加载配置
    config = Config()
    config.load()
    log = Logger()

    # 2. 硬件检测
    print("[1/4] 硬件检测中...")
    hw_detector = HardwareDetector()
    hw = hw_detector.detect()
    print(f"  GPU: {hw['gpu_name'] or '无（CPU 推理）'}")
    print(f"  VRAM: {hw['vram_gb']:.1f} GB")
    print(f"  RAM: {hw['ram_gb']:.1f} GB")
    print(f"  CPU 核心: {hw['cpu_cores']}")
    print(f"  平台: {hw['platform']}")
    print()

    # 3. Ollama 健康检查
    print("[2/4] Ollama 服务检测中...")
    client = OllamaClient(
        base_url=config.ollama_base_url
    )
    if client.health_check():
        print(f"  Ollama 服务运行正常 ({config.ollama_base_url})")
    else:
        print(f"  Ollama 服务未响应 ({config.ollama_base_url})")
        print("  请确保已安装并启动 Ollama (ollama serve)")
        return 1
    print()

    # 4. 模型扫描
    print("[3/4] 已下载模型扫描中...")
    available = client.list_available_models()
    if not available:
        print("  未检测到已下载的模型。请使用 ollama pull <model> 下载模型。")
        return 1

    detector = ModelDetector(client)
    detected = detector.detect_all_available()

    print(f"  发现 {len(detected)} 个模型：")
    print(f"  {'模型名称':40s} {'参数量':>8s}  {'量化':>8s}  "
          f"{'等级':>4s}  {'能力分':>6s}")
    print("  " + "-" * 76)
    for d in detected:
        print(
            f"  {d['model_name']:40s} "
            f"{d['param_size_billions']:7.2f}B "
            f"{d['quantization']:>8s}  "
            f"{d['grade']:>4s}  "
            f"{d['capability_score']:.4f}"
        )
    print()

    # 5. 推荐
    print("[4/4] 模型推荐...")
    rec = hw_detector.get_hardware_adapted_recommendations()
    rec_model = detector.recommend_model(available, hw)

    print(f"  硬件适配等级: {rec['recommended_grade']} 级")
    print(f"  推荐模型范围: {rec['recommended_model']}")
    if rec_model:
        print(f"  当前最佳可用: {rec_model}")
        # 检测其详情
        info = detector.detect(rec_model)
        print(f"    等级: {info['grade']}, 能力分: {info['capability_score']}")
    else:
        print("  未找到与硬件兼容的已下载模型。")
    print()

    log.info("--check completed", models_found=len(detected))
    return 0


def cmd_init() -> int:
    """执行 --init 命令：首次运行设置。

    检测 Ollama → 扫描硬件 → 列出模型 → 设置默认模型 → 保存配置。

    Returns:
        0 成功，1 失败。
    """
    _print_header()
    print()
    print("首次运行设置向导")
    print("=" * 50)
    print()

    config = Config()
    config.load()
    log = Logger()

    # Step 1: 检查 Ollama
    print("[1/5] 检查 Ollama 服务...")
    client = OllamaClient(base_url=config.ollama_base_url)
    if not client.health_check():
        print("  错误: Ollama 服务未响应。")
        print("  请安装并启动 Ollama: ollama serve")
        return 1
    print(f"  Ollama 正常: {config.ollama_base_url}")
    print()

    # Step 2: 硬件检测
    print("[2/5] 硬件检测...")
    hw = HardwareDetector().detect()
    print(f"  GPU: {hw['gpu_name'] or '无'}")
    print(f"  VRAM: {hw['vram_gb']:.1f} GB / RAM: {hw['ram_gb']:.1f} GB")
    print()

    # Step 3: 模型扫描
    print("[3/5] 扫描已下载模型...")
    available = client.list_available_models()
    detector = ModelDetector(client)
    detected = detector.detect_all_available()
    if not detected:
        print("  未找到模型。推荐下载: ollama pull qwen2.5-coder:7b")
        config.save()
        return 0
    for d in detected:
        print(
            f"  {d['model_name']} — {d['grade']} 级"
            f" ({d['param_size_billions']}B, {d['quantization']})"
        )
    print()

    # Step 4: 自动推荐/选择默认模型
    print("[4/5] 设置默认模型...")
    default_model = detector.recommend_model(available, hw)
    if default_model and not config.default_model:
        config.set("ollama.default_model", default_model)
        print(f"  已设置默认模型: {default_model}")
    elif config.default_model:
        print(f"  当前默认模型: {config.default_model}")
    else:
        print("  未设置默认模型（稍后可手动设置）")
    print()

    # Step 5: 保存
    print("[5/5] 保存配置...")
    config.save()
    print(f"  配置已保存到: {config.config_path}")
    print()
    print("设置完成。运行 loop-ollama --check 查看完整报告。")

    log.info("--init completed")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 主入口。

    Args:
        argv: 命令行参数列表。None 则使用 sys.argv。

    Returns:
        退出码。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --version
    if args.version:
        return cmd_version()

    # --init
    if args.init:
        return cmd_init()

    # --check
    if args.check:
        return cmd_check()

    # 无参数时打印帮助
    if not any([args.task, args.model]):
        parser.print_help()
        print(f"\nloop-ollama v{__version__}")
        return 0

    # --task + --model 模式（Phase 2 之后实现）
    cfg = Config()
    cfg.load()
    print("loop-ollama ReAct loop 尚未实现（Phase 2 计划中）。")
    print(f"  任务: {args.task or '(无)'}")
    print(f"  模型: {args.model or cfg.default_model or '(未设置)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
