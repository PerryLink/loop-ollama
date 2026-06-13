"""
loop-ollama 硬件检测器。

自动检测 GPU 型号、VRAM、系统 RAM、CPU 核心数等信息，
用于模型推荐和硬件适配矩阵匹配。

Classes:
    HardwareDetector: 硬件检测器。
"""

import os
import platform
import subprocess
from typing import Any, Optional


class HardwareDetector:
    """硬件检测器 —— 自动扫描 GPU / VRAM / RAM / CPU。

    核心方法:
        detect() -> dict: 返回完整硬件信息字典。
    """

    def detect(self) -> dict[str, Any]:
        """执行完整硬件检测流程。

        Returns:
            {
                "gpu_name": str | None,
                "vram_gb": float,
                "ram_gb": float,
                "cpu_cores": int,
                "platform": str,
                "gpu_vendor": str | None,   # "nvidia" / "amd" / "apple" / None
            }
        """
        result: dict[str, Any] = {
            "gpu_name": None,
            "vram_gb": 0.0,
            "ram_gb": 0.0,
            "cpu_cores": os.cpu_count() or 4,
            "platform": platform.system(),
            "gpu_vendor": None,
        }

        # ── GPU 检测 ─────────────────────────────────────────
        gpu_info = self._detect_nvidia()
        if gpu_info is None:
            gpu_info = self._detect_amd()
        if gpu_info is None:
            gpu_info = self._detect_apple_silicon()

        if gpu_info is not None:
            result["gpu_name"] = gpu_info["name"]
            result["vram_gb"] = gpu_info["vram_gb"]
            result["gpu_vendor"] = gpu_info["vendor"]

        # ── RAM 检测 ─────────────────────────────────────────
        result["ram_gb"] = self._detect_ram()

        return result

    # ── NVIDIA GPU 检测 ──────────────────────────────────────

    def _detect_nvidia(self) -> Optional[dict[str, Any]]:
        """通过 nvidia-smi 检测 NVIDIA GPU。

        Returns:
            {"name": str, "vram_gb": float, "vendor": "nvidia"} 或 None。
        """
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return None
            line = proc.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) < 2:
                return None
            name = parts[0]
            vram_mib = float(parts[1])
            return {
                "name": name,
                "vram_gb": round(vram_mib / 1024.0, 2),
                "vendor": "nvidia",
            }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    # ── AMD GPU 检测 ─────────────────────────────────────────

    def _detect_amd(self) -> Optional[dict[str, Any]]:
        """通过 rocminfo 检测 AMD GPU。

        Returns:
            {"name": str, "vram_gb": float, "vendor": "amd"} 或 None。
        """
        try:
            proc = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return None
            name = "AMD GPU (ROCm)"
            vram = 0.0
            for line in proc.stdout.split("\n"):
                if "VRAM" in line or "Memory" in line:
                    pass  # rocminfo 输出不稳定，使用默认值
            # AMD VRAM 回退：常见 ROCm GPU 默认 8GB
            vram = 8.0
            return {
                "name": name,
                "vram_gb": vram,
                "vendor": "amd",
            }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    # ── Apple Silicon 检测 ───────────────────────────────────

    def _detect_apple_silicon(self) -> Optional[dict[str, Any]]:
        """在 macOS 上检测 Apple Silicon (M 系列芯片)。

        Returns:
            {"name": str, "vram_gb": float, "vendor": "apple"} 或 None。
        """
        if platform.system() != "Darwin":
            return None
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0:
                return None
            brand = proc.stdout.strip()
            if "Apple" not in brand:
                return None
            # Apple Silicon: 统一内存，GPU 可使用约 60-70% 总 RAM
            ram_gb = self._detect_ram()
            vram_gb = round(ram_gb * 0.65, 2)
            return {
                "name": brand,
                "vram_gb": vram_gb,
                "vendor": "apple",
            }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    # ── RAM 检测 ─────────────────────────────────────────────

    def _detect_ram(self) -> float:
        """检测系统总 RAM (GB)。

        依次尝试 psutil、sysctl、/proc/meminfo。

        Returns:
            系统 RAM 总量 (GB)。
        """
        # 尝试 psutil
        try:
            import psutil  # type: ignore[import]
            return round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            pass

        # macOS fallback
        if platform.system() == "Darwin":
            try:
                proc = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    return round(int(proc.stdout.strip()) / (1024**3), 2)
            except Exception:
                pass

        # Linux fallback
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
        except (FileNotFoundError, Exception):
            pass

        return 8.0  # 默认 8GB（安全下限）

    # ── 模型推荐表 ───────────────────────────────────────────

    def get_hardware_adapted_recommendations(self) -> dict[str, Any]:
        """根据检测到的硬件生成模型推荐表。

        Returns:
            含 model_recommendations 和 grade 的字典。
        """
        hw = self.detect()
        vram = hw["vram_gb"]

        if vram >= 48:
            grade = "S"
            rec = "Qwen2.5-Coder-32B Q8_0 / DeepSeek-Coder-V2 Q4_K_M"
        elif vram >= 24:
            grade = "S"
            rec = "Qwen2.5-Coder-32B Q4_K_M"
        elif vram >= 16:
            grade = "A"
            rec = "Qwen2.5-Coder-14B Q4_K_M 或 Qwen2.5-Coder-7B Q8_0"
        elif vram >= 12:
            grade = "A"
            rec = "Qwen2.5-Coder-7B Q4_K_M"
        elif vram >= 8:
            grade = "B"
            rec = "Qwen2.5-Coder-3B Q4_K_M 或 Qwen2.5-Coder-7B Q4_0"
        elif vram >= 4:
            grade = "C"
            rec = "Qwen2.5-Coder-1.5B Q4_K_M"
        else:
            grade = "D"
            rec = "TinyLlama-1.1B (仅概念验证)"

        return {
            "hardware": hw,
            "recommended_grade": grade,
            "recommended_model": rec,
        }
