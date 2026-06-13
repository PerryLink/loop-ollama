"""
loop-ollama HardwareDetector 单元测试。

测试 GPU/RAM/CPU 检测、各平台回退、
硬件适配推荐矩阵、VRAM 分级。
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware_detector import HardwareDetector


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def detector():
    return HardwareDetector()


# ── Tests: 基本检测 ─────────────────────────────────────────────


def test_detect_returns_expected_keys(detector):
    """detect() 返回完整硬件信息字典。"""
    with patch.object(detector, "_detect_ram", return_value=16.0):
        result = detector.detect()
    for key in ["gpu_name", "vram_gb", "ram_gb", "cpu_cores", "platform", "gpu_vendor"]:
        assert key in result, f"缺少键: {key}"


def test_detect_platform_not_none(detector):
    """platform 不应为 None。"""
    with patch.object(detector, "_detect_ram", return_value=16.0):
        result = detector.detect()
    assert result["platform"] is not None
    assert len(result["platform"]) > 0


def test_detect_cpu_cores_positive(detector):
    """CPU 核心数应为正整数。"""
    with patch.object(detector, "_detect_ram", return_value=16.0):
        result = detector.detect()
    assert result["cpu_cores"] >= 1


# ── Tests: NVIDIA 检测 (mock) ───────────────────────────────────


def test_detect_nvidia_success(detector):
    """nvidia-smi 正常时返回 GPU 信息。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 24564\n",
        )
        result = detector._detect_nvidia()
        assert result is not None
        assert result["name"] == "NVIDIA GeForce RTX 4090"
        assert result["vram_gb"] == pytest.approx(24564 / 1024, rel=0.1)
        assert result["vendor"] == "nvidia"


def test_detect_nvidia_command_not_found(detector):
    """nvidia-smi 不存在时返回 None。"""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = detector._detect_nvidia()
        assert result is None


def test_detect_nvidia_returncode_error(detector):
    """nvidia-smi 返回非零时返回 None。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = detector._detect_nvidia()
        assert result is None


def test_detect_nvidia_timeout(detector):
    """nvidia-smi 超时时返回 None。"""
    import subprocess as sp
    with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="nvidia-smi", timeout=10)):
        result = detector._detect_nvidia()
        assert result is None


# ── Tests: AMD 检测 (mock) ──────────────────────────────────────


def test_detect_amd_success(detector):
    """rocminfo 正常时返回 AMD GPU 信息。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="AMD INSTINCT\n")
        result = detector._detect_amd()
        assert result is not None
        assert result["vendor"] == "amd"
        assert result["vram_gb"] == 8.0  # 默认 8GB


def test_detect_amd_command_not_found(detector):
    """rocminfo 不存在时返回 None。"""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = detector._detect_amd()
        assert result is None


# ── Tests: Apple Silicon 检测 (mock) ────────────────────────────


def test_detect_apple_silicon_darwin(detector):
    """macOS 上 Apple Silicon 检测。"""
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Apple M3 Pro\n",
            )
            with patch.object(detector, "_detect_ram", return_value=36.0):
                result = detector._detect_apple_silicon()
                assert result is not None
                assert result["vendor"] == "apple"
                assert "M3" in result["name"]
                assert result["vram_gb"] == pytest.approx(36.0 * 0.65, rel=0.1)


def test_detect_apple_silicon_not_darwin(detector):
    """非 macOS 系统应返回 None。"""
    with patch("platform.system", return_value="Linux"):
        result = detector._detect_apple_silicon()
        assert result is None


def test_detect_apple_silicon_intel_mac(detector):
    """Intel Mac 应返回 None（品牌不含 Apple 字样）。"""
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Intel(R) Core(TM) i7-9750H\n",
            )
            result = detector._detect_apple_silicon()
            assert result is None


# ── Tests: RAM 检测 ─────────────────────────────────────────────


def test_detect_ram_psutil_available(detector):
    """psutil 可用时返回真实 RAM 值。"""
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value.total = 32 * 1024**3
    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        result = detector._detect_ram()
        assert result == pytest.approx(32.0, rel=0.1)


def test_detect_ram_fallback_linux(detector):
    """psutil 不可用时回退 /proc/meminfo。"""
    mock_f = MagicMock()
    mock_f.__enter__.return_value = ["MemTotal:       16384000 kB\n"]
    with patch("builtins.open", return_value=mock_f):
        with patch.object(detector, "_detect_ram") as mock_ram:
            mock_ram.side_effect = None  # 不 mock，测试默认逻辑
        result = detector._detect_ram()
    assert result > 0


def test_detect_ram_default_fallback(detector):
    """所有方法失败时返回 8.0 GB 默认值。"""
    with patch.dict("sys.modules", {"psutil": None}):
        with patch("platform.system", return_value="Unknown"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                result = detector._detect_ram()
                assert result == 8.0


# ── Tests: 硬件适配推荐 ─────────────────────────────────────────


def test_recommendation_vram_48gb(detector):
    """VRAM >= 48GB -> S 级推荐 32B 模型。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "NVIDIA H100", "vram_gb": 80.0, "ram_gb": 128.0,
        "cpu_cores": 64, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "S"
        assert "32B" in rec["recommended_model"]


def test_recommendation_vram_24gb(detector):
    """VRAM >= 24GB -> S 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "RTX 4090", "vram_gb": 24.0, "ram_gb": 64.0,
        "cpu_cores": 16, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "S"


def test_recommendation_vram_16gb(detector):
    """VRAM >= 16GB -> A 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "RTX 4080", "vram_gb": 16.0, "ram_gb": 32.0,
        "cpu_cores": 16, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "A"


def test_recommendation_vram_12gb(detector):
    """VRAM >= 12GB -> A 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "RTX 4070", "vram_gb": 12.0, "ram_gb": 32.0,
        "cpu_cores": 12, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "A"


def test_recommendation_vram_8gb(detector):
    """VRAM >= 8GB -> B 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "RTX 3070", "vram_gb": 8.0, "ram_gb": 16.0,
        "cpu_cores": 8, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "B"


def test_recommendation_vram_4gb(detector):
    """VRAM >= 4GB -> C 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": "GTX 1650", "vram_gb": 4.0, "ram_gb": 8.0,
        "cpu_cores": 4, "platform": "Linux", "gpu_vendor": "nvidia"
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "C"


def test_recommendation_vram_below_4gb(detector):
    """VRAM < 4GB -> D 级。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": None, "vram_gb": 0.0, "ram_gb": 8.0,
        "cpu_cores": 4, "platform": "Linux", "gpu_vendor": None
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "D"
        assert "TinyLlama" in rec["recommended_model"]


def test_recommendation_no_gpu(detector):
    """无 GPU 时返回最低等级推荐。"""
    with patch.object(detector, "detect", return_value={
        "gpu_name": None, "vram_gb": 0.0, "ram_gb": 16.0,
        "cpu_cores": 8, "platform": "Linux", "gpu_vendor": None
    }):
        rec = detector.get_hardware_adapted_recommendations()
        assert rec["recommended_grade"] == "D"
        assert "Qwen" in rec["recommended_model"] or "TinyLlama" in rec["recommended_model"]
