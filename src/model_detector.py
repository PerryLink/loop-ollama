"""
loop-ollama 模型能力检测与分级模块。

通过 Ollama /api/show 获取模型参数/量化/context_window，
计算有效参数量与能力分数，输出 S/A/B/C/D 五级评级。

Classes:
    ModelDetector: 模型检测器与分级引擎。
"""

import re
from typing import Any, Optional

from .ollama_client import OllamaClient


# ── 量化折扣查找表 ────────────────────────────────────────────────

QUANTIZATION_PENALTY: dict[str, float] = {
    "F16": 1.0,
    "F32": 1.0,
    "Q8_0": 0.95,
    "Q6_K": 0.92,
    "Q6_K_M": 0.92,
    "Q5_K_M": 0.88,
    "Q5_K": 0.88,
    "Q4_K_M": 0.82,
    "Q4_K": 0.82,
    "Q4_0": 0.75,
    "Q3_K_M": 0.65,
    "Q3_K": 0.65,
    "Q2_K": 0.50,
    "Q2_K_M": 0.50,
}

_UNKNOWN_QUANT_PENALTY = 0.80

# ── 分级阈值 ─────────────────────────────────────────────────────

GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (32.0, "S"),
    (7.0, "A"),
    (3.0, "B"),
    (1.0, "C"),
    (0.0, "D"),
]


class ModelDetector:
    """模型能力检测与分级引擎。

    通过 Ollama API 获取模型参数信息，计算有效参数量与能力分数，
    输出 S/A/B/C/D 评级及模型推荐。

    Attributes:
        ollama_client: Ollama REST API 客户端实例。
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """初始化 ModelDetector。

        Args:
            ollama_client: OllamaClient 实例。
        """
        self.ollama_client = ollama_client

    # ── 完整检测管道 ─────────────────────────────────────────

    def detect(self, model_name: str) -> dict[str, Any]:
        """执行完整的模型能力检测。

        管道: show_model -> 提取参数量 -> 量化折扣 ->
              计算有效参数 -> 分级 -> 能力分数。

        Args:
            model_name: 模型名称（如 "qwen2.5-coder:7b"）。

        Returns:
            {
                "model_name": str,
                "param_size_billions": float,
                "quantization": str,
                "quantization_penalty": float,
                "effective_params": float,
                "grade": str,
                "capability_score": float,
                "context_window": int,
                "vram_estimate_gb": float,
            }
        """
        try:
            model_info = self.ollama_client.show_model(model_name)
        except Exception:
            return self._unknown_model_result(model_name)

        # 提取参数量
        param_size = self.extract_param_size_billions(model_info)

        # 提取量化级别
        details = model_info.get("details", {})
        quantization = details.get("quantization_level", "unknown")

        # 量化折扣
        quant_penalty = QUANTIZATION_PENALTY.get(
            quantization, _UNKNOWN_QUANT_PENALTY
        )

        # 有效参数量
        effective_params = param_size * quant_penalty

        # 分级
        grade = self.compute_grade(effective_params)

        # 能力分数 (0.0-1.0)
        capability_score = self.compute_capability_score(effective_params)

        # 上下文窗口
        context_window = self._extract_context_window(model_info)

        # VRAM 估算
        vram_estimate = self.estimate_vram_required(param_size)

        return {
            "model_name": model_name,
            "param_size_billions": param_size,
            "quantization": quantization,
            "quantization_penalty": quant_penalty,
            "effective_params": effective_params,
            "grade": grade,
            "capability_score": capability_score,
            "context_window": context_window,
            "vram_estimate_gb": vram_estimate,
        }

    def detect_all_available(self) -> list[dict[str, Any]]:
        """扫描所有已下载模型，执行完整检测并按能力分数降序排列。

        Returns:
            检测结果列表，按 capability_score 降序。
        """
        models = self.ollama_client.list_available_models()
        results = []
        for m in models:
            name = m.get("name", "")
            if name:
                result = self.detect(name)
                results.append(result)
        results.sort(key=lambda r: r["capability_score"], reverse=True)
        return results

    # ── 参数量提取 ───────────────────────────────────────────

    @staticmethod
    def extract_param_size_billions(model_info: dict[str, Any]) -> float:
        """从 /api/show 响应中提取参数量（十亿为单位）。

        支持格式: "7.6B", "236B", "360M", "1.1B", "70B"。

        Args:
            model_info: /api/show 响应字典。

        Returns:
            参数量（B），无法识别返回 0.0。
        """
        details = model_info.get("details", {})
        param_str = details.get(
            "parameter_size", ""
        )

        if not param_str:
            # 尝试从 model_info 字段推断
            mi = model_info.get("model_info", {})
            for key, val in mi.items():
                if "parameter" in key.lower():
                    param_str = str(val)
                    break

        if not param_str:
            return 0.0

        param_str = param_str.strip().upper()

        # 匹配 MoE 格式 "8x7B" / "8x22B"
        m = re.match(r"(\d+)\s*[xX]\s*([\d.]+)\s*B", param_str)
        if m:
            return float(m.group(1)) * float(m.group(2))

        # 匹配 "7.6B" / "236B"
        m = re.match(r"([\d.]+)\s*B", param_str)
        if m:
            return float(m.group(1))

        # 匹配 "360M" / "500M"
        m = re.match(r"([\d.]+)\s*M", param_str)
        if m:
            return float(m.group(1)) / 1000.0

        # 纯数字
        try:
            return float(param_str)
        except ValueError:
            return 0.0

    # ── 分级计算 ─────────────────────────────────────────────

    @staticmethod
    def compute_grade(effective_params: float) -> str:
        """根据有效参数量计算模型等级。

        阈值:
            >= 32 → S
            >= 7  → A
            >= 3  → B
            >= 1  → C
            < 1   → D

        Args:
            effective_params: 有效参数量（B，含量化折扣）。

        Returns:
            模型等级 (S/A/B/C/D)。
        """
        for threshold, grade in GRADE_THRESHOLDS:
            if effective_params >= threshold:
                return grade
        return "D"

    @staticmethod
    def compute_capability_score(effective_params: float) -> float:
        """计算能力分数 (0.0-1.0)。

        以 70B 有效参数为满分基准。

        Args:
            effective_params: 有效参数量（B）。

        Returns:
            能力分数，范围 [0.0, 1.0]。
        """
        if effective_params <= 0:
            return 0.0
        return round(min(1.0, effective_params / 70.0), 4)

    # ── 上下文窗口提取 ───────────────────────────────────────

    @staticmethod
    def _extract_context_window(model_info: dict[str, Any]) -> int:
        """从模型信息中提取上下文窗口大小。

        Args:
            model_info: /api/show 响应。

        Returns:
            Context window token 数，默认 4096。
        """
        mi = model_info.get("model_info", {})
        for key in (
            "context_length",
            "qwen2.context_length",
            "llama.context_length",
            "max_position_embeddings",
        ):
            # 遍历 model_info 中的 key，匹配包含 context_length 的
            for k, v in mi.items():
                if "context_length" in k.lower() or key in k.lower():
                    try:
                        return int(v)
                    except (ValueError, TypeError):
                        pass

        # 通用搜索 model_info 中包含 "context" 的键
        for k, v in mi.items():
            if "context" in k.lower():
                try:
                    return int(v)
                except (ValueError, TypeError):
                    pass

        return 4096

    # ── VRAM 估算 ────────────────────────────────────────────

    @staticmethod
    def estimate_vram_required(param_billions: float) -> float:
        """估算模型运行所需 VRAM (GB)。

        使用公式: param_billions * 2 / 1e9 * 1.15 (安全余量)。

        Args:
            param_billions: 模型参数量（B）。

        Returns:
            估算 VRAM 需求 (GB)。
        """
        # 粗略估算: 1B 参数 ≈ 2GB (FP16) 或 0.5GB (Q4)，
        # 取保守值 + 15% 开销
        if param_billions > 10:
            bytes_per_param = 1.0  # Q4 ≈ 0.5 字节/参数 + 开销
        else:
            bytes_per_param = 1.5
        vram = param_billions * bytes_per_param * 1.15
        return round(vram, 2)

    # ── 模型推荐 ─────────────────────────────────────────────

    def recommend_model(
        self,
        available_models: list[dict[str, Any]],
        hardware: dict[str, Any],
        prefer_grade: Optional[str] = None,
    ) -> Optional[str]:
        """根据硬件和偏好推荐最优模型。

        筛选逻辑:
            1. 按 VRAM 过滤（模型 VRAM 估算 <= 可用 VRAM）。
            2. GPU 不可用时降低推荐等级。
            3. prefer_grade 不高于硬件上限时优先匹配。
            4. 按 capability_score 降序返回。

        Args:
            available_models: 模型列表（/api/tags 返回格式）。
            hardware: 硬件检测结果。
            prefer_grade: 偏好等级。

        Returns:
            推荐模型名称，无可用模型时返回 None。
        """
        vram_gb = hardware.get("vram_gb", 0.0)
        gpu_available = hardware.get("gpu_name") is not None

        candidates: list[tuple[str, float]] = []
        for m in available_models:
            name = m.get("name", "")
            if not name:
                continue
            result = self.detect(name)

            # VRAM 筛选
            if (
                gpu_available
                and result["vram_estimate_gb"] > vram_gb * 0.95
            ):
                continue
            if (
                not gpu_available
                and result["param_size_billions"] > 3.0
            ):
                continue  # CPU 推理跳过超过 3B 的模型

            candidates.append((name, result["capability_score"]))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)

        # 如果有 prefer_grade，在候选列表中优先匹配
        if prefer_grade:
            for name, score in candidates:
                try:
                    g = self.detect(name)["grade"]
                    if g == prefer_grade:
                        return name
                except Exception:
                    pass

        return candidates[0][0] if candidates else None

    def get_best_available_model(
        self,
        current_model: str,
        available_models: list[dict[str, Any]],
        hardware: dict[str, Any],
    ) -> Optional[str]:
        """获取比当前模型更强、且硬件可运行的最佳模型。

        用于模型升级决策。

        Args:
            current_model: 当前使用的模型名称。
            available_models: /api/tags 返回的可用模型列表。
            hardware: 硬件检测结果。

        Returns:
            更好的模型名称，无可升级模型时返回 None。
        """
        try:
            current_info = self.detect(current_model)
        except Exception:
            return None

        current_score = current_info["capability_score"]

        vram_gb = hardware.get("vram_gb", 0.0)

        best: Optional[tuple[str, float]] = None
        for m in available_models:
            name = m.get("name", "")
            if name == current_model:
                continue
            try:
                info = self.detect(name)
            except Exception:
                continue

            if info["capability_score"] <= current_score:
                continue
            if info["vram_estimate_gb"] > vram_gb * 0.95:
                continue

            if best is None or info["capability_score"] > best[1]:
                best = (name, info["capability_score"])

        return best[0] if best else None

    # ── 辅助 ─────────────────────────────────────────────────

    @staticmethod
    def _unknown_model_result(model_name: str) -> dict[str, Any]:
        """返回未知模型的默认检测结果。"""
        return {
            "model_name": model_name,
            "param_size_billions": 0.0,
            "quantization": "unknown",
            "quantization_penalty": _UNKNOWN_QUANT_PENALTY,
            "effective_params": 0.0,
            "grade": "D",
            "capability_score": 0.0,
            "context_window": 4096,
            "vram_estimate_gb": 0.0,
        }
