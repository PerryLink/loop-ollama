"""
timeout_manager.py —— 动态超时管理器（EMA 算法）

核心功能：
- 基于指数移动平均（EMA）动态调整 API 请求超时时间
- 统计每个模型的响应延迟，按模型粒度独立跟踪
- 提供健康检查接口，判定模型是否响应过慢
- 支持配置：平滑因子 alpha、最小/最大超时边界、窗口大小

设计思路：
  固定超时难以适应不同模型和网络环境。实际延迟 = 模型推理 + 网络传输 + 队列等待。
  EMA 平滑地跟踪最新趋势（近期延迟权重高），在响应变慢时逐步放宽超时，
  在响应恢复后逐步收紧，避免因偶发抖动而过早超时。

算法：
  ema_{n} = alpha * sample_{n} + (1 - alpha) * ema_{n-1}
  timeout = min(max(ema * multiplier, min_timeout), max_timeout)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.logger import Logger

# ═══════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════


@dataclass
class ModelTimeoutStats:
    """单个模型的超时统计。"""

    model_name: str
    ema: float = 0.0  # 当前 EMA 值（秒）
    sample_count: int = 0
    last_sample_time: float = 0.0
    current_timeout: float = 10.0  # 当前推荐超时（秒）
    min_latency: float = float("inf")
    max_latency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "ema_seconds": round(self.ema, 3),
            "samples": self.sample_count,
            "current_timeout": round(self.current_timeout, 1),
            "min_latency": round(self.min_latency, 3) if self.min_latency != float("inf") else None,
            "max_latency": round(self.max_latency, 3),
        }


# ═══════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════

_DEFAULT_ALPHA = 0.3  # EMA 平滑因子（0<alpha<=1，越大越敏感）
_DEFAULT_MULTIPLIER = 3.0  # 超时 = EMA * multiplier
_DEFAULT_MIN_TIMEOUT = 5.0  # 最小超时（秒）
_DEFAULT_MAX_TIMEOUT = 120.0  # 最大超时（秒）
_DEFAULT_WINDOW_SIZE = 100  # 保留最近 N 个样本（统计用）


# ═══════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════


class TimeoutManager:
    """
    动态超时管理器。

    为每个模型独立跟踪 API 延迟，使用 EMA 算法平滑计算推荐超时时间。
    线程安全，支持多模型并发调用。

    Usage:
        tm = TimeoutManager(alpha=0.3, multiplier=3.0)
        timeout = tm.get_timeout("qwen2.5-coder:14b")
        # ... 执行 API 调用，记录延迟 ...
        tm.record("qwen2.5-coder:14b", elapsed_seconds=2.3)
    """

    def __init__(
        self,
        alpha: float = _DEFAULT_ALPHA,
        multiplier: float = _DEFAULT_MULTIPLIER,
        min_timeout: float = _DEFAULT_MIN_TIMEOUT,
        max_timeout: float = _DEFAULT_MAX_TIMEOUT,
    ):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        if min_timeout <= 0 or max_timeout <= min_timeout:
            raise ValueError(
                f"invalid timeout bounds: min={min_timeout}, max={max_timeout}"
            )

        self._alpha = alpha
        self._multiplier = multiplier
        self._min_timeout = min_timeout
        self._max_timeout = max_timeout
        self._lock = threading.Lock()
        self._stats: dict[str, ModelTimeoutStats] = {}
        self._recent_samples: dict[str, list[float]] = {}  # 滚动窗口
        self._logger = Logger.get_instance()

    # ── 公共 API ──

    def get_timeout(self, model_name: str) -> float:
        """获取指定模型的推荐超时时间（秒）。"""
        with self._lock:
            if model_name not in self._stats:
                # 初次调用，使用默认超时
                return self._min_timeout
            stats = self._stats[model_name]
            return stats.current_timeout

    def record(
        self,
        model_name: str,
        elapsed_seconds: float,
        success: bool = True,
    ) -> None:
        """
        记录一次 API 调用耗时，更新 EMA。

        Args:
            model_name: 模型名称。
            elapsed_seconds: 实际耗时（秒）。
            success: 调用是否成功。失败时不更新 EMA（避免错误延迟污染）。
        """
        if not success or elapsed_seconds <= 0:
            return

        with self._lock:
            # 初始化统计
            if model_name not in self._stats:
                self._stats[model_name] = ModelTimeoutStats(
                    model_name=model_name,
                    ema=elapsed_seconds,
                    current_timeout=self._min_timeout,
                )
                self._recent_samples[model_name] = []

            stats = self._stats[model_name]
            stats.sample_count += 1
            stats.last_sample_time = time.time()

            # 更新 min/max
            if elapsed_seconds < stats.min_latency:
                stats.min_latency = elapsed_seconds
            if elapsed_seconds > stats.max_latency:
                stats.max_latency = elapsed_seconds

            # EMA 更新
            stats.ema = (
                self._alpha * elapsed_seconds
                + (1 - self._alpha) * stats.ema
            )

            # 计算推荐超时
            raw_timeout = stats.ema * self._multiplier
            stats.current_timeout = min(
                max(raw_timeout, self._min_timeout),
                self._max_timeout,
            )

            # 维护滚动窗口
            samples = self._recent_samples[model_name]
            samples.append(elapsed_seconds)
            if len(samples) > _DEFAULT_WINDOW_SIZE:
                samples.pop(0)

    def is_healthy(self, model_name: str) -> bool:
        """检查模型是否健康（延迟在可接受范围内）。"""
        with self._lock:
            stats = self._stats.get(model_name)
            if stats is None or stats.sample_count < 3:
                return True  # 样本不足，默认健康
            # 如果当前推荐超时 >= 最大超时的 80%，视为异常
            return stats.current_timeout < self._max_timeout * 0.8

    def get_stats(self, model_name: str) -> Optional[ModelTimeoutStats]:
        """获取模型的超时统计信息。"""
        with self._lock:
            return self._stats.get(model_name)

    def get_all_stats(self) -> list[dict]:
        """获取所有模型的统计摘要。"""
        with self._lock:
            return [s.to_dict() for s in self._stats.values()]

    def reset(self, model_name: Optional[str] = None) -> None:
        """
        重置统计信息。

        Args:
            model_name: 指定模型名称，None 表示重置所有。
        """
        with self._lock:
            if model_name:
                self._stats.pop(model_name, None)
                self._recent_samples.pop(model_name, None)
            else:
                self._stats.clear()
                self._recent_samples.clear()

    def predict_timeout(self, model_name: str) -> float:
        """
        预测下次调用的超时时间。

        基于 EMA 和近期样本的标准差，给出一个更激进的超时预测。
        公式：predicted = ema + 2 * recent_stddev
        """
        with self._lock:
            stats = self._stats.get(model_name)
            if stats is None or stats.sample_count < 5:
                return self._min_timeout

            samples = self._recent_samples.get(model_name, [])
            if len(samples) < 2:
                return stats.current_timeout

            mean = sum(samples) / len(samples)
            variance = sum((x - mean) ** 2 for x in samples) / len(samples)
            stddev = variance**0.5

            predicted = stats.ema + 2 * stddev
            return min(max(predicted, self._min_timeout), self._max_timeout)

    def get_recent_latency_trend(self, model_name: str, window: int = 10) -> list[float]:
        """获取最近 N 次延迟记录。"""
        with self._lock:
            samples = self._recent_samples.get(model_name, [])
            return samples[-window:] if samples else []
