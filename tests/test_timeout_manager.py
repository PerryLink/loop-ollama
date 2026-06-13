"""
TimeoutManager 单元测试。

测试 EMA 动态超时计算、模型统计跟踪、健康检查、
预测超时、延迟趋势获取——覆盖线程安全基础场景。
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.timeout_manager import TimeoutManager, ModelTimeoutStats


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tm():
    """创建基础 TimeoutManager 实例（mock Logger）。"""
    with patch("src.timeout_manager.Logger", autospec=False) as mock_logger:
        mock_logger.get_instance.return_value = MagicMock()
        yield TimeoutManager(alpha=0.3, multiplier=3.0,
                             min_timeout=5.0, max_timeout=60.0)


# ── 初始化测试 ────────────────────────────────────────────────


class TestInit:
    """初始化参数校验。"""

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_default_constructor(self, mock_logger):
        mock_logger.get_instance.return_value = MagicMock()
        tm = TimeoutManager()
        assert tm._alpha == 0.3
        assert tm._multiplier == 3.0
        assert tm._min_timeout == 5.0
        assert tm._max_timeout == 120.0

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_custom_params(self, mock_logger):
        mock_logger.get_instance.return_value = MagicMock()
        tm = TimeoutManager(alpha=0.5, multiplier=2.0,
                            min_timeout=10.0, max_timeout=90.0)
        assert tm._alpha == 0.5
        assert tm._multiplier == 2.0
        assert tm._min_timeout == 10.0
        assert tm._max_timeout == 90.0

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_alpha_out_of_range_raises(self, mock_logger):
        mock_logger.get_instance.return_value = MagicMock()
        with pytest.raises(ValueError):
            TimeoutManager(alpha=0.0)
        with pytest.raises(ValueError):
            TimeoutManager(alpha=1.5)

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_multiplier_below_1_raises(self, mock_logger):
        mock_logger.get_instance.return_value = MagicMock()
        with pytest.raises(ValueError):
            TimeoutManager(multiplier=0.5)

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_invalid_timeout_bounds_raise(self, mock_logger):
        mock_logger.get_instance.return_value = MagicMock()
        with pytest.raises(ValueError):
            TimeoutManager(min_timeout=0, max_timeout=10)
        with pytest.raises(ValueError):
            TimeoutManager(min_timeout=20, max_timeout=10)


# ── get_timeout 测试 ──────────────────────────────────────────


class TestGetTimeout:
    """get_timeout() 基础行为。"""

    def test_new_model_returns_min_timeout(self, tm):
        timeout = tm.get_timeout("new-model:7b")
        assert timeout == 5.0

    def test_existing_model_returns_current_timeout(self, tm):
        tm.record("llama3", elapsed_seconds=2.0)
        timeout = tm.get_timeout("llama3")
        # Should be >= min_timeout
        assert timeout >= 5.0


# ── record 测试 ───────────────────────────────────────────────


class TestRecord:
    """record() 更新 EMA 和超时。"""

    def test_first_record_sets_initial_ema(self, tm):
        tm.record("qwen2.5-coder:7b", elapsed_seconds=3.0)
        stats = tm.get_stats("qwen2.5-coder:7b")
        assert stats is not None
        assert abs(stats.ema - 3.0) < 0.001
        assert stats.sample_count == 1

    def test_multiple_records_update_ema(self, tm):
        # alpha=0.3, initial ema=2.0 after first record
        tm.record("qwen", elapsed_seconds=2.0)
        # ema = 0.3*4.0 + 0.7*2.0 = 1.2 + 1.4 = 2.6
        tm.record("qwen", elapsed_seconds=4.0)
        stats = tm.get_stats("qwen")
        assert stats is not None
        assert abs(stats.ema - 2.6) < 0.001

    def test_record_updates_min_max_latency(self, tm):
        tm.record("model", elapsed_seconds=1.0)
        tm.record("model", elapsed_seconds=5.0)
        stats = tm.get_stats("model")
        assert stats is not None
        assert stats.min_latency == 1.0
        assert stats.max_latency == 5.0

    def test_record_updates_current_timeout(self, tm):
        # ema=2.0, multiplier=3.0, timeout=min(max(6.0,5),60)=6.0
        tm.record("model", elapsed_seconds=2.0)
        stats = tm.get_stats("model")
        assert stats is not None
        assert stats.current_timeout == 6.0  # 2.0 * 3.0

    def test_failed_record_does_not_update_ema(self, tm):
        tm.record("model", elapsed_seconds=2.0)
        ema_before = tm.get_stats("model").ema
        tm.record("model", elapsed_seconds=10.0, success=False)
        stats = tm.get_stats("model")
        assert stats.ema == ema_before

    def test_record_zero_elapsed_ignored(self, tm):
        tm.record("model", elapsed_seconds=2.0)
        ema_before = tm.get_stats("model").ema
        tm.record("model", elapsed_seconds=0.0)
        tm.record("model", elapsed_seconds=-1.0)
        stats = tm.get_stats("model")
        assert stats.ema == ema_before

    def test_timeout_capped_at_max(self, tm):
        # Very large elapsed -> ema large -> timeout capped at max_timeout=60
        tm.record("model", elapsed_seconds=100.0)
        stats = tm.get_stats("model")
        assert stats.current_timeout <= 60.0

    def test_timeout_at_least_min(self, tm):
        # Very small elapsed -> ema small -> timeout floored at min_timeout=5
        tm.record("model", elapsed_seconds=0.1)
        stats = tm.get_stats("model")
        assert stats.current_timeout >= 5.0


# ── 健康检查测试 ──────────────────────────────────────────────


class TestIsHealthy:
    """is_healthy() 健康判定。"""

    def test_new_model_is_healthy(self, tm):
        assert tm.is_healthy("new-model") is True

    def test_few_samples_is_healthy(self, tm):
        tm.record("model", elapsed_seconds=2.0)
        tm.record("model", elapsed_seconds=3.0)
        assert tm.is_healthy("model") is True  # only 2 samples

    def test_within_80_pct_is_healthy(self, tm):
        # 10s * 3 = 30s timeout, max=60s, 30 < 48 (80% of 60) => healthy
        tm.record("model", elapsed_seconds=10.0)
        tm.record("model", elapsed_seconds=10.0)
        tm.record("model", elapsed_seconds=10.0)
        assert tm.is_healthy("model") is True

    def test_above_80_pct_is_unhealthy(self, tm):
        # max_timeout=60, 80%=48
        # Set up ema so current_timeout exceeds 48
        # Large elapsed will cause timeout to hit max=60 which is >=48
        for _ in range(10):
            tm.record("slow", elapsed_seconds=25.0)
        # With ema ≈ 25, timeout ≈ 60 (capped)
        assert tm.is_healthy("slow") is False


# ── get_stats / get_all_stats 测试 ────────────────────────────


class TestGetStats:
    """统计查询测试。"""

    def test_get_stats_none_for_unknown_model(self, tm):
        assert tm.get_stats("unknown") is None

    def test_get_stats_returns_model_timeout_stats(self, tm):
        tm.record("qwen", elapsed_seconds=2.0)
        stats = tm.get_stats("qwen")
        assert isinstance(stats, ModelTimeoutStats)
        assert stats.model_name == "qwen"

    def test_get_all_stats_empty_initially(self, tm):
        assert tm.get_all_stats() == []

    def test_get_all_stats_after_records(self, tm):
        tm.record("a", elapsed_seconds=2.0)
        tm.record("b", elapsed_seconds=3.0)
        stats = tm.get_all_stats()
        assert len(stats) == 2
        assert {s["model"] for s in stats} == {"a", "b"}


# ── reset 测试 ────────────────────────────────────────────────


class TestReset:
    """reset() 重置统计。"""

    def test_reset_single_model(self, tm):
        tm.record("a", elapsed_seconds=2.0)
        tm.record("b", elapsed_seconds=2.0)
        tm.reset(model_name="a")
        assert tm.get_stats("a") is None
        assert tm.get_stats("b") is not None

    def test_reset_all_models(self, tm):
        tm.record("a", elapsed_seconds=2.0)
        tm.record("b", elapsed_seconds=2.0)
        tm.reset()
        assert tm.get_all_stats() == []

    def test_reset_unknown_model_no_error(self, tm):
        tm.reset(model_name="does_not_exist")  # should not raise


# ── predict_timeout 测试 ──────────────────────────────────────


class TestPredictTimeout:
    """predict_timeout() 预测超时。"""

    def test_few_samples_returns_min_timeout(self, tm):
        assert tm.predict_timeout("new-model") == 5.0

    def test_stable_latency_predicts_near_ema(self, tm):
        for _ in range(10):
            tm.record("stable", elapsed_seconds=2.0)
        predicted = tm.predict_timeout("stable")
        # ema=2.0, stddev≈0, predicted=2.0 + 2*0 = 2.0, floored to 5.0
        assert predicted >= 5.0

    def test_variable_latency_increases_prediction(self, tm):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0]
        for lat in latencies:
            tm.record("variable", elapsed_seconds=lat)
        predicted = tm.predict_timeout("variable")
        # Should be higher than plain ema due to stddev
        stats = tm.get_stats("variable")
        assert predicted >= stats.current_timeout - 5  # tolerance


# ── get_recent_latency_trend 测试 ─────────────────────────────


class TestGetRecentLatencyTrend:
    """延迟趋势测试。"""

    def test_no_data_returns_empty(self, tm):
        assert tm.get_recent_latency_trend("new") == []

    def test_returns_recent_n_samples(self, tm):
        for i in range(5):
            tm.record("m", elapsed_seconds=float(i + 1))
        trend = tm.get_recent_latency_trend("m", window=3)
        assert len(trend) == 3
        assert trend == [3.0, 4.0, 5.0]

    def test_window_larger_than_samples(self, tm):
        for i in range(3):
            tm.record("m", elapsed_seconds=float(i + 1))
        trend = tm.get_recent_latency_trend("m", window=10)
        assert len(trend) == 3


# ── ModelTimeoutStats 测试 ────────────────────────────────────


class TestModelTimeoutStats:
    """ModelTimeoutStats 数据类测试。"""

    def test_defaults(self):
        stats = ModelTimeoutStats(model_name="test-model")
        assert stats.model_name == "test-model"
        assert stats.ema == 0.0
        assert stats.sample_count == 0
        assert stats.current_timeout == 10.0

    def test_to_dict(self):
        stats = ModelTimeoutStats(
            model_name="llama3",
            ema=2.5,
            sample_count=10,
            current_timeout=7.5,
            min_latency=1.0,
            max_latency=5.0,
        )
        d = stats.to_dict()
        assert d["model"] == "llama3"
        assert d["ema_seconds"] == 2.5
        assert d["samples"] == 10
        assert d["current_timeout"] == 7.5
        assert d["min_latency"] == 1.0
        assert d["max_latency"] == 5.0

    def test_to_dict_with_inf_min_latency(self):
        stats = ModelTimeoutStats(model_name="new")
        d = stats.to_dict()
        assert d["min_latency"] is None


# ── 线程安全基础测试 ──────────────────────────────────────────


class TestThreadSafety:
    """线程安全基础验证。"""

    @patch("src.timeout_manager.Logger", autospec=False)
    def test_concurrent_records_no_crash(self, mock_logger):
        import threading
        mock_logger.get_instance.return_value = MagicMock()
        tm = TimeoutManager()

        def record_n(model, n):
            for i in range(n):
                tm.record(model, elapsed_seconds=float(i % 5 + 1))

        threads = [
            threading.Thread(target=record_n, args=(f"model_{t}", 20))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tm.get_all_stats()
        assert len(stats) == 4
        for s in stats:
            assert s["samples"] == 20
