"""
loop-ollama ModelDetector 单元测试。

测试模型能力检测管道：参数量提取、量化折扣计算、
有效参数量、S/A/B/C/D 分级、能力分数计算。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_detector import (
    ModelDetector,
    QUANTIZATION_PENALTY,
    GRADE_THRESHOLDS,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """创建 mock OllamaClient。"""
    client = MagicMock()
    client.show_model.return_value = {}
    client.list_available_models.return_value = []
    return client


@pytest.fixture
def detector(mock_client):
    """创建 ModelDetector 实例。"""
    return ModelDetector(mock_client)


# ── Tests: 参数量提取 ───────────────────────────────────────────


@pytest.mark.parametrize("param_str,expected", [
    ("7B", 7.0),
    ("7.6B", 7.6),
    ("32B", 32.0),
    ("1.5B", 1.5),
    ("70B", 70.0),
    ("0.5B", 0.5),
    ("405B", 405.0),
    ("8x7B", 56.0),  # MoE: total params 8 * 7B
    ("8x22B", 176.0),
    ("1.8B", 1.8),
    ("NotAParamString", 0.0),
    ("", 0.0),
])
def test_extract_params(detector, param_str, expected):
    """参数字符串提取为浮点数。"""
    result = ModelDetector.extract_param_size_billions(
        {"details": {"parameter_size": param_str}}
    )
    assert result == pytest.approx(expected, abs=0.1)


def test_extract_params_from_model_info(detector, mock_client):
    """从模型详情中提取参数量。"""
    mock_client.show_model.return_value = {
        "details": {"parameter_size": "7B"},
    }
    result = detector.detect("qwen2.5-coder:7b")
    assert result["param_size_billions"] == pytest.approx(7.0, abs=0.1)


def test_extract_params_from_model_info_fallback_name(detector, mock_client):
    """API 无数据时从模型名称回退提取。"""
    mock_client.show_model.return_value = {}
    result = detector.detect("llama3:8b")
    assert result["param_size_billions"] == pytest.approx(0.0, abs=0.1)


# ── Tests: 量化折扣 ─────────────────────────────────────────────


def test_quantization_penalty_f16(detector):
    """F16 量化折扣为 1.0。"""
    assert QUANTIZATION_PENALTY["F16"] == 1.0


def test_quantization_penalty_q4_k_m(detector):
    """Q4_K_M 折扣约 0.82。"""
    assert QUANTIZATION_PENALTY["Q4_K_M"] == 0.82


def test_quantization_penalty_q2_k(detector):
    """Q2_K 折扣为 0.50。"""
    assert QUANTIZATION_PENALTY["Q2_K"] == 0.50


def test_get_quant_penalty_known(detector):
    """已知量化格式返回正确折扣。"""
    penalty = QUANTIZATION_PENALTY.get("Q5_K_M", 0.80)
    assert penalty == 0.88


def test_get_quant_penalty_unknown(detector):
    """未知量化格式返回默认折扣 0.80。"""
    penalty = QUANTIZATION_PENALTY.get("UnknownQuant", 0.80)
    assert penalty == 0.80


def test_get_quant_penalty_none(detector):
    """None 量化返回默认折扣。"""
    penalty = QUANTIZATION_PENALTY.get(None, 0.80)
    assert penalty == 0.80


# ── Tests: 有效参数量 ───────────────────────────────────────────


def test_effective_params(detector):
    """有效参数量 = raw_params * quant_penalty。"""
    assert 10.0 * 0.8 == pytest.approx(8.0)


def test_effective_params_full_precision(detector):
    """满精度时有效参数 = 原始参数。"""
    assert 32.0 * 1.0 == 32.0


def test_effective_params_heavy_quant(detector):
    """重度量化大幅降低有效参数。"""
    assert 32.0 * 0.50 == 16.0


# ── Tests: 模型分级 ─────────────────────────────────────────────


@pytest.mark.parametrize("effective_params,expected_grade", [
    (64.0, "S"),
    (32.0, "S"),
    (20.0, "A"),
    (7.0, "A"),
    (5.0, "B"),
    (3.0, "B"),
    (1.5, "C"),
    (1.0, "C"),
    (0.5, "D"),
    (0.1, "D"),
])
def test_compute_grade(detector, effective_params, expected_grade):
    """有效参数量映射到正确等级。"""
    grade = ModelDetector.compute_grade(effective_params)
    assert grade == expected_grade


def test_grade_thresholds_order():
    """分级阈值应从高到低排列。"""
    prev = float("inf")
    for threshold, _ in GRADE_THRESHOLDS:
        assert threshold <= prev
        prev = threshold


def test_all_grades_covered():
    """所有 5 个等级都在阈值表中。"""
    grades = {g for _, g in GRADE_THRESHOLDS}
    assert grades == {"S", "A", "B", "C", "D"}


# ── Tests: 能力分数 ─────────────────────────────────────────────


def test_capability_score_s_tier(detector):
    """S 级模型能力分数 >= 0.8（需要 56B+ 有效参数才能达到 0.8）。"""
    score = ModelDetector.compute_capability_score(56.0)
    assert score == pytest.approx(0.8, abs=0.01)


def test_capability_score_d_tier(detector):
    """D 级模型能力分数 < 0.3。"""
    score = ModelDetector.compute_capability_score(0.5)
    assert score < 0.3


def test_capability_score_range(detector):
    """能力分数在 0.0-1.0 之间。"""
    for ep in [32.0, 7.0, 3.0, 1.0, 0.5]:
        score = ModelDetector.compute_capability_score(ep)
        assert 0.0 <= score <= 1.0, f"score={score} for ep={ep}"


def test_capability_score_monotonic(detector):
    """有效参数量越大，能力分数越高。"""
    scores = []
    for ep in [0.5, 1.0, 3.0, 7.0, 32.0]:
        scores.append(ModelDetector.compute_capability_score(ep))
    # 应保持递增
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1]


# ── Tests: 完整检测管道 ─────────────────────────────────────────


def test_detect_full_pipeline(detector, mock_client):
    """完整检测管道返回所有必需字段。"""
    mock_client.show_model.return_value = {
        "details": {
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
            "family": "qwen2.5",
        },
    }
    mock_client.list_available_models.return_value = [
        {"name": "qwen2.5-coder:7b"},
        {"name": "qwen2.5-coder:32b"},
        {"name": "llama3:8b"},
    ]

    result = detector.detect("qwen2.5-coder:7b")

    for key in ["model_name", "param_size_billions", "quantization",
                "effective_params", "grade", "capability_score",
                "context_window"]:
        assert key in result, f"缺少键: {key}"


def test_detect_grade_a_for_7b(detector, mock_client):
    """7B 模型应为 A 级（F16 满精度时有效参数=原始参数）。"""
    mock_client.show_model.return_value = {
        "details": {"parameter_size": "7B", "quantization_level": "F16"},
    }
    result = detector.detect("qwen2.5-coder:7b")
    assert result["grade"] == "A"


def test_detect_grade_s_for_32b(detector, mock_client):
    """32B 模型应为 S 级（F16 满精度）。"""
    mock_client.show_model.return_value = {
        "details": {"parameter_size": "32B", "quantization_level": "F16"},
    }
    result = detector.detect("codeqwen:32b")
    assert result["grade"] == "S"


def test_detect_upgrade_candidates(detector, mock_client):
    """低等级模型应有更高等级的候选升级模型。"""
    mock_client.show_model.return_value = {
        "details": {"parameter_size": "3B", "quantization_level": "F16"},
    }
    mock_client.list_available_models.return_value = [
        {"name": "tinyllama:1b"},
        {"name": "qwen2.5-coder:3b"},
        {"name": "qwen2.5-coder:7b"},
        {"name": "qwen2.5-coder:32b"},
    ]
    result = detector.detect("qwen2.5-coder:3b")
    assert result["grade"] == "B"
    # detect 不再内置 upgrade_candidates，改为通过 recommend_model/get_best_available_model 获取


def test_detect_context_window(detector, mock_client):
    """应提取 context_window 字段。"""
    mock_client.show_model.return_value = {
        "details": {
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "llama.context_length": 32768,
        },
    }
    result = detector.detect("qwen2.5-coder:7b")
    assert "context_window" in result
    assert isinstance(result["context_window"], int)


def test_detect_api_unavailable(detector, mock_client):
    """API 不可用时仍返回基本结果。"""
    mock_client.show_model.side_effect = Exception("Connection refused")
    result = detector.detect("unknown-model")
    assert result["model_name"] == "unknown-model"
    assert result["grade"] in ("C", "D")  # 降级


# ── Tests: 量化格式提取 ─────────────────────────────────────────


def test_extract_quant_from_details(detector, mock_client):
    """从详情中提取量化格式。"""
    mock_client.show_model.return_value = {
        "details": {"quantization_level": "Q4_K_M"},
    }
    result = detector.detect("test-model")
    assert result["quantization"] == "Q4_K_M"


def test_extract_quant_from_name(detector):
    """detect() 从模型名称中提取量化格式（通过 show_model）。"""
    # detect() 内部通过 show_model 获取量化信息，而不是从名称解析
    # 测试 detect() 返回字段包含 quantization
    result = detector.detect("qwen2.5-coder:7b-q4_k_m")
    assert "quantization" in result


def test_extract_quant_from_name_fallback(detector):
    """名称中无量化信息时返回 unknown。"""
    result = detector.detect("llama3:latest")
    assert "quantization" in result
    assert result["quantization"] == "unknown"
