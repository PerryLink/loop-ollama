"""
loop-ollama Logger 单元测试。

测试 Logger 单例模式、分级日志、结构化字段、
turn 事件、容错追踪、线程安全。
"""

import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import Logger


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """每个测试前重置 Logger 单例状态。"""
    Logger._instance = None
    yield
    Logger._instance = None


@pytest.fixture
def logger(temp_dir):
    """在临时目录中创建 Logger 实例。"""
    return Logger(log_dir=temp_dir, level="DEBUG")


# ── Tests: 单例模式 ─────────────────────────────────────────────


def test_logger_singleton_same_instance():
    """两次创建 Logger 应为同一实例。"""
    log1 = Logger()
    log2 = Logger()
    assert log1 is log2


def test_logger_singleton_same_log_file():
    """单例 Logger 应共享 log_file 属性。"""
    log1 = Logger(log_dir="/tmp/singleton_test_1")
    log2 = Logger()
    assert log1.log_file == log2.log_file


# ── Tests: 初始化 ───────────────────────────────────────────────


def test_logger_creates_log_dir(temp_dir):
    """Logger 自动创建日志目录。"""
    log_path = os.path.join(temp_dir, "subdir", "logs")
    log = Logger(log_dir=log_path)
    assert os.path.isdir(log_path)


def test_logger_creates_log_file(logger, temp_dir):
    """Log event 后应创建 runs.log 文件。"""
    logger.log_event("INFO", "test message")
    log_file = os.path.join(temp_dir, "runs.log")
    assert os.path.isfile(log_file)


def test_logger_default_level_is_info():
    """默认日志级别为 INFO。"""
    log = Logger()
    assert log.level == "INFO"


# ── Tests: 分级日志 ─────────────────────────────────────────────


def test_log_info_writes(logger, temp_dir):
    """INFO 级别日志应写入文件。"""
    logger.log_event("INFO", "hello world")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "hello world" in content
    assert "INFO" in content


def test_log_warn_writes(logger, temp_dir):
    """WARN 级别日志应写入文件。"""
    logger.log_event("WARN", "warning message")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "warning message" in content
    assert "WARN" in content


def test_log_error_writes(logger, temp_dir):
    """ERROR 级别日志应写入文件。"""
    logger.log_event("ERROR", "error occurred")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "error occurred" in content
    assert "ERROR" in content


def test_log_debug_writes_when_level_debug(logger, temp_dir):
    """DEBUG 级别在 level=DEBUG 时应写入。"""
    logger.log_event("DEBUG", "debug info")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "debug info" in content


def test_log_debug_filtered_when_level_info(temp_dir):
    """DEBUG 在 level=INFO 时应被过滤。"""
    log = Logger(log_dir=temp_dir, level="INFO")
    log.log_event("DEBUG", "should not appear")
    log_file = os.path.join(temp_dir, "runs.log")
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            content = f.read()
        assert "should not appear" not in content


# ── Tests: 结构化字段 ───────────────────────────────────────────


def test_log_with_extra_fields(logger, temp_dir):
    """日志应包含额外结构化字段。"""
    logger.log_event("INFO", "turn start", turn=5, model="llama3")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "turn" in content
    assert "llama3" in content


def test_log_timestamp_present(logger, temp_dir):
    """每条日志应包含时间戳。"""
    logger.log_event("INFO", "timestamped")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    # 检查包含 ISO 8601 时间格式
    assert "202" in content


def test_log_multiple_events(logger, temp_dir):
    """多条日志事件应全部写入。"""
    for i in range(10):
        logger.log_event("INFO", f"event_{i}")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 10


# ── Tests: 便捷方法 ─────────────────────────────────────────────


def test_log_info_shorthand(logger, temp_dir):
    """Logger.info() 快捷方法。"""
    logger.info("short info")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "short info" in content
    assert "INFO" in content


def test_log_warn_shorthand(logger, temp_dir):
    """Logger.warn() 快捷方法。"""
    logger.warn("short warn")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "short warn" in content
    assert "WARN" in content


def test_log_error_shorthand(logger, temp_dir):
    """Logger.error() 快捷方法。"""
    logger.error("short error")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "short error" in content
    assert "ERROR" in content


def test_log_debug_shorthand(logger, temp_dir):
    """Logger.debug() 快捷方法。"""
    logger.debug("short debug")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "short debug" in content
    assert "DEBUG" in content


# ── Tests: Turn 生命周期 ────────────────────────────────────────


def test_log_turn_start(logger, temp_dir):
    """log_turn 记录 turn 事件。"""
    logger.log_turn(turn=3, tool="read_file", result="ok", duration_ms=0, model="test-model")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "turn" in content.lower()


def test_log_turn_end(logger, temp_dir):
    """log_turn 记录 turn 结束及耗时。"""
    logger.log_turn(turn=3, tool="write_file", result="ok", duration_ms=1500)
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "1500" in content


# ── Tests: fault 事件 ───────────────────────────────────────────


def test_log_fault_event(logger, temp_dir):
    """Tier-1 容错事件应记录修复详情。"""
    logger.log_tier1_repair(
        turn=1, pattern_id="T1-001", snippet='{"bad json'
    )
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "TIER1" in content or "tier" in content.lower()


def test_log_fault_tier3(logger, temp_dir):
    """Tier-3 退化事件应记录。"""
    logger.log_tier3_extraction(
        turn=12, rule_id="T3-005", confidence=0.65
    )
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    assert "TIER3" in content or "tier" in content.lower()


# ── Tests: 线程安全 ─────────────────────────────────────────────


def test_logger_thread_safety(logger, temp_dir):
    """并发写入不应丢失数据或崩溃。"""
    errors = []

    def write_logs(thread_id):
        try:
            for i in range(20):
                logger.log_event("INFO", f"thread_{thread_id}_msg_{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_logs, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 80  # 4 threads * 20 messages


# ── Tests: 特殊字符 ─────────────────────────────────────────────


def test_log_unicode_message(logger, temp_dir):
    """Unicode 消息应正确写入。"""
    logger.log_event("INFO", "中文日志消息 — 日本語テスト")
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "中文日志消息" in content


def test_log_special_chars(logger, temp_dir):
    """特殊字符应正确转义。"""
    logger.log_event("INFO", '{"key": "value", "list": [1,2,3]}')
    log_file = os.path.join(temp_dir, "runs.log")
    with open(log_file, "r") as f:
        content = f.read()
    # JSON 片段应原样保存
    assert "key" in content
    assert "value" in content
