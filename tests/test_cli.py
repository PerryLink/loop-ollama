"""Tests for loop-ollama CLI entry module (src/cli.py)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.cli import (
    _build_parser,
    _print_header,
    cmd_version,
    main,
)


class TestBuildParser:
    """Tests for CLI argument parser construction."""

    def test_parser_returns_argparser(self):
        parser = _build_parser()
        assert parser.prog == "loop-ollama"

    def test_parser_version_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_check_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--check"])
        assert args.check is True

    def test_parser_init_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--init"])
        assert args.init is True

    def test_parser_model_option_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.model is None

    def test_parser_model_option_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--model", "qwen2.5-coder:7b"])
        assert args.model == "qwen2.5-coder:7b"

    def test_parser_config_option(self):
        parser = _build_parser()
        args = parser.parse_args(["--config", "/custom/config.json"])
        assert args.config == "/custom/config.json"

    def test_parser_task_option(self):
        parser = _build_parser()
        args = parser.parse_args(["--task", "fix the bug in main.py"])
        assert args.task == "fix the bug in main.py"

    def test_parser_safe_mode_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.safe is False
        assert args.auto is True

    def test_parser_safe_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--safe"])
        assert args.safe is True
        assert args.unsafe is False

    def test_parser_unsafe_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--unsafe"])
        assert args.unsafe is True

    def test_parser_mutually_exclusive_safe_auto_error(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--safe", "--auto"])

    def test_parser_help_message(self):
        parser = _build_parser()
        help_text = parser.format_help()
        assert "--version" in help_text
        assert "--check" in help_text
        assert "--init" in help_text
        assert "--model" in help_text
        assert "--task" in help_text
        assert "--safe" in help_text
        assert "--unsafe" in help_text


class TestPrintHeader:
    """Tests for the _print_header banner function."""

    def test_print_header_outputs_string(self, capsys):
        _print_header()
        captured = capsys.readouterr()
        assert "loop-ollama" in captured.out

    def test_print_header_contains_version(self, capsys):
        _print_header()
        captured = capsys.readouterr()
        assert "loop-ollama v" in captured.out


class TestCmdVersion:
    """Tests for the --version command handler."""

    def test_cmd_version_returns_zero(self):
        assert cmd_version() == 0

    def test_cmd_version_prints_version(self, capsys):
        cmd_version()
        captured = capsys.readouterr()
        assert "loop-ollama" in captured.out


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_main_version_exits_zero(self):
        assert main(["--version"]) == 0

    def test_main_no_args_prints_help(self, capsys):
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "loop-ollama" in captured.out

    @patch("src.cli.Config")
    def test_main_with_task_and_model(self, mock_config_cls):
        mock_config = MagicMock()
        mock_config.default_model = "qwen2.5-coder:7b"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.return_value = mock_config

        result = main(["--model", "qwen2.5-coder:7b", "--task", "hello"])
        assert result == 0

    @patch("src.cli.HardwareDetector")
    @patch("src.cli.OllamaClient")
    @patch("src.cli.ModelDetector")
    @patch("src.cli.Config")
    def test_main_check_flag_runs_check(
        self, mock_config_cls, mock_model_detector_cls,
        mock_client_cls, mock_hw_cls
    ):
        mock_config = MagicMock()
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.return_value = mock_config

        mock_hw = MagicMock()
        mock_hw.detect.return_value = {
            "gpu_name": "NVIDIA RTX 4090",
            "vram_gb": 24.0,
            "ram_gb": 64.0,
            "cpu_cores": 16,
            "platform": "linux",
        }
        mock_hw_cls.return_value = mock_hw

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.list_available_models.return_value = [
            {"name": "qwen2.5-coder:7b"}
        ]
        mock_client_cls.return_value = mock_client

        mock_detector = MagicMock()
        mock_detector.detect_all_available.return_value = [
            {
                "model_name": "qwen2.5-coder:7b",
                "param_size_billions": 7.6,
                "quantization": "Q4_K_M",
                "grade": "A",
                "capability_score": 0.75,
            }
        ]
        mock_detector.recommend_model.return_value = "qwen2.5-coder:7b"
        mock_model_detector_cls.return_value = mock_detector

        result = main(["--check"])
        assert result == 0
        mock_client.health_check.assert_called_once()

    @patch("src.cli.HardwareDetector")
    @patch("src.cli.OllamaClient")
    @patch("src.cli.ModelDetector")
    @patch("src.cli.Config")
    def test_main_check_health_failure_returns_1(
        self, mock_config_cls, mock_model_detector_cls,
        mock_client_cls, mock_hw_cls
    ):
        mock_config = MagicMock()
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.return_value = mock_config

        mock_hw = MagicMock()
        mock_hw.detect.return_value = {
            "gpu_name": None,
            "vram_gb": 0.0,
            "ram_gb": 8.0,
            "cpu_cores": 4,
            "platform": "linux",
        }
        mock_hw_cls.return_value = mock_hw

        mock_client = MagicMock()
        mock_client.health_check.return_value = False
        mock_client_cls.return_value = mock_client

        result = main(["--check"])
        assert result == 1

    @patch("src.cli.HardwareDetector")
    @patch("src.cli.OllamaClient")
    @patch("src.cli.ModelDetector")
    @patch("src.cli.Config")
    def test_main_check_no_models_returns_1(
        self, mock_config_cls, mock_model_detector_cls,
        mock_client_cls, mock_hw_cls
    ):
        mock_config = MagicMock()
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.return_value = mock_config

        mock_hw = MagicMock()
        mock_hw.detect.return_value = {
            "gpu_name": None, "vram_gb": 0.0,
            "ram_gb": 8.0, "cpu_cores": 4, "platform": "linux",
        }
        mock_hw_cls.return_value = mock_hw

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.list_available_models.return_value = []
        mock_client_cls.return_value = mock_client

        result = main(["--check"])
        assert result == 1

    @patch("src.cli.HardwareDetector")
    @patch("src.cli.OllamaClient")
    @patch("src.cli.ModelDetector")
    @patch("src.cli.Config")
    def test_main_init_succeeds(
        self, mock_config_cls, mock_model_detector_cls,
        mock_client_cls, mock_hw_cls
    ):
        mock_config = MagicMock()
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.default_model = None
        mock_config_cls.return_value = mock_config

        mock_hw = MagicMock()
        mock_hw.detect.return_value = {
            "gpu_name": "NVIDIA RTX 3060",
            "vram_gb": 12.0,
            "ram_gb": 32.0,
            "cpu_cores": 8,
            "platform": "linux",
        }
        mock_hw_cls.return_value = mock_hw

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.list_available_models.return_value = [
            {"name": "qwen2.5-coder:7b"}
        ]
        mock_client_cls.return_value = mock_client

        mock_detector = MagicMock()
        mock_detector.detect_all_available.return_value = [
            {
                "model_name": "qwen2.5-coder:7b",
                "param_size_billions": 7.6,
                "quantization": "Q4_K_M",
                "grade": "A",
                "capability_score": 0.75,
            }
        ]
        mock_detector.recommend_model.return_value = "qwen2.5-coder:7b"
        mock_model_detector_cls.return_value = mock_detector

        result = main(["--init"])
        assert result == 0

    @patch("src.cli.HardwareDetector")
    @patch("src.cli.OllamaClient")
    @patch("src.cli.ModelDetector")
    @patch("src.cli.Config")
    def test_main_init_no_models_returns_0(
        self, mock_config_cls, mock_model_detector_cls,
        mock_client_cls, mock_hw_cls
    ):
        mock_config = MagicMock()
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.return_value = mock_config

        mock_hw = MagicMock()
        mock_hw.detect.return_value = {
            "gpu_name": None, "vram_gb": 0.0,
            "ram_gb": 8.0, "cpu_cores": 4, "platform": "linux",
        }
        mock_hw_cls.return_value = mock_hw

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.list_available_models.return_value = []
        mock_client_cls.return_value = mock_client

        mock_detector = MagicMock()
        mock_detector.detect_all_available.return_value = []
        mock_model_detector_cls.return_value = mock_detector

        result = main(["--init"])
        assert result == 0

    @patch("src.cli.OllamaClient")
    def test_main_init_ollama_down_returns_1(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.health_check.return_value = False
        mock_client_cls.return_value = mock_client

        with patch("src.cli.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.ollama_base_url = "http://localhost:11434"
            mock_config_cls.return_value = mock_config

            result = main(["--init"])
            assert result == 1

    def test_main_sys_argv_integration(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["loop-ollama", "--version"])
        assert main() == 0
