"""
loop-ollama —— 本地 AI 编程 Agent 包。

基于 Ollama REST API 的自建 ReAct 自主编程 agent。
零 API 费用、零数据外泄、三层容错对抗弱模型幻觉。
"""

__version__ = "0.1.0"
__author__ = "loop-ollama contributors"
__license__ = "Apache-2.0"

# 公开 API 组件，方便外部导入
from .config import Config  # noqa: F401
from .logger import Logger  # noqa: F401
from .timeout_manager import TimeoutManager, ModelTimeoutStats  # noqa: F401
from .context_manager import ContextManager  # noqa: F401
from .state_manager import StateManager  # noqa: F401
from .model_detector import ModelDetector  # noqa: F401
from .ollama_client import OllamaClient, ChatResponse  # noqa: F401
from .react_loop import ReactLoop  # noqa: F401
from .fault_tolerance import FaultToleranceEngine  # noqa: F401
from .guard_layer import GuardLayer  # noqa: F401
from .convergence_controller import ConvergenceController  # noqa: F401
