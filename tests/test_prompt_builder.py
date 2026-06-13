"""
PromptBuilder 单元测试。

测试系统提示构建、轮次提示构建、工具定义格式化、
收敛提示和降级提示——覆盖 S/A/B/C/D 五级差异化。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.prompt_builder import PromptBuilder


# ── 示例工具定义 ──────────────────────────────────────────────

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
]


# ── 初始化测试 ────────────────────────────────────────────────


class TestInit:
    """PromptBuilder 初始化测试。"""

    def test_default_grade_is_A(self):
        pb = PromptBuilder()
        assert pb.g == "A"

    def test_grade_uppercased(self):
        pb = PromptBuilder(model_grade="b")
        assert pb.g == "B"

    def test_empty_grade_defaults_to_A(self):
        pb = PromptBuilder(model_grade="")
        assert pb.g == "A"

    def test_none_grade_defaults_to_A(self):
        pb = PromptBuilder(model_grade=None)
        assert pb.g == "A"

    def test_stores_tools(self):
        pb = PromptBuilder(tools=SAMPLE_TOOLS)
        assert pb.t == SAMPLE_TOOLS
        assert len(pb.t) == 2

    def test_default_tools_is_empty(self):
        pb = PromptBuilder()
        assert pb.t == []


# ── 等级判定测试 ──────────────────────────────────────────────


class TestGradeLevel:
    """模型等级 -> 详细度映射测试。"""

    def test_S_grade_is_h(self):
        pb = PromptBuilder(model_grade="S")
        assert pb._l() == "h"

    def test_A_grade_is_h(self):
        pb = PromptBuilder(model_grade="A")
        assert pb._l() == "h"

    def test_B_grade_is_m(self):
        pb = PromptBuilder(model_grade="B")
        assert pb._l() == "m"

    def test_C_grade_is_l(self):
        pb = PromptBuilder(model_grade="C")
        assert pb._l() == "l"

    def test_D_grade_is_l(self):
        pb = PromptBuilder(model_grade="D")
        assert pb._l() == "l"


# ── 阶段映射测试 ──────────────────────────────────────────────


class TestPhaseMapping:
    """阶段名词/描述映射测试。"""

    def test_init_phase(self):
        pb = PromptBuilder()
        assert pb._n("init") == "理解"
        assert pb._d("init") == "拆解需求"

    def test_analyzing_phase(self):
        pb = PromptBuilder()
        assert pb._n("analyzing") == "规划"
        assert pb._d("analyzing") == "分析代码定计划"

    def test_executing_phase(self):
        pb = PromptBuilder()
        assert pb._n("executing") == "执行"
        assert pb._d("executing") == "读写文件运行命令"

    def test_converging_phase(self):
        pb = PromptBuilder()
        assert pb._n("converging") == "验证"
        assert pb._d("converging") == "检查结果确认完成"

    def test_unknown_phase_falls_back_to_init(self):
        pb = PromptBuilder()
        assert pb._n("unknown") == "理解"
        assert pb._d("unknown") == "拆解需求"


# ── 系统提示构建测试 ──────────────────────────────────────────


class TestBuildSystemPrompt:
    """系统提示构建测试。"""

    def test_high_grade_prompt_contains_security(self):
        pb = PromptBuilder(model_grade="S")
        prompt = pb.build_system_prompt(phase="init", task="测试")
        assert "安全" in prompt
        assert "不删不危" in prompt or "禁删" in prompt

    def test_high_grade_prompt_contains_phase(self):
        pb = PromptBuilder(model_grade="A")
        prompt = pb.build_system_prompt(phase="analyzing", task="分析代码")
        assert "规划" in prompt

    def test_mid_grade_prompt_shorter(self):
        pb_h = PromptBuilder(model_grade="A")
        pb_m = PromptBuilder(model_grade="B")
        ph = pb_h.build_system_prompt(phase="init", task="t")
        pm = pb_m.build_system_prompt(phase="init", task="t")
        assert len(pm) < len(ph)

    def test_low_grade_prompt_has_examples(self):
        pb = PromptBuilder(model_grade="D")
        prompt = pb.build_system_prompt(phase="init", task="任务")
        assert "示例" in prompt
        assert "read file_path" in prompt

    def test_prompt_includes_task(self):
        pb = PromptBuilder(model_grade="S")
        prompt = pb.build_system_prompt(phase="init", task="写一个排序函数")
        assert "写一个排序函数" in prompt

    def test_prompt_includes_tools_when_provided(self):
        pb = PromptBuilder(model_grade="A", tools=SAMPLE_TOOLS)
        prompt = pb.build_system_prompt(phase="init", task="t")
        assert "可用工具" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt

    def test_prompt_no_tools_section_when_empty(self):
        pb = PromptBuilder(model_grade="A", tools=[])
        prompt = pb.build_system_prompt(phase="init", task="t")
        assert "可用工具" not in prompt


# ── 工具定义格式化测试 ────────────────────────────────────────


class TestGetToolDefinitionsText:
    """工具定义文本格式化测试。"""

    def test_empty_tools_returns_empty_string(self):
        pb = PromptBuilder(tools=[])
        assert pb.get_tool_definitions_text() == ""

    def test_formats_tool_name_and_description(self):
        pb = PromptBuilder(tools=SAMPLE_TOOLS)
        text = pb.get_tool_definitions_text()
        assert "read_file" in text
        assert "读取文件内容" in text
        assert "write_file" in text

    def test_formats_required_params_with_asterisk(self):
        pb = PromptBuilder(tools=SAMPLE_TOOLS)
        text = pb.get_tool_definitions_text()
        assert "*" in text  # required params marked with *

    def test_no_function_key_falls_back(self):
        tool = {"name": "simple_tool", "description": "just a tool"}
        pb = PromptBuilder(tools=[tool])
        text = pb.get_tool_definitions_text()
        assert "simple_tool" in text


# ── 轮次提示测试 ──────────────────────────────────────────────


class TestBuildTurnPrompt:
    """轮次提示构建测试。"""

    def test_basic_turn_prompt(self):
        pb = PromptBuilder(model_grade="A")
        state = {"task": "写代码", "turn": 1, "max_turns": 10, "phase": "init",
                 "modified_files": [], "converged": False}
        prompt = pb.build_turn_prompt(state)
        assert "写代码" in prompt
        assert "回合 1/10" in prompt
        assert "未收敛" in prompt

    def test_converged_state_in_turn_prompt(self):
        pb = PromptBuilder()
        state = {"task": "t", "turn": 3, "max_turns": 5, "phase": "converging",
                 "modified_files": [], "converged": True}
        prompt = pb.build_turn_prompt(state)
        assert "已收敛" in prompt or "converged" in prompt.lower()

    def test_modified_files_listed(self):
        pb = PromptBuilder()
        state = {"task": "t", "turn": 2, "max_turns": 5, "phase": "executing",
                 "modified_files": ["/a.py", "/b.py"], "converged": False}
        prompt = pb.build_turn_prompt(state)
        assert "/a.py" in prompt
        assert "/b.py" in prompt

    def test_last_observation_included(self):
        pb = PromptBuilder()
        state = {"task": "t", "turn": 2, "max_turns": 5, "phase": "executing",
                 "modified_files": [], "converged": False}
        prompt = pb.build_turn_prompt(state, last_observation="文件已修改")
        assert "文件已修改" in prompt


# ── 收敛提示测试 ──────────────────────────────────────────────


class TestBuildConvergencePrompt:
    """收敛提示构建测试。"""

    def test_convergence_prompt_contains_task(self):
        pb = PromptBuilder()
        state = {"task": "实现排序算法", "phase": "converging"}
        prompt = pb.build_convergence_prompt(state)
        assert "实现排序算法" in prompt
        assert "CONVERGED" in prompt

    def test_convergence_prompt_has_checklist(self):
        pb = PromptBuilder()
        state = {"task": "t", "phase": "converging"}
        prompt = pb.build_convergence_prompt(state)
        assert "满足" in prompt
        assert "遗漏" in prompt


# ── 降级提示测试 ──────────────────────────────────────────────


class TestBuildDegradedPrompt:
    """降级提示构建测试。"""

    def test_degraded_prompt_contains_task(self):
        pb = PromptBuilder()
        state = {"task": "创建配置文件", "turn": 3, "modified_files": []}
        prompt = pb.build_degraded_prompt(state)
        assert "降级模式" in prompt
        assert "创建配置文件" in prompt
        assert "回合" in prompt or str(3) in prompt

    def test_degraded_prompt_lists_modified_files(self):
        pb = PromptBuilder()
        state = {"task": "t", "turn": 4,
                 "modified_files": ["/tmp/a.py", "/tmp/b.py"]}
        prompt = pb.build_degraded_prompt(state)
        assert "/tmp/a.py" in prompt or "a.py" in prompt

    def test_degraded_prompt_no_files(self):
        pb = PromptBuilder()
        state = {"task": "t", "turn": 2, "modified_files": []}
        prompt = pb.build_degraded_prompt(state)
        assert prompt  # should not crash
