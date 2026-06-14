# loop-ollama -- Self-Built ReAct Agent Loop for Local Ollama Models

> 在给定硬件上榨取本地模型的最高编程效能——零 API 费用、零数据外泄、离线可用。 / Squeeze maximum coding performance from local models on your hardware -- zero API cost, zero data leaks, fully offline.

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/PerryLink/loop-ollama)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

## Navigation / 导航

- [English](#english) -- Features, Quick Start, FAQ
- [中文文档](#chinese-docs) -- 功能特性、快速开始、常见问题

---

<a name="english"></a>
## English

**loop-ollama** is a self-built ReAct agent loop for local development with Ollama models -- zero API cost, zero data leaks, fully offline.

### Features

- **Self-built ReAct loop** -- Thought/Action/Observation cycle with regex extraction + JSON Schema validation, no LangChain/CrewAI/AutoGPT
- **7-tool registry** -- read_file, write_file, edit_file, run_command, glob_search, grep_search, task_complete in Ollama tool_calls format
- **Three-tier fault tolerance** -- Tier-1 format repair (12 regex patterns) -> Tier-2 simplified retry -> Tier-3 degraded plain-text extraction
- **Model capability detector** -- queries `/api/show` for params/quantization/context_window, computes capability score (0.0-1.0), assigns grade S/A/B/C/D
- **5-tier model grading** -- S (>=32B) through D (<1B) influences max_turns, tool whitelist, fault tolerance strategy, and guard strictness
- **Auto model upgrade** -- B/C/D models that hit P1 issues auto-switch to the strongest available model on current hardware
- **Dynamic timeout** -- `timeout = base + estimated_tokens / current_tps * 2.0 + tier_penalty` with EMA-smoothed real-time token/s tracking
- **Model unload auto-recovery** -- pre-request `/api/ps` check; if unloaded, sends warmup request with `keep_alive=-1`

### Quick Start

```bash
# Prerequisites: Ollama installed and running, Python >= 3.10
git clone https://github.com/PerryLink/loop-ollama.git
cd loop-ollama
pip install -r requirements.txt

# Run directly
python -m src.cli --goal "Build a simple Flask API" --model llama3

# Or build standalone binary with PyInstaller
pip install pyinstaller
pyinstaller --onefile src/cli.py --name loop-ollama
./dist/loop-ollama --goal "Refactor the auth module"
```

Requirements: Python >= 3.10, Ollama installed and running (`ollama serve`).

### FAQ

**Q: What models work best with loop-ollama?**
A: Grade S models (>=32B parameters, e.g., Qwen 2.5 32B, Llama 3 70B) deliver near-cloud-quality results. Grade A models (7-14B, e.g., Llama 3 8B, Mistral 7B) are usable with more conservative turn limits. Grade B and below (sub-7B) are experimental -- they trigger the three-tier fault tolerance engine frequently and are best for simple, single-file tasks.

**Q: Why does the fault tolerance engine have three tiers?**
A: Local models produce malformed JSON tool calls far more often than cloud APIs. Tier-1 (12 regex patterns) fixes common JSON syntax errors. If that fails, Tier-2 retries with a simplified prompt and strict schema constraints. If that also fails, Tier-3 falls back to plain-text extraction using 12 heuristic rules. This layered approach maximizes the chance of a successful parse without wasting tokens on endless retries.

**Q: How does model auto-upgrade work?**
A: When a B/C/D-graded model encounters a P1-severity issue (core functionality failure), the `ModelDetector` queries Ollama for all locally available models, ranks them by capability score, and prompts the user to switch. If `--auto-upgrade` is set, it switches automatically for the remainder of the session.

---

<a name="chinese-docs"></a>
## 中文文档 / Chinese Docs

**loop-ollama** 在给定硬件上榨取本地模型的最高编程效能——零 API 费用、零数据外泄、离线可用。基于 Ollama REST API 的自建 ReAct agent 循环，编译为单个 PyInstaller 二进制文件。

### 功能特性

- **自建 ReAct 循环** — Thought/Action/Observation 循环，纯正则提取 + JSON Schema 校验，无 LangChain/CrewAI/AutoGPT 依赖
- **7 工具注册表** — read_file、write_file、edit_file、run_command、glob_search、grep_search、task_complete，使用 Ollama tool_calls 格式
- **三级容错** — Tier-1 格式修复（12 正则模式）→ Tier-2 简化重试 → Tier-3 降级纯文本提取
- **模型能力检测器** — 查询 `/api/show` 获取参数/量化/上下文窗口，计算能力分数（0.0-1.0），分级 S/A/B/C/D
- **五级模型分级** — S（>=32B）到 D（<1B），影响 max_turns、工具白名单、容错策略和 guard 严格度
- **自动模型升级** — B/C/D 级模型遇到 P1 问题时自动切换到当前硬件上最强的可用模型
- **动态超时** — `timeout = base + estimated_tokens / current_tps * 2.0 + tier_penalty`，基于 EMA 平滑的实时 token/s 追踪
- **模型卸载自动恢复** — 请求前检查 `/api/ps`；若已卸载，发送 `keep_alive=-1` 的热身请求

### 快速开始

```bash
git clone https://github.com/PerryLink/loop-ollama.git
cd loop-ollama
pip install -r requirements.txt

# 直接运行
python -m src.cli --goal "构建一个简单的 Flask API" --model llama3

# 或使用 PyInstaller 构建独立二进制文件
pip install pyinstaller
pyinstaller --onefile src/cli.py --name loop-ollama
./dist/loop-ollama --goal "重构认证模块"
```

环境要求：Python >= 3.10，Ollama 已安装并运行（`ollama serve`）。

### 常见问题

**问：哪些模型最适合 loop-ollama？**
答：S 级模型（>=32B 参数，如 Qwen 2.5 32B、Llama 3 70B）可提供接近云端质量的结果。A 级模型（7-14B，如 Llama 3 8B、Mistral 7B）在较保守的轮次限制下可用。B 级及以下（<7B）属于实验性质——会频繁触发三级容错引擎，最适合简单的单文件任务。

**问：为什么容错引擎有三个层级？**
答：本地模型产生格式错误的 JSON 工具调用的频率远高于云端 API。Tier-1（12 个正则模式）修复常见的 JSON 语法错误。若失败，Tier-2 使用简化的提示词和严格的 schema 约束重试。若仍失败，Tier-3 降级为使用 12 条启发式规则进行纯文本提取。这种分层方法在不过度浪费 token 的情况下最大化成功解析的概率。

**问：模型自动升级是如何工作的？**
答：当 B/C/D 级模型遇到 P1 严重问题（核心功能故障）时，`ModelDetector` 查询 Ollama 获取所有本地可用模型，按能力分数排序，并提示用户切换。若设置了 `--auto-upgrade`，则自动切换并在当前会话的剩余时间内继续使用。

---

## Related Projects / 相关项目

- [loop-superpowers](https://github.com/PerryLink/loop-superpowers) — pure Skill mini-loops for Claude Code
- [loop-opencode](https://github.com/PerryLink/loop-opencode) — closed-loop driver for OpenCode CLI
- [loop-codex](https://github.com/PerryLink/loop-codex) — dual-channel (JSON-RPC + CDP) driver for Codex Desktop
- [loop-copilot](https://github.com/PerryLink/loop-copilot) — closed-loop driver for GitHub Copilot SDK
- [loop-cursor](https://github.com/PerryLink/loop-cursor) — closed-loop driver for Cursor IDE SDK
- [loop-deepseek](https://github.com/PerryLink/loop-deepseek) — self-built ReAct agent loop for DeepSeek API
- [loop-antigravity](https://github.com/PerryLink/loop-antigravity) — closed-loop driver for Google Antigravity / Gemini
- [loop-openclaw](https://github.com/PerryLink/loop-openclaw) — multi-agent loop config generator for OpenClaw Gateway

## License / 许可证

[Apache License 2.0](./LICENSE) -- see [LICENSE](./LICENSE) for full text.

Copyright 2026 Perry Link
