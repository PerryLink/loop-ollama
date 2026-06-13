# loop-ollama

*A [**Loop Engineering**](https://github.com/PerryLink/loop-everything) autonomous coding loop engine — turn goals into production code.*

> Zero-cloud, air-gapped autonomous coding loops on local LLMs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![CI](https://github.com/PerryLink/loop-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/PerryLink/loop-ollama/actions)

**LLMO Entity Definition**: This project is an **autonomous coding agent** that **wraps Ollama local LLMs with a self-built ReAct loop engine**, optimized for **fully-local, air-gapped autonomous coding loops** using **Python 3.10+ and the Ollama REST API**.


---

## ✨ Core Features

| Module | Description |
|--------|-------------|
| **ReAct Loop Engine** | Thought / Action / Observation cycle with regex extraction + JSON Schema validation. Zero dependencies on LangChain, CrewAI, or AutoGPT. |
| **3-Tier Fault Tolerance** | Tier-1: 12 regex patterns repair malformed JSON tool calls. Tier-2: simplified prompt retry with strict schema. Tier-3: plain-text extraction via 12 heuristic rules. |
| **5-Level Model Grading** | Queries `/api/show` for params, quantization, and context window. Computes capability score (0.0–1.0) and assigns grade S (>=32B) / A (7–14B) / B (3–7B) / C (1–3B) / D (<1B). |
| **Context Manager** | Manages `keep_alive`, warmup requests, and model lifecycle. Pre-request `/api/ps` check detects unloaded models and auto-recovers with `keep_alive=-1`. |
| **EMA Timeout Management** | Dynamic timeout formula: `base + estimated_tokens / current_tps * 2.0 + tier_penalty`. Real-time token/s tracked with exponential moving average (EMA) smoothing. |
| **Guard Layer** | Safety validation on generated code and tool calls. Strictness scales with model grade — Grade S models get relaxed guards; Grade C/D get strict whitelisting. |
| **7-Tool Registry** | `read_file`, `write_file`, `edit_file`, `run_command`, `glob_search`, `grep_search`, `task_complete` in Ollama-compatible `tool_calls` format. |
| **Auto Model Upgrade** | When a B/C/D-graded model hits a P1-severity issue (core function failure), the detector queries all locally available models and switches to the strongest one (`--auto-upgrade`). |

---

## 🚀 Quick Start

```bash
# Install
pip install loop-ollama

# Run (Ollama must be running: ollama serve)
loop-ollama run --model "llama3" --goal "Build a simple Flask API"
```

**Prerequisites**: Python >= 3.10, Ollama installed and serving (`ollama serve`).

---

## 🙋 FAQ

### Q: Which models deliver the best results?
A: **Grade S** (>=32B params, e.g. Qwen 2.5 32B, Llama 3 70B) delivers near-cloud-quality output. **Grade A** (7–14B, e.g. Llama 3 8B, Mistral 7B) is usable with conservative turn limits. **Grade B and below** (sub-7B) frequently trigger the 3-tier fault tolerance engine — suitable for simple, single-file tasks only.

### Q: Why does the fault tolerance engine need three tiers?
A: Local models produce malformed JSON tool calls far more often than cloud-hosted APIs. Tier-1 (12 regex patterns) fixes common syntax errors. Tier-2 retries with a simplified prompt and strict schema. Tier-3 falls back to plain-text extraction. This layered approach maximizes parse success without wasting tokens on endless retries.

### Q: How does model auto-upgrade decide when to switch?
A: When a B/C/D model encounters a P1-severity issue (task-critical failure), the `ModelDetector` queries Ollama for all locally available models, ranks them by capability score, and submits the recommendation. With `--auto-upgrade`, it switches silently for the remainder of the session.

### Q: What makes loop-ollama different from cloud-based coding agents?
A: **Zero API cost, zero data exfiltration, full offline readiness.** The entire agent — ReAct loop, tool registry, fault tolerance, grading, guards — runs locally against your Ollama instance. It compiles to a single PyInstaller binary for air-gapped environments.

### Q: Can it recover if Ollama unloads the model mid-session?
A: Yes. Before every request, the engine checks `/api/ps`. If the target model is unloaded, it sends a warmup request with `keep_alive=-1` to reload it, then proceeds with the original task.

---

## 🌐 Related Projects

| Project | Description |
|---------|-------------|
| [loop-everything](https://github.com/PerryLink/loop-everything) | Ecosystem hub — meta-repo for all loop projects |
| [loop-aider](https://github.com/PerryLink/loop-aider) | Aider CLI autonomous coding loop |
| [loop-superpowers](https://github.com/PerryLink/loop-superpowers) | Pure Skill mini-loops for Claude Code |
| [loop-hermes](https://github.com/PerryLink/loop-hermes) | Hermes SDK autonomous coding loop |
| [loop-antigravity](https://github.com/PerryLink/loop-antigravity) | Google Gemini API autonomous coding loop |
| [loop-codex](https://github.com/PerryLink/loop-codex) | Codex Desktop CDP autonomous coding loop |
| [loop-copilot](https://github.com/PerryLink/loop-copilot) | GitHub Copilot SDK autonomous coding loop |
| [loop-cursor](https://github.com/PerryLink/loop-cursor) | Cursor IDE SDK autonomous coding loop |
| [loop-opencode](https://github.com/PerryLink/loop-opencode) | OpenCode CLI autonomous coding loop |
| [loop-openclaw](https://github.com/PerryLink/loop-openclaw) | OpenClaw Gateway multi-agent loop generator |
| [loop-deepseek](https://github.com/PerryLink/loop-deepseek) | DeepSeek API autonomous coding loop |
| [loop-claudecode](https://github.com/PerryLink/loop-claudecode) | Reference implementation with OS-level safety gates |

---

## 📄 License

Apache 2.0 © 2026 Perry Link

---

## 中文说明

**loop-ollama** 是一个自主编程智能体，将 Ollama 本地大模型封装为自建 ReAct 循环引擎，专为**全本地、气隙隔离的自主编程开发**优化，零 API 费用、零数据外泄、离线可用。

### 核心功能

| 模块 | 说明 |
|------|------|
| **ReAct 循环引擎** | Thought / Action / Observation 循环，纯正则提取 + JSON Schema 校验，零 LangChain/CrewAI/AutoGPT 依赖 |
| **3 级容错** | Tier-1：12 个正则模式修复畸形 JSON → Tier-2：简化 prompt 重试 → Tier-3：12 个启发式规则纯文本提取 |
| **5 级模型分级** | 查询 `/api/show` 获取参数/量化/上下文窗口，计算能力分数 (0.0–1.0)，分级 S(>=32B)/A(7-14B)/B(3-7B)/C(1-3B)/D(<1B) |
| **Context Manager** | 管理 `keep_alive`、预热请求与模型生命周期；请求前检测 `/api/ps`，自动恢复已卸载模型 |
| **EMA 超时管理** | 动态超时公式：`base + estimated_tokens / current_tps * 2.0 + tier_penalty`，实时 token/s 采用指数移动平均平滑追踪 |
| **Guard Layer 安全** | 对生成代码和工具调用进行安全校验；严格度随模型等级自适应 — S 级放宽，C/D 级严格白名单 |
| **7 工具注册表** | `read_file`、`write_file`、`edit_file`、`run_command`、`glob_search`、`grep_search`、`task_complete` |
| **自动模型升级** | B/C/D 级模型遇到 P1 严重问题时，自动检测本地最强可用模型并切换 (`--auto-upgrade`) |

### 快速开始

```bash
# 安装
pip install loop-ollama

# 运行 (需先启动 Ollama: ollama serve)
loop-ollama run --model "llama3" --goal "构建一个简单的 Flask API"
```

**运行环境**：Python >= 3.10，Ollama 已安装并运行 (`ollama serve`)。

### 常见问题

**问：什么模型效果最好？**
答：**S 级**（>=32B 参数，如 Qwen 2.5 32B、Llama 3 70B）可达到接近云端质量。**A 级**（7–14B，如 Llama 3 8B、Mistral 7B）在保守轮次限制下可用。**B 级及以下**（<7B）频繁触发三级容错，仅适合简单单文件任务。

**问：为什么需要三级容错？**
答：本地模型产生畸形 JSON 工具调用的概率远高于云端 API。Tier-1（12 个正则模式）修复常见语法错误；失败则 Tier-2 以简化 prompt 和严格 schema 重试；仍失败则 Tier-3 降级为纯文本提取。分层策略在不浪费 token 的前提下最大化解析成功率。

**问：模型自动升级如何触发？**
答：当 B/C/D 级模型遇到 P1 严重问题（核心功能故障）时，`ModelDetector` 查询所有本地可用模型，按能力分数排序提交推荐。使用 `--auto-upgrade` 时静默切换，当前会话剩余任务均使用新模型。

**问：与云端编程智能体有何区别？**
答：**零 API 费用、零数据外泄、完全离线可用。** 整个智能体 — ReAct 循环、工具注册表、容错、分级、安全守卫 — 都在本地 Ollama 实例上运行，并可编译为单个 PyInstaller 二进制文件，适用于气隙环境。

**问：Ollama 中途卸载模型能恢复吗？**
答：能。每次请求前引擎检查 `/api/ps`；若目标模型已卸载，自动发送 `keep_alive=-1` 预热请求重新加载，然后继续原任务。

### 相关项目

| 项目 | 说明 |
|------|------|
| [loop-everything](https://github.com/PerryLink/loop-everything) | 生态系统中枢 — 所有 loop 项目的元仓库 |
| [loop-aider](https://github.com/PerryLink/loop-aider) | Aider CLI 自主编程循环 |
| [loop-superpowers](https://github.com/PerryLink/loop-superpowers) | Claude Code 的纯 Skill 微型循环 |
| [loop-hermes](https://github.com/PerryLink/loop-hermes) | Hermes SDK 自主编程循环 |
| [loop-antigravity](https://github.com/PerryLink/loop-antigravity) | Google Gemini API 自主编程循环 |
| [loop-codex](https://github.com/PerryLink/loop-codex) | Codex Desktop CDP 自主编程循环 |
| [loop-copilot](https://github.com/PerryLink/loop-copilot) | GitHub Copilot SDK 自主编程循环 |
| [loop-cursor](https://github.com/PerryLink/loop-cursor) | Cursor IDE SDK 自主编程循环 |
| [loop-opencode](https://github.com/PerryLink/loop-opencode) | OpenCode CLI 自主编程循环 |
| [loop-openclaw](https://github.com/PerryLink/loop-openclaw) | OpenClaw Gateway 多智能体循环生成器 |
| [loop-deepseek](https://github.com/PerryLink/loop-deepseek) | DeepSeek API 自主编程循环 |
| [loop-claudecode](https://github.com/PerryLink/loop-claudecode) | 参考实现，含操作系统级安全闸门 |

### 完成度

当前完成度：**85%**。已完成：ReAct 循环引擎、7 工具注册表、三级容错、5 级模型分级、自动模型升级、EMA 超时管理、模型卸载自动恢复、CLI 入口。待完善：复杂多步任务的收敛检测准确度需更多真实场景验证；C/D 级模型在长上下文场景下的容错成功率需增加启发式规则。

### 许可证

Apache 2.0 © 2026 Perry Link
