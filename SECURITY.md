# Security Policy / 安全策略

## Supported Versions / 支持的版本

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

---

## 1. Ollama Local API Security Model / Ollama 本地 API 安全模型

### English

loop-ollama communicates exclusively with a **local Ollama instance** via its REST API (default `http://127.0.0.1:11434`). This architecture provides the following security guarantees:

- **No cloud dependency** — All inference runs on local hardware. No API keys, no external network calls, no third-party intermediaries.
- **Air-gapped by default** — Ollama binds to `127.0.0.1` (localhost) out of the box. loop-ollama does not reconfigure this binding. Ensure your Ollama instance is not exposed to `0.0.0.0` unless you have explicitly configured authentication and TLS.
- **No credential storage** — loop-ollama never reads, stores, or transmits API keys, tokens, or passwords. The Ollama API is unauthenticated by design (localhost-only).
- **Tool execution sandboxing** — The `run_command` tool executes shell commands in the same environment as the agent. **Always review goals before running loop-ollama**, especially when using `--auto-approve` mode, which bypasses human confirmation for tool calls.

**Recommendations:**

1. Never bind Ollama to `0.0.0.0` without a reverse proxy that adds authentication (e.g., nginx + basic auth) and TLS termination.
2. Run loop-ollama in a dedicated user account or container with minimal filesystem permissions.
3. Use `--dry-run` first to audit the agent's planned actions before execution.
4. Set `--max-turns` to a conservative value (default: 20 for S-tier, 10 for A-tier, 5 for B/C/D) to bound agent autonomy.

### 中文

loop-ollama 仅与**本地 Ollama 实例**通过 REST API（默认 `http://127.0.0.1:11434`）通信。该架构提供以下安全保障：

- **无云依赖** — 所有推理在本地硬件上运行。无 API 密钥、无外部网络调用、无第三方中介。
- **默认气隙隔离** — Ollama 开箱即绑定到 `127.0.0.1`（localhost）。loop-ollama 不会重新配置此绑定。请确保你的 Ollama 实例未暴露到 `0.0.0.0`，除非你已显式配置身份验证和 TLS。
- **不存储凭据** — loop-ollama 从不读取、存储或传输 API 密钥、令牌或密码。Ollama API 设计上无身份验证（仅限 localhost）。
- **工具执行沙箱** — `run_command` 工具在与 agent 相同的环境中执行 shell 命令。**运行 loop-ollama 前务必审查目标**，尤其是在使用 `--auto-approve` 模式时，该模式会绕过工具调用的人工确认。

**建议：**

1. 不要将 Ollama 绑定到 `0.0.0.0`，除非前面有反向代理（如 nginx + basic auth）提供身份验证和 TLS 终止。
2. 在专用用户账户或容器中运行 loop-ollama，并给予最小文件系统权限。
3. 先用 `--dry-run` 审计 agent 计划的操作，再正式执行。
4. 将 `--max-turns` 设置为保守值（默认：S 级 20，A 级 10，B/C/D 级 5），以限制 agent 自主权。

---

## 2. Model Security / 模型安全

### English

- **Untrusted model files** — Only run models pulled from Ollama's official library or trusted sources. Malicious GGUF files can embed arbitrary code in model metadata, custom tokenizers, or chat templates. loop-ollama's `ModelDetector` queries `/api/show` to read model parameters and metadata; a crafted response from a compromised model could exploit deserialization paths. Always verify model provenance via `ollama show <model> --modelfile` before use.
- **Prompt injection** — The ReAct loop concatenates user goals, tool outputs, and model responses into a single context window. Model outputs (including tool-generated file contents) are fed back into subsequent turns. A file on disk containing adversarial content (e.g., "IGNORE ALL PREVIOUS INSTRUCTIONS") could influence model behavior when read by `read_file` or `grep_search`. loop-ollama's `GuardLayer` applies output sanitization (line-length truncation, binary detection) before re-feeding content, but this is a heuristic defense, not a guarantee.
- **Quantization risks** — Quantized models (Q4_K_M, Q2_K, etc.) may exhibit degraded reasoning, increasing the likelihood of unsafe tool calls. loop-ollama's capability detector flags quantization levels and downgrades the model grade accordingly, which tightens guard strictness and reduces tool availability.
- **Model output trust boundary** — The model's `run_command` suggestions are executed verbatim by the shell. Never run loop-ollama as root or with elevated privileges. Use `--allowed-tools` to restrict the tool registry to safe subsets (e.g., `read_file,write_file` only, no `run_command`).

### 中文

- **不安全的模型文件** — 仅运行从 Ollama 官方库或可信来源拉取的模型。恶意 GGUF 文件可在模型元数据、自定义分词器或聊天模板中嵌入任意代码。loop-ollama 的 `ModelDetector` 查询 `/api/show` 读取模型参数和元数据；被攻破的模型返回的伪造响应可能利用反序列化路径。使用前务必通过 `ollama show <model> --modelfile` 验证模型来源。
- **提示注入** — ReAct 循环将用户目标、工具输出和模型响应串联到单个上下文窗口中。模型输出（包括工具生成的文件内容）会被反馈到后续轮次中。磁盘上包含对抗性内容的文件（如 "忽略所有之前的指令"）在被 `read_file` 或 `grep_search` 读取时可能影响模型行为。loop-ollama 的 `GuardLayer` 在重新输入内容前会对输出进行清理（行长度截断、二进制检测），但这是启发式防御，并非保证。
- **量化风险** — 量化模型（Q4_K_M、Q2_K 等）可能表现出退化的推理能力，增加不安全工具调用的可能性。loop-ollama 的能力检测器会标记量化级别并相应下调模型等级，从而收紧防护严格度并减少可用工具。
- **模型输出信任边界** — 模型建议的 `run_command` 会被 shell 逐字执行。切勿以 root 或提升权限运行 loop-ollama。使用 `--allowed-tools` 将工具注册表限制为安全子集（如仅 `read_file,write_file`，不含 `run_command`）。

---

## 3. Fault Tolerance Security / 故障容忍安全

### English

loop-ollama's three-tier fault tolerance engine is designed for resilience, not security obfuscation. Each tier has distinct security implications:

- **Tier 1 — Format Repair (12 regex patterns):** Repairs malformed JSON tool calls (missing brackets, unquoted keys, trailing commas, truncated strings). This is a purely syntactic fix — it does not alter the semantic content of the tool call. However, excessively broken JSON could cause the regex engine to enter catastrophic backtracking. loop-ollama caps regex execution time at 500ms per pattern.

- **Tier 2 — Simplified Retry:** Re-issues the prompt with stricter schema constraints and reduced context. The retry uses a fresh Ollama `/api/chat` request — no accumulated state from the failed attempt is carried over except the original goal. This prevents the model from "learning" from its own malformed output.

- **Tier 3 — Degraded Plain-Text Extraction:** Falls back to 12 heuristic rules to extract tool name and arguments from unstructured text. This is the highest-risk tier — the heuristic parser may misinterpret natural language as tool calls. loop-ollama applies the `GuardLayer` output sanitizer to all Tier-3 extracted content and forces `--require-approval` mode (even if `--auto-approve` was set).

- **No data leakage during retries:** Failed parse outputs are logged locally (structured JSON to `logs/`) but never transmitted externally. The `Logger` component redacts paths matching `$HOME` and `$PWD` patterns by default. Sensitive file paths can be added to a deny-list via `Config.sensitive_paths`.

- **Graceful degradation:** When all three tiers fail for a single turn, the agent records a `TIER3_FAILURE` state entry, increments the fault counter, and — if the fault counter exceeds `max_consecutive_faults` (default: 3) — terminates the session with a structured error log. It does not retry indefinitely, preventing denial-of-service on the local Ollama instance.

### 中文

loop-ollama 的三级容错引擎旨在提供韧性，而非安全混淆。每一级都有其特定的安全影响：

- **Tier 1 — 格式修复（12 正则模式）：** 修复格式错误的 JSON 工具调用（缺失括号、未加引号的键、尾部逗号、截断字符串）。这纯粹是语法修正——不会改变工具调用的语义内容。然而，严重破损的 JSON 可能导致正则引擎进入灾难性回溯。loop-ollama 将每条正则的执行时间限制在 500ms。

- **Tier 2 — 简化重试：** 使用更严格的模式约束和精简的上下文重新发送提示。重试使用全新的 Ollama `/api/chat` 请求——除了原始目标外，不会携带失败尝试的任何累积状态。这防止了模型从自身错误输出中"学习"。

- **Tier 3 — 降级纯文本提取：** 退回到 12 条启发式规则，从非结构化文本中提取工具名称和参数。这是风险最高的一级——启发式解析器可能将自然语言误判为工具调用。loop-ollama 对所有 Tier-3 提取的内容应用 `GuardLayer` 输出清理器，并强制启用 `--require-approval` 模式（即使设置了 `--auto-approve`）。

- **重试期间无数据泄露：** 失败的解析输出会记录到本地（结构化 JSON 到 `logs/`），但不会传输到外部。`Logger` 组件默认会脱敏匹配 `$HOME` 和 `$PWD` 模式的路径。敏感文件路径可通过 `Config.sensitive_paths` 添加到黑名单。

- **优雅降级：** 当单轮中所有三级都失败时，agent 会记录 `TIER3_FAILURE` 状态条目，递增故障计数器，如果故障计数器超过 `max_consecutive_faults`（默认：3），则终止会话并输出结构化错误日志。它不会无限重试，防止对本地 Ollama 实例造成拒绝服务。

---

## Reporting a Vulnerability / 报告漏洞

### English

If you discover a security vulnerability in loop-ollama, please **do not** open a public GitHub issue. Instead, email:

**novelnexusai@outlook.com**

Please include:

- A clear description of the vulnerability
- Steps to reproduce (including Ollama version, model, OS, and Python version)
- Whether the vulnerability requires local access, a compromised model file, or a specific Ollama configuration
- Any suggested mitigations

We will acknowledge your report within 72 hours and provide an initial assessment within 7 days. We follow coordinated disclosure: we will agree on a disclosure timeline with you before publishing any advisory.

### 中文

如果你在 loop-ollama 中发现安全漏洞，请**不要**在 GitHub 上公开提 issue。请发送邮件至：

**novelnexusai@outlook.com**

请包含：

- 漏洞的清晰描述
- 复现步骤（包括 Ollama 版本、模型、操作系统和 Python 版本）
- 该漏洞是否需要本地访问、被攻破的模型文件或特定的 Ollama 配置
- 任何建议的缓解措施

我们将在 72 小时内确认收到你的报告，并在 7 天内提供初步评估。我们遵循协调披露原则：在发布任何公告前，我们会与你商定披露时间表。

---

## Security Acknowledgments / 安全致谢

We thank all security researchers and community members who responsibly disclose vulnerabilities. Contributors who report valid security issues will be acknowledged here (with permission).

我们感谢所有负责任地披露漏洞的安全研究人员和社区成员。报告有效安全问题的贡献者将在此处得到致谢（经许可）。

---

Copyright 2026 Perry Link | GitHub: [PerryLink](https://github.com/PerryLink) | Contact: novelnexusai@outlook.com

Licensed under the Apache License, Version 2.0.
