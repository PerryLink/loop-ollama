# Changelog

All notable changes to loop-ollama will be documented in this file.

## [0.1.0] - 2026-06-13

### Added

- Self-built ReAct loop engine with Thought/Action/Observation cycle
- 7-tool registry: read_file, write_file, edit_file, run_command, glob_search, grep_search, task_complete
- Three-tier fault tolerance: JSON repair (12 regex) -> simplified retry -> degraded plain-text extraction
- Model capability detector via Ollama `/api/show` endpoint
- 5-tier model grading (S/A/B/C/D) influencing max_turns, tool whitelist, and fault tolerance strategy
- Auto model upgrade for B/C/D models hitting P1 severity issues
- Dynamic timeout calculation with EMA-smoothed TPS tracking
- Model unload auto-recovery with pre-request `/api/ps` check and warmup
- Guard layer with L0-L4 safety tiers (catastrophic, system-dangerous, filesystem, network, path protection)
- Hardware detector (GPU VRAM, RAM, CPU, platform detection)
- Convergence controller with counter-based termination logic
- Context manager with state-based conversation trimming
- Structured logger with singleton pattern and thread safety
- State manager with atomic JSON persistence
- CLI entry point with full argument parsing
- PyInstaller single-binary compilation support
- 167 passing unit tests covering all core modules
- REGEX pattern library (heuristics.json, patterns.json)
- Comprehensive pytest conftest with shared fixtures and mocks
- Benchmark suite in `.benchmarks/` directory

### Infrastructure

- GitHub Actions CI/CD pipeline (Python 3.10-3.12, ubuntu/windows/macos)
- pytest configuration with tier markers (S/A/B/C/D tier)
- Ruff linting integration
- Coverage reporting with Codecov
- PyInstaller build verification in CI
- Apache 2.0 LICENSE
- CONTRIBUTING.md with development guidelines
- DESIGN.md architecture documentation
- IMPLEMENTATION_PLAN.md detailed planning document

### Supported Models

- S-tier (>=32B): Qwen 2.5 32B, Llama 3 70B — near-cloud-quality results
- A-tier (7-14B): Llama 3 8B, Mistral 7B — usable with conservative settings
- B-tier (3-7B): Phi-3 Mini — experimental, single-file tasks
- C-tier (1-3B) and D-tier (<1B) — basic command execution only

### Compatibility

- Python >= 3.10
- Ollama >= 0.1.0 with `/api/chat` endpoint support
- Linux, macOS, Windows (all tested in CI)
