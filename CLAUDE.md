# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Data Scientist is a multi-agent framework for data science tasks built on Google's Agent Development Kit (ADK) and Claude Agent SDK. It separates planning from execution, validates work continuously, and adapts its approach based on progress.

## Common Commands

```bash
# Install with dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/unit/test_tools.py -v

# Run a specific test
uv run pytest tests/unit/test_tools.py::test_function_name -v

# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Run the CLI
uv run agentic-data-scientist --mode simple "your query"
uv run agentic-data-scientist --mode orchestrated "complex task" --files data.csv
```

## Architecture

### Multi-Agent Workflow (Orchestrated Mode)

The workflow is a `SequentialAgent` with these phases:

1. **Planning Loop** (`NonEscalatingLoopAgent`): `plan_maker` → `plan_reviewer` → `plan_review_confirmation` (iterates until plan approved)
2. **Plan Parser**: Converts natural language plan into structured stages and success criteria
3. **Stage Orchestrator**: For each stage runs:
   - **Implementation Loop**: `coding_agent` (Claude Code) → `review_agent` → `review_confirmation`
   - **Criteria Checker**: Updates which success criteria are met
   - **Stage Reflector**: Adapts remaining stages based on progress
4. **Summary Agent**: Generates final report

### Key Agent Types

- `LoopDetectionAgent`: Extends ADK's `LlmAgent` with automatic loop detection to prevent infinite generation
- `NonEscalatingLoopAgent`: Allows iterative refinement without propagating rejection signals upward
- `ClaudeCodeAgent`: Wraps Claude Code SDK for implementation (in `agents/claude_code/`)
- `StageOrchestratorAgent`: Custom orchestrator managing stage-by-stage execution (in `agents/adk/stage_orchestrator.py`)

### Code Structure

```
src/agentic_data_scientist/
├── core/           # DataScientist API and event handling
│   ├── api.py      # Main DataScientist class
│   └── events.py   # Event processing utilities
├── agents/
│   ├── adk/        # ADK multi-agent workflow
│   │   ├── agent.py              # Agent factory (create_agent, create_app)
│   │   ├── stage_orchestrator.py # Stage-by-stage execution
│   │   ├── implementation_loop.py# Coding + review loop
│   │   ├── loop_detection.py     # LoopDetectionAgent
│   │   ├── event_compression.py  # Context window management
│   │   └── utils.py              # Model config (DEFAULT_MODEL, REVIEW_MODEL)
│   └── claude_code/              # Claude Code SDK integration
│       └── agent.py              # ClaudeCodeAgent wrapper
├── prompts/        # Jinja2 prompt templates
│   ├── base/       # Core agent prompts (plan_maker.md, coding_review.md, etc.)
│   └── domain/     # Domain-specific prompts (bioinformatics/)
├── tools/          # Built-in tools for ADK agents
│   ├── file_ops.py # Read-only file operations (sandboxed)
│   └── web_ops.py  # fetch_url tool
└── cli/            # Click CLI interface
```

### Session State Keys

The workflow uses these state keys for coordination:
- `high_level_plan`: The approved plan text
- `high_level_stages`: List of stage dicts with `title`, `description`, `completed`, `implementation_result`
- `high_level_success_criteria`: List of criteria dicts with `criteria`, `met`, `evidence`
- `current_stage`: Current stage being implemented
- `implementation_summary`: Latest implementation output
- `review_feedback`: Review agent feedback

### Context Window Management

Event compression is critical for long-running analyses. Key parameters in `event_compression.py`:
- `EVENT_THRESHOLD`: 30 events (when compression triggers)
- `EVENT_OVERLAP`: 10 events (kept as recent context)
- `MAX_EVENTS`: 50 events (hard limit)

Compression uses LLM summarization of old events, then direct assignment to `session.events`.

## Environment Variables

**Required:**
- `NEXUS_URL`: URL for Nexus LiteLLM Proxy (default: `https://nexus-master.lmndstaging.com`)
- `LITELLM_PROXY_API_KEY`: API key for Nexus proxy (default: `sk-12345`)
- `ANTHROPIC_API_KEY`: For Claude Code coding agent

**Optional:**
- `DEFAULT_MODEL`: Model for planning/review (default: `google/gemini-2.5-pro`)
- `CODING_MODEL`: Model for coding agent (default: `claude-sonnet-4-5-20250929`)
- `DISABLE_NETWORK_ACCESS`: Set to `true` to disable web tools

## Code Style

- Python 3.12+ features, type hints required
- NumPy-style docstrings
- Line length 120 chars (configured in `pyproject.toml`)
- Tests use pytest with `asyncio_mode = "auto"`
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
