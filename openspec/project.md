# Project Context

## Purpose
Minimal, runnable scaffold for research on Test-Time Scaling (TTS) with temperature sampling. It orchestrates multi-temperature sampling, provides two-stage voting (intra- then cross-temperature), verifies boxed integer answers, and reports Pass@K and Avg@N.

## Tech Stack
- Python 3.10+
- CLI entry: `python -m src.runner` (primary), `python main.py` (smoke)
- Optional: HuggingFace Transformers + Accelerate (sampler), SymPy (expression verifier)
- Dev tooling (recommended): uv, Black, Ruff, pytest

## Project Conventions

### Code Style
- Python, 4-space indent, UTF-8, type hints for public APIs
- Naming: files/modules/functions `snake_case`, classes `PascalCase`
- Keep modules import-safe (no side effects). I/O centralized in `src/runner.py` and `scripts/`.
- Prefer Black and Ruff (configure in `pyproject.toml` if added)

### Architecture Patterns
- Orchestration: `src/runner.py` parses args, iterates temps/rounds, logs JSONL per question
- Samplers: `BaseSampler` with `DummySampler` and optional `TransformersSampler`
- Voting: `TwoStageVoter` with `tau_intra` and `tau_cross`
- Verifiers: `verifier/` with integer and optional sympy-based equality
- Metrics/utilities: `metrics.py`, `voting.py`, `prompts.py`, `scripts/aggregate_results.py`

### Testing Strategy
- Use `pytest`; place tests under `tests/` (e.g., `tests/test_voting.py`)
- Unit-test: `metrics.pass_at_k`, `voting.TwoStageVoter`, `runner.parse_temps`, verifiers
- Avoid network; mock samplers; target >=80% coverage on core logic

### Git Workflow
- Conventional Commits (e.g., `feat: add two-stage voter debug`)
- Feature branches; small focused PRs with sample command, affected files, and runs path

## Domain Context
- Inputs are math questions; final answers extracted from `\boxed{...}`
- Early exit via two-stage voting; logs stored under `runs/<name>/logs/*.jsonl`
- Summary via `scripts/aggregate_results.py` producing `summary.json`

## Important Constraints
- Keep runs, large artifacts, and model weights out of git
- Optional deps only; project should run with dummy sampler offline
- Deterministic metrics; reproducibility favored via explicit seeds/configs

## External Dependencies
- Transformers/Accelerate (optional for real models)
- SymPy (optional verifier)
