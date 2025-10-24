<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core code (`runner.py`, `sampling.py`, `voting.py`, `metrics.py`, `prompts.py`, `verifier/`).
- `data/`: small example datasets (e.g., `data/aime_toy.jsonl`).
- `configs/`: lightweight configs (e.g., temperature presets).
- `scripts/`: utilities like `scripts/aggregate_results.py`.
- `runs/`: created on execution for logs and summaries.
- `main.py`: trivial smoke entry; primary entry is `python -m src.runner`.

## Build, Test, and Development Commands
- Setup (Python 3.10+): `python -m venv .venv && source .venv/bin/activate && pip install -U pip`
- Optional deps (samplers/verifiers): `pip install transformers accelerate sympy`
- Toy run: `python -m src.runner --model dummy --dataset data/aime_toy.jsonl --prompt_template AIME --temps 0.0:0.9:0.3 --samples_per_temp 16 --max_tokens 256 --verifier integer --use_two_stage_voting --tau_intra 0.8 --tau_cross 1.0 --output runs/aime_toy_dummy`
- Aggregate: `python scripts/aggregate_results.py runs/aime_toy_dummy`
- Smoke check: `python main.py`

## Coding Style & Naming Conventions
- Python, 4-space indentation, UTF-8, one import per line.
- Use type hints and docstrings for public functions/classes.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.
- Keep modules import-safe (no side effects at import). Prefer pure functions; isolate I/O in `runner.py` or `scripts/`.
- If adding tooling, prefer Black and Ruff with config in `pyproject.toml`.

## Testing Guidelines
- Use `pytest`. Place tests in `tests/` with names like `tests/test_voting.py`, `tests/test_metrics.py`.
- Run tests: `pytest -q` (avoid network; mock Transformers if needed).
- Cover core logic: `metrics.pass_at_k`, `voting.TwoStageVoter`, `verifier` extractors, and `parse_temps`.

## Commit & Pull Request Guidelines
- Commit style: follow Conventional Commits (e.g., `feat: add two-stage voter debug`), small focused changes.
- PRs must include: clear description, linked issue (if any), sample command used, artifacts path (e.g., `runs/...`), and before/after notes for metrics behavior.
- Do not commit large artifacts, model weights, or `runs/` logs; include minimal snippets or `summary.json` as text when necessary.
- Update README if flags, CLI, or defaults change.

## Security & Configuration Tips
- Keep credentials and API keys out of the repo and logs.
- Optional components: Transformers and SymPy are not required; guard imports and document fallback behavior.
