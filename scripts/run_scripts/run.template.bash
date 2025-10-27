#!/usr/bin/env bash
set -euo pipefail

# Template for running temperature-tts sampling jobs.
# Copy this file into runs/<run_name>/run.bash and fill in the command below.
# Required placeholders are marked with TODOs.

# UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

# uv run --cache-dir "${UV_CACHE_DIR}" python -m src.runner \
uv run python -m src.runner \
  --model dummy \
  --dataset data/aime_toy.jsonl \
  --prompt_template AIME \
  --temps 0.0:0.6:0.3 \
  --samples_per_temp 4 \
  --max_tokens 32 \
  --batch_size 1 \
  --verifier integer \
  --use_two_stage_voting \
  --tau_intra 0.8  \
  --tau_cross 1.0 \
  --output todo_output_directory

  # --output "$(dirname "$(realpath "$0")")"
