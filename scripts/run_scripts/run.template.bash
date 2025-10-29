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

  # --- VLLM BACKEND (optional) ---
  # To switch to vLLM, uncomment the following flags and set a real model id/path:
  # --backend vllm \
  # --model Qwen/Qwen2.5-0.5B-Instruct \
  # Engine config:
  # --vllm-tensor-parallel-size 1 \
  # --vllm-gpu-mem-utilization 0.9 \
  # --vllm-swap-space 4 \
  # --vllm-max-model-len 32768 \
  # Sampling defaults (temperature and max_tokens are already set above):
  # --vllm-top-p 0.95 \
  # --vllm-top-k 50 \
  # --vllm-repetition-penalty 1.1 \
  # --vllm-presence-penalty 0.0 \
  # --vllm-frequency-penalty 0.0 \

  # --output "$(dirname "$(realpath "$0")")"
