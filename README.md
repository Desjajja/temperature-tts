# temperature-tts (minimal skeleton)

Reproduction scaffold for **"On the Role of Temperature Sampling in Test-Time Scaling"**.
This is a **minimal, runnable** project that:
- Implements **multi-temperature sampling orchestration** (with a dummy sampler so you can run immediately).
- Includes the **two-stage voting + early-exit** algorithm: intra-temperature (τ_intra) then cross-temperature (τ_cross).
- Computes **Pass@K** (combinatorial formula) and **Avg@N**.
- Provides a **simple AIME-style integer verifier** (`\boxed{...}`).

> Swap the dummy sampler with your preferred backend (vLLM / SGLang / HuggingFace Transformers) in `src/sampling.py`.

## Quickstart (toy demo)
```bash
python -m src.runner \
  --model dummy \
  --dataset data/aime_toy.jsonl \
  --prompt_template AIME \
  --temps 0.0:0.9:0.3 \
  --samples_per_temp 16 \
  --max_tokens 256 \
  --verifier integer \
  --use_two_stage_voting \
  --tau_intra 0.8 --tau_cross 1.0 \
  --output runs/aime_toy_dummy
```

## Switch to Transformers (example)
Edit `src/sampling.py` and set `get_sampler` to return `TransformersSampler`. Then run:
```bash
python -m src.runner \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/aime_toy.jsonl \
  --prompt_template AIME \
  --temps 0.0:1.2:0.2 \
  --samples_per_temp 64 \
  --max_tokens 512 \
  --verifier integer \
  --use_two_stage_voting \
  --tau_intra 0.8 --tau_cross 1.0 \
  --output runs/aime_toy_transformers
```

## OpenAI-Compatible Endpoints (LiteLLM/vLLM)
Use an OpenAI-compatible proxy (e.g., LiteLLM) with streaming:
```bash
python -m src.runner \
  --model my-provider/my-model \
  --dataset data/aime_toy.jsonl \
  --prompt_template AIME \
  --temps 0.0:0.9:0.3 \
  --samples_per_temp 8 \
  --max_tokens 256 \
  --verifier integer \
  --use_two_stage_voting \
  --output runs/aime_toy_openai \
  --base-url http://localhost:4000/chat/v1 \
  --ak sk-xxxxx
```
Notes:
- If `--ak` is omitted, `OPENAI_API_KEY` is used when set.
- If only `--ak` is provided, default base URL is `http://localhost:4000/chat/v1`.
- If neither `--base-url` nor `--ak` are provided, the runner uses the existing dummy/transformers samplers.

## Layout
```
temperature-tts/
├─ configs/
├─ data/
│  └─ aime_toy.jsonl
├─ scripts/
│  └─ aggregate_results.py
├─ src/
│  ├─ runner.py
│  ├─ sampling.py
│  ├─ voting.py
│  ├─ metrics.py
│  ├─ prompts.py
│  └─ verifier/
│     ├─ integer_verify.py
│     └─ sympy_verify.py
└─ runs/  (created on execution)
```
