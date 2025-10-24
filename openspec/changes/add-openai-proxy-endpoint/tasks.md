1. Add CLI flags `--base-url` and `--ak` in `src/runner.py`.
2. Implement `OpenAISampler` using `openai` with streaming concatenation.
3. Update `get_sampler` to select `OpenAISampler` when `base_url` or `api_key` provided.
4. Wire flags through runner to `get_sampler` and env var fallback (`OPENAI_API_KEY`).
5. Update README with usage examples and notes.
6. Validate no imports have side-effects and dummy mode remains functional.

