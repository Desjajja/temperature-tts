Title: Add OpenAI-compatible endpoint assignment and streaming sampler

Summary
- Add CLI flags `--base-url` and `--ak` to route runs to an OpenAI-compatible proxy (e.g., LiteLLM) and use a streaming chat completions client.
- Extend sampler selection to use the OpenAI client when endpoint/key is provided; otherwise keep existing dummy/transformers behavior.

Motivation
- Allow assigning each run to a specific LLM endpoint via a single proxy while keeping the CLI and runner simple.

Scope
- Affects runner CLI and sampling implementation only. No changes to voting/metrics.

Affected specs
- runner-cli
- sampling-backends

Backwards compatibility
- Default behavior remains unchanged unless `--base-url` or `--ak` is provided.

