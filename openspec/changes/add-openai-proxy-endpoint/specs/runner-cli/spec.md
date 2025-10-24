## MODIFIED Requirements

### Requirement: Runner accepts endpoint assignment flags
- The runner SHALL accept `--base-url <string>` to specify an OpenAI-compatible endpoint (e.g., LiteLLM proxy path like `http://localhost:4000/chat/v1`).
- The runner SHALL accept `--ak <string>` as the API key for the endpoint. If omitted, the runner SHALL fall back to `OPENAI_API_KEY` when present.
- The runner SHALL use `http://localhost:4000/chat/v1` as the default base URL when OpenAI mode is selected without an explicit `--base-url`.

#### Scenario: Base URL + API key provided
- WHEN the user passes both `--base-url` and `--ak`
- THEN the runner selects the OpenAI-compatible sampler and uses the provided values.

#### Scenario: Only API key provided
- WHEN the user passes `--ak` but not `--base-url`
- THEN the runner selects the OpenAI-compatible sampler and uses the default base URL `http://localhost:4000/chat/v1`.

#### Scenario: Only base URL provided
- WHEN the user passes `--base-url` but not `--ak`
- THEN the runner selects the OpenAI-compatible sampler and uses no API key (unless `OPENAI_API_KEY` is present).

#### Scenario: Neither base URL nor API key provided
- WHEN the user passes neither flag
- THEN the runner retains existing sampler behavior (dummy/transformers).

## ADDED Requirements

### Requirement: Streaming concatenation for OpenAI-compatible sampler
- The OpenAI-compatible sampler SHALL request chat completions with `stream=True`.
- The sampler SHALL concatenate streamed deltas to a single string and return it as the generation output.

#### Scenario: Streamed chunks
- GIVEN a sequence of streamed chunks with `choices[0].delta.content`
- WHEN reading the stream
- THEN the sampler concatenates all non-empty delta content into the final response string.

