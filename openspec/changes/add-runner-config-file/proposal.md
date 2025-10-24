## Why
Long CLI invocations are error-prone and hard to reproduce. A first-class config file enables shareable, versioned run settings and easier automation.

## What Changes
- Add `--config <path>` to `src.runner` to load parameters from YAML or JSON (TOML optional).
- Merge strategy: values from CLI override values from config file.
- Persist resolved parameters to `[output]/config.json` for reproducibility.
- Validate required fields (`model`, `dataset`, `output`) and fail with clear errors.
- Backward compatible: all existing flags continue to work without `--config`.

## Impact
- Affected specs: `runner-cli`
- Affected code: `src/runner.py` (argparse + loader), README examples (add config usage)
