## ADDED Requirements

### Requirement: Runner Config File Support
The CLI runner SHALL accept a config file describing run parameters and SHALL merge it with CLI flags, with CLI taking precedence.

#### Scenario: Load YAML config
- **WHEN** the user passes `--config path/to/run.yaml`
- **THEN** parameters are loaded from the YAML file and applied as defaults

#### Scenario: Load JSON config
- **WHEN** the user passes `--config path/to/run.json`
- **THEN** parameters are loaded from the JSON file and applied as defaults

#### Scenario: CLI overrides config
- **WHEN** both a config file and specific CLI flags are provided
- **THEN** CLI flag values override config values

#### Scenario: Persist resolved config
- **WHEN** a run starts (after argument resolution)
- **THEN** the fully resolved parameters are written to `[output]/config.json`

#### Scenario: Validate required fields
- **WHEN** required fields (e.g., `model`, `dataset`, `output`) are missing after resolution
- **THEN** the runner exits non-zero with a clear error message
