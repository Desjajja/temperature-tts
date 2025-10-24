## 1. Implementation
- [ ] 1.1 Add `--config` arg to argparse
- [ ] 1.2 Implement loader: YAML (if PyYAML installed) or JSON fallback; TOML optional
- [ ] 1.3 Merge config with CLI (CLI has precedence)
- [ ] 1.4 Persist resolved params to `[output]/config.json`
- [ ] 1.5 Validate required fields; exit non-zero with helpful message
- [ ] 1.6 Update README with example `config.yaml` and command

## 2. Testing
- [ ] 2.1 Unit test merge precedence
- [ ] 2.2 Unit test minimal YAML and JSON configs
- [ ] 2.3 Unit test missing required fields produces clear error

## 3. Docs
- [ ] 3.1 Add sample `configs/run_example.yaml`
- [ ] 3.2 Document `--config` in AGENTS.md/README
