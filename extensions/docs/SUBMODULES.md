# Submodules inside `extensions/`

This file gives a quick overview of the primary subprojects contained by `extensions/` and their purpose.

- `custom_mean/` — Custom mean functions for models. Contains `custom_basegp_prior_mean.py` and sample INI files.
- `custom_likelihood/` — Configurable likelihoods (e.g. configurable gaussian likelihoods for specialized noise models).
- `custom_factory/` — Factory helpers and mixed-model factory code (useful when combining residual models or mixed strategies).
- `custom_generators/` — Custom generators (pool-based and warmup strategies). Contains `custom_pool_based_generator.py` and utilities.
- `config_builder/` — Tools and templates for building INI configurations and verifying configuration outputs.

Notes:

- Many of the subprojects contain archives/backups for experimental or earlier work; those are intentionally left in `archives/` and `backup/`.
- These submodules are meant to be maintained inside the `extensions/` umbrella — no separate repo is required for now.

Where to go next:

- Implementation docs and examples live in `extensions/test/` and `extensions/backup/` for historical/legacy material.
- For how to contribute or run tests, see `CONTRIBUTING.md`.
