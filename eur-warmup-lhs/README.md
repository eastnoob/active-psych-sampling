# EUR-Warmup LHS Management System

A modular, CLI-driven tool for managing the **Warmup Phase** of psychological experiments with LHS-based sampling.

## Features
- **Interactive CLI**: English-based command-line interface for step-by-step or batch execution.
- **Modular Architecture**: Easily extendable via the `modules/` directory.
- **TOML Configuration**: Unified configuration with Chinese comments and automatic editor integration.
- **Templates & Persistence**: Choose between default templates, existing config files, or create new named configurations for reuse.
- **LHS Sampling**: Uses global Latin Hypercube Sampling with optional shared points and subject-wise allocation.
- **Automated Data Flow**: Seamlessly passes data between sampling, simulation, analysis, and modeling steps.
- **Robust Logging**: Structured logging with `loguru`.

## Quick Start

### 1. Setup Environment
This project uses [pixi](https://pixi.sh) for package management.
```bash
pixi install
```

### 2. Run the CLI
```bash
pixi run cli
```

### 3. Available Steps
- `1`: **Sampling** - Generate LHS-based subject sampling plans.
- `1.5`: **Simulation** - Simulate responses (useful for testing).
- `2`: **Analysis** - Extract Phase 2 parameters ($\lambda$, $\gamma$, interaction pairs).
- `3`: **Base GP** - Train a prior Gaussian Process model and scan the design space.
- `all`: Run all steps in sequence.

## Output Structure
The LHS version keeps the original workflow layout unchanged:

- `output_root/timestamp/step1`
- `output_root/timestamp/step1_5`
- `output_root/timestamp/step2`
- `output_root/timestamp/step3`

You may change only the root directory through `step1.output_root`. The step folder names remain fixed so downstream modules and existing tooling continue to work unchanged.

Example:

```text
output/
	20260320_064032/
		step1/
			subject_1.csv
			subject_2.csv
			README_sampling.txt
		step1_5/
			subject_1.csv
			subject_2.csv
			MODEL_SUMMARY.txt
		step2/
			analysis_results.json
			model_spec.json
			model_spec_summary.md
		step3/
			base_gp_state.pth
			key_points.json
			scan_results.csv
```

## Ready-Made Recipes
- `config/lhs_budget_5.toml`: extreme low-budget recipe for 5 trials per subject.
- `config/lhs_budget_10.toml`: default cold-start recipe for 10 trials per subject.

Recommended starting points:
- 5 trials per subject: `allocation_mode = "disjoint"`, `shared_points = 0`
- 10 trials per subject: `allocation_mode = "disjoint"`, `shared_points = 1`

## Project Structure
- `main.py`: CLI Entry point.
- `core/`: Core infrastructure (Context, BaseModule, ConfigManager).
- `modules/`: Implementation of each warmup step.
- `config/`: Configuration templates and temporary files.
- `logs/`: Execution logs.
- `tests/`: Integration and unit tests.

## Development
To add a new step:
1. Create a new file in `modules/` (e.g., `step4.py`).
2. Inherit from `BaseModule` and implement the required methods.
3. Register the module in `main.py`.
