# EUR Experiment Runner

A modular, composable framework for testing and comparing EUR (Expected Uncertainty Reduction) acquisition methods.

## Overview

EUR Experiment Runner provides a flexible architecture for running psychophysical experiments with different acquisition strategies. It separates concerns into three composable components:

- **Roles**: Acquisition methods (EUR, Sobol, Random, etc.)
- **Behaviors**: Experiment workflows (single run, comparison, batch, etc.)
- **Evaluators**: Analysis and reporting (metrics, visualization, statistics)

## Quick Start

### Installation

```bash
# Clone and navigate to project
cd eur-experiment-runner

# Install dependencies with pixi
pixi install

# Verify installation
pixi run python main.py --help
```

### Run Your First Experiment

```bash
# Single EUR test
pixi run python main.py --config config/eur_single.toml

# Compare multiple methods
pixi run python main.py --config config/comparison.toml
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  EUR Experiment Runner                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   ROLES      │  │  BEHAVIORS   │  │  EVALUATORS  │  │
│  │ (Acquisition)│  │ (Experiment) │  │  (Analysis)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
│         └──────────────────┴──────────────────┘          │
│                         │                                 │
│                    ┌────▼────┐                           │
│                    │ Context │                           │
│                    └─────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### Components

**Roles (Acquisition Methods)**
- `eur`: Expected Uncertainty Reduction
- `sobol`: Quasi-random Sobol sampling
- `random`: Random sampling
- Custom: Extend `BaseRole` for your own methods

**Behaviors (Experiment Types)**
- `single_run`: Test one method on one design space
- `comparison`: Compare multiple methods on same space
- `batch_run`: Run multiple experiments with different parameters
- `reproduce`: Replay from saved subject data

**Evaluators (Analysis)**
- `performance_metrics`: Accuracy, efficiency, convergence
- `comparison_report`: Side-by-side method comparison
- `visualization`: Plots and charts
- `statistical_analysis`: Significance tests

## Configuration

Experiments are defined using TOML configuration files:

```toml
# config/eur_single.toml

[experiment]
name = "EUR Single Run Test"
output_dir = "output/eur_single"

[role]
type = "eur"

[role.eur]
eur_weight = 0.5
exploration_bonus = 0.1

[behavior]
type = "single_run"

[behavior.single_run]
design_space_csv = "data/design_space.csv"
n_subjects = 1
trials_per_subject = 20
budget = 60
sobol_init = 10

[evaluator]
types = ["performance_metrics", "visualization"]
```

## Programmatic API

```python
from eur_experiment_runner import Experiment
from eur_experiment_runner.roles import EURRole, SobolRole
from eur_experiment_runner.behaviors import Comparison
from eur_experiment_runner.evaluators import ComparisonReport

# Create experiment
exp = Experiment(name="EUR vs Sobol")

# Add components
exp.add_role(EURRole(eur_weight=0.5))
exp.add_role(SobolRole(scramble=True))
exp.add_behavior(Comparison(
    design_space_csv="data/space.csv",
    n_subjects=5,
    trials_per_subject=20
))
exp.add_evaluator(ComparisonReport(include_plots=True))

# Run
context = exp.run()
print(f"Results: {context.output_dir}")
```

## Extending with Custom Components

### Custom Acquisition Method

```python
# roles/my_custom_role.py
from core.base_role import BaseRole
import torch

class MyCustomRole(BaseRole):
    def get_name(self) -> str:
        return "my_custom"

    def initialize(self, config, context):
        self.param = config.get("param", 1.0)

    def generate_next_point(self, model, context):
        # Your acquisition logic here
        return torch.randn(1, model.dim)

    def update(self, x, y, context):
        pass
```

### Custom Behavior

```python
# behaviors/my_custom_behavior.py
from core.base_behavior import BaseBehavior

class MyCustomBehavior(BaseBehavior):
    def get_name(self) -> str:
        return "my_custom"

    def run(self, roles, config, context):
        # Your experiment logic here
        return context
```

## Project Structure

```
eur-experiment-runner/
├── core/                   # Core abstractions
│   ├── context.py         # Shared state
│   ├── base_role.py       # Role interface
│   ├── base_behavior.py   # Behavior interface
│   └── base_evaluator.py  # Evaluator interface
│
├── roles/                  # Acquisition methods
│   ├── eur_role.py
│   ├── sobol_role.py
│   └── random_role.py
│
├── behaviors/              # Experiment workflows
│   ├── single_run.py
│   ├── comparison.py
│   └── batch_run.py
│
├── evaluators/             # Analysis tools
│   ├── performance_metrics.py
│   ├── comparison_report.py
│   └── visualization.py
│
├── utils/                  # Utilities
│   ├── subject_simulator.py
│   ├── design_space.py
│   └── data_loader.py
│
├── config/                 # Config templates
│   ├── default.toml
│   ├── eur_single.toml
│   └── comparison.toml
│
└── output/                 # Results (gitignored)
```

## Examples

### Example 1: Single EUR Test

```bash
pixi run python main.py --config config/eur_single.toml
```

Output:
```
output/eur_single/
├── summary.json           # Experiment metadata
├── results.csv            # Trial-by-trial data
├── metrics.json           # Performance metrics
└── plots/                 # Visualizations
    ├── trajectory.png
    └── uncertainty.png
```

### Example 2: Method Comparison

```bash
pixi run python main.py --config config/comparison.toml
```

Output:
```
output/comparison/
├── summary.json
├── eur_results.csv
├── sobol_results.csv
├── random_results.csv
├── comparison_report.md   # Side-by-side analysis
└── plots/
    ├── method_comparison.png
    └── statistical_tests.png
```

### Example 3: Programmatic Batch Run

```python
from eur_experiment_runner import Experiment
from eur_experiment_runner.roles import EURRole
from eur_experiment_runner.behaviors import BatchRun

# Test different EUR weights
for weight in [0.3, 0.5, 0.7]:
    exp = Experiment(name=f"EUR_weight_{weight}")
    exp.add_role(EURRole(eur_weight=weight))
    exp.add_behavior(BatchRun(
        design_space_csv="data/space.csv",
        n_subjects=10,
        trials_per_subject=20
    ))
    exp.run()
```

## Development

### Running Tests

```bash
pixi run test
```

### Adding a New Component

1. Create file in appropriate directory (`roles/`, `behaviors/`, or `evaluators/`)
2. Inherit from base class (`BaseRole`, `BaseBehavior`, or `BaseEvaluator`)
3. Implement required methods
4. Add tests in `tests/`
5. Update documentation

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions focused

## Comparison with KEY_CESHI

| Feature | KEY_CESHI | EUR Experiment Runner |
|---------|-----------|----------------------|
| Architecture | Script-based | Modular components |
| Configuration | Hardcoded | TOML-based |
| Extensibility | Manual editing | Plugin system |
| API | None | Programmatic API |
| Comparison | Manual | Built-in |
| Reporting | Basic | Comprehensive |

## Roadmap

- [x] Core architecture design
- [ ] Basic role implementations (EUR, Sobol, Random)
- [ ] Single run behavior
- [ ] Performance metrics evaluator
- [ ] CLI interface
- [ ] Comparison behavior
- [ ] Advanced evaluators
- [ ] Programmatic API
- [ ] Documentation and examples
- [ ] Integration tests

## References

- **Architecture pattern**: `eur-warmup` (modular TOML-based CLI)
- **Functionality**: `KEY_CESHI` (EUR testing and evaluation)
- **Implementation plan**: See `IMPLEMENTATION_PLAN.md`

## License

[Your License Here]

## Contact

eastnoob <tianfengxu1997@outlook.com>
