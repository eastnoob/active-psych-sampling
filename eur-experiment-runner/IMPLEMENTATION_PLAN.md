# EUR Experiment Runner - Implementation Plan

## 1. Project Overview

**Purpose**: Modular experiment runner for testing and comparing EUR acquisition methods. Replicates KEY_CESHI functionality with improved architecture.

**Reference Projects**:
- Architecture: `eur-warmup` (modular TOML-based CLI)
- Functionality: `KEY_CESHI` (EUR testing with AEPsych Strategy system)

**Core Philosophy**: Composable abstractions - Roles (acquisition methods) + Behaviors (experiment orchestration) + Evaluators (analysis) = runnable experiments.

**Key Integration**: Works WITH AEPsych's INI-based configuration and Strategy execution. Does not replace AEPsych - orchestrates it at a higher level.

---

## 2. Architecture Design

### 2.1 Three-Component System

**ROLES (Acquisition Methods)**: Generate/modify INI config sections for different acquisition strategies
- EUR: DynamicEURGenerator + DynamicEURFunction
- Sobol: SobolGenerator
- OptimizeAcqf: OptimizeAcqfGenerator with various acqf
- Custom: User-defined generators

**BEHAVIORS (Experiment Orchestration)**: Control Strategy execution flow
- SingleRun: One Strategy + one Oracle
- Comparison: Multiple Strategies (different roles) + same Oracle
- BatchRun: Parameter sweeps across multiple experiments
- Reproduce: Replay from saved data

**EVALUATORS (Post-Processing)**: Analyze results
- PerformanceMetrics: Accuracy, efficiency, convergence
- ComparisonReport: Side-by-side comparison
- Visualization: Plots
- StatisticalAnalysis: Significance tests

### 2.2 Integration with AEPsych

**The tool orchestrates AEPsych, not replaces it**:

1. **INI Config**: Roles generate INI sections for acquisition methods
2. **Strategy**: Behaviors create and execute AEPsych Strategy
3. **Oracle**: Provides subject simulators for testing
4. **Results**: Collects Strategy data for Evaluators

**Flow**:
```
User TOML → Role generates INI → Behavior creates Strategy →
Oracle responds → Strategy executes → Evaluator analyzes
```

---

## 3. Directory Structure

```
eur-experiment-runner/
├── pixi.toml                    # Dependencies
├── main.py                      # CLI entry
├── README.md
├── IMPLEMENTATION_PLAN.md
│
├── core/                        # Core abstractions
│   ├── __init__.py
│   ├── context.py              # Shared state
│   ├── config_manager.py       # TOML handler
│   ├── base_role.py            # Role interface
│   ├── base_behavior.py        # Behavior interface
│   ├── base_evaluator.py       # Evaluator interface
│   └── experiment.py           # Orchestrator
│
├── roles/                       # Acquisition methods
│   ├── __init__.py
│   ├── eur_role.py             # EUR (DynamicEURGenerator)
│   ├── sobol_role.py           # Sobol
│   ├── optimize_acqf_role.py   # OptimizeAcqf variants
│   └── custom_role.py          # Template
│
├── behaviors/                   # Experiment orchestration
│   ├── __init__.py
│   ├── single_run.py           # Single Strategy test
│   ├── comparison.py           # Multi-Strategy comparison
│   ├── batch_run.py            # Parameter sweeps
│   └── reproduce.py            # Replay from data
│
├── evaluators/                  # Analysis
│   ├── __init__.py
│   ├── performance_metrics.py
│   ├── comparison_report.py
│   ├── visualization.py
│   └── statistical_analysis.py
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── oracle.py               # Subject simulators
│   ├── design_space.py         # Design space utilities
│   ├── ini_builder.py          # INI generation helpers
│   └── data_loader.py          # I/O
│
├── config/                      # Config templates
│   ├── default.toml
│   ├── eur_single.toml
│   └── comparison.toml
│
├── tests/
└── output/                      # Results (gitignored)
```

---

## 4. Core Abstractions

### 4.1 Context (Shared State)

```python
# core/context.py
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

class Context:
    """Shared state passed between components."""

    def __init__(self):
        self.design_space: Optional[pd.DataFrame] = None
        self.oracle: Optional[Any] = None  # Subject simulator
        self.strategy: Optional[Any] = None  # AEPsych Strategy
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.output_dir: Optional[Path] = None
        self.ini_config_path: Optional[Path] = None

    def set(self, key: str, value: Any):
        self.metadata[key] = value

    def get(self, key: str, default=None) -> Any:
        return self.metadata.get(key, default)
```

### 4.2 BaseRole (Acquisition Method)

```python
# core/base_role.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseRole(ABC):
    """Abstract base for acquisition methods."""

    @abstractmethod
    def get_name(self) -> str:
        """Return role name (e.g., 'eur', 'sobol')."""
        pass

    @abstractmethod
    def get_default_config(self) -> str:
        """Return default TOML config."""
        pass

    @abstractmethod
    def generate_ini_section(self, config: Dict[str, Any]) -> str:
        """Generate INI config section for this acquisition method.

        Returns:
            INI string with [strategy_name] and [generator_name] sections
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for INI (e.g., 'eur_strat')."""
        pass
```

### 4.3 BaseBehavior (Experiment Orchestration)

```python
# core/base_behavior.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from .context import Context

class BaseBehavior(ABC):
    """Abstract base for experiment behaviors."""

    @abstractmethod
    def get_name(self) -> str:
        """Return behavior name."""
        pass

    @abstractmethod
    def get_default_config(self) -> str:
        """Return default TOML config."""
        pass

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        pass

    @abstractmethod
    def run(self, roles: List['BaseRole'], config: Dict[str, Any],
            context: Context) -> Context:
        """Execute experiment.

        Typical flow:
        1. Build INI config from roles
        2. Create AEPsych Strategy
        3. Run warmup (if specified)
        4. Execute sampling loop with Oracle
        5. Collect results
        """
        pass
```

### 4.4 BaseEvaluator (Post-Processing)

```python
# core/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from .context import Context

class BaseEvaluator(ABC):
    """Abstract base for evaluators."""

    @abstractmethod
    def get_name(self) -> str:
        """Return evaluator name."""
        pass

    @abstractmethod
    def get_default_config(self) -> str:
        """Return default TOML config."""
        pass

    @abstractmethod
    def evaluate(self, config: Dict[str, Any], context: Context) -> Dict[str, Any]:
        """Perform evaluation."""
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any], context: Context):
        """Generate and save report."""
        pass
```

---

## 5. Configuration System

### 5.1 TOML Structure (User-Facing)

```toml
# config/eur_single.toml

[experiment]
name = "EUR Single Run Test"
description = "Test EUR acquisition"
output_dir = "output/eur_single"

[design_space]
# Option 1: Use CSV file
csv_path = "data/design_space.csv"

# Option 2: Define inline (for simple cases)
# parameters = [
#   {name = "x1", type = "continuous", lower = 0.0, upper = 1.0},
#   {name = "x2", type = "ordinal", values = ["low", "medium", "high"]}
# ]

[oracle]
type = "linear"  # linear, from_cluster, custom
seed = 42
noise_std = 0.5
output_type = "likert"  # continuous, likert, binary
likert_levels = 5

[oracle.linear]
weights = [0.3, 0.2, -0.4, 0.1, 0.25, -0.15]
bias = 0.1
interactions = [{indices = [1, 2], weight = 0.15}]

[role]
type = "eur"  # Role to use

[role.eur]
# EUR-specific parameters (will be converted to INI)
eur_weight = 0.5
max_uncertainty = 0.95
max_divergence = 0.95
weight_update_lr = 0.1

[behavior]
type = "single_run"

[behavior.single_run]
budget = 60
warmup_points = 10  # Initial Sobol points
warmup_from_basegp = false  # Use BaseGP keypoints if true

[evaluator]
types = ["performance_metrics", "visualization"]

[evaluator.performance_metrics]
metrics = ["accuracy", "efficiency", "convergence"]

[evaluator.visualization]
plot_types = ["trajectory", "uncertainty"]
```

### 5.2 INI Generation (Internal)

Roles generate INI sections that get combined into full AEPsych config:

```python
# Example: EURRole.generate_ini_section()
def generate_ini_section(self, config):
    eur_weight = config.get("eur_weight", 0.5)
    max_unc = config.get("max_uncertainty", 0.95)
    # ... other params

    return f"""
[eur_strat]
min_asks = {config.get('budget', 50)}
generator = DynamicEURGenerator
acqf = DynamicEURFunction
model = MonotonicRejectionGP
refit_every = 1

[DynamicEURGenerator]
max_uncertainty = {max_unc}
max_divergence = {config.get('max_divergence', 0.95)}
initial_uncertainty_weight = {eur_weight}
initial_divergence_weight = {1.0 - eur_weight}
weight_update_lr = {config.get('weight_update_lr', 0.1)}
"""
```

### 5.3 Comparison Config

```toml
# config/comparison.toml

[experiment]
name = "EUR vs Sobol Comparison"
output_dir = "output/comparison"

[design_space]
csv_path = "data/design_space.csv"

[oracle]
type = "linear"
seed = 42

[roles]
# Multiple roles for comparison
types = ["eur", "sobol", "optimize_acqf"]

[roles.eur]
eur_weight = 0.5

[roles.sobol]
# Sobol has no special params

[roles.optimize_acqf]
acqf_type = "qLogNEI"  # qNEI, qLogNEI, qUCB, etc.

[behavior]
type = "comparison"

[behavior.comparison]
budget = 60
warmup_points = 10
n_replicates = 5  # Run each method 5 times

[evaluator]
types = ["comparison_report", "statistical_analysis"]
```

---

## 6. Implementation Steps

### Phase 1: Core Infrastructure

1. **Project setup**
   - Create directory structure
   - Initialize pixi.toml
   - Setup logging (loguru)

2. **Core abstractions**
   - `core/context.py`
   - `core/base_role.py`
   - `core/base_behavior.py`
   - `core/base_evaluator.py`

3. **Configuration**
   - `core/config_manager.py`: TOML parser
   - `utils/ini_builder.py`: INI generation helpers

4. **Experiment orchestrator**
   - `core/experiment.py`: Main runner
   - Component discovery

### Phase 2: Basic Components

5. **Oracle (Subject Simulator)**
   - `utils/oracle.py`: LinearOracle, ClusterOracle
   - Port from KEY_CESHI

6. **Basic Role**
   - `roles/sobol_role.py`: Simplest (just SobolGenerator)

7. **Basic Behavior**
   - `behaviors/single_run.py`: Single Strategy execution
   - INI building from role
   - Strategy creation
   - Sampling loop with Oracle

8. **Basic Evaluator**
   - `evaluators/performance_metrics.py`: Basic metrics

### Phase 3: EUR and Advanced Features

9. **EUR Role**
   - `roles/eur_role.py`: DynamicEURGenerator config

10. **Comparison Behavior**
    - `behaviors/comparison.py`: Multi-Strategy comparison

11. **Advanced Evaluators**
    - `evaluators/comparison_report.py`
    - `evaluators/visualization.py`

### Phase 4: CLI and Polish

12. **CLI**
    - `main.py`: Rich CLI interface
    - Module discovery
    - Interactive mode

13. **Documentation**
    - README.md
    - Examples

14. **Testing**
    - Unit tests
    - Integration tests

---

## 7. Key Design Decisions

### 7.1 Why Work WITH AEPsych INI?

- **Leverage existing system**: AEPsych's Strategy/INI system is mature
- **Compatibility**: Can use existing INI configs
- **Flexibility**: Roles just generate INI sections, easy to extend
- **No reinvention**: Don't reimplement Strategy logic

### 7.2 Role Responsibility

Roles do NOT implement acquisition logic - they generate INI config for AEPsych components that do. This keeps the tool lightweight and maintainable.

### 7.3 Behavior Responsibility

Behaviors orchestrate the experiment flow:
- Build complete INI from roles + base config
- Create Strategy
- Manage Oracle
- Execute sampling loop
- Collect results

### 7.4 Oracle Integration

Oracle simulates subject responses. Key types:
- **LinearOracle**: Simple linear model (from KEY_CESHI)
- **ClusterOracle**: Load from generated subject cluster
- **CustomOracle**: User-defined

---

## 8. Example Usage

### Example 1: Single EUR Test

```bash
pixi run python main.py --config config/eur_single.toml
```

Output:
```
output/eur_single/
├── experiment_summary.json
├── eur_strat_results.csv
├── eur_strat_config.ini  # Generated INI
├── metrics.json
└── plots/
```

### Example 2: Comparison

```bash
pixi run python main.py --config config/comparison.toml
```

Output:
```
output/comparison/
├── experiment_summary.json
├── eur_results.csv
├── sobol_results.csv
├── optimize_acqf_results.csv
├── comparison_report.md
└── plots/
```

### Example 3: Programmatic API

```python
from eur_experiment_runner import Experiment
from eur_experiment_runner.roles import EURRole, SobolRole
from eur_experiment_runner.behaviors import Comparison
from eur_experiment_runner.utils.oracle import LinearOracle

exp = Experiment(name="EUR vs Sobol")
exp.add_role(EURRole(eur_weight=0.5))
exp.add_role(SobolRole())
exp.set_oracle(LinearOracle(seed=42))
exp.add_behavior(Comparison(budget=60, n_replicates=5))
exp.run()
```

---

## 9. Dependencies (pixi.toml)

```toml
[project]
name = "eur-experiment-runner"
version = "0.1.0"
channels = ["conda-forge", "pytorch"]
platforms = ["win-64"]

[tasks]
run = {cmd = "python main.py", env = {PYTHONPATH = "."}}
test = {cmd = "pytest tests/", env = {PYTHONPATH = "."}}

[dependencies]
python = "3.10.*"
pytorch = ">=2.9.1,<3"
gpytorch = ">=1.14.3,<2"
botorch = ">=0.16.1,<0.17"
numpy = ">=2.2.6,<3"
pandas = ">=2.3.3,<3"
scipy = ">=1.15.2,<2"
toml = ">=0.10.2,<0.11"
loguru = ">=0.7.3,<0.8"
rich = ">=14.2.0,<15"
click = ">=8.3.1,<9"
matplotlib = ">=3.9.0,<4"
seaborn = ">=0.13.0,<0.14"
pytest = ">=8.0.0,<9"
```

---

## 10. Success Criteria

### Functional
- Single acquisition method testing works
- Multi-method comparison works
- Evaluation generates useful metrics
- Custom roles can be added
- Works with existing AEPsych INI configs

### Non-Functional
- Code is modular and maintainable
- Configuration is intuitive
- API is simple
- Performance acceptable
- Error messages clear

---

## 11. Implementation Notes for AI

### Code Style
- PEP 8
- Type hints
- Docstrings
- Small focused functions

### Error Handling
- Use loguru
- Specific exceptions with clear messages
- Validate early

### Testing
- Write tests alongside implementation
- Use pytest fixtures
- Mock external dependencies

### Key Files to Reference
- `KEY_CESHI/20251212/scripts/run_eur_isolated_test.py`: Oracle and Strategy execution
- `KEY_CESHI/20251212/configs/eur_residual_test.ini`: INI structure
- `eur-warmup/main.py`: Module discovery pattern
- `eur-warmup/core/base_module.py`: Abstract base pattern

---

## 12. Quick Start for Implementation

**Step 1**: Create structure
```bash
mkdir -p eur-experiment-runner/{core,roles,behaviors,evaluators,utils,config,tests,output}
touch eur-experiment-runner/{core,roles,behaviors,evaluators,utils,tests}/__init__.py
```

**Step 2**: Initialize pixi
```bash
cd eur-experiment-runner
pixi init
# Edit pixi.toml with dependencies
```

**Step 3**: Implement core
- Context
- BaseRole, BaseBehavior, BaseEvaluator
- ConfigManager

**Step 4**: Implement Oracle
- LinearOracle (port from KEY_CESHI)

**Step 5**: Implement SobolRole + SingleRun
- Simplest role
- Basic behavior with Strategy execution

**Step 6**: Test end-to-end
- Create simple config
- Run experiment
- Verify output

**Step 7**: Expand
- Add EURRole
- Add Comparison behavior
- Add evaluators

---

**End of Implementation Plan**
