# EUR-Warmup Restructuring Plan (v3 - Detailed)

## 1. Core Concept

A modular, CLI-driven tool for managing the **Warmup Phase** of psychological experiments.

- **CLI**: English interactive interface using `rich` for UI.
- **Config**: TOML with Chinese comments, auto-popup for editing, validation before run.
- **Extensibility**: Module-based API in `modules/`.
- **Logging**: `loguru` for all outputs.

## 2. Warmup Steps (Phases)

### Step 1: Sampling (`step1.py`)

- **Purpose**: Generate subject-specific sampling plans.
- **Logic**:
  - Uses `WarmupBudgetEstimator` to check adequacy.
  - Implements "Five-Step Sampling": Core-1 (strategic), Boundary (diversity), LHS (coverage).
  - Supports `interaction_mode`: `free`, `specified_only`, `hybrid`.
- **Input**: Design CSV, `n_subjects`, `trials_per_subject`.
- **Output**: `sample/subject_*.csv`.

### Step 1.5: Simulation (`step1_5.py`)

- **Purpose**: Simulate subject responses for testing/validation.
- **Logic**:
  - Uses `SingleOutputLatentSubject` (Mixed Effects model).
  - Supports `likert` (tanh/percentile) or `continuous` outputs.
  - Configurable population/individual variance and interaction effects.
- **Input**: Step 1 CSVs.
- **Output**: `result/subject_*.csv` (with response column `y`).

### Step 2: Analysis (`step2.py`)

- **Purpose**: Extract Phase 2 parameters from warmup data.
- **Logic**:
  - ANOVA-based interaction selection.
  - $\lambda$ (interaction weight) and $\gamma$ (coverage weight) estimation.
  - Selection methods: `elbow`, `bic_threshold`, `top_k`.
- **Input**: Result CSVs (from Step 1.5 or real experiment).
- **Output**: `analysis_output/phase2_params.json`, reports.

### Step 3: Base GP (`step3.py`)

- **Purpose**: Train a prior GP model and scan design space.
- **Logic**:
  - Matern 2.5 Kernel + ARD.
  - Z-score normalization per subject.
  - Identifies `x_best`, `x_worst`, `x_max_std`.
- **Input**: Result CSVs, Design CSV.
- **Output**: `base_gp_output/model.pth`, `key_points.json`, `scan_results.csv`.

## 3. Configuration Structure (TOML)

Each module provides a default TOML block. Example for Step 1:

```toml
[step1]
design_csv = "data/design.csv" # 设计空间文件路径
n_subjects = 10 # 被试数量
trials_per_subject = 20 # 每个被试的实验次数
interaction_mode = "hybrid" # 交互模式: free/specified_only/hybrid
```

*Note: Every parameter MUST have a Chinese comment.*

## 4. CLI Interaction Flow

1. **Command Entry**: User types `1,2,3` or `all`.
2. **Logic Check**:
    - `1,2` is invalid without `1.5` (unless real data exists).
    - `2,3` is valid if data is provided.
3. **Config Generation**:
    - System merges `get_default_config()` from all selected modules.
    - Saves to `config/temp_config.toml`.
4. **Editor Pop-up**:
    - System opens `temp_config.toml` using `EDITOR` env var or `notepad`.
    - CLI waits: "Please edit the config file. Save and close to continue..."
5. **Validation**:
    - On return, system runs `module.validate(config)`.
    - If fails, shows errors and asks to re-edit or exit.
6. **Execution**:
    - Sequential run using a shared `Context` object.
    - `Context` stores: `design_df`, `subject_data_paths`, `output_dirs`, etc.

## 5. Module API (The `BaseModule` Interface)

```python
class BaseModule:
    def get_default_config(self) -> dict:
        """Return dict with comments for TOML generation."""
        pass
    
    def validate(self, config: dict) -> Tuple[bool, List[str]]:
        """Check config integrity and logic."""
        pass
    
    def run(self, config: dict, context: Context) -> Context:
        """Execute logic and update context."""
        pass
```

## 6. Data Flow (The `Context` Object)

The `Context` object is passed between modules:

- `design_space`: Loaded DataFrame of the design space.
- `subject_files`: List of paths to generated/collected CSVs.
- `analysis_results`: Dictionary of parameters from Step 2.
- `model_path`: Path to the trained GP model from Step 3.

## 7. Implementation Details

- **Loguru**: Configured to log to both console (colorized) and `logs/warmup_{timestamp}.log`.
- **Git**:
  - Local repo in `eur-warmup/`.
  - Commit after each major feature (CLI, Step 1, etc.).
- **Tests**:
  - `tests/test_modules.py`: Unit tests for each module.
  - `tests/test_integration.py`: Test `all` flow with simulation.

## 8. Key Source Files (Reference)

- `warmup_sampler.py` -> `modules/step1.py`
- `simulation_runner.py` -> `modules/step1_5.py`
- `analyze_phase1.py` -> `modules/step2.py`
- `phase1_step3_base_gp.py` -> `modules/step3.py`
