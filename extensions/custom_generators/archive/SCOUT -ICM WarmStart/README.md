# SCOUT Warm-up Generator for AEPsych

This is a Phase-1 "Warm-up" Sampler (Generator) for AEPsych that automatically chooses sampling parameters and produces a Phase-1 design plan and sampled trial list.

## Features

- Accepts only a design grid (full-factorial or candidate set) as minimal input
- Automatically chooses sampling parameters and produces a Phase-1 design plan and sampled trial list
- Targets: measurement calibration (ICC, batch effects, test-retest), coarse main effects estimation, initial GP model training, space coverage
- Emits warnings when dimensionality exceeds thresholds (d>10: warning; d>12: strong warning), but still runs
- Modular, testable, and documented

## Installation

The generator is ready to use with AEPsych. Simply ensure that the `SCOUT -ICM WarmStart` directory is in the `extensions/custom_generators` path of your AEPsych installation.

## Usage with AEPsych INI Configuration

To use this generator with AEPsych, specify it in your INI configuration file:

```ini
[common]
parnames = f1, f2, f3
lb = 0, 0, 0
ub = 1, 1, 1

[experiment]
design_file = path/to/your/design.csv

[generator]
name = SCOUT -ICM WarmStart.scout_warmup_generator.WarmupAEPsychGenerator
n_subjects = 10
total_budget = 350
n_batches = 3
seed = 42
```

### Parameters

- `n_subjects`: Number of subjects (default=10)
- `total_budget`: Total number of trials (default=350)
- `n_batches`: Number of batches (default=3)
- `seed`: Random seed for reproducibility (optional)
- `candidate_interactions`: Optional list of interaction pairs (not directly configurable via INI)

## Dimensionality-Adaptive Defaults

The generator automatically adapts to the dimensionality of your design:

- **d ≤ 8**: Core-1 (20-23%), Core-2 (42-45%), Individual (32-36%)
- **9-10**: Core-1 (22-25%), Core-2 (45-50%), Individual (28-33%)
- **11-12**: Core-1 (~25%), Core-2 (45-50%), Individual (25-30%)

## Outputs

The generator produces:

1. **trial_schedule_df**: A DataFrame with columns:
   - `subject_id`, `batch_id`, `is_bridge`, `block_type`
   - `is_core1`, `is_core2`, `is_individual`, `is_boundary`, `is_lhs`
   - `interaction_pair_id` (if applicable)
   - `design_row_id` referencing the row in design_df
   - Factor values (f1, f2, ..., fd)

2. **summary JSON**: Phase1Output-like metadata & config for downstream

## Testing

To run the tests:

```bash
cd extensions/custom_generators/SCOUT\ -ICM\ WarmStart/test
python run_all_tests.py
```

This will run:

- Basic functionality tests
- High dimensionality warning tests
- Validation hook tests
- INI configuration integration tests

## Implementation Details

The generator implements the following components:

- **Core-1 points**: Global skeleton sampled by all subjects
- **Core-2 points**: Main effects coverage + interaction screening trials
- **Individual points**: Boundary extremes + stratified LHS fill-in
- **Batch allocation**: With bridge subjects across batches

## Warnings

The generator emits warnings for high dimensionality:

- **d > 10**: "Warning: d>10. Phase-1 warm-up becomes sample-hungry..."
- **d > 12**: "Strong Warning: d>12. Efficiency degrades..."

## Requirements

- numpy
- pandas
- scikit-learn
- scipy

## Example Usage

```python
import pandas as pd
from scout_warmup_generator import WarmupAEPsychGenerator

# Create a design DataFrame
design_data = {
    'f1': [0.1, 0.5, 0.9],
    'f2': [0.2, 0.6, 0.8],
    'f3': [0.3, 0.4, 0.7]
}
design_df = pd.DataFrame(design_data)

# Create generator
gen = WarmupAEPsychGenerator(design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42)

# Generate trials
gen.fit_planning()
trials = gen.generate_trials()
summary = gen.summarize()
