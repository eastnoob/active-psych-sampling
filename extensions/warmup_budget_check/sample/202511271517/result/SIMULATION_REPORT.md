# Simulation Report

**Generated:** 2025-11-27T15:52:06.907040

## Configuration Summary

### Step 1: Sampling Configuration

- Design CSV path: `D:\WORKSPACE\python\aepsych-source\data\only_independences\data\only_independences\6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv`
- Number of subjects: 5
- Trials per subject: 25
- Skip interaction (in budget evaluation): True
- Output directory: `extensions\warmup_budget_check\sample\202511172026`
- Merge subjects in sampling: False

### Step 2: Simulation Configuration

- Random seed: 42
- Output mode: combined
- Use latent variables: false
- Output type: likert
- Likert levels: 5
- Likert mapping mode: tanh
- Likert sensitivity: 1.2
- Population mean: 0.0
- Population std: 0.3
- Individual std (percent): 0.3
- Individual correlation: 0.0
- Clean prior results: True

## Data Summary

- Total number of subject files processed: 5
- Total observations (rows across all subjects): 125
- Output range: Likert 1-5

### Subject Files

- subject_1.csv: 25 rows
- subject_2.csv: 25 rows
- subject_3.csv: 25 rows
- subject_4.csv: 25 rows
- subject_5.csv: 25 rows

## Generated Output Files

### Result Files

- **combined_results.csv**: Aggregated results from all subjects (total: 125 rows)

### Model Documentation Files

- **subject_1_model.md**: Model specification and weights for subject
- **subject_2_model.md**: Model specification and weights for subject
- **subject_3_model.md**: Model specification and weights for subject
- **subject_4_model.md**: Model specification and weights for subject
- **subject_5_model.md**: Model specification and weights for subject
- **model_spec.txt**: Final model specification snapshot

### Summary & Analysis Files

- **subjects_parameters_summary.md**: Summary of subject parameter differences and pairwise distances
- **likert_distribution.txt**: Distribution of Likert values across output
- **fixed_weights_auto.json**: Auto-generated fixed weights (if not imported)
- **SIMULATION_REPORT.md**: This file

## Results & Statistics

### Likert Distribution

| Likert Level | Count | Percentage |
| --- | ---: | ---: |
| 1 | 16 | 12.8% |
| 2 | 30 | 24.0% |
| 3 | 23 | 18.4% |
| 4 | 31 | 24.8% |
| 5 | 25 | 20.0% |
| **Total** | **125** | **100%** |

## Subject Parameters Summary

- Number of subjects: 5
- Shared population weights: Yes (all subjects use same pop_weights)
- Individual deviations: Present (generated per-subject with unique seeds)

## Model Details

### Deterministic Fixed-Weight Model

Each subject has:
- **Fixed weights**: Direct linear mapping from input features to output
- **No latent variables**: Output is computed as y = X · w (dot product)
- **Per-subject weights**: May vary across subjects if individually specified
- **Individual deviations**: Applied to fixed weights (subject-specific perturbations)

## Reproducibility Notes

- Random seed: **42** (use same seed to reproduce results)
- Subject-specific seeds: seed + subject_index (e.g., seed=42 → subject_1 uses 43)
- Fixed weights reproducibility: Auto-generated weights use population seed 42
- Percentile binning: Uses global quantiles computed across all 125 observations

---

*For detailed model specifications per subject, see `subject_*_model.md` files.*
*For per-subject parameter comparisons, see `subjects_parameters_summary.md`.*
