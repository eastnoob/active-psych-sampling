# Simulation result

This folder contains scripts to simulate dependent variable (y) for subject CSVs in the parent folder.

Files:

- `single_output_subject.py`: wrapper around `MixedEffectsLatentSubject` to force a single output.
- `run_simulation.py`: script to read `subject_*` CSVs and produce `*_result.csv` with a new `y` column. Also writes `combined_results.csv` and `model_spec.txt`.

Usage:

1. Open a PowerShell terminal at the `202511161637` folder and run:

```powershell
python .\result\run_simulation.py --input_dir . --seed 42
```

2. Each `subject_*.csv` will generate `subject_*_result.csv` in the `result` folder.

Notes:

- The script tries to preserve model parameters across subjects by sharing a randomly-generated `population_weights` (fixed effects). Individual deviations are different per subject (seed offset).
- To reproduce exact individual deviations, set `--seed` to the same value each time.
- If `pandas` is installed, the script uses it for IO; otherwise falls back to the standard `csv` library.
