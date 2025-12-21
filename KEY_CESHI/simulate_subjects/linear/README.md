# simulate_subjects/linear

Tools to generate and load linear subject clusters for testing and simulation.

Usage:

1. Generate a cluster (50 realistic subjects by default):

```bash
# Using your local Python environment:
python generate_cluster.py --n-subjects 50 --out-dir runs/50_subjects --population-std 0.25 --individual-std-percent 0.4

# Use provided design space CSV (project data)
python generate_cluster.py --n-subjects 50 --design-space ../../data/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv --out-dir runs/50_subjects_from_csv --population-std 0.25 --individual-std-percent 0.4

# Or, using pixi (recommended in this project):
pixi run python KEY_CESHI/simulate_subjects/linear/generate_cluster.py --n-subjects 50 --out-dir KEY_CESHI/simulate_subjects/linear/runs/50_subjects --population-std 0.25 --individual-std-percent 0.4
```

1. Inspect outputs in `runs/50_subjects/`:

- `subject_1_spec.json`, ... `subject_N_spec.json`
- `subject_1.csv`, ... (responses on design space)
- `cluster_summary.json`, `run_meta.json`, `design_space.csv`

1. Load a subject:

```py
from simulate_subjects.linear.utils import load_subject_spec
sub = load_subject_spec('runs/50_subjects/subject_1_spec.json')
y = sub(x)
```

1. Use the generated subject in the EUR isolated test (pixi recommended):

```bash
# Example: run isolated test using subject 1 from the generated cluster
pixi run python scripts/run_eur_isolated_test.py --cluster-dir KEY_CESHI/simulate_subjects/linear/runs/50_subjects --subject-id 1 --budget 30
```
