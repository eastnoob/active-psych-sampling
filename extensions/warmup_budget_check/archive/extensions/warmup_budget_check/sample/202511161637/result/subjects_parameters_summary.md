# Subjects parameter summary

This file summarizes whether subjects share population weights and whether individual deviations differ.

## Population weights comparison

This simulation used deterministic fixed-weight outputs for subjects (no latent variables).

## Individual deviations check

Below is a list of per-subject `individual_deviation` filenames and whether they are identical across all subjects.

- subject_1: subject_1_model.md
- subject_2: subject_2_model.md
- subject_3: subject_3_model.md
- subject_4: subject_4_model.md
- subject_5: subject_5_model.md

## Pairwise subject deviation distances

Pairwise Euclidean distances between individual deviations:

| Subject A | Subject B | Distance |
| --- | --- | ---: |
| subject_1 | subject_2 | 0.203601 |
| subject_1 | subject_3 | 0.187545 |
| subject_1 | subject_4 | 0.166815 |
| subject_1 | subject_5 | 0.201627 |
| subject_2 | subject_3 | 0.138628 |
| subject_2 | subject_4 | 0.143244 |
| subject_2 | subject_5 | 0.217862 |
| subject_3 | subject_4 | 0.144389 |
| subject_3 | subject_5 | 0.232097 |
| subject_4 | subject_5 | 0.163024 |

## Quick conclusion

- Subjects use deterministic fixed_weights; pairwise distances above are computed across fixed_weights when latent weights are not available.
