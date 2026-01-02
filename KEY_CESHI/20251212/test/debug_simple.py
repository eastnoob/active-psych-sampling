#!/usr/bin/env python3
"""Simple debug to check Oracle vs expected output"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

os.environ['PYTHONIOENCODING'] = 'utf-8'

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modules"))

from single_output_subject import SingleOutputLatentSubject

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# Load combined_results (first 3 points from subject_1)
combined_path = PROJECT_ROOT / "extensions/warmup_budget_check/sample/202511301011/result/combined_results.csv"
df = pd.read_csv(combined_path)
subject_1_df = df[df['subject'] == 'subject_1'].head(3)

# Load item_params
with open(Path(__file__).parent / "item_params.json", 'r') as f:
    item_params = json.load(f)

subj_params = item_params['subject_1']

# Create Oracle
weights = np.array([[-0.09457, 0.34170, 0.18657, 0.03447, -0.24101, -0.39357]])
interaction_weights = {tuple(map(int, k.split(','))): v for k, v in subj_params['interaction_weights'].items()}

oracle = SingleOutputLatentSubject(
    seed=42,
    bias=0.0,
    noise_std=0.0,
    interaction_pairs=[],
    interaction_scale=0.25,
    likert_levels=5,
    likert_sensitivity=2.0,
    use_latent=False,
    fixed_weights=weights,
    num_features=6,
    item_biases=np.array(subj_params['item_biases']),
    item_noises=np.zeros(1),
)
oracle.interaction_weights = interaction_weights

print(f"interaction_weights: {oracle.interaction_weights}")
print(f"item_biases: {oracle.item_biases}")
print()

# Define mappings
height_map = {2.8: 0, 4.0: 1, 8.5: 2}
grid_map = {6.5: 0, 8.0: 1}
furniture_map = {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
visual_map = {'Solid': 0, 'Translucent': 1, 'Color': 2}
physical_map = {'Closed': 0, 'Open': 1}

for idx, row in subject_1_df.iterrows():
    x = np.array([
        height_map[row['x1_CeilingHeight']],
        grid_map[row['x2_GridModule']],
        furniture_map[row['x3_OuterFurniture']],
        visual_map[row['x4_VisualBoundary']],
        physical_map[row['x5_PhysicalBoundary']],
        furniture_map[row['x6_InnerFurniture']],
    ], dtype=float)

    y_pred = oracle(x)
    y_expected = row['y']

    # Manual calc
    main = np.dot(weights[0], x)
    int_sum = sum(weight * x[i] * x[j] for (i, j), weight in interaction_weights.items())
    y_linear = main + int_sum + oracle.item_biases[0]

    print(f"Point {idx}:")
    print(f"  x = {x}")
    print(f"  main = {main:.4f}, int = {int_sum:.4f}, bias = {oracle.item_biases[0]:.4f}")
    print(f"  y_linear = {y_linear:.4f}")
    print(f"  y_pred = {y_pred}, y_expected = {y_expected}, match = {y_pred == y_expected}")
    print()
