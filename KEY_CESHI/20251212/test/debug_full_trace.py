#!/usr/bin/env python3
"""Full trace of a single prediction to find the bug"""

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

# Load the first design point from combined_results.csv
combined_path = (
    PROJECT_ROOT
    / "extensions/warmup_budget_check/sample/202511301011/result/combined_results.csv"
)
df = pd.read_csv(combined_path)
subject_1_df = df[df['subject'] == 'subject_1']
first_row = subject_1_df.iloc[0]

print("First design point from combined_results.csv (subject_1):")
print(f"  x1_CeilingHeight: {first_row['x1_CeilingHeight']}")
print(f"  x2_GridModule: {first_row['x2_GridModule']}")
print(f"  x3_OuterFurniture: {first_row['x3_OuterFurniture']}")
print(f"  x4_VisualBoundary: {first_row['x4_VisualBoundary']}")
print(f"  x5_PhysicalBoundary: {first_row['x5_PhysicalBoundary']}")
print(f"  x6_InnerFurniture: {first_row['x6_InnerFurniture']}")
print(f"  y (expected): {first_row['y']}")
print()

# Transform to canonical
height_map = {2.8: 0, 4.0: 1, 8.5: 2}
grid_map = {6.5: 0, 8.0: 1}
furniture_map = {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
visual_map = {'Solid': 0, 'Translucent': 1, 'Color': 2}
physical_map = {'Closed': 0, 'Open': 1}

x = np.array([
    height_map[first_row['x1_CeilingHeight']],
    grid_map[first_row['x2_GridModule']],
    furniture_map[first_row['x3_OuterFurniture']],
    visual_map[first_row['x4_VisualBoundary']],
    physical_map[first_row['x5_PhysicalBoundary']],
    furniture_map[first_row['x6_InnerFurniture']],
], dtype=float)

print(f"Canonical form: {x}")
print()

# Load item parameters
with open(Path(__file__).parent / "item_params.json", 'r') as f:
    item_params = json.load(f)

subject_1_params = item_params['subject_1']
item_biases = np.array(subject_1_params['item_biases'])
item_noises = np.array(subject_1_params['item_noises'])

print(f"Subject 1 item_biases: {item_biases}")
print(f"Subject 1 item_noises: {item_noises}")
print()

# Create oracle
weights = np.array([[-0.09457, 0.34170, 0.18657, 0.03447, -0.24101, -0.39357]])

oracle = SingleOutputLatentSubject(
    seed=42,
    bias=0.0,
    noise_std=0.0,
    interaction_pairs=[(3, 4), (0, 1)],
    interaction_scale=0.25,
    likert_levels=5,
    likert_sensitivity=2.0,
    use_latent=False,
    fixed_weights=weights,
    num_features=6,
    item_biases=item_biases,
    item_noises=np.zeros_like(item_noises),
)

print(f"Oracle interaction_weights: {oracle.interaction_weights}")
print()

# Manual calculation
print("Manual calculation:")
main = np.dot(weights[0], x)
print(f"  Main effects: {main:.6f}")

int_sum = 0.0
for (i, j), weight in oracle.interaction_weights.items():
    int_val = x[i] * x[j]
    int_sum += weight * int_val
    print(f"  Interaction x{i}*x{j}: {x[i]}*{x[j]} = {int_val}, weight={weight:.6f}, contrib={weight*int_val:.6f}")
print(f"  Total interactions: {int_sum:.6f}")

y_linear = 0.0 + main + int_sum + item_biases[0]
print(f"  Linear (with bias): {y_linear:.6f}")

tanh_val = np.tanh(y_linear * 2.0)
likert_float = tanh_val * 2 + 3
likert_manual = round(likert_float)
print(f"  tanh({y_linear:.6f} * 2.0) = {tanh_val:.6f}")
print(f"  likert_float = {tanh_val:.6f} * 2 + 3 = {likert_float:.6f}")
print(f"  likert_manual = {likert_manual}")
print()

# Oracle prediction
y_pred = oracle(x)
print(f"Oracle prediction: {y_pred}")
print(f"Expected: {first_row['y']}")
print(f"Match: {y_pred == first_row['y']}")
