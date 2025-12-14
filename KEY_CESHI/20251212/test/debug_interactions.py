#!/usr/bin/env python3
"""Debug interaction weights initialization"""

import sys
import os
from pathlib import Path
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modules"))

from single_output_subject import SingleOutputLatentSubject

# Create oracle with fixed weights
fixed_weights = np.array([[-0.09457, 0.34170, 0.18657, 0.03447, -0.24101, -0.39357]])

oracle = SingleOutputLatentSubject(
    seed=42,
    bias=0.0,
    noise_std=0.1,
    interaction_pairs=[(3, 4), (0, 1)],
    interaction_scale=0.25,
    likert_levels=5,
    likert_sensitivity=2.0,
    use_latent=False,
    fixed_weights=fixed_weights,
    num_features=6,
)

print("Oracle created")
print(f"use_latent: {oracle.use_latent}")
print(f"fixed_weights: {oracle.fixed_weights}")
print(f"interaction_pairs config: {oracle._interaction_pairs_config}")
print(f"interaction_scale: {oracle.interaction_scale}")
print(f"interaction_weights: {oracle.interaction_weights}")
print(f"item_biases: {oracle.item_biases}")
print(f"item_noises: {oracle.item_noises}")
print(f"num_observed_vars: {oracle.num_observed_vars}")

# Test a simple call
x = np.array([0, 0, 0, 2, 1, 2])
print(f"\nTest input: {x}")

# Manual calculation
main = np.dot(fixed_weights[0], x)
print(f"Main effects: {main:.6f}")

if oracle.interaction_weights:
    int_sum = 0.0
    for (i, j), weight in oracle.interaction_weights.items():
        int_val = x[i] * x[j]
        int_sum += weight * int_val
        print(f"Interaction x{i}*x{j}: {x[i]}*{x[j]} = {int_val}, weight={weight:.6f}, contrib={weight*int_val:.6f}")
    print(f"Total interactions: {int_sum:.6f}")
    y_linear = 0.0 + main + int_sum
else:
    print("NO INTERACTION WEIGHTS!")
    y_linear = 0.0 + main

print(f"Linear output: {y_linear:.6f}")

# Oracle call
y_pred = oracle(x)
print(f"\nOracle prediction: {y_pred}")
print(f"Expected: 1")
