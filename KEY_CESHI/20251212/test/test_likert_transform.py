#!/usr/bin/env python3
"""Test Likert transform directly"""

import numpy as np

def apply_likert_transform_correct(value, levels=5, sensitivity=2.0):
    """Correct formula"""
    tanh_val = np.tanh(value * sensitivity)
    likert_float = tanh_val * (levels - 1) / 2 + (levels + 1) / 2
    likert = int(np.round(likert_float))
    likert = max(1, min(levels, likert))
    return likert

# Test with the values from debug_simple.py
test_values = [-0.8491, 0.5501, 0.9676]

print("Testing Likert transform:")
for val in test_values:
    tanh_val = np.tanh(val * 2.0)
    likert_float = tanh_val * 2 + 3
    likert = round(likert_float)
    likert_clamped = max(1, min(5, likert))

    print(f"\ny_linear = {val:.4f}")
    print(f"  tanh({val:.4f} * 2) = {tanh_val:.4f}")
    print(f"  likert_float = {tanh_val:.4f} * 2 + 3 = {likert_float:.4f}")
    print(f"  round({likert_float:.4f}) = {likert}")
    print(f"  clamped = {likert_clamped}")
    print(f"  apply_likert_transform_correct = {apply_likert_transform_correct(val)}")
