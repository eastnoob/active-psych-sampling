#!/usr/bin/env python3
"""
Test: Hybrid method V2 - use SMALLER interaction weights
The previous test showed negative interaction weights were too strong
Now test smaller absolute values
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from collections import Counter

print("=" * 80)
print("Test: Hybrid Method V2 - Smaller Interaction Weights")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Load design space
df = pd.read_csv(design_space_csv)

# Convert categorical variables
categorical_cols = ['x3_OuterFurniture', 'x4_VisualBoundary', 'x5_PhysicalBoundary', 'x6_InnerFurniture']
for col in categorical_cols:
    if col in df.columns:
        unique_vals = sorted(df[col].unique())
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)

# Extract numeric features
X_base = df[['x1_CeilingHeight', 'x2_GridModule', 'x3_OuterFurniture', 'x4_VisualBoundary', 'x5_PhysicalBoundary', 'x6_InnerFurniture']].values

# Create interaction features
interaction_x3x4 = X_base[:, 2] * X_base[:, 3]
interaction_x0x1 = X_base[:, 0] * X_base[:, 1]

# Extended design space
X_extended = np.column_stack([X_base, interaction_x3x4, interaction_x0x1])

print(f"Extended features shape: {X_extended.shape}")
print()

# Test different configurations with SMALLER interaction weights
test_configs = [
    {
        'name': 'Config 1: Main(0.0, 0.3), Interact(-0.05, 0.03)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_mean': -0.05,
        'interact_std': 0.03
    },
    {
        'name': 'Config 2: Main(0.0, 0.3), Interact(-0.1, 0.05)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_mean': -0.1,
        'interact_std': 0.05
    },
    {
        'name': 'Config 3: Main(0.0, 0.3), Interact(0.0, 0.05)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_mean': 0.0,
        'interact_std': 0.05
    },
    {
        'name': 'Config 4: Main(-0.05, 0.3), Interact(-0.05, 0.03)',
        'main_mean': -0.05,
        'main_std': 0.3,
        'interact_mean': -0.05,
        'interact_std': 0.03
    },
    {
        'name': 'Config 5: Main(-0.1, 0.3), Interact(0.0, 0.05)',
        'main_mean': -0.1,
        'main_std': 0.3,
        'interact_mean': 0.0,
        'interact_std': 0.05
    },
    {
        'name': 'Config 6: Main(0.0, 0.3), Interact(-0.08, 0.04)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_mean': -0.08,
        'interact_std': 0.04
    },
    {
        'name': 'Config 7: Main(0.0, 0.25), Interact(-0.05, 0.03)',
        'main_mean': 0.0,
        'main_std': 0.25,
        'interact_mean': -0.05,
        'interact_std': 0.03
    },
    {
        'name': 'Config 8: Main(-0.08, 0.3), Interact(-0.03, 0.02)',
        'main_mean': -0.08,
        'main_std': 0.3,
        'interact_mean': -0.03,
        'interact_std': 0.02
    },
]

results = []
sensitivity = 2.0

for config in test_configs:
    print("=" * 80)
    print(f"{config['name']}")
    print("=" * 80)

    # Sample weights separately
    np.random.seed(99)

    # Main effects (6 features)
    main_weights = np.random.normal(
        config['main_mean'],
        config['main_std'],
        size=6
    )

    # Interaction features (2 features)
    interact_weights = np.random.normal(
        config['interact_mean'],
        config['interact_std'],
        size=2
    )

    # Combine all weights
    all_weights = np.concatenate([main_weights, interact_weights])

    print(f"Main weights: {main_weights}")
    print(f"Interaction weights: {interact_weights}")
    print()

    # Calculate continuous output
    continuous_output = X_extended @ all_weights

    print(f"Continuous output: range=[{continuous_output.min():.2f}, {continuous_output.max():.2f}], mean={continuous_output.mean():.2f}, std={continuous_output.std():.2f}")

    # Auto-calculate bias
    bias = -continuous_output.mean()
    print(f"Auto-calculated bias: {bias:.2f}")
    print()

    # Apply bias and convert to Likert
    continuous_output_centered = continuous_output + bias

    # Tanh transformation
    tanh_output = np.tanh(sensitivity * continuous_output_centered)

    # Map to Likert 1-5
    likert_output = np.clip(
        np.round((tanh_output + 1) * 2 + 1),
        1, 5
    ).astype(int)

    # Analyze distribution
    counter = Counter(likert_output)
    coverage = len([l for l in counter if counter[l] > 0])
    max_ratio = max(counter.values()) / len(likert_output)
    mean = likert_output.mean()
    std = likert_output.std()

    # Relaxed criteria
    is_reasonable = coverage >= 4 and max_ratio <= 0.7 and 2.0 <= mean <= 4.5

    # Print distribution
    print("Likert Distribution:")
    for level in range(1, 6):
        count = counter.get(level, 0)
        pct = count / len(likert_output) * 100
        bar = '#' * int(pct / 5)
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"  Coverage: {coverage}, Max ratio: {max_ratio*100:.1f}%")

    if is_reasonable:
        print(f"\n  [OK] REASONABLE DISTRIBUTION!")
    else:
        print(f"\n  [FAIL] Too skewed")

    print()

    # Save result
    results.append({
        'name': config['name'],
        'main_mean': config['main_mean'],
        'main_std': config['main_std'],
        'interact_mean': config['interact_mean'],
        'interact_std': config['interact_std'],
        'coverage': coverage,
        'max_ratio': max_ratio,
        'mean': mean,
        'std': std,
        'is_reasonable': is_reasonable,
        'distribution': dict(counter)
    })

# Summary
print()
print("=" * 80)
print("Summary of Results")
print("=" * 80)
print()

print(f"{'#':<4} {'Main Params':<15} {'Interact Params':<15} {'Mean':<8} {'Coverage':<10} {'MaxRatio':<10} {'Status':<15}")
print("-" * 95)

for i, r in enumerate(results, 1):
    status = "[OK] GOOD" if r['is_reasonable'] else "[FAIL] SKEWED"
    main_params = f"({r['main_mean']:.2f},{r['main_std']:.2f})"
    interact_params = f"({r['interact_mean']:.2f},{r['interact_std']:.2f})"
    print(f"{i:<4} {main_params:<15} {interact_params:<15} {r['mean']:<8.2f} {r['coverage']:<10} {r['max_ratio']*100:>5.1f}%{' '*4} {status}")

print()

# Recommendations
reasonable_results = [r for r in results if r.get('is_reasonable', False)]

if reasonable_results:
    print("=" * 80)
    print("RECOMMENDED CONFIGURATIONS:")
    print("=" * 80)
    print()

    # Sort by mean closeness to 3.0
    reasonable_results.sort(key=lambda r: abs(r['mean'] - 3.0))

    for i, r in enumerate(reasonable_results[:3], 1):
        print(f"#{i}: {r['name']}")
        print(f"   Main effects: N({r['main_mean']}, {r['main_std']})")
        print(f"   Interactions: N({r['interact_mean']}, {r['interact_std']})")
        print(f"   Distribution: {r['distribution']}")
        print(f"   Mean: {r['mean']:.2f}, Std: {r['std']:.2f}, Max ratio: {r['max_ratio']*100:.1f}%")
        print()
else:
    print("=" * 80)
    print("No perfect config found, showing best attempts:")
    print("=" * 80)
    print()

    # Sort by combined score: mean closeness + max_ratio
    results.sort(key=lambda r: abs(r['mean'] - 3.0) * 0.5 + r['max_ratio'] * 0.5)

    for i, r in enumerate(results[:3], 1):
        print(f"#{i}: {r['name']}")
        print(f"   Distribution: {r['distribution']}")
        print(f"   Mean: {r['mean']:.2f}, Coverage: {r['coverage']}, Max ratio: {r['max_ratio']*100:.1f}%")
        print()

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
