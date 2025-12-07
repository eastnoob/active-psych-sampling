#!/usr/bin/env python3
"""
Test: Asymmetric interaction weights
- One strong interaction (detectable in experiments)
- One weak interaction (balance distribution)
Goal: At least one interaction is significant enough to be discovered
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from collections import Counter

print("=" * 80)
print("Test: Asymmetric Interaction Weights - One Strong, One Weak")
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
interaction_x3x4 = X_base[:, 2] * X_base[:, 3]  # Categorical interaction (range 0-4)
interaction_x0x1 = X_base[:, 0] * X_base[:, 1]  # Continuous interaction (range 18-68)

# Extended design space
X_extended = np.column_stack([X_base, interaction_x3x4, interaction_x0x1])

print(f"Extended features shape: {X_extended.shape}")
print(f"Interaction feature ranges:")
print(f"  x3*x4 (categorical): [{interaction_x3x4.min():.1f}, {interaction_x3x4.max():.1f}]")
print(f"  x0*x1 (continuous): [{interaction_x0x1.min():.1f}, {interaction_x0x1.max():.1f}]")
print()

# Test different asymmetric configurations
test_configs = [
    {
        'name': 'Config 1: Strong x3x4 (0.15), Weak x0x1 (-0.03)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_x3x4': 0.15,  # Strong categorical interaction
        'interact_x0x1': -0.03   # Weak continuous interaction (to balance)
    },
    {
        'name': 'Config 2: Strong x3x4 (0.20), Weak x0x1 (-0.05)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_x3x4': 0.20,
        'interact_x0x1': -0.05
    },
    {
        'name': 'Config 3: Strong x3x4 (0.25), Weak x0x1 (-0.08)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_x3x4': 0.25,
        'interact_x0x1': -0.08
    },
    {
        'name': 'Config 4: Strong x3x4 (0.15), Weak x0x1 (-0.03), Negative main (-0.05)',
        'main_mean': -0.05,
        'main_std': 0.3,
        'interact_x3x4': 0.15,
        'interact_x0x1': -0.03
    },
    {
        'name': 'Config 5: Strong x3x4 (0.18), Weak x0x1 (-0.04)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_x3x4': 0.18,
        'interact_x0x1': -0.04
    },
    {
        'name': 'Config 6: Strong x3x4 (0.20), Weak x0x1 (-0.04), Negative main (-0.03)',
        'main_mean': -0.03,
        'main_std': 0.3,
        'interact_x3x4': 0.20,
        'interact_x0x1': -0.04
    },
    {
        'name': 'Config 7: Moderate x3x4 (0.12), Very weak x0x1 (-0.02)',
        'main_mean': 0.0,
        'main_std': 0.3,
        'interact_x3x4': 0.12,
        'interact_x0x1': -0.02
    },
    {
        'name': 'Config 8: Strong x3x4 (0.22), Weak x0x1 (-0.06), Smaller main std (0.25)',
        'main_mean': 0.0,
        'main_std': 0.25,
        'interact_x3x4': 0.22,
        'interact_x0x1': -0.06
    },
]

results = []
sensitivity = 2.0

for config in test_configs:
    print("=" * 80)
    print(f"{config['name']}")
    print("=" * 80)

    # Sample main effect weights
    np.random.seed(99)
    main_weights = np.random.normal(
        config['main_mean'],
        config['main_std'],
        size=6
    )

    # Set specific interaction weights (asymmetric)
    interact_weights = np.array([
        config['interact_x3x4'],  # x3*x4
        config['interact_x0x1']   # x0*x1
    ])

    # Combine all weights
    all_weights = np.concatenate([main_weights, interact_weights])

    print(f"Main weights: {main_weights}")
    print(f"Interaction weights: x3*x4={interact_weights[0]:.3f}, x0*x1={interact_weights[1]:.3f}")
    print()

    # Calculate continuous output
    continuous_output = X_extended @ all_weights

    # Calculate interaction contributions separately for analysis
    interact_contrib_x3x4 = config['interact_x3x4'] * interaction_x3x4
    interact_contrib_x0x1 = config['interact_x0x1'] * interaction_x0x1

    print(f"Continuous output: range=[{continuous_output.min():.2f}, {continuous_output.max():.2f}], mean={continuous_output.mean():.2f}, std={continuous_output.std():.2f}")
    print(f"Interaction x3*x4 contribution: range=[{interact_contrib_x3x4.min():.2f}, {interact_contrib_x3x4.max():.2f}]")
    print(f"Interaction x0*x1 contribution: range=[{interact_contrib_x0x1.min():.2f}, {interact_contrib_x0x1.max():.2f}]")

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

    # Check if x3*x4 interaction is "detectable"
    # A reasonable threshold: contribution range > 0.3 (enough to shift ~0.5 Likert levels)
    x3x4_detectable = (interact_contrib_x3x4.max() - interact_contrib_x3x4.min()) > 0.3

    # Print distribution
    print("Likert Distribution:")
    for level in range(1, 6):
        count = counter.get(level, 0)
        pct = count / len(likert_output) * 100
        bar = '#' * int(pct / 5)
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"  Coverage: {coverage}, Max ratio: {max_ratio*100:.1f}%")

    if is_reasonable and x3x4_detectable:
        print(f"\n  [OK] EXCELLENT! Reasonable distribution + detectable interaction")
    elif is_reasonable:
        print(f"\n  [OK] REASONABLE distribution (but interaction may be weak)")
    elif x3x4_detectable:
        print(f"\n  [FAIL] Detectable interaction but distribution too skewed")
    else:
        print(f"\n  [FAIL] Poor distribution and weak interaction")

    print()

    # Save result
    results.append({
        'name': config['name'],
        'main_mean': config['main_mean'],
        'main_std': config['main_std'],
        'interact_x3x4': config['interact_x3x4'],
        'interact_x0x1': config['interact_x0x1'],
        'coverage': coverage,
        'max_ratio': max_ratio,
        'mean': mean,
        'std': std,
        'is_reasonable': is_reasonable,
        'x3x4_detectable': x3x4_detectable,
        'distribution': dict(counter)
    })

# Summary
print()
print("=" * 80)
print("Summary of Results")
print("=" * 80)
print()

print(f"{'#':<4} {'x3x4':<8} {'x0x1':<8} {'Mean':<8} {'MaxRatio':<10} {'x3x4 Detect':<12} {'Status':<20}")
print("-" * 95)

for i, r in enumerate(results, 1):
    if r['is_reasonable'] and r['x3x4_detectable']:
        status = "[EXCELLENT]"
    elif r['is_reasonable']:
        status = "[OK] Weak interact"
    else:
        status = "[FAIL] Skewed"

    detect_mark = "YES" if r['x3x4_detectable'] else "NO"
    print(f"{i:<4} {r['interact_x3x4']:<8.2f} {r['interact_x0x1']:<8.2f} {r['mean']:<8.2f} {r['max_ratio']*100:>5.1f}%{' '*4} {detect_mark:<12} {status}")

print()

# Recommendations
excellent_results = [r for r in results if r.get('is_reasonable', False) and r.get('x3x4_detectable', False)]

if excellent_results:
    print("=" * 80)
    print("RECOMMENDED CONFIGURATIONS (Detectable + Balanced):")
    print("=" * 80)
    print()

    # Sort by mean closeness to 3.0
    excellent_results.sort(key=lambda r: abs(r['mean'] - 3.0))

    for i, r in enumerate(excellent_results[:3], 1):
        print(f"#{i}: {r['name']}")
        print(f"   Main effects: N({r['main_mean']}, {r['main_std']})")
        print(f"   Interaction x3*x4 (categorical, STRONG): {r['interact_x3x4']:.3f}")
        print(f"   Interaction x0*x1 (continuous, weak): {r['interact_x0x1']:.3f}")
        print(f"   Distribution: {r['distribution']}")
        print(f"   Mean: {r['mean']:.2f}, Std: {r['std']:.2f}, Max ratio: {r['max_ratio']*100:.1f}%")
        print()

    print("Why this works:")
    print("  - x3*x4 (categorical) has small value range (0-4)")
    print("  - So a weight of ~0.15-0.25 is DETECTABLE but doesn't dominate")
    print("  - x0*x1 (continuous) has huge value range (18-68)")
    print("  - So a small negative weight (-0.03 to -0.08) balances the distribution")
    print("  - This gives you one strong, explorable interaction!")
else:
    print("=" * 80)
    print("No perfect configuration found")
    print("=" * 80)
    print()

    # Show reasonable ones
    reasonable_results = [r for r in results if r.get('is_reasonable', False)]
    if reasonable_results:
        print("Reasonable distributions (but interaction may be weak):")
        for r in reasonable_results[:3]:
            print(f"  {r['name']}")
            print(f"    Mean: {r['mean']:.2f}, Distribution: {r['distribution']}")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
