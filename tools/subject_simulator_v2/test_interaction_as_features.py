#!/usr/bin/env python3
"""
Test: Treat interactions as explicit features (user's suggestion)
Instead of: y = w·x + interaction_weight·(xi*xj)
Use: y = w·[x, xi*xj]  (interactions are just regular features)
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
import numpy as np
from collections import Counter

print("=" * 80)
print("Test: Interactions as Explicit Features")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Load design space
df = pd.read_csv(design_space_csv)

# Convert categorical variables (same mapping as warmup_adapter)
categorical_cols = ['x3_OuterFurniture', 'x4_VisualBoundary', 'x5_PhysicalBoundary', 'x6_InnerFurniture']
for col in categorical_cols:
    if col in df.columns:
        unique_vals = sorted(df[col].unique())
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)

# Extract numeric features
X_base = df[['x1_CeilingHeight', 'x2_GridModule', 'x3_OuterFurniture', 'x4_VisualBoundary', 'x5_PhysicalBoundary', 'x6_InnerFurniture']].values

print(f"Base features shape: {X_base.shape}")
print(f"Feature ranges:")
print(f"  x0 (x1_CeilingHeight): [{X_base[:, 0].min():.2f}, {X_base[:, 0].max():.2f}]")
print(f"  x1 (x2_GridModule): [{X_base[:, 1].min():.2f}, {X_base[:, 1].max():.2f}]")
print(f"  x3 (x3_OuterFurniture): [{X_base[:, 2].min():.0f}, {X_base[:, 2].max():.0f}]")
print(f"  x4 (x4_VisualBoundary): [{X_base[:, 3].min():.0f}, {X_base[:, 3].max():.0f}]")
print()

# Create interaction features
interaction_x3x4 = X_base[:, 2] * X_base[:, 3]  # x3 * x4
interaction_x0x1 = X_base[:, 0] * X_base[:, 1]  # x0 * x1

print(f"Interaction feature ranges:")
print(f"  x3*x4: [{interaction_x3x4.min():.2f}, {interaction_x3x4.max():.2f}]")
print(f"  x0*x1: [{interaction_x0x1.min():.2f}, {interaction_x0x1.max():.2f}]")
print()

# Extended design space: [x0, x1, x2, x3, x4, x5, x3*x4, x0*x1]
X_extended = np.column_stack([X_base, interaction_x3x4, interaction_x0x1])

print(f"Extended features shape: {X_extended.shape}")
print(f"  Original features: 6")
print(f"  Interaction features: 2")
print(f"  Total: 8")
print()

# Test different configurations
test_configs = [
    {
        'name': 'Standard: population_mean=0.0, population_std=0.3',
        'population_mean': 0.0,
        'population_std': 0.3,
        'likert_sensitivity': 2.0
    },
    {
        'name': 'Negative mean: population_mean=-0.2, population_std=0.3',
        'population_mean': -0.2,
        'population_std': 0.3,
        'likert_sensitivity': 2.0
    },
    {
        'name': 'Smaller std: population_mean=0.0, population_std=0.15',
        'population_mean': 0.0,
        'population_std': 0.15,
        'likert_sensitivity': 2.0
    },
    {
        'name': 'Higher sensitivity: population_mean=0.0, population_std=0.3, sensitivity=3.0',
        'population_mean': 0.0,
        'population_std': 0.3,
        'likert_sensitivity': 3.0
    },
]

results = []

for config in test_configs:
    print("=" * 80)
    print(f"Config: {config['name']}")
    print("=" * 80)
    print()

    # Sample population weights (all features including interactions)
    np.random.seed(99)
    population_weights = np.random.normal(
        config['population_mean'],
        config['population_std'],
        size=X_extended.shape[1]
    )

    print(f"Population weights (8 features):")
    print(f"  Main effects (x0-x5): {population_weights[:6]}")
    print(f"  Interaction x3*x4: {population_weights[6]:.4f}")
    print(f"  Interaction x0*x1: {population_weights[7]:.4f}")
    print()

    # Calculate continuous output
    continuous_output = X_extended @ population_weights

    print(f"Continuous output range: [{continuous_output.min():.2f}, {continuous_output.max():.2f}]")
    print(f"Continuous output mean: {continuous_output.mean():.2f}")
    print()

    # Auto-calculate bias
    bias = -continuous_output.mean()
    print(f"Auto-calculated bias: {bias:.2f}")
    print()

    # Apply bias and convert to Likert
    continuous_output_centered = continuous_output + bias

    # Tanh transformation
    sensitivity = config['likert_sensitivity']
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

    # Normality check
    is_normal = coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0

    # Print distribution
    print("Likert Distribution:")
    for level in range(1, 6):
        count = counter.get(level, 0)
        pct = count / len(likert_output) * 100
        bar = '#' * int(pct / 5)
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Mean: {mean:.2f}")
    print(f"  Coverage: {coverage}")
    print(f"  Max ratio: {max_ratio*100:.1f}%")

    if is_normal:
        print(f"\n  [OK] NORMAL DISTRIBUTION!")
    else:
        print(f"\n  [FAIL] Not normal")
        reasons = []
        if coverage < 3:
            reasons.append(f"coverage={coverage} < 3")
        if max_ratio > 0.6:
            reasons.append(f"max_ratio={max_ratio*100:.1f}% > 60%")
        if not (2.0 <= mean <= 4.0):
            reasons.append(f"mean={mean:.2f} not in [2.0, 4.0]")
        print(f"  Reasons: {'; '.join(reasons)}")

    print()

    # Save result
    results.append({
        'name': config['name'],
        'population_mean': config['population_mean'],
        'population_std': config['population_std'],
        'sensitivity': config['likert_sensitivity'],
        'coverage': coverage,
        'max_ratio': max_ratio,
        'mean': mean,
        'is_normal': is_normal,
        'distribution': dict(counter)
    })

# Summary
print()
print("=" * 80)
print("Summary of Results")
print("=" * 80)
print()

for r in results:
    normal_status = "[OK] NORMAL" if r['is_normal'] else "[FAIL] NOT NORMAL"
    print(f"{r['name']}:")
    print(f"  Parameters: mean={r['population_mean']}, std={r['population_std']}, sensitivity={r['sensitivity']}")
    print(f"  Distribution: {r['distribution']}")
    print(f"  Coverage: {r['coverage']}, Max ratio: {r['max_ratio']*100:.1f}%, Mean: {r['mean']:.2f}")
    print(f"  Status: {normal_status}")
    print()

# Recommendations
normal_results = [r for r in results if r.get('is_normal', False)]

if normal_results:
    print("=" * 80)
    print("SUCCESS! RECOMMENDED CONFIGURATION:")
    print("=" * 80)
    print()

    best = normal_results[0]
    print(f"Use this approach in warmup_adapter:")
    print(f"  Treat interactions as explicit features")
    print(f"  population_mean={best['population_mean']}")
    print(f"  population_std={best['population_std']}")
    print(f"  likert_sensitivity={best['sensitivity']}")
    print()
    print(f"This produces:")
    print(f"  Distribution: {best['distribution']}")
    print(f"  Mean: {best['mean']:.2f}")
    print(f"  Coverage: {best['coverage']} levels")
    print(f"  Max single level: {best['max_ratio']*100:.1f}%")
else:
    print("=" * 80)
    print("COMPARISON WITH CURRENT METHOD:")
    print("=" * 80)
    print()
    print("The 'interactions as features' approach still doesn't achieve normality.")
    print("This confirms the fundamental issue is the large value ranges of continuous")
    print("features, not the method of handling interactions.")
    print()
    print("However, this approach is MATHEMATICALLY CLEANER:")
    print("  1. Unified weight distribution (no separate interaction_scale)")
    print("  2. Simpler implementation (just extend feature matrix)")
    print("  3. Standard statistical modeling approach")
    print()
    print("We should adopt this method regardless of normality, then use:")
    print("  ensure_normality=False")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
