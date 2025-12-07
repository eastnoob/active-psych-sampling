#!/usr/bin/env python3
"""
Demo: How to recreate a subject from fixed_weights_auto.json
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2 import LinearSubject

print("=" * 80)
print("Demo: Recreate Subject from fixed_weights_auto.json")
print("=" * 80)
print()

# Load fixed_weights_auto.json
fixed_weights_file = Path(__file__).parent.parent.parent / "extensions/warmup_budget_check/sample/202511301011/result/fixed_weights_auto.json"

if not fixed_weights_file.exists():
    print(f"[Error] File not found: {fixed_weights_file}")
    print("Please run Step 1.5 first to generate fixed_weights_auto.json")
    sys.exit(1)

with open(fixed_weights_file, 'r') as f:
    fixed_data = json.load(f)

print("Loaded fixed_weights_auto.json:")
print(json.dumps(fixed_data, indent=2))
print()

# Extract parameters
weights = np.array(fixed_data['global'][0])

# Convert interactions format: "3,4" -> (3, 4)
interaction_weights = {}
for key, value in fixed_data['interactions'].items():
    i, j = map(int, key.split(','))
    interaction_weights[(i, j)] = value

bias = fixed_data['bias']

print("Extracted parameters:")
print(f"  weights: {weights}")
print(f"  interaction_weights: {interaction_weights}")
print(f"  bias: {bias}")
print()

# Create subject
subject = LinearSubject(
    weights=weights,
    interaction_weights=interaction_weights,
    bias=bias,
    noise_std=0.0,
    likert_levels=5,
    likert_sensitivity=2.0,
    seed=42
)

print("Created subject successfully!")
print()

# Test on some points
print("=" * 80)
print("Test: Subject responses on example points")
print("=" * 80)
print()

# Example points (numeric values after categorical mapping)
# Categorical mappings:
#   x3_OuterFurniture: {'Chaos': 0, 'Rotated': 1, 'Strict': 2}
#   x4_VisualBoundary: {'Color': 0, 'Solid': 1, 'Translucent': 2}
#   x5_PhysicalBoundary: {'Closed': 0, 'Open': 1}
#   x6_InnerFurniture: {'Chaos': 0, 'Rotated': 1, 'Strict': 2}

test_points = [
    {
        "x": [2.8, 6.5, 2, 0, 1, 0],  # Numeric values
        "desc": "x1=2.8, x2=6.5, x3=Strict, x4=Color, x5=Open, x6=Chaos"
    },
    {
        "x": [4.0, 8.0, 1, 1, 0, 2],
        "desc": "x1=4.0, x2=8.0, x3=Rotated, x4=Solid, x5=Closed, x6=Strict"
    },
    {
        "x": [8.5, 6.5, 0, 2, 1, 1],
        "desc": "x1=8.5, x2=6.5, x3=Chaos, x4=Translucent, x5=Open, x6=Rotated"
    },
]

for i, point in enumerate(test_points, 1):
    x = np.array(point["x"])
    response = subject(x)

    print(f"Point {i}:")
    print(f"  Description: {point['desc']}")
    print(f"  Numeric: {x}")
    print(f"  Response: Likert {response}")
    print()

print("=" * 80)
print("Demo: Create multiple similar subjects with individual variations")
print("=" * 80)
print()

# Population parameters
population_weights = weights
individual_std = 0.3 * 0.3  # population_std * individual_std_percent

print(f"Population weights: {population_weights}")
print(f"Individual std: {individual_std}")
print()

# Create 3 new subjects
new_subjects = []
for subject_id in range(1, 4):
    np.random.seed(100 + subject_id)

    # Individual weights = population weights + random deviation
    deviation = np.random.normal(0, individual_std, size=len(population_weights))
    individual_weights = population_weights + deviation

    new_subject = LinearSubject(
        weights=individual_weights,
        interaction_weights=interaction_weights,
        bias=bias,
        noise_std=0.0,
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=100 + subject_id
    )

    new_subjects.append(new_subject)

    print(f"New Subject {subject_id}:")
    print(f"  Individual weights: {individual_weights}")
    print(f"  Deviation from population: {deviation}")

    # Test on first point
    response = new_subject(np.array(test_points[0]["x"]))
    print(f"  Response on test point 1: Likert {response}")
    print()

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("1. Loaded fixed_weights_auto.json successfully")
print(f"2. Created base subject with population parameters")
print(f"3. Tested subject on {len(test_points)} example points")
print(f"4. Created {len(new_subjects)} new subjects with individual variations")
print()
print("You can now use these subjects in:")
print("  - Oracle simulation")
print("  - Real-time interactive experiments")
print("  - Monte Carlo simulations")
print("  - Any custom research code")
print()
print("=" * 80)
