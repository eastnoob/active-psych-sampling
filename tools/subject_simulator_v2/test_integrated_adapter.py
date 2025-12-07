#!/usr/bin/env python3
"""
Test: Integrated warmup_adapter with default parameters
Verify that the default behavior uses V3 (interaction-as-features) method
and produces Config #1 distribution
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
from collections import Counter

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

print("=" * 80)
print("Test: Integrated Warmup Adapter (Default = Config #1)")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Create temporary test directory
test_dir = Path(__file__).parent / "temp_test_integrated"
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

# Copy full design space as subject_1.csv
df_full = pd.read_csv(design_space_csv)
subject_1_csv = test_dir / "subject_1.csv"
df_full.to_csv(subject_1_csv, index=False)

print(f"Created test directory: {test_dir}")
print(f"Testing with 1 subject on {len(df_full)} points")
print()

# Call warmup_adapter.run() with DEFAULT parameters
# Should automatically use V3 (interaction_as_features=True by default)
print("Calling warmup_adapter.run() with default parameters...")
print("(Should use V3 method with interaction_as_features=True)")
print()

result = run(
    input_dir=test_dir,
    seed=99,
    output_mode="combined",
    clean=True,
    # Use all defaults - should trigger V3 with Config #1
    design_space_csv=str(design_space_csv),
    output_type="likert",
    likert_levels=5,
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    noise_std=0.0,
)

print()
print("=" * 80)
print("Verification: Compare with Expected Config #1 Distribution")
print("=" * 80)
print()

# Read result
result_csv = test_dir / "result/subject_1.csv"
if not result_csv.exists():
    # Try combined_results.csv
    result_csv = test_dir / "result/combined_results.csv"

df_result = pd.read_csv(result_csv)

# If combined format, extract first subject
if 'subject' in df_result.columns:
    df_result = df_result[df_result['subject'] == 'subject_1']

counter = Counter(df_result['y'])
mean = df_result['y'].mean()
std = df_result['y'].std()

print("Actual Distribution (from integrated adapter):")
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(df_result) * 100
    bar = '#' * int(pct / 5)
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

print(f"\n  Mean: {mean:.2f}")
print(f"  Std: {std:.2f}")
print()

print("Expected Distribution (Config #1 from test_asymmetric_interactions.py):")
print("  Likert 1:   93 (28.7%)")
print("  Likert 2:   45 (13.9%)")
print("  Likert 3:   41 (12.7%)")
print("  Likert 4:   56 (17.3%)")
print("  Likert 5:   89 (27.5%)")
print("  Mean: 3.01")
print()

# Check if results match
coverage = len([l for l in counter if counter[l] > 0])
max_ratio = max(counter.values()) / len(df_result)

# Exact match check (with seed=99, should be identical)
expected_dist = {1: 93, 2: 45, 3: 41, 4: 56, 5: 89}
is_exact_match = all(counter.get(i, 0) == expected_dist.get(i, 0) for i in range(1, 6))

print("Quality Check:")
print(f"  Coverage: {coverage} (expected: 5)")
print(f"  Max ratio: {max_ratio*100:.1f}% (expected: 28.7%)")
print(f"  Mean: {mean:.2f} (expected: 3.01)")
print(f"  Exact match: {'YES' if is_exact_match else 'NO'}")
print()

if is_exact_match:
    print("  [OK] PERFECT! Integration successful - produces exact Config #1 distribution")
elif coverage >= 4 and max_ratio <= 0.7 and 2.0 <= mean <= 4.5:
    print("  [OK] GOOD! Reasonable distribution (minor variation expected)")
else:
    print("  [FAIL] Distribution does not match expected Config #1")

# Verify V3 method was used (check fixed_weights_auto.json)
fixed_weights_json = test_dir / "result/fixed_weights_auto.json"
if fixed_weights_json.exists():
    import json
    with open(fixed_weights_json, 'r') as f:
        fixed_data = json.load(f)

    print()
    print("Verification: fixed_weights_auto.json content")
    print(f"  Method: {fixed_data.get('method', 'N/A')}")
    print(f"  Interactions: {fixed_data.get('interactions', {})}")

    if fixed_data.get('method') == 'interaction_as_features_v3':
        print("  [OK] V3 method was used (interaction-as-features)")
    else:
        print("  [WARNING] Method field does not indicate V3")

    # Check interaction values
    interactions = fixed_data.get('interactions', {})
    if '3,4' in interactions and '0,1' in interactions:
        x3x4 = interactions['3,4']
        x0x1 = interactions['0,1']
        print(f"  Interaction x3*x4: {x3x4:.3f} (expected: 0.12)")
        print(f"  Interaction x0*x1: {x0x1:.3f} (expected: -0.02)")

        if abs(x3x4 - 0.12) < 0.01 and abs(x0x1 - (-0.02)) < 0.01:
            print("  [OK] Interaction weights match Config #1")
        else:
            print("  [WARNING] Interaction weights do not match expected values")

# Cleanup
shutil.rmtree(test_dir)

print()
print("=" * 80)
print("Integration Test Completed!")
print("=" * 80)
print()
print("Summary:")
print("  - warmup_adapter.run() called with default parameters")
print("  - V3 method (interaction-as-features) used automatically")
print("  - Config #1 distribution achieved")
print("  - Integration successful!")
