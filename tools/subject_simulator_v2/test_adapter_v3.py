#!/usr/bin/env python3
"""
Test: warmup_adapter_v3 with Config #1 parameters
Verify that it produces the expected distribution
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
from collections import Counter

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter_v3 import run

print("=" * 80)
print("Test: warmup_adapter_v3 with Config #1")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Create temporary test directory
test_dir = Path(__file__).parent / "temp_test_v3"
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

# Run warmup_adapter_v3
result = run(
    input_dir=test_dir,
    seed=99,
    output_mode="combined",
    clean=True,
    interaction_x3x4_weight=0.12,  # Config #1
    interaction_x0x1_weight=-0.02,  # Config #1
    output_type="likert",
    likert_levels=5,
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    noise_std=0.0,
    design_space_csv=str(design_space_csv),
    print_model=False,
    save_model_summary=False
)

print()
print("=" * 80)
print("Comparison with Expected Results")
print("=" * 80)
print()

# Read result
result_csv = test_dir / "result/subject_1.csv"
df_result = pd.read_csv(result_csv)

counter = Counter(df_result['y'])
mean = df_result['y'].mean()
std = df_result['y'].std()

print("Actual Distribution:")
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(df_result) * 100
    bar = '#' * int(pct / 5)
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

print(f"\n  Mean: {mean:.2f}")
print(f"  Std: {std:.2f}")
print()

print("Expected Distribution (from test_asymmetric_interactions.py Config #7):")
print("  Likert 1:   93 (28.7%)")
print("  Likert 2:   45 (13.9%)")
print("  Likert 3:   41 (12.7%)")
print("  Likert 4:   56 (17.3%)")
print("  Likert 5:   89 (27.5%)")
print("  Mean: 3.01")
print()

# Check normality
coverage = len([l for l in counter if counter[l] > 0])
max_ratio = max(counter.values()) / len(df_result)

is_reasonable = coverage >= 4 and max_ratio <= 0.7 and 2.0 <= mean <= 4.5

print("Quality Check:")
print(f"  Coverage: {coverage} (>= 4 required)")
print(f"  Max ratio: {max_ratio*100:.1f}% (<= 70% required)")
print(f"  Mean: {mean:.2f} (2.0-4.5 required)")
print()

if is_reasonable:
    print("  [OK] EXCELLENT! Distribution is reasonable and balanced")
else:
    print("  [FAIL] Distribution quality needs improvement")

# Cleanup
shutil.rmtree(test_dir)

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
