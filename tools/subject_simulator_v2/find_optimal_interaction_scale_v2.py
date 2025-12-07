#!/usr/bin/env python3
"""
Find suitable interaction_scale value for normal distribution
Test MUCH smaller values: 0.001 to 0.008
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
print("Finding optimal interaction_scale for normal distribution (V2)")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Test MUCH smaller interaction_scale values
test_scales = [0.001, 0.002, 0.003, 0.005, 0.008]

print(f"Testing {len(test_scales)} very small interaction_scale values:")
print(f"  {test_scales}")
print()
print("Interaction pairs: [(3, 4), (0, 1)]")
print()

results = []

for scale in test_scales:
    print("=" * 80)
    print(f"Testing interaction_scale = {scale}")
    print("=" * 80)

    # Create temporary test directory
    test_dir = Path(__file__).parent / f"temp_test_scale_{scale}"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    # Copy full design space as subject_1.csv
    df_full = pd.read_csv(design_space_csv)
    subject_1_csv = test_dir / "subject_1.csv"
    df_full.to_csv(subject_1_csv, index=False)

    # Run warmup_adapter
    try:
        run(
            input_dir=test_dir,
            seed=99,
            output_mode="combined",
            clean=True,
            interaction_pairs=[(3, 4), (0, 1)],
            interaction_scale=scale,
            output_type="likert",
            likert_levels=5,
            likert_mode="tanh",
            likert_sensitivity=2.0,
            population_mean=0.0,
            population_std=0.3,
            individual_std_percent=0.3,
            ensure_normality=False,
            bias=0.0,  # Auto-calculate
            noise_std=0.0,
            design_space_csv=str(design_space_csv),
            print_model=False,
            save_model_summary=False
        )

        # Read result
        result_csv = test_dir / "result/subject_1.csv"
        df_result = pd.read_csv(result_csv)

        counter = Counter(df_result['y'])

        # Calculate statistics
        coverage = len([l for l in counter if counter[l] > 0])
        max_ratio = max(counter.values()) / len(df_result)
        mean = df_result['y'].mean()

        # Normality check
        is_normal = coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0

        # Save result
        result = {
            'scale': scale,
            'coverage': coverage,
            'max_ratio': max_ratio,
            'mean': mean,
            'is_normal': is_normal,
            'distribution': dict(counter)
        }
        results.append(result)

        # Print distribution
        print()
        for level in range(1, 6):
            count = counter.get(level, 0)
            pct = count / len(df_result) * 100
            bar = '#' * int(pct / 5)
            print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n  Mean: {mean:.2f}")
        print(f"  Coverage: {coverage}")
        print(f"  Max ratio: {max_ratio*100:.1f}%")

        if is_normal:
            print(f"\n  [OK] NORMAL DISTRIBUTION!")
        else:
            print(f"\n  [FAIL] Not normal")

        print()

    except Exception as e:
        print(f"[Error] Failed with scale={scale}: {e}")
        results.append({
            'scale': scale,
            'error': str(e)
        })

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)

# Summary
print()
print("=" * 80)
print("Summary of Results")
print("=" * 80)
print()

print(f"{'Scale':<8} {'Coverage':<10} {'Max Ratio':<12} {'Mean':<8} {'Normal?':<10}")
print("-" * 60)

for r in results:
    if 'error' in r:
        print(f"{r['scale']:<8} ERROR: {r['error']}")
    else:
        normal_mark = "[OK] YES" if r['is_normal'] else "[FAIL] NO"
        print(f"{r['scale']:<8} {r['coverage']:<10} {r['max_ratio']*100:>5.1f}%{' '*6} {r['mean']:<8.2f} {normal_mark}")

# Find best values
normal_results = [r for r in results if r.get('is_normal', False)]

if normal_results:
    print()
    print("=" * 80)
    print("Recommended interaction_scale values:")
    print("=" * 80)
    print()

    for r in normal_results:
        print(f"  interaction_scale = {r['scale']}")
        print(f"    Distribution: {r['distribution']}")
        print(f"    Mean: {r['mean']:.2f}, Coverage: {r['coverage']}, Max ratio: {r['max_ratio']*100:.1f}%")
        print()
else:
    print()
    print("[Warning] No scale produced a normal distribution!")
    print("The interaction effect on continuous variables may be fundamentally incompatible")
    print("with normality requirement. Consider:")
    print("  1. Using interactions only on categorical variables")
    print("  2. Increasing population_std or likert_sensitivity")
    print("  3. Disabling normality requirement")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
