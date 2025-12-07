#!/usr/bin/env python3
"""
Test: Use only categorical interactions to achieve normal distribution
Remove (0,1) continuous interaction, keep only (3,4) and add (5,6)
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
print("Test: Categorical interactions only for normal distribution")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Test different categorical interaction configurations
test_configs = [
    {
        'name': 'Only (3,4) - single categorical interaction',
        'pairs': [(3, 4)],
        'scale': 0.25
    },
    {
        'name': 'Both (3,4) and (5,6) - two categorical interactions',
        'pairs': [(3, 4), (5, 6)],
        'scale': 0.25
    },
    {
        'name': '(3,4) with smaller scale',
        'pairs': [(3, 4)],
        'scale': 0.10
    },
]

print(f"Testing {len(test_configs)} different categorical interaction configurations:")
print()

results = []

for config in test_configs:
    print("=" * 80)
    print(f"Config: {config['name']}")
    print(f"  Pairs: {config['pairs']}")
    print(f"  Scale: {config['scale']}")
    print("=" * 80)

    # Create temporary test directory
    test_dir = Path(__file__).parent / f"temp_test_catonly_{len(results)}"
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
            interaction_pairs=config['pairs'],
            interaction_scale=config['scale'],
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
            'name': config['name'],
            'pairs': config['pairs'],
            'scale': config['scale'],
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
            reasons = []
            if coverage < 3:
                reasons.append(f"coverage={coverage} < 3")
            if max_ratio > 0.6:
                reasons.append(f"max_ratio={max_ratio*100:.1f}% > 60%")
            if not (2.0 <= mean <= 4.0):
                reasons.append(f"mean={mean:.2f} not in [2.0, 4.0]")
            print(f"  Reasons: {'; '.join(reasons)}")

        print()

    except Exception as e:
        print(f"[Error] Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': config['name'],
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

for r in results:
    if 'error' in r:
        print(f"{r['name']}:")
        print(f"  ERROR: {r['error']}")
    else:
        normal_status = "[OK] NORMAL" if r['is_normal'] else "[FAIL] NOT NORMAL"
        print(f"{r['name']}:")
        print(f"  Pairs: {r['pairs']}, Scale: {r['scale']}")
        print(f"  Distribution: {r['distribution']}")
        print(f"  Coverage: {r['coverage']}, Max ratio: {r['max_ratio']*100:.1f}%, Mean: {r['mean']:.2f}")
        print(f"  Status: {normal_status}")
    print()

# Recommendations
normal_results = [r for r in results if r.get('is_normal', False)]

if normal_results:
    print("=" * 80)
    print("RECOMMENDED CONFIGURATION:")
    print("=" * 80)
    print()

    best = normal_results[0]
    print(f"Use the following configuration in Step 1.5:")
    print(f"  interaction_pairs={best['pairs']}")
    print(f"  interaction_scale={best['scale']}")
    print()
    print(f"This produces:")
    print(f"  Distribution: {best['distribution']}")
    print(f"  Mean: {best['mean']:.2f}")
    print(f"  Coverage: {best['coverage']} levels")
    print(f"  Max single level: {best['max_ratio']*100:.1f}%")
else:
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print()
    print("None of the tested configurations produced normal distribution.")
    print()
    print("The fundamental issue is that this design space with these parameters")
    print("naturally skews towards high Likert values. Possible solutions:")
    print()
    print("1. Disable normality requirement (set ensure_normality=False)")
    print("2. Increase likert_sensitivity to spread responses more")
    print("3. Adjust population_mean to shift the center")
    print("4. Accept the natural distribution of the model")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
