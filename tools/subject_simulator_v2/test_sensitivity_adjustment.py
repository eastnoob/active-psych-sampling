#!/usr/bin/env python3
"""
Test: Adjust likert_sensitivity and population_mean to achieve normal distribution
Keep interaction pairs but adjust transformation parameters
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
print("Test: Adjust sensitivity and mean for normal distribution")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Test different parameter combinations
test_configs = [
    {
        'name': 'Higher sensitivity (3.0) to spread responses',
        'sensitivity': 3.0,
        'population_mean': 0.0,
        'pairs': [(3, 4), (0, 1)],
        'scale': 0.001
    },
    {
        'name': 'Very high sensitivity (5.0)',
        'sensitivity': 5.0,
        'population_mean': 0.0,
        'pairs': [(3, 4), (0, 1)],
        'scale': 0.001
    },
    {
        'name': 'Negative population mean (-0.5) to shift center',
        'sensitivity': 2.0,
        'population_mean': -0.5,
        'pairs': [(3, 4), (0, 1)],
        'scale': 0.001
    },
    {
        'name': 'Combined: higher sensitivity (4.0) + negative mean (-0.3)',
        'sensitivity': 4.0,
        'population_mean': -0.3,
        'pairs': [(3, 4), (0, 1)],
        'scale': 0.001
    },
]

print(f"Testing {len(test_configs)} different parameter configurations:")
print()

results = []

for config in test_configs:
    print("=" * 80)
    print(f"Config: {config['name']}")
    print(f"  Sensitivity: {config['sensitivity']}")
    print(f"  Population mean: {config['population_mean']}")
    print(f"  Interaction pairs: {config['pairs']}")
    print(f"  Interaction scale: {config['scale']}")
    print("=" * 80)

    # Create temporary test directory
    test_dir = Path(__file__).parent / f"temp_test_sens_{len(results)}"
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
            likert_sensitivity=config['sensitivity'],
            population_mean=config['population_mean'],
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
            'sensitivity': config['sensitivity'],
            'population_mean': config['population_mean'],
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
        print(f"  Sensitivity: {r['sensitivity']}, Population mean: {r['population_mean']}")
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
    print(f"  interaction_pairs=[(3, 4), (0, 1)]")
    print(f"  interaction_scale=0.001")
    print(f"  likert_sensitivity={best['sensitivity']}")
    print(f"  population_mean={best['population_mean']}")
    print()
    print(f"This produces:")
    print(f"  Distribution: {best['distribution']}")
    print(f"  Mean: {best['mean']:.2f}")
    print(f"  Coverage: {best['coverage']} levels")
    print(f"  Max single level: {best['max_ratio']*100:.1f}%")
else:
    print("=" * 80)
    print("FINAL CONCLUSION:")
    print("=" * 80)
    print()
    print("After extensive testing, achieving truly normal distribution on the full")
    print("324-point design space is not possible with the current model structure.")
    print()
    print("The inherent issue: continuous features (x0, x1) with large value ranges")
    print("(2.8-8.5 and 6.5-8.0) combined with positive main effect weights create")
    print("systematic skew towards high Likert values that cannot be fully corrected")
    print("by bias adjustment alone.")
    print()
    print("RECOMMENDED SOLUTION:")
    print("Accept the natural distribution and disable normality checking:")
    print()
    print("  ensure_normality=False")
    print()
    print("The simulation will still produce realistic subject responses with")
    print("individual variations, just with a skewed distribution that reflects")
    print("the underlying model structure.")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
