#!/usr/bin/env python3
"""
测试Step 1.5：生成5个被试并查看每个被试的响应分布
"""

from pathlib import Path
import sys
import pandas as pd
from collections import Counter

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

# 测试参数
input_dir = Path(__file__).parent.parent.parent / "extensions/warmup_budget_check/sample/202511301011"
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

print("=" * 80)
print("Step 1.5: Generate 5 Subjects with Auto-Bias")
print("=" * 80)
print()

if not input_dir.exists():
    print(f"[Error] Input directory not found: {input_dir}")
    sys.exit(1)

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# 运行Step 1.5
run(
    input_dir=input_dir,
    seed=99,
    output_mode="both",  # 同时保存individual和combined
    clean=True,
    interaction_pairs=[(3, 4), (0, 1)],
    interaction_scale=0.25,
    output_type="likert",
    likert_levels=5,
    likert_mode="tanh",
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    ensure_normality=True,
    bias=0.0,  # Auto-calculate based on full design space
    noise_std=0.0,
    design_space_csv=str(design_space_csv),  # Use 324-point design space
    print_model=True,
    save_model_summary=True,
    model_summary_format="both"
)

print()
print("=" * 80)
print("Individual Subject Response Distributions")
print("=" * 80)
print()

# 分析每个被试的响应分布
result_dir = input_dir / "result"

for i in range(1, 6):
    subject_csv = result_dir / f"subject_{i}.csv"
    if not subject_csv.exists():
        print(f"[Warning] subject_{i}.csv not found")
        continue

    df = pd.read_csv(subject_csv)
    counter = Counter(df['y'])

    print(f"Subject {i} (n={len(df)}):")
    for level in range(1, 6):
        count = counter.get(level, 0)
        pct = count / len(df) * 100
        bar = '#' * int(pct / 2)  # Simple bar chart
        print(f"  Likert {level}: {count:3d} ({pct:5.1f}%) {bar}")
    print(f"  Mean: {df['y'].mean():.2f}")

    # 正态性检查
    coverage = len([l for l in counter if counter[l] > 0])
    max_ratio = max(counter.values()) / len(df) if counter else 0
    mean = df['y'].mean()

    passed = coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0
    status = "[OK]" if passed else "[FAIL]"
    print(f"  Normality: {status} (coverage={coverage}, max_ratio={max_ratio*100:.1f}%, mean={mean:.2f})")
    print()

# 整体分布
print("=" * 80)
print("Combined Response Distribution (All 5 Subjects)")
print("=" * 80)
print()

combined_csv = result_dir / "combined_results.csv"
df_combined = pd.read_csv(combined_csv)
counter_all = Counter(df_combined['y'])

print(f"Total responses: {len(df_combined)}")
print()
for level in range(1, 6):
    count = counter_all.get(level, 0)
    pct = count / len(df_combined) * 100
    bar = '#' * int(pct / 2)
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")
print(f"  Mean: {df_combined['y'].mean():.2f}")
print()

# 正态性检查
coverage_all = len([l for l in counter_all if counter_all[l] > 0])
max_ratio_all = max(counter_all.values()) / len(df_combined)
mean_all = df_combined['y'].mean()

print("Overall Normality Check:")
print(f"  Coverage (>=3 required): {coverage_all}")
print(f"  Max single level ratio (<=60% required): {max_ratio_all*100:.1f}%")
print(f"  Mean (2-4 preferred): {mean_all:.2f}")
print()

if coverage_all >= 3 and max_ratio_all <= 0.6 and 2.0 <= mean_all <= 4.0:
    print("[OK] Overall distribution is normal!")
else:
    print("[FAIL] Overall distribution is not normal")

print()
print("=" * 80)
print("Generated Files:")
print("=" * 80)
print()
print(f"  {result_dir}/subject_1.csv ... subject_5.csv")
print(f"  {result_dir}/combined_results.csv")
print(f"  {result_dir}/subject_1_spec.json ... subject_5_spec.json")
print(f"  {result_dir}/fixed_weights_auto.json (use this to recreate subjects)")
print(f"  {result_dir}/cluster_summary.json")
print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
