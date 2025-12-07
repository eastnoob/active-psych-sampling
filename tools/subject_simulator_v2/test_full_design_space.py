#!/usr/bin/env python3
"""
测试使用完整设计空间CSV的功能
"""

from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

# 测试参数
input_dir = Path(__file__).parent.parent.parent / "extensions/warmup_budget_check/sample/202511301011"
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

print("=" * 80)
print("Test with Full Design Space CSV")
print("=" * 80)
print()

if not input_dir.exists():
    print(f"[Error] Input directory not found: {input_dir}")
    sys.exit(1)

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

print(f"Input directory: {input_dir}")
print(f"Design space CSV: {design_space_csv}")
print()

# 第一次运行：生成数据（无fixed_weights）
print("=" * 80)
print("STEP 1: Generate subjects with auto-calculated bias (using 324-point design space)")
print("=" * 80)
print()

run(
    input_dir=input_dir,
    seed=99,
    output_mode="combined",
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
    bias=0.0,  # Auto-calculate
    noise_std=0.0,
    design_space_csv=str(design_space_csv),  # Use full 324-point design space
    print_model=True,
    save_model_summary=True,
    model_summary_format="both"
)

# 读取生成的fixed_weights_auto.json
import json
fixed_weights_file = input_dir / "result/fixed_weights_auto.json"
with open(fixed_weights_file, 'r') as f:
    fixed_data = json.load(f)

print()
print("=" * 80)
print("Generated fixed_weights_auto.json:")
print("=" * 80)
print(json.dumps(fixed_data, indent=2))
print()

# 读取并分析结果
import pandas as pd
result_csv = input_dir / "result/combined_results.csv"
df = pd.read_csv(result_csv)

print("=" * 80)
print("Response Distribution (First Run)")
print("=" * 80)
print(f"Total responses: {len(df)}")
print()
print("Response distribution:")
from collections import Counter
counter = Counter(df['y'])
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(df) * 100
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%)")
print(f"Mean: {df['y'].mean():.2f}")
print()

# 检查正态性
coverage = len([l for l in counter if counter[l] > 0])
max_ratio = max(counter.values()) / len(df)
mean = df['y'].mean()

print("Normality Check:")
print(f"  Coverage (>=3 required): {coverage}")
print(f"  Max single level ratio (<=60% required): {max_ratio*100:.1f}%")
print(f"  Mean (2-4 preferred): {mean:.2f}")
print()

if coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0:
    print("[OK] Normality check passed!")
else:
    print("[FAIL] Normality check failed")

print()
print("=" * 80)
print("STEP 2: Reload with fixed_weights (should reproduce exact same results)")
print("=" * 80)
print()

# 第二次运行：使用fixed_weights重现
run(
    input_dir=input_dir,
    seed=99,
    output_mode="combined",
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
    ensure_normality=False,  # Disable normality check for fixed weights
    bias=0.0,  # Will use bias from fixed_weights
    noise_std=0.0,
    fixed_weights_file=str(fixed_weights_file),
    design_space_csv=str(design_space_csv),
    print_model=False,
    save_model_summary=False
)

# 分析第二次结果
df2 = pd.read_csv(result_csv)

print()
print("=" * 80)
print("Response Distribution (Second Run with fixed_weights)")
print("=" * 80)
print(f"Total responses: {len(df2)}")
print()
print("Response distribution:")
counter2 = Counter(df2['y'])
for level in range(1, 6):
    count = counter2.get(level, 0)
    pct = count / len(df2) * 100
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%)")
print(f"Mean: {df2['y'].mean():.2f}")
print()

# 比较两次结果
if (df['y'] == df2['y']).all():
    print("[OK] Results are identical! Fixed weights work correctly.")
else:
    print("[FAIL] Results differ!")
    diff_count = (df['y'] != df2['y']).sum()
    print(f"  Different responses: {diff_count}/{len(df)}")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
