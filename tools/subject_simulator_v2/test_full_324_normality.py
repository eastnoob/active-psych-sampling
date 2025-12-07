#!/usr/bin/env python3
"""
测试：1个被试在完整324点设计空间上作答，响应分布是否正态
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
from collections import Counter

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

print("=" * 80)
print("Test: Single subject answering on full 324-point design space")
print("=" * 80)
print()

# 设计空间CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# 创建临时测试目录
test_dir = Path(__file__).parent / "temp_test_324"
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

# 复制完整设计空间作为subject_1.csv
df_full = pd.read_csv(design_space_csv)
subject_1_csv = test_dir / "subject_1.csv"
df_full.to_csv(subject_1_csv, index=False)

print(f"Created test directory: {test_dir}")
print(f"Subject will answer on {len(df_full)} points from full design space")
print()

# 运行warmup_adapter
print("Generating subject and responses...")
print()

run(
    input_dir=test_dir,
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
    bias=0.0,  # Auto-calculate based on 324 points
    noise_std=0.0,
    design_space_csv=str(design_space_csv),  # Same 324 points
    print_model=False,
    save_model_summary=False
)

# 读取结果
result_csv = test_dir / "result/subject_1.csv"
df_result = pd.read_csv(result_csv)

print()
print("=" * 80)
print("Response Distribution (324 points)")
print("=" * 80)
print()

counter = Counter(df_result['y'])

for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(df_result) * 100
    bar = '#' * int(pct / 2)
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

mean = df_result['y'].mean()
print(f"\n  Mean: {mean:.2f}")

# 正态性检查
coverage = len([l for l in counter if counter[l] > 0])
max_ratio = max(counter.values()) / len(df_result)

print()
print("Normality Check:")
print(f"  Coverage (>=3 required): {coverage}")
print(f"  Max single level ratio (<=60% required): {max_ratio*100:.1f}%")
print(f"  Mean (2-4 preferred): {mean:.2f}")
print()

if coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0:
    print("[OK] Distribution is approximately normal!")
else:
    print("[FAIL] Distribution is NOT normal")
    if coverage < 3:
        print(f"  - Coverage too low: {coverage} < 3")
    if max_ratio > 0.6:
        print(f"  - Single level ratio too high: {max_ratio*100:.1f}% > 60%")
    if not (2.0 <= mean <= 4.0):
        print(f"  - Mean out of range: {mean:.2f} not in [2.0, 4.0]")

# 额外统计
import numpy as np
print()
print("Additional Statistics:")
print(f"  Std Dev: {np.std(df_result['y']):.2f}")
print(f"  Min: {df_result['y'].min()}")
print(f"  Max: {df_result['y'].max()}")
print(f"  Median: {df_result['y'].median():.1f}")

# 清理
print()
print(f"Cleaning up test directory: {test_dir}")
shutil.rmtree(test_dir)

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
