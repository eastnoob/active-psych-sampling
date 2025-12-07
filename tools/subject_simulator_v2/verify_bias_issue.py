#!/usr/bin/env python3
"""
验证bias计算问题
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# 1. 读取subject_1.csv（25个采样点）
subject_csv = Path("../../extensions/warmup_budget_check/sample/202511301011/subject_1.csv")
df = pd.read_csv(subject_csv)

# 转换分类变量
categorical_mappings = {}
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = sorted(df[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        categorical_mappings[col] = mapping
        df[col] = df[col].map(mapping)
        print(f"  {col}: {mapping}")

X_25points = df.values.astype(float)

# 2. 读取fixed_weights
fixed_weights_file = Path("../../extensions/warmup_budget_check/sample/202511301011/result/fixed_weights_auto.json")
with open(fixed_weights_file, 'r') as f:
    fixed_data = json.load(f)
    weights = np.array(fixed_data["global"][0])

print(f"\nFixed weights: {weights}")
print()

# 3. 计算25个点的连续输出
continuous_25 = X_25points @ weights
print("25 sampling points (subject_1.csv):")
print(f"  Range: [{continuous_25.min():.2f}, {continuous_25.max():.2f}]")
print(f"  Mean: {continuous_25.mean():.2f}")
print(f"  Auto-calculated bias: {-continuous_25.mean():.2f}")
print()

# 4. 加上bias后的输出
bias_auto = -continuous_25.mean()
with_bias_25 = continuous_25 + bias_auto
print("After applying bias (on 25 points):")
print(f"  Range: [{with_bias_25.min():.2f}, {with_bias_25.max():.2f}]")
print(f"  Mean: {with_bias_25.mean():.2f}")
print()

# 5. tanh转换
tanh_25 = np.tanh(with_bias_25 * 2.0)
print("After tanh (sensitivity=2.0):")
print(f"  Range: [{tanh_25.min():.4f}, {tanh_25.max():.4f}]")
print(f"  Mean: {tanh_25.mean():.4f}")
print()

# 6. Likert转换
likert_floats = tanh_25 * (5 - 1) / 2 + (5 + 1) / 2
likert_ints = np.round(likert_floats).astype(int)
likert_ints = np.clip(likert_ints, 1, 5)

print("Likert distribution (on 25 points):")
from collections import Counter
counter = Counter(likert_ints)
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(likert_ints) * 100
    print(f"  Likert {level}: {count:2d} ({pct:5.1f}%)")
print()

# 7. 与完整324点比较
design_csv = Path("../../data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv")
df_full = pd.read_csv(design_csv)

# 转换
for col in df_full.columns:
    if df_full[col].dtype == 'object':
        unique_vals = sorted(df_full[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df_full[col] = df_full[col].map(mapping)

X_324points = df_full.values.astype(float)

continuous_324 = X_324points @ weights
print("=" * 80)
print("Full 324 design space:")
print(f"  Range: [{continuous_324.min():.2f}, {continuous_324.max():.2f}]")
print(f"  Mean: {continuous_324.mean():.2f}")
print(f"  Optimal bias: {-continuous_324.mean():.2f}")
print()

# 8. 使用324点的optimal bias在25点上测试
bias_optimal = -continuous_324.mean()
with_bias_optimal = continuous_25 + bias_optimal
tanh_optimal = np.tanh(with_bias_optimal * 2.0)
likert_floats_optimal = tanh_optimal * (5 - 1) / 2 + (5 + 1) / 2
likert_ints_optimal = np.round(likert_floats_optimal).astype(int)
likert_ints_optimal = np.clip(likert_ints_optimal, 1, 5)

print("Likert distribution (25 points with 324-optimal bias):")
counter_optimal = Counter(likert_ints_optimal)
for level in range(1, 6):
    count = counter_optimal.get(level, 0)
    pct = count / len(likert_ints_optimal) * 100
    print(f"  Likert {level}: {count:2d} ({pct:5.1f}%)")
print()

print("=" * 80)
print("Conclusion:")
print(f"  25-point bias: {bias_auto:.2f} (too small!)")
print(f"  324-point optimal bias: {bias_optimal:.2f}")
print(f"  Difference: {abs(bias_optimal - bias_auto):.2f}")
print()
print("Problem: Auto-bias is calculated from 25 sampling points, not full design space!")
print("Solution: Need to use full design space CSV for bias calculation.")
