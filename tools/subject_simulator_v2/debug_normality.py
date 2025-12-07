#!/usr/bin/env python3
"""
调试正态性问题：分析为什么会产生99%+ Likert=5的极度偏斜分布
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2 import LinearSubject

# 1. 读取设计空间
design_csv = Path("../data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv")
if not design_csv.exists():
    design_csv = Path("../../data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv")

df = pd.read_csv(design_csv)
print(f"设计空间：{len(df)} 个点")
print(f"特征：{df.columns.tolist()}")
print()

# 2. 转换分类变量
categorical_mappings = {}
df_numeric = df.copy()

for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = sorted(df[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        categorical_mappings[col] = mapping
        df_numeric[col] = df_numeric[col].map(mapping)
        print(f"分类映射 {col}: {mapping}")

print()

# 3. 数值范围
print("特征数值范围：")
for col in df_numeric.columns:
    values = df_numeric[col].values
    print(f"  {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, std={values.std():.2f}")

print()

# 4. 使用202511301011的固定权重
weights_from_202511301011 = np.array([-0.04271, 0.61717, 0.08498, 0.39894, -0.04639, -0.02071])
print(f"权重（来自202511301011）: {weights_from_202511301011}")
print()

# 5. 计算连续输出（无交互项、无bias、无noise）
X = df_numeric.values.astype(float)
continuous_outputs = X @ weights_from_202511301011

print("连续输出统计：")
print(f"  min: {continuous_outputs.min():.4f}")
print(f"  max: {continuous_outputs.max():.4f}")
print(f"  mean: {continuous_outputs.mean():.4f}")
print(f"  std: {continuous_outputs.std():.4f}")
print(f"  25%: {np.percentile(continuous_outputs, 25):.4f}")
print(f"  50%: {np.percentile(continuous_outputs, 50):.4f}")
print(f"  75%: {np.percentile(continuous_outputs, 75):.4f}")
print()

# 6. 测试不同的Likert转换参数
print("=" * 80)
print("测试不同的Likert转换参数")
print("=" * 80)
print()

test_configs = [
    {"bias": 0.0, "sensitivity": 1.0, "desc": "原始参数"},
    {"bias": -0.3, "sensitivity": 2.0, "desc": "202511301011使用的参数"},
    {"bias": -1.0, "sensitivity": 1.0, "desc": "强负bias"},
    {"bias": -2.0, "sensitivity": 1.0, "desc": "极强负bias"},
    {"bias": 0.0, "sensitivity": 0.5, "desc": "降低灵敏度"},
    {"bias": -0.5, "sensitivity": 0.5, "desc": "负bias + 低灵敏度"},
]

for config in test_configs:
    bias = config["bias"]
    sensitivity = config["sensitivity"]

    subject = LinearSubject(
        weights=weights_from_202511301011,
        bias=bias,
        noise_std=0.0,
        likert_levels=5,
        likert_sensitivity=sensitivity,
        seed=42
    )

    responses = [subject(x) for x in X]

    from collections import Counter
    counter = Counter(responses)

    print(f"{config['desc']} (bias={bias}, sensitivity={sensitivity}):")
    for level in range(1, 6):
        count = counter.get(level, 0)
        pct = count / len(responses) * 100
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%)")
    print(f"  Mean: {np.mean(responses):.2f}")
    print()

# 7. 手动计算Likert转换看看阈值
print("=" * 80)
print("手动计算Likert转换（bias=-0.3, sensitivity=2.0）")
print("=" * 80)
print()

# 加上bias
continuous_with_bias = continuous_outputs - 0.3

# tanh转换
tanh_values = np.tanh(continuous_with_bias * 2.0)

# 映射到Likert 1-5
# formula: tanh_val * (levels - 1) / 2 + (levels + 1) / 2
likert_floats = tanh_values * (5 - 1) / 2 + (5 + 1) / 2

print("变换后的统计：")
print(f"  continuous_with_bias: min={continuous_with_bias.min():.4f}, max={continuous_with_bias.max():.4f}")
print(f"  tanh_values: min={tanh_values.min():.4f}, max={tanh_values.max():.4f}")
print(f"  likert_floats: min={likert_floats.min():.4f}, max={likert_floats.max():.4f}")
print()

# 显示一些样本
print("前20个点的详细计算：")
print(f"{'continuous':>12} {'with_bias':>12} {'tanh':>12} {'likert_float':>12} {'likert_int':>10}")
for i in range(20):
    cont = continuous_outputs[i]
    with_bias = continuous_with_bias[i]
    tanh_val = tanh_values[i]
    lik_float = likert_floats[i]
    lik_int = int(np.round(lik_float))
    lik_int = max(1, min(5, lik_int))
    print(f"{cont:12.4f} {with_bias:12.4f} {tanh_val:12.4f} {lik_float:12.4f} {lik_int:10d}")

print()
print("=" * 80)
print("诊断结论：")
print("=" * 80)
print()

# 计算需要什么样的bias才能让输出居中
target_continuous = 0.0  # tanh(0)=0 对应Likert=3
current_mean = continuous_outputs.mean()
needed_bias = -current_mean

print(f"当前连续输出均值：{current_mean:.4f}")
print(f"建议的bias值（使tanh输入居中于0）：{needed_bias:.4f}")
print()

# 测试建议的bias
print(f"测试建议的bias={needed_bias:.2f}:")
subject_suggested = LinearSubject(
    weights=weights_from_202511301011,
    bias=needed_bias,
    noise_std=0.0,
    likert_levels=5,
    likert_sensitivity=2.0,
    seed=42
)

responses_suggested = [subject_suggested(x) for x in X]
counter_suggested = Counter(responses_suggested)

for level in range(1, 6):
    count = counter_suggested.get(level, 0)
    pct = count / len(responses_suggested) * 100
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%)")
print(f"  Mean: {np.mean(responses_suggested):.2f}")
