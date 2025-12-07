#!/usr/bin/env python3
"""
调试subject响应问题
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2 import LinearSubject

# 1. 从spec加载subject
spec_file = Path("../../extensions/warmup_budget_check/sample/202511301011/result/subject_1_spec.json")
with open(spec_file, 'r') as f:
    spec = json.load(f)

print("Subject spec:")
print(f"  weights: {spec['weights']}")
print(f"  bias: {spec['bias']}")
print(f"  interaction_weights: {spec['interaction_weights']}")
print(f"  likert_levels: {spec['likert_levels']}")
print(f"  likert_sensitivity: {spec['likert_sensitivity']}")
print()

# 2. 创建subject
subject = LinearSubject.from_dict(spec)

print("Created subject:")
print(f"  weights: {subject.weights}")
print(f"  bias: {subject.bias}")
print(f"  interaction_weights: {subject.interaction_weights}")
print()

# 3. 读取subject_1.csv
csv_file = Path("../../extensions/warmup_budget_check/sample/202511301011/subject_1.csv")
df = pd.read_csv(csv_file)

# 转换分类变量
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = sorted(df[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)

X = df.values.astype(float)

print(f"Loaded {len(X)} data points from subject_1.csv")
print()

# 4. 手动计算第一个点
x0 = X[0]
print(f"First data point: {x0}")

# 主效应
y_main = subject.bias + np.dot(subject.weights, x0)
print(f"  Main effect: {subject.bias} + {np.dot(subject.weights, x0):.4f} = {y_main:.4f}")

# 交互效应
y_interaction = 0.0
for (i, j), weight in subject.interaction_weights.items():
    y_interaction += weight * x0[i] * x0[j]
    print(f"  Interaction x{i}*x{j}: {weight:.4f} * {x0[i]} * {x0[j]} = {weight * x0[i] * x0[j]:.4f}")

y_continuous = y_main + y_interaction
print(f"  Total continuous: {y_continuous:.4f}")

# tanh
tanh_val = np.tanh(y_continuous * 2.0)
print(f"  tanh({y_continuous:.4f} * 2.0) = {tanh_val:.4f}")

# Likert
likert_float = tanh_val * (5 - 1) / 2 + (5 + 1) / 2
likert_int = int(np.round(likert_float))
likert_int = max(1, min(5, likert_int))
print(f"  Likert: {likert_float:.4f} -> {likert_int}")
print()

# 5. 使用subject生成所有响应
responses = [subject(x) for x in X]

print("Generated responses:")
from collections import Counter
counter = Counter(responses)
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(responses) * 100
    print(f"  Likert {level}: {count:2d} ({pct:5.1f}%)")
print(f"  Mean: {np.mean(responses):.2f}")
print()

# 6. 与result比较
result_csv = Path("../../extensions/warmup_budget_check/sample/202511301011/result/subject_1.csv")
df_result = pd.read_csv(result_csv)

print("From result/subject_1.csv:")
counter_result = Counter(df_result['y'])
for level in range(1, 6):
    count = counter_result.get(level, 0)
    pct = count / len(df_result) * 100
    print(f"  Likert {level}: {count:2d} ({pct:5.1f}%)")
print(f"  Mean: {df_result['y'].mean():.2f}")
