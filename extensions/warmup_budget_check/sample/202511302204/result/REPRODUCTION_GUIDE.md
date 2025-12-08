# 被试行为复现指南

本文档说明如何复现与当前模拟被试行为类似的新被试。

## 当前模拟结果摘要

**生成时间**: 2025-11-30
**方法**: V3 (Interaction-as-Features)
**分布质量**:
```
Likert 1:  38 (25.3%)
Likert 2:  10 ( 6.7%)
Likert 3:  13 ( 8.7%)
Likert 4:  23 (15.3%)
Likert 5:  66 (44.0%)
Mean: 3.46, Std: 1.67
```

**关键参数** (见 `fixed_weights_auto.json`):
- 主效应权重: `[0.199, -0.055, 0.259, 0.609, -0.094, -0.094]`
- 交互权重: `x3×x4=0.12, x0×x1=-0.02`
- Bias: `-0.72`
- 方法标记: `interaction_as_features_v3`

---

## 方法1: 完全相同复现（精确克隆）

**适用场景**: 需要完全相同的被试（用于验证、测试）

**步骤**:

1. 使用相同的配置参数运行 warmup_adapter:

```python
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path.cwd() / "tools"))
from subject_simulator_v2.adapters.warmup_adapter import run

# 配置参数（与生成时完全一致）
run(
    input_dir="path/to/sampling/plan",  # 新的采样方案目录
    seed=42,  # ⭐ 关键：相同的种子
    output_mode="combined",
    clean=True,
    # V3 方法参数（默认启用）
    interaction_as_features=True,
    interaction_x3x4_weight=0.12,
    interaction_x0x1_weight=-0.02,
    # 模型参数（与生成时一致）
    output_type="likert",
    likert_levels=5,
    likert_mode="tanh",
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.4,  # ⚠ 当前使用的值
    individual_std_percent=0.3,
    noise_std=0.0,
    design_space_csv="data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv",
)
```

**结果**: 生成的被试将具有**完全相同**的权重和行为模式。

---

## 方法2: 统计上类似（推荐用于新实验）

**适用场景**: 需要新的被试，但保持相同的分布特性

### 2.1 使用 fixed_weights_file (保持群体参数)

**步骤**:

1. 复制 `fixed_weights_auto.json` 到新实验目录
2. 运行时指定该文件:

```python
run(
    input_dir="path/to/NEW/sampling/plan",
    seed=99,  # ⭐ 不同的种子 → 不同的个体偏差
    fixed_weights_file="extensions/warmup_budget_check/sample/202511302204/result/fixed_weights_auto.json",
    # 其他参数保持一致
    output_type="likert",
    likert_levels=5,
    likert_sensitivity=2.0,
    population_std=0.4,
    individual_std_percent=0.3,
    design_space_csv="data/...",
)
```

**结果**:
- ✅ 主效应权重: 相同 (`[0.199, -0.055, ...]`)
- ✅ 交互权重: 相同 (`x3×x4=0.12, x0×x1=-0.02`)
- ✅ Bias: 相同 (`-0.72`)
- ✨ 个体偏差: **新采样** (因为seed不同)
- 📊 分布: 统计上类似，但略有差异

### 2.2 使用相同配置参数（新群体）

**步骤**:

```python
run(
    input_dir="path/to/NEW/sampling/plan",
    seed=123,  # ⭐ 完全不同的种子 → 新群体
    # 不指定 fixed_weights_file
    # 保持关键配置参数
    interaction_as_features=True,  # V3方法
    interaction_x3x4_weight=0.12,  # 保持交互权重
    interaction_x0x1_weight=-0.02,
    population_mean=0.0,
    population_std=0.4,  # 保持群体分布
    individual_std_percent=0.3,
    # 其他参数...
)
```

**结果**:
- ✨ 主效应权重: **新采样** `N(0.0, 0.4)`
- ✅ 交互权重: 相同 (`0.12, -0.02`)
- ✨ Bias: **重新计算** (基于新的主效应权重)
- 📊 分布: 统计上类似，但会有更大的变异

---

## 方法3: 调优获得更理想分布

**适用场景**: 希望改进分布质量（更接近 Mean=3.0）

### 建议调整

当前配置导致 Mean=3.46（稍高），可以调整：

```python
run(
    input_dir="path/to/sampling/plan",
    seed=99,  # ⭐ 尝试seed=99（测试时的完美分布）
    population_std=0.3,  # ⭐ 降低到0.3（更稳定）
    individual_std_percent=0.3,
    # 其他参数保持不变
    interaction_x3x4_weight=0.12,
    interaction_x0x1_weight=-0.02,
    likert_sensitivity=2.0,
    # ...
)
```

**预期结果** (基于测试经验):
```
Likert 1:  ~29%
Likert 2:  ~14%
Likert 3:  ~13%
Likert 4:  ~17%
Likert 5:  ~28%
Mean: ~3.0 (更接近理想)
```

---

## 关键参数说明

### 控制相同性的参数

| 参数 | 作用 | 固定以保持相似性 |
|------|------|------------------|
| `seed` | 随机种子 | ✅ 完全相同 → 完全相同被试 |
| `population_std` | 群体权重分布宽度 | ✅ 相同 → 相似的权重范围 |
| `interaction_x3x4_weight` | 强交互权重 | ✅ **必须**固定为 0.12 |
| `interaction_x0x1_weight` | 弱交互权重 | ✅ **必须**固定为 -0.02 |
| `likert_sensitivity` | Likert转换灵敏度 | ✅ 相同 → 相似的分布形状 |
| `fixed_weights_file` | 固定权重文件 | ✅ 使用 → 保持群体参数 |

### 允许变化的参数

| 参数 | 作用 | 可以调整 |
|------|------|----------|
| `seed` | 随机种子 | ✨ 不同 → 新的个体偏差 |
| `n_subjects` | 被试数量 | ✨ 可以增减 |
| `noise_std` | 试次内噪声 | ✨ 可以添加 (增加真实性) |

---

## 快速复现命令

### 在 quick_start.py 中配置

编辑 `extensions/warmup_budget_check/quick_start.py`:

```python
# 方法1: 完全相同复现
STEP1_5_CONFIG = {
    "input_dir": "extensions\\warmup_budget_check\\sample\\NEW_DIR",
    "seed": 42,  # 相同种子
    # ... 其他参数与当前完全一致
}

# 方法2: 统计上类似（推荐）
STEP1_5_CONFIG = {
    "input_dir": "extensions\\warmup_budget_check\\sample\\NEW_DIR",
    "seed": 99,  # 不同种子
    "fixed_weights_file": "extensions\\warmup_budget_check\\sample\\202511302204\\result\\fixed_weights_auto.json",
    "population_std": 0.3,  # 可选：调优
    # ... 其他参数
}
```

然后运行:
```bash
python extensions/warmup_budget_check/quick_start.py
```

---

## 验证复现质量

生成新被试后，验证分布质量:

```python
import pandas as pd
from collections import Counter

# 读取新结果
df_new = pd.read_csv("NEW_DIR/result/combined_results.csv")

# 读取原始结果
df_old = pd.read_csv("extensions/warmup_budget_check/sample/202511302204/result/combined_results.csv")

# 对比分布
print("原始分布:")
print(df_old['y'].value_counts().sort_index())
print(f"Mean: {df_old['y'].mean():.2f}")

print("\n新分布:")
print(df_new['y'].value_counts().sort_index())
print(f"Mean: {df_new['y'].mean():.2f}")

# 统计相似性检验（可选）
from scipy.stats import ks_2samp
statistic, pvalue = ks_2samp(df_old['y'], df_new['y'])
print(f"\nKS检验: p={pvalue:.3f} (>0.05表示分布相似)")
```

---

## 常见问题

### Q1: 为什么不同seed生成的分布会有差异？

**A**: seed控制：
1. 主效应权重的采样 (如果未使用 fixed_weights_file)
2. 个体偏差的采样
3. Bias的计算（间接受主效应权重影响）

使用 `fixed_weights_file` 可以固定住主效应和交互权重，减少变异。

### Q2: 如何确保交互效应保持一致？

**A**: 必须固定以下参数：
```python
interaction_as_features=True
interaction_x3x4_weight=0.12
interaction_x0x1_weight=-0.02
```

或使用 `fixed_weights_file`（包含这些参数）。

### Q3: 能否只复现某一个具体的被试？

**A**: 可以，但需要：
1. 从 `subject_X_spec.json` 读取该被试的完整参数
2. 手动创建 LinearSubject 对象并调用

具体方法见下方"高级用法"。

---

## 高级用法：精确复现单个被试

如果需要复现 `subject_1` 的精确行为：

```python
import json
import numpy as np
from pathlib import Path
from subject_simulator_v2 import LinearSubject

# 读取被试规格
spec_file = Path("extensions/warmup_budget_check/sample/202511302204/result/subject_1_spec.json")
with open(spec_file, 'r') as f:
    spec = json.load(f)

# 读取 fixed_weights
fixed_file = Path("extensions/warmup_budget_check/sample/202511302204/result/fixed_weights_auto.json")
with open(fixed_file, 'r') as f:
    fixed_data = json.load(f)

# 重建被试（V3方法直接使用8维权重）
main_weights = np.array(spec['weights'])  # 6个主效应
interaction_weights = np.array([
    fixed_data['interactions']['3,4'],
    fixed_data['interactions']['0,1']
])
weights_extended = np.concatenate([main_weights, interaction_weights])
bias = fixed_data['bias']

# 手动计算响应
def predict(X_base):
    # X_base: (n, 6) 基础特征
    interact_x3x4 = X_base[:, 2] * X_base[:, 3]
    interact_x0x1 = X_base[:, 0] * X_base[:, 1]
    X_extended = np.column_stack([X_base, interact_x3x4, interact_x0x1])

    continuous = X_extended @ weights_extended + bias
    tanh_output = np.tanh(2.0 * continuous)  # sensitivity=2.0
    return np.clip(np.round((tanh_output + 1) * 2 + 1), 1, 5).astype(int)

# 使用
# y = predict(new_design_points)
```

---

## 总结

| 需求 | 推荐方法 | 关键参数 |
|------|---------|---------|
| **完全相同克隆** | 方法1 | seed=42 + 所有参数一致 |
| **统计上类似（推荐）** | 方法2.1 | fixed_weights_file + 不同seed |
| **新群体但相似分布** | 方法2.2 | 保持 population_std + 交互权重 |
| **改进分布质量** | 方法3 | seed=99 + population_std=0.3 |

**核心原则**:
- ✅ **必须固定**: `interaction_x3x4_weight=0.12`, `interaction_x0x1_weight=-0.02`
- ✅ **建议固定**: `population_std`, `likert_sensitivity`
- ✨ **可以变化**: `seed`, `n_subjects`

---

生成日期: 2025-11-30
方法版本: V3 (Interaction-as-Features)
参考文档: `tools/subject_simulator_v2/INTEGRATION_SUMMARY.md`
