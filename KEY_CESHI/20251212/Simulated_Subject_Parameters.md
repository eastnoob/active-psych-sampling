# 模拟被试参数分布指南

**数据来源**: `extensions/warmup_budget_check/sample/202511271517/result/`

---

## 1. 5个被试的原始参数

| Subject | β₀ | β₁ | β₂ | β₃ | β₄ | β₅ | β₆ | noise |
|---------|-----|-----|-----|-----|-----|-----|-----|-------|
| 1 | 1.95 | -0.51 | 0.78 | -0.12 | -1.17 | 0.08 | -0.32 | 0.85 |
| 2 | 2.50 | 0.27 | 1.28 | 0.09 | -1.67 | 0.17 | 0.67 | 1.15 |
| 3 | 1.71 | 0.41 | 0.76 | 0.03 | -0.79 | -0.19 | -0.14 | 0.80 |
| 4 | 3.94 | -0.13 | 0.59 | 0.21 | -1.40 | 0.33 | -0.35 | 0.55 |
| 5 | 4.53 | -0.42 | 0.27 | 0.33 | -1.97 | 0.04 | 0.03 | 0.50 |

---

## 2. 参数分布 (生成新被试用)

| 参数 | 变量 | Mean | Std | 分布 |
|------|------|------|-----|------|
| β₀ | 截距 | 2.93 | 1.18 | N(2.93, 1.18²) |
| β₁ | x1_binary | -0.08 | 0.40 | N(-0.08, 0.40²) |
| β₂ | x2_5level | **0.74** | 0.37 | N(0.74, 0.37²) |
| β₃ | x3_decimal | 0.11 | 0.17 | N(0.11, 0.17²) |
| β₄ | x4_4level | **-1.40** | 0.44 | N(-1.40, 0.44²) |
| β₅ | x5_3level | 0.09 | 0.19 | N(0.09, 0.19²) |
| β₆ | x6_binary | -0.02 | 0.41 | N(-0.02, 0.41²) |
| noise | 噪声 | 0.77 | 0.26 | \|N(0.77, 0.26²)\| |

---

## 3. 生成代码

```python
import numpy as np

def generate_subject(seed=None):
    if seed: np.random.seed(seed)
    return {
        'beta_0': np.random.normal(2.93, 1.18),
        'beta_1': np.random.normal(-0.08, 0.40),
        'beta_2': np.random.normal(0.74, 0.37),   # 正效应
        'beta_3': np.random.normal(0.11, 0.17),
        'beta_4': np.random.normal(-1.40, 0.44),  # 最强负效应
        'beta_5': np.random.normal(0.09, 0.19),
        'beta_6': np.random.normal(-0.02, 0.41),
        'noise': abs(np.random.normal(0.77, 0.26))
    }
```

---

## 4. 效应强度与BaseGP对应

| 排名 | 变量 | \|β\| mean | BaseGP ls | 一致 |
|------|------|-----------|-----------|------|
| 1 | x4_4level | **1.40** | **0.437** | ✅ |
| 2 | x2_5level | **0.74** | **1.466** | ✅ |
| 3 | x1_binary | 0.35 | 1.939 | ✅ |
| 4 | x6_binary | 0.30 | 3.615 | ✅ |
| 5 | x3_decimal | 0.16 | 3.893 | ✅ |
| 6 | x5_3level | 0.16 | 4.238 | ✅ |

**结论**: 效应强度 |β| 与 1/lengthscale 完美对应

---

## 5. 关键特征

- **β₄ (x4_4level)**: 始终为负，最强效应
- **β₂ (x2_5level)**: 始终为正，第二强
- **β₀ (截距)**: 变异最大 (个体差异)
- **β₃, β₅**: 变异最小，效应微弱

---

## 6. 文件索引

| 文件 | 路径 |
|------|------|
| 被试参数汇总 | `sample/202511271517/result/subjects_parameters_summary.md` |
| 单被试详情 | `sample/202511271517/result/subject_*_model.md` |
| 模拟报告 | `sample/202511271517/result/SIMULATION_REPORT.md` |
| BaseGP对照 | `phase1_analysis_output/202511271557/base_gp/base_gp_lengthscales.json` |

*路径前缀: `F:\Github\aepsych-source\extensions\warmup_budget_check\`*
