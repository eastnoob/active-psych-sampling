# 两阶段实验独立工作流指南

本指南说明如何使用两个独立脚本完成从预热采样到Phase 2参数生成的完整流程。

## 概述

整个流程分为**两个独立步骤**：

1. **Step 1**: 使用 `warmup_sampler.py` 生成预热采样方案
2. **Step 2**: 使用 `analyze_phase1.py` 分析实验数据并生成Phase 2参数

这两个步骤完全独立，中间由用户自行执行实际实验。

---

## Step 1: 生成预热采样方案

### 1.1 准备设计空间CSV

准备包含所有因子的全组合设计CSV（**只包含自变量，不包含因变量**）。

**示例文件** (`design_space.csv`)：
```csv
density,height,greenery,street_width,landmark,style
1,1,1,1,1,1
1,1,1,1,1,2
1,1,1,1,1,3
...
5,5,5,5,5,5
```

### 1.2 运行采样规划器

```bash
python warmup_sampler.py
```

### 1.3 交互式输入

脚本会要求输入以下信息：

```
请输入设计空间CSV路径（或按Enter使用默认 'design_space.csv'）: design_space.csv

请输入预算参数:
  被试数量: 14
  每个被试的最大承受次数: 25
  是否跳过交互效应探索？(y/N): N
```

### 1.4 预算评估

系统会自动评估预算充足性：

```
================================================================================
预算评估
================================================================================

输入参数:
  被试数: 14人
  每人trials: 25次
  总预算: 350次

预算分配方案:
  Core-1 (重复点):  112次
  Core-2a (主效应): 95次
  Core-2b (交互):   66次
  边界点:          30次
  LHS填充:         47次

充足性评估: 【刚好】

[OK] 做得好的方面:
  - Core-1重复数充足：14次/配置（ICC估计可靠）
  - 主效应预算合理：95次（充分估计）
  - 交互预算合理：66次（合理初筛）
  - 探索预算适中：77次（22.0%）
```

### 1.5 确认并生成采样文件

如果评估满意，确认生成：

```
是否生成采样方案？(Y/n): Y

输出配置:
  输出目录（默认 'sample'）: sample
  是否合并为单个CSV？(y/N): N
```

### 1.6 输出结果

生成的文件：

```
sample/
├── subject_1.csv       # 被试1的采样方案
├── subject_2.csv       # 被试2的采样方案
├── ...
├── subject_14.csv      # 被试14的采样方案
└── README.txt          # 实验说明文档
```

每个CSV包含该被试需要测试的所有配置（只有因子列）：

**示例** (`subject_1.csv`)：
```csv
density,height,greenery,street_width,landmark,style
3,2,5,1,4,2
1,5,3,4,2,5
...
```

---

## 中间步骤: 执行实验（用户自行完成）

1. 按照生成的CSV文件，为每个被试依次执行实验
2. 记录每个配置的响应值（因变量）
3. 将响应值添加到CSV中，或准备包含响应列的新CSV

**执行实验后的数据示例** (`warmup_data.csv`)：
```csv
subject_id,density,height,greenery,street_width,landmark,style,response
1,3,2,5,1,4,2,7.2
1,1,5,3,4,2,5,8.1
...
14,2,4,1,5,3,2,6.8
```

---

## Step 2: 分析实验数据并生成Phase 2参数

### 2.1 准备实验数据

确保数据CSV包含：
- **被试编号列**（默认: `subject_id`）
- **响应变量列**（默认: `response`）
- **所有因子列**

### 2.2 运行分析脚本

```bash
python analyze_phase1.py
```

### 2.3 交互式输入

```
请输入实验数据CSV路径（或按Enter使用默认 'warmup_data.csv'）: warmup_data.csv

请指定列名:
  被试编号列名（默认 'subject_id'）: subject_id
  响应变量列名（默认 'response'）: response

请配置分析参数:
  最多选择交互对数量（默认 5）: 5
  最少选择交互对数量（默认 3）: 3
  选择方法 (elbow/bic_threshold/top_k，默认 elbow): elbow
```

### 2.4 数据分析

系统会自动：
1. 评估所有可能的交互对
2. 使用指定方法筛选重要交互对
3. 估计λ参数（交互权重）
4. 计算主效应和交互效应

**输出示例**：
```
================================================================================
Phase 1数据分析
================================================================================

[1/3] 混合效应模型拟合...
  [OK] ICC = 0.234

[2/3] 评估所有交互对...
  候选交互对: 15个
  评估完成

[3/3] 筛选交互对（方法: elbow）...
  [OK] 选出3个交互对（肘部法则）

================================================================================
筛选结果
================================================================================

筛选出的交互对: 3个
  1. (density, greenery): score=142.460
  2. (height, style): score=48.072
  3. (density, style): score=0.140

λ估计（交互/总方差）: 0.897
```

### 2.5 配置Phase 2参数

```
================================================================================
Phase 2配置
================================================================================

请输入Phase 2参数:
  被试数量: 18
  每个被试的测试次数: 25
  λ调整系数（默认 1.2）: 1.2
```

### 2.6 生成配置文件

```
输出目录（默认 'phase1_analysis_output'）: phase1_analysis_output
文件名前缀（默认 'phase1'）: phase1
```

### 2.7 输出结果

生成4个文件：

```
phase1_analysis_output/
├── phase1_phase2_config.json      # JSON格式配置（供程序读取）
├── phase1_phase2_schedules.npz    # NumPy格式动态调度（供程序读取）
├── phase1_analysis_report.txt     # 人类可读的详细报告
└── PHASE2_USAGE_GUIDE.txt         # Phase 2使用指南
```

**JSON配置示例** (`phase1_phase2_config.json`)：
```json
{
  "interaction_pairs": [[0, 2], [1, 5], [0, 5]],
  "lambda_init": 0.950,
  "lambda_end": 0.200,
  "gamma_init": 0.300,
  "gamma_end": 0.060,
  "total_budget": 450,
  "mid_diagnostic_trial": 301
}
```

---

## Phase 2: 在EUR-ANOVA中使用参数

### 加载配置

```python
import numpy as np
import json

# 加载JSON配置
with open('phase1_analysis_output/phase1_phase2_config.json') as f:
    config = json.load(f)

# 加载动态调度
data = np.load('phase1_analysis_output/phase1_phase2_schedules.npz')
lambda_schedule = data['lambda_schedule']  # (n_trials, 2): (trial, lambda_t)
gamma_schedule = data['gamma_schedule']    # (n_trials, 2): (trial, gamma_t)
interaction_pairs = data['interaction_pairs'].tolist()
```

### 初始化EUR-ANOVA

```python
from eur_anova_pair import EURAnovaPairAcqf

acqf = EURAnovaPairAcqf(
    model=your_gp_model,
    gamma=config['gamma_init'],
    lambda_min=0.1,
    lambda_max=1.0,
    interaction_pairs=config['interaction_pairs'],  # 从Phase 1筛选
    tau1=0.7,
    tau2=0.3,
    tau_n_max=config['total_budget'],
    gamma_min=config['gamma_end']
)
```

### 主动学习循环

```python
for trial in range(1, config['total_budget'] + 1):
    # 获取当前λ和γ
    lambda_t = lambda_schedule[trial - 1, 1]
    gamma_t = gamma_schedule[trial - 1, 1]

    # EUR-ANOVA采集
    scores = acqf(X_candidates)
    next_idx = scores.argmax()
    X_next = X_candidates[next_idx]

    # 执行实验
    y_next = conduct_experiment(X_next)

    # 更新GP模型
    your_gp_model.update(X_next, y_next)

    # 中期诊断
    if trial == config['mid_diagnostic_trial']:
        run_mid_phase_diagnostic()
```

---

## 关键参数说明

### `selection_method`（交互对选择方法）

- **`elbow`**（推荐）：肘部法则，自动找拐点，在3-5个之间确定最优数量
- **`bic_threshold`**：BIC阈值法，只选择统计显著的交互对
- **`top_k`**：固定选择top-K个（不推荐，可能选入噪声）

### `lambda_adjustment`（λ调整系数）

- **1.0**：不调整，直接使用Phase 1估计
- **1.2**（默认）：Phase 2前期增强20%交互探索
- **1.5**：大幅增强交互探索（如果Phase 1显示强交互）

### `skip_interaction`（是否跳过交互探索）

- **False**（默认）：包含Core-2b交互探索，推荐
- **True**：跳过交互探索，仅在确定无交互时使用

---

## 文件流转图

```
设计空间CSV (只有自变量)
    ↓
[warmup_sampler.py]
    ↓
subject_N.csv文件 (采样方案)
    ↓
【用户执行实验】
    ↓
实验数据CSV (添加响应列)
    ↓
[analyze_phase1.py]
    ↓
Phase 2配置文件 (JSON + NPZ)
    ↓
【Phase 2 EUR-ANOVA主动学习】
```

---

## 常见问题

### Q1: 采样文件可以合并为单个CSV吗？

**可以**。在生成采样方案时选择合并：

```
是否合并为单个CSV？(y/N): y
被试编号列名（默认 'subject_id'）: subject_id
```

生成的合并文件会包含 `subject_id` 列标识每个样本属于哪个被试。

### Q2: 如果预算评估显示"不足"怎么办？

根据系统提示调整：
- 增加被试数
- 增加每个被试的测试次数
- 如果只关注主效应，可以设置 `skip_interaction=True`

### Q3: Phase 1必须选出3-5个交互对吗？

**不一定**。使用 `selection_method='elbow'` 时，系统会自动确定最优数量。如果数据只显示2个显著交互，会只选2个。

### Q4: 如果Phase 1数据质量很差怎么办？

系统会给出警告。如果拟合失败，会使用默认λ=0.5。建议：
- 检查数据质量（是否有缺失值、异常值）
- 检查响应变量是否正确
- 增加Phase 1样本数

### Q5: 可以修改λ/γ的衰减策略吗？

**可以**。修改 `analyze_phase1.py` 中的 `_compute_lambda_schedule()` 和 `_compute_gamma_schedule()` 方法。

当前使用指数衰减：
```python
λ_t = λ_init × exp(decay_rate × (t - 1))
```

可以改为线性衰减、分段常数等。

---

## 技术支持

如有问题，请参考：
- **完整技术文档**: `README_TWO_PHASE.md`
- **快速开始指南**: `QUICKSTART.md`
- **示例代码**: `example_two_phase_workflow.py`
