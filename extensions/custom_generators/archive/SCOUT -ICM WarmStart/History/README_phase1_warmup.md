# Phase 1 预热采样实现文档

## 概述

`scout_warmup_251113.py` 实现了基于设计文档 `phase1_warmup_strategy_new.md` 的Phase 1预热采样策略。

## 功能特性

### 1. 采样策略实现

#### ✅ Core-1 固定重复点（8个点）

- **策略**: 角点 + 中心点
- **实现**: `select_core1_points()`
- **特点**:
  - 全取最小值、最大值角点
  - 4种交替模式角点
  - 中位数点 + 扰动中位数点
  - 每个受试者都测试这8个点

#### ✅ Core-2a 主效应覆盖（45次）

- **策略**: D-optimal设计
- **实现**: `select_core2_main_effects()`
- **特点**:
  - 最大化信息矩阵行列式
  - 确保每个因子水平有足够采样
  - 排除Core-1已选点

#### ✅ Core-2b 交互初筛（25次）

- **策略**: 象限均衡采样
- **实现**: `select_core2_interactions()`
- **特点**:
  - 测试5个交互对，每对5次
  - 4个象限各采样1次
  - 支持自定义优先交互对

#### ✅ 个体点a 边界极端（20次）

- **策略**: 分层极端点
- **实现**: `select_boundary_points()`
- **特点**:
  - 单维极端：每维2个点（只该因子极值，其他中位）
  - 全局极端：全最小、全最大
  - 使用maximin补充不足

#### ✅ 个体点b 分层LHS（29次）

- **策略**: 约束LHS + 最近邻匹配
- **实现**: `select_lhs_points()`
- **特点**:
  - 拉丁超立方采样保证空间覆盖
  - 映射到实际设计空间
  - 避免与已选点重复

### 2. 预算分配

```
总预算: 7人 × 25次 = 175次

分配:
- Core-1: 8点 × 7人 = 56次 (32%)
- Core-2主效应: 45次 (26%)  
- Core-2交互: 25次 (14%)
- 边界极端: 20次 (11%)
- 分层LHS: 29次 (17%)
```

## 使用方法

### 基本用法

```python
import numpy as np
import pandas as pd
from scout_warmup_251113 import Phase1WarmupSampler

# 1. 准备设计空间DataFrame
design_df = pd.read_csv('design_space.csv')  # 必须包含f1, f2, ..., fd列

# 2. 创建采样器
sampler = Phase1WarmupSampler(
    design_df=design_df,
    n_subjects=7,              # 受试者数量
    trials_per_subject=25,     # 每人试验次数
    seed=42                    # 随机种子
)

# 3. 执行采样
results = sampler.run_sampling()

# 4. 获取结果
trials_df = results['trials']          # 试验清单
core1_points = results['core1_points'] # Core-1固定点
```

### 使用优先交互对

```python
# 指定要测试的交互对（因子索引，0-based）
priority_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

sampler = Phase1WarmupSampler(
    design_df=design_df,
    n_subjects=7,
    trials_per_subject=25,
    priority_pairs=priority_pairs,  # 最多5对
    seed=42
)
```

### 导出结果

```python
# 运行采样
results = sampler.run_sampling()
sampler.trials = results['trials']  # 保存试验清单

# 导出到文件
sampler.export_results(output_dir='./phase1_output')

# 生成文件:
# - phase1_trials.csv: 完整试验清单
# - phase1_core1_points.csv: Core-1固定点
# - phase1_summary.json: 采样摘要
```

## 输出格式

### 试验清单 (phase1_trials.csv)

| 列名 | 说明 |
|------|------|
| trial_id | 试验编号 |
| subject_id | 受试者编号（0-based） |
| block_type | 块类型：core1/core2_main/core2_inter/boundary/lhs |
| design_idx | 在design_df中的索引 |
| pair_id | 交互对标识（仅core2_inter有值） |
| f1, f2, ... | 因子值 |

### 采样摘要 (phase1_summary.json)

```json
{
  "n_subjects": 7,
  "trials_per_subject": 25,
  "total_budget": 175,
  "d": 5,
  "factor_names": ["f1", "f2", "f3", "f4", "f5"],
  "budget": {
    "core1_unique": 8,
    "core1_total": 56,
    "core2_main": 45,
    "core2_inter": 25,
    "boundary": 20,
    "lhs": 29
  },
  "selected_counts": {
    "core1": 8,
    "core2_main": 45,
    "core2_inter": 25,
    "boundary": 20,
    "lhs": 29
  }
}
```

## 测试

运行测试套件：

```bash
python extensions/custom_generators/SCOUT\ -ICM\ WarmStart/test/test_phase1_warmup.py
```

测试包括：

1. 基本采样功能测试
2. 完整工作流测试
3. 优先交互对测试
4. 结果导出测试

## 设计原则

1. **忠实原设计文档**: 严格按照`phase1_warmup_strategy_new.md`实现
2. **确定性**: 使用随机种子确保可重现
3. **避免重复**: 各阶段采样互不重复
4. **空间覆盖**: 结合极端点、D-optimal和LHS保证覆盖
5. **灵活性**: 支持不同维度、预算和交互对配置

## 依赖项

```python
numpy
pandas
scipy
sklearn
```

## 限制与注意事项

1. **设计空间格式**: 必须包含f1, f2, ...命名的因子列
2. **最少配置数**: 建议至少200个配置以保证采样质量
3. **交互对数量**: 优先交互对最多5对
4. **因子类型**: 当前假设所有因子为连续型

## 后续扩展

根据设计文档，Phase 1完成后还需要实现：

### 数据分析模块（待实现）

1. **ICC估计**: 混合效应模型估计组内相关系数
2. **主效应估计**: 固定效应回归获取β和SE
3. **交互对筛选**:
   - p值检验
   - 效应/SE比率
   - 象限范围检验
   - BIC改善
4. **GP训练**: Matérn-5/2核 + ARD
5. **不确定性地图**: 预测标准差地图

### Phase1Output格式（待实现）

```json
{
  "measurement_quality": {
    "icc": 0.45,
    "ci_95": [0.38, 0.52]
  },
  "factor_effects": {
    "main_effects": {...},
    "lambda_main": 0.85
  },
  "interaction_screening": {
    "tested_pairs": [...],
    "selected_for_phase2": [3个],
    "lambda_interaction": 0.78
  },
  "gp_model": {
    "model_file": "phase1_gp.pkl",
    "cv_rmse": 0.82
  },
  "spatial_coverage": {
    "coverage_rate": 0.15,
    "uncertainty_map_file": "uncertainty_map.csv"
  }
}
```

## 作者

基于SCOUT -ICM WarmStart项目设计文档实现

## 版本

v1.0 (2025-11-13) - 初始实现

- 完成所有采样策略
- 支持试验清单生成
- 支持结果导出
