# EUR Residual Test Tool

EUR（Expected Uncertainty Reduction）采集函数残差测试工具。使用BaseGP关键点作为warmup，测试EUR采集函数性能。

## 使用方法

```bash
python run_eur_residual.py --budget 50 --seed 42
```

### 参数

- `--budget`: EUR采样预算（默认50）
- `--seed`: 随机种子（默认42）

## 配置说明

配置文件：`configs/eur_residual_test.ini`

### 参数类型定义

**数值离散参数**（使用 `custom_ordinal_mono`）：
- x1_CeilingHeight: [2.8, 4.0, 8.5]
- x2_GridModule: [6.5, 8.0]

**字符串分类参数**（使用 `categorical`）：
- x3_OuterFurniture: ['Chaos', 'Rotated', 'Strict']
- x4_VisualBoundary: ['Color', 'Solid', 'Translucent']
- x5_PhysicalBoundary: ['Closed', 'Open']
- x6_InnerFurniture: ['Chaos', 'Rotated', 'Strict']

### Warmup策略

使用BaseGP关键点（3个黄金点）：
1. **Best** (mean=1.366): [2.8, 6.5, 'Strict', 'Translucent', 'Closed', 'Chaos']
2. **Worst** (mean=-1.588): [4.0, 6.5, 'Chaos', 'Color', 'Open', 'Strict']
3. **MaxStd** (std=0.796): [8.5, 8.0, 'Strict', 'Translucent', 'Open', 'Chaos']

数据来源：`extensions/warmup_budget_check/phase1_analysis_output/202512081445/step3/base_gp_key_points.json`

### EUR采样策略

- 采集函数：MCPosteriorVariance
- Exploration目标
- 每次迭代后重新拟合模型

## 输出

结果保存在 `results/<timestamp>/` 目录：
- `data_files/sampling_history.npy`: 采样历史
- `data_files/interaction_logs.json`: 交互日志
- `data_files/warmup_logs.json`: Warmup日志
- `data_files/summary.json`: 实验摘要

## 依赖

- aepsych
- numpy
- pandas
- torch
- loguru

## 注意事项

1. **参数类型**：数值离散参数必须使用 `custom_ordinal_mono`，字符串分类参数使用 `categorical`
2. **ManualGenerator**：使用实际值（包括字符串），不使用索引
3. **设计空间**：来自 `data/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv`
