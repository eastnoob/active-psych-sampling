# 使用指南：一键式实验规划

## 快速开始（3步）

### 1. 准备配置

```bash
# 复制配置模板
cp config_template.py config.py

# 编辑配置文件
nano config.py  # 或使用你喜欢的编辑器
```

**关键参数**（在`config.py`中修改）：

```python
# 你的设计空间CSV路径
DESIGN_CSV_PATH = "your_design_space.csv"

# Phase 1被试数和测试次数
PHASE1_CONFIG = {
    "n_subjects": 14,           # 根据你的资源调整
    "trials_per_subject": 25,   # 每人测试次数
    "skip_interaction": False,  # 是否包含交互探索
}

# Phase 2被试数和测试次数
PHASE2_CONFIG = {
    "n_subjects": 18,           # 剩余被试数
    "trials_per_subject": 25,
    "lambda_adjustment": 1.2,   # 交互权重调整（1.0-1.5）
}
```

### 2. 运行流程

```bash
python run_full_pipeline.py
```

**如果是演示/测试**（使用模拟数据）：

```python
# 在config.py中设置
SIMULATION_CONFIG = {
    "enabled": True,  # 启用模拟模式
}
```

**如果是实际实验**（使用真实数据）：

1. 收集Phase 1数据（按照五步采样法）
2. 将数据保存为`phase1_data.npz`：
   ```python
   import numpy as np
   np.savez('phase1_data.npz',
            X=X_warmup,           # 形状: (n_samples, n_factors)
            y=y_warmup,           # 形状: (n_samples,)
            subject_ids=subject_ids)  # 形状: (n_samples,)
   ```
3. 运行流程脚本

### 3. 查看输出

脚本会生成以下文件（在`experiment_outputs/`目录）：

```
my_experiment.json          # JSON格式结果
my_experiment_full.pkl      # 完整pickle结果
my_experiment_data.npz      # Phase 1原始数据
my_experiment_report.txt    # 人类可读报告
```

---

## 详细流程

脚本自动执行以下6个步骤：

### 步骤1: 规划Phase 1

- 输入：Phase 1配置参数
- 输出：预算分配方案
- 示例输出：
  ```
  总预算: 350次
  Core-1: 112次
  Core-2a: 95次
  Core-2b: 66次
  边界: 30次
  LHS: 47次
  ```

### 步骤2: 收集Phase 1数据

**模拟模式**：自动生成模拟数据

**实际实验模式**：
1. 按照规划执行五步采样
2. 收集数据：`X_warmup`, `y_warmup`, `subject_ids`
3. 保存为`phase1_data.npz`

### 步骤3: 分析Phase 1数据

- 自动筛选显著交互对（3-5个）
- 估算λ参数（主效应vs交互效应权重）
- 估计主效应和交互效应
- 示例输出：
  ```
  筛选出的交互对: 3个
    1. (density, greenery): score=142.460
    2. (height, style): score=48.072
    3. (density, style): score=0.140
  λ估计: 0.897
  ```

### 步骤4: 规划Phase 2

- 自动从Phase 1继承参数
- 生成λ和γ的动态schedule
- 示例输出：
  ```
  总预算: 450次
  λ初始: 0.950 (从Phase 1的0.897调整)
  γ初始: 0.300
  λ衰减: 0.950 → 0.200
  ```

### 步骤5: 导出所有输出

生成4个文件（见上文）

### 步骤6: 显示总结

显示整体统计、Phase 1→2衔接信息、下一步行动

---

## 配置参数详解

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_subjects` (Phase 1) | 14 | 5-20人（太少ICC不足，太多边际收益递减） |
| `trials_per_subject` (Phase 1) | 25 | 20-30次 |
| `n_subjects` (Phase 2) | 18 | 根据总预算和Phase 1确定 |
| `lambda_adjustment` | 1.2 | 1.0=不调整，1.5=大幅增强交互探索 |
| `selection_method` | `'elbow'` | 肘部法则（推荐），也可用`'bic_threshold'` |
| `max_pairs` | 5 | 交互对上限 |
| `min_pairs` | 3 | 交互对下限 |

### 高级参数

在`config_template.py`中有详细注释，一般无需修改。

---

## 常见问题

### Q1: 如何只规划Phase 1，不分析数据？

将流程拆分：

```python
from two_phase_planner import TwoPhaseExperimentPlanner

planner = TwoPhaseExperimentPlanner("design.csv")

# 只规划Phase 1
phase1_plan = planner.plan_phase1(
    n_subjects=14,
    trials_per_subject=25
)

# ... 然后收集数据 ...
```

### Q2: 如何从已有的Phase 1输出开始？

```python
from two_phase_planner import TwoPhaseExperimentPlanner

# 加载Phase 1输出
phase1_analysis = TwoPhaseExperimentPlanner.load_phase1_output(
    "experiment_outputs/my_experiment_full.pkl"
)

# 创建新的planner并设置Phase 1分析结果
planner = TwoPhaseExperimentPlanner("design.csv")
planner.phase1_analysis = phase1_analysis

# 规划Phase 2
phase2_plan = planner.plan_phase2(n_subjects=18, trials_per_subject=25)
```

### Q3: 如何调整λ和γ的衰减策略？

修改`two_phase_planner.py`中的`_compute_lambda_schedule()`和`_compute_gamma_schedule()`方法。

### Q4: 配置验证失败怎么办？

运行配置验证：

```bash
python config_template.py
```

会显示所有错误和警告。修正后再运行主流程。

---

## 输出文件说明

### 1. JSON文件（`*.json`）

轻量级，适合跨语言交换：

```json
{
  "selected_pairs": [[0, 2], [1, 4]],
  "lambda_init": 0.897,
  "main_effects": {...},
  "interaction_effects": {...}
}
```

### 2. Pickle文件（`*_full.pkl`）

完整的Python对象，包含所有中间结果：

```python
import pickle
with open("my_experiment_full.pkl", "rb") as f:
    data = pickle.load(f)

print(data['selected_pairs'])
print(data['diagnostics'])
```

### 3. 数据文件（`*_data.npz`）

Phase 1的原始数据：

```python
import numpy as np
data = np.load("my_experiment_data.npz")
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
```

### 4. 文本报告（`*_report.txt`）

人类可读的分析报告，包含：
- 筛选出的交互对及评分
- λ估计和方差分解
- 主效应和交互效应估计

---

## 在Phase 2中使用

```python
# 假设你已经运行了run_full_pipeline.py
# 现在要在EUR-ANOVA中使用结果

import pickle

# 加载Phase 1输出
with open("experiment_outputs/my_experiment_full.pkl", "rb") as f:
    phase1 = pickle.load(f)

# 获取交互对
interaction_pairs = phase1['selected_pairs']  # [(0, 2), (1, 4), ...]

# 获取λ和γ的schedule
# 这些应该已经在phase2_plan中生成了
# 如果需要重新生成：
from two_phase_planner import TwoPhaseExperimentPlanner

planner = TwoPhaseExperimentPlanner("design.csv")
planner.phase1_analysis = phase1

phase2_plan = planner.plan_phase2(n_subjects=18, trials_per_subject=25)

lambda_schedule = phase2_plan['lambda_schedule']
gamma_schedule = phase2_plan['gamma_schedule']

# 在EUR-ANOVA循环中使用
for trial in range(1, 451):
    lambda_t = lambda_schedule[trial - 1][1]
    gamma_t = gamma_schedule[trial - 1][1]

    # EUR-ANOVA采集
    scores = EUR_ANOVA(
        X_candidates,
        gp_model,
        interaction_pairs,
        lambda_t,
        gamma_t
    )

    X_next = X_candidates[argmax(scores)]
    # ... 执行实验 ...
```

---

## 故障排除

### 问题: "未找到config.py"

**解决**：复制`config_template.py`为`config.py`

### 问题: "设计空间文件不存在"

**解决**：确保`DESIGN_CSV_PATH`路径正确，或脚本会自动创建示例文件

### 问题: "Phase 1数据加载失败"

**解决**：
1. 检查`phase1_data.npz`是否存在
2. 或启用模拟模式：`SIMULATION_CONFIG['enabled'] = True`

### 问题: "Mixed model fitting failed"

**解决**：这是警告，不影响主流程。系统会使用默认λ=0.5

---

## 进阶使用

### 批量运行多个配置

```bash
# 创建多个配置文件
cp config_template.py config_14p_25t.py
cp config_template.py config_20p_30t.py

# 修改run_full_pipeline.py，循环加载不同配置
# 或使用命令行参数
```

### 自定义评分函数

修改`phase1_analyzer.py`中的`_select_interaction_pairs()`：

```python
# 自定义评分权重
score = 0.5 * pattern_score + 0.3 * bic_gain + 0.2 * var_gain
```

---

## 版本信息

- 基于: AEPsych warmup_budget_estimator
- 扩展: 两阶段实验规划系统
- 文档更新: 2025
