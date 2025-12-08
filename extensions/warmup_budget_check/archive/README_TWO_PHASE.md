# 两阶段实验规划系统

## 概述

本系统整合了 **Phase 1（预热阶段）** 和 **Phase 2（主动学习阶段）** 的完整实验规划流程，解决了预热方法与后续主动学习（EUR-ANOVA）之间的衔接问题。

### 核心功能

1. **Phase 1预算规划**：基于五步采样法的预算估算和评估
2. **Phase 1数据分析**：
   - 交互对筛选（基于残差分析 + BIC + 方差分解）
   - λ参数估计（主效应 vs 交互效应权重）
   - 主效应和交互效应估计
3. **Phase 2参数初始化**：
   - 自动从Phase 1继承交互对
   - 自动估算λ和γ的初始值
   - 生成λ和γ的动态调整schedule
4. **完整的工作流支持**：导出/加载Phase 1输出

---

## 文件结构

```
extensions/warmup_budget_check/
├── warmup_budget_estimator.py      # 原有的预算估算器
├── phase1_analyzer.py              # 新增：Phase 1数据分析
├── two_phase_planner.py            # 新增：两阶段规划器
├── example_two_phase_workflow.py   # 新增：使用示例
└── README_TWO_PHASE.md             # 本文档
```

---

## 快速开始

### 1. 运行示例

```bash
python example_two_phase_workflow.py
```

这将演示完整的工作流，包括：
- 创建示例设计空间
- 规划Phase 1
- 模拟Phase 1数据收集
- 分析Phase 1数据
- 规划Phase 2
- 导出Phase 1输出

### 2. 实际使用流程

```python
from two_phase_planner import TwoPhaseExperimentPlanner
import numpy as np

# Step 1: 初始化规划器
planner = TwoPhaseExperimentPlanner("your_design_space.csv")

# Step 2: 规划Phase 1
phase1_plan = planner.plan_phase1(
    n_subjects=14,
    trials_per_subject=25,
    skip_interaction=False
)

# Step 3: [执行Phase 1数据收集]
# ... 收集X_warmup, y_warmup, subject_ids ...

# Step 4: 分析Phase 1数据
phase1_analysis = planner.analyze_phase1_data(
    X_warmup=X_warmup,
    y_warmup=y_warmup,
    subject_ids=subject_ids,
    max_pairs=5,
    min_pairs=3,
    selection_method='elbow'
)

# Step 5: 规划Phase 2
phase2_plan = planner.plan_phase2(
    n_subjects=18,
    trials_per_subject=25,
    use_phase1_estimates=True
)

# Step 6: 导出Phase 1输出
exported_files = planner.export_phase1_output(
    output_dir="phase1_outputs",
    prefix="my_experiment"
)

# Step 7: [执行Phase 2主动学习]
# 使用phase2_plan中的参数初始化EUR-ANOVA
```

---

## 核心组件详解

### 1. Phase 1 数据分析 (`phase1_analyzer.py`)

**交互对筛选方法**：

```python
analyze_phase1_data(
    X_warmup, y_warmup, subject_ids,
    max_pairs=5,        # 最多选择5个交互对
    min_pairs=3,        # 最少选择3个交互对
    selection_method='elbow'  # 选择方法
)
```

**支持的选择方法**：
- `'elbow'`：肘部法则（推荐）- 自动找拐点
- `'bic_threshold'`：BIC阈值法 - 只选BIC增益显著的
- `'top_k'`：固定选择top-K

**评分机制**（综合三种方法）：
1. **残差模式评分**（30%权重）：四象限分析
2. **BIC增益**（50%权重）：模型改进程度
3. **方差解释**（20%权重）：额外解释的方差占比

### 2. λ参数估计

**原理**：
```
λ = 交互效应方差 / (主效应方差 + 交互效应方差)
```

**调整策略**：
- Phase 1估算λ₀
- Phase 2初始λ = λ₀ × 1.2（增强交互探索）
- 动态衰减：λ_t 从初始值逐渐降低到0.2

### 3. γ参数估计

**自适应策略**：
```python
if phase1_coverage < 0.20:
    gamma_init = 0.4    # 覆盖率很低，增加探索
elif phase1_coverage < 0.30:
    gamma_init = 0.3    # 覆盖率偏低
else:
    gamma_init = 0.2    # 覆盖率可接受
```

### 4. 动态Schedule

**λ的三阶段衰减**：
```
前40%: λ_init → λ_init × 0.7
中40%: λ_init × 0.7 → λ_init × 0.4
后20%: λ_init × 0.4 → 0.2
```

**γ的线性衰减**：
```
γ_t = γ_init × (1 - 0.8 × progress)
```

---

## Phase 1 输出详解

### 导出文件

调用 `planner.export_phase1_output()` 会生成：

1. **JSON文件** (`phase1_output.json`)
   - 筛选出的交互对
   - λ估计
   - 主效应和交互效应估计
   - 诊断信息

2. **Pickle文件** (`phase1_output_full.pkl`)
   - 完整的分析结果（包含所有中间数据）

3. **数据文件** (`phase1_output_data.npz`)
   - Phase 1的原始数据（X, y, subject_ids）

4. **文本报告** (`phase1_output_report.txt`)
   - 人类可读的分析报告

### 在Phase 2中使用

```python
from two_phase_planner import TwoPhaseExperimentPlanner

# 加载Phase 1输出
phase1_analysis = TwoPhaseExperimentPlanner.load_phase1_output(
    "phase1_outputs/my_experiment_full.pkl"
)

# 使用Phase 1的结果
selected_pairs = phase1_analysis['selected_pairs']
lambda_init = phase1_analysis['lambda_init']
```

---

## 与EUR-ANOVA的集成

### Phase 2初始化示例

```python
# 从Phase 2规划中获取参数
interaction_pairs = phase2_plan['interaction_pairs']  # [(0,3), (1,4), ...]
lambda_schedule = phase2_plan['lambda_schedule']      # [(1, 0.84), (2, 0.83), ...]
gamma_schedule = phase2_plan['gamma_schedule']        # [(1, 0.30), (2, 0.29), ...]

# 在EUR-ANOVA循环中使用
for trial in range(1, total_budget + 1):
    # 获取当前trial的λ和γ
    lambda_t = lambda_schedule[trial - 1][1]
    gamma_t = gamma_schedule[trial - 1][1]

    # EUR-ANOVA采集
    scores = EUR_ANOVA(
        X_candidates,
        gp_model,
        interaction_pairs,  # 从Phase 1继承
        lambda_t,
        gamma_t
    )

    X_next = X_candidates[argmax(scores)]
    # ... 执行实验 ...
```

---

## 关键改进点

### 相比原方案的优势

| 方面 | 原方案 | 新方案 |
|------|--------|--------|
| **交互对选择** | 预设5个（来源不明） | 数据驱动筛选3-5个 |
| **λ估计** | 无估计，盲目设定 | 从Phase 1数据估算 |
| **γ估计** | 固定0.3 | 根据覆盖率自适应 |
| **Phase衔接** | 断裂 | 完整工作流 |
| **可追溯性** | 无记录 | 完整导出+报告 |

### 解决的核心问题

1. ✅ **判定性 vs 探索性**：从"盲目覆盖"到"数据驱动筛选"
2. ✅ **参数初始化**：λ和γ不再是"拍脑袋"
3. ✅ **工作流完整性**：Phase 1 → Phase 2无缝衔接
4. ✅ **可复现性**：所有决策有数据支撑

---

## 常见问题

### Q1: Phase 1必须选出3-5个交互对吗？

不一定。`selection_method='elbow'` 会自动确定最优数量。如果数据显示只有2个显著交互，也会只选2个。`min_pairs`和`max_pairs`只是约束范围。

### Q2: 如果Phase 1数据质量很差怎么办？

系统会给出警告。如果拟合混合效应模型失败，会使用默认λ=0.5。建议检查数据质量后重新收集。

### Q3: 可以跳过Phase 1直接进入Phase 2吗？

不建议。Phase 2的EUR-ANOVA需要知道哪些交互对重要，否则会浪费大量预算在无关交互上。

### Q4: λ的衰减策略可以自定义吗？

可以。修改`_compute_lambda_schedule()`方法中的衰减曲线即可。

---

## 技术依赖

```python
# 核心依赖
numpy
pandas
scikit-learn
statsmodels      # 用于混合效应模型

# 可选（用于GP模型，如果需要在Phase 1训练GP）
aepsych
```

---

## 贡献者

基于原`warmup_budget_estimator.py`扩展开发。

---

## 许可证

与AEPsych项目保持一致。
