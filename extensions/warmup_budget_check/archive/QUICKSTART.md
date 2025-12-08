# 快速开始指南

## 安装依赖

```bash
pip install numpy pandas scikit-learn statsmodels
```

## 快速测试

```bash
# 运行完整示例
python example_two_phase_workflow.py
```

这将创建示例数据，演示完整的两阶段工作流。

## 实际使用（5步流程）

### Step 1: 准备设计空间CSV

确保你的CSV包含所有因子列，例如：

```csv
density,height,greenery,street_width,landmark,style
1,1,1,1,1,1
1,1,1,1,1,2
...
```

### Step 2: 规划Phase 1

```python
from two_phase_planner import TwoPhaseExperimentPlanner

# 初始化
planner = TwoPhaseExperimentPlanner("your_design_space.csv")

# 规划Phase 1
phase1_plan = planner.plan_phase1(
    n_subjects=14,           # Phase 1被试数
    trials_per_subject=25,   # 每人测试次数
    skip_interaction=False   # 包含交互效应探索
)

# 输出：
#   Core-1: 112次
#   Core-2a: 95次
#   Core-2b: 66次
#   边界: 30次
#   LHS: 47次
#   总计: 350次
```

### Step 3: 收集Phase 1数据

按照五步采样法收集数据，得到：
- `X_warmup`: 形状 (350, 6) - 因子值
- `y_warmup`: 形状 (350,) - 响应变量
- `subject_ids`: 形状 (350,) - 被试ID

### Step 4: 分析Phase 1数据（关键步骤！）

```python
# 分析Phase 1数据
phase1_analysis = planner.analyze_phase1_data(
    X_warmup=X_warmup,
    y_warmup=y_warmup,
    subject_ids=subject_ids,
    max_pairs=5,              # 最多选5个交互对
    min_pairs=3,              # 最少选3个
    selection_method='elbow'  # 使用肘部法则自动确定数量
)

# 输出示例：
#   筛选出的交互对: 3个
#     1. (density, greenery): score=142.460
#     2. (height, style): score=48.072
#     3. (density, style): score=0.140
#   λ估计: 0.897
```

### Step 5: 规划Phase 2

```python
# 规划Phase 2（自动从Phase 1继承参数）
phase2_plan = planner.plan_phase2(
    n_subjects=18,           # Phase 2被试数
    trials_per_subject=25,   # 每人测试次数
    use_phase1_estimates=True,  # 使用Phase 1的估计
    lambda_adjustment=1.2    # λ调整系数（增强交互探索）
)

# 输出示例：
#   总预算: 450次
#   继承的交互对: 3个
#   λ初始: 0.950 (从Phase 1的0.897调整)
#   γ初始: 0.300
#   λ衰减: 0.950 → 0.200
#   γ衰减: 0.300 → 0.060
```

### Step 6: 导出Phase 1输出

```python
# 导出所有Phase 1结果（供Phase 2使用）
exported_files = planner.export_phase1_output(
    output_dir="phase1_outputs",
    prefix="my_experiment"
)

# 生成4个文件：
#   my_experiment.json - JSON格式结果
#   my_experiment_full.pkl - 完整pickle结果
#   my_experiment_data.npz - 原始数据
#   my_experiment_report.txt - 人类可读报告
```

### Step 7: 在Phase 2中使用

```python
# 获取Phase 2的EUR-ANOVA参数
interaction_pairs = phase2_plan['interaction_pairs']  # [(0, 2), (1, 4), (0, 4)]
lambda_schedule = phase2_plan['lambda_schedule']      # [(1, 0.948), (2, 0.946), ...]
gamma_schedule = phase2_plan['gamma_schedule']        # [(1, 0.299), (2, 0.298), ...]

# 在EUR-ANOVA循环中使用
for trial in range(1, 451):
    # 获取当前的λ和γ
    lambda_t = lambda_schedule[trial - 1][1]
    gamma_t = gamma_schedule[trial - 1][1]

    # EUR-ANOVA采集
    scores = EUR_ANOVA(
        X_candidates,
        gp_model,
        interaction_pairs,  # 从Phase 1筛选出的交互对
        lambda_t,           # 动态λ权重
        gamma_t             # 动态γ权重
    )

    # 选择最高分配置
    X_next = X_candidates[argmax(scores)]

    # 执行实验...
    y_next = conduct_experiment(X_next)

    # 更新GP模型...
    gp_model.update(X_next, y_next)

    # 中期诊断
    if trial == phase2_plan['mid_diagnostic_trial']:
        run_mid_phase_diagnostic()
```

## 关键参数说明

### `selection_method`（交互对选择方法）

- `'elbow'` - **推荐**：肘部法则，自动找拐点
- `'bic_threshold'` - BIC阈值法，只选显著的
- `'top_k'` - 固定选top-K（不推荐）

### `lambda_adjustment`（λ调整系数）

- `1.0` - 不调整，直接使用Phase 1估计
- `1.2` - **默认**：Phase 2前期增强20%交互探索
- `1.5` - 大幅增强交互探索（如果Phase 1显示强交互）

### `skip_interaction`

- `False` - **默认**：包含Core-2b交互探索
- `True` - 跳过交互（如果只关注主效应）

## 输出文件说明

### JSON文件（`my_experiment.json`）

```json
{
  "selected_pairs": [[0, 2], [1, 4]],
  "lambda_init": 0.897,
  "main_effects": {...},
  "interaction_effects": {...}
}
```

### 文本报告（`my_experiment_report.txt`）

人类可读的完整分析报告，包括：
- 筛选出的交互对及评分
- λ估计和方差分解
- 主效应和交互效应估计

## 常见问题

### Q: Phase 1必须选出3-5个交互对吗？

不一定。`selection_method='elbow'`会自动确定最优数量。如果数据只显示2个显著交互，会只选2个。

### Q: 如果Phase 1数据质量很差怎么办？

系统会给出警告。如果拟合失败，会使用默认λ=0.5。建议检查数据质量。

### Q: 可以修改λ/γ的衰减策略吗？

可以。修改`TwoPhaseExperimentPlanner`类中的`_compute_lambda_schedule()`和`_compute_gamma_schedule()`方法。

## 技术支持

如有问题，请参考：
- 完整文档：`README_TWO_PHASE.md`
- 示例代码：`example_two_phase_workflow.py`
