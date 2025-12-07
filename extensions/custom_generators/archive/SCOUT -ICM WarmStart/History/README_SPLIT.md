# SCOUT Warmup Generator - Modular Architecture

## 背景与拆分动机

AEPsych采用"一被试一个generator"的架构设计。为了适应这一设计，我们将原有的单体`WarmupAEPsychGenerator`拆分为两个独立模块：

1. **WarmupSubjectGenerator** (单被试生成器) - 面向单个被试的试验生成
2. **StudyCoordinator** (全局协调器) - 面向整个研究的全局规划与约束

这种拆分使得：

- 单被试生成器可以独立运行，不依赖跨被试信息
- 全局协调器负责高层规划，生成subject_plan供单被试生成器使用
- 两模块职责清晰，便于测试、维护和扩展

## 模块对比

### WarmupSubjectGenerator (单被试生成器)

**文件**: `warmup_subject_generator.py`

**定位**: 为单个被试生成Phase-1 warmup试验列表

**输入**:

- `design_df`: DataFrame，包含所有候选刺激（列: f1...fd + 可选元数据）
- `subject_plan`: Dict，包含:

  ```python
  {
      "subject_id": str or int,
      "quotas": {
          "core1": int,      # Core-1点数量
          "main": int,       # 主效应点数量
          "inter": int,      # 交互点数量
          "boundary": int,   # 边界点数量
          "lhs": int         # LHS填充点数量
      },
      "constraints": {
          "must_include_design_ids": List[int],         # 必须包含的设计点
          "per_factor_min_counts": Dict[str, int],      # 每个因子的最小计数
          "interactions": List[Dict]                     # 交互规格
      }
  }
  ```

- `seed`: Optional[int]

**输出**:

- `trials_df`: DataFrame，包含列:
  - `subject_id`: 被试ID
  - `block_type`: 试验块类型 (must_include/core1/main_effects/interaction/boundary/lhs)
  - `trial_index`: 被试内试验序号
  - `f1...fd`: 因子值
  - `design_row_id`: 对应design_df的行索引

**核心方法**:

```python
generator = WarmupSubjectGenerator(design_df, subject_plan, seed=42)
trials_df = generator.generate_trials()
summary = generator.summarize(trials_df)
```

**特点**:

- 不处理跨被试约束（桥接、批次等）
- 专注于单被试层面的点选择与去重
- 使用maximin和D-optimal策略确保空间覆盖
- 支持LHS采样的距离优化

### StudyCoordinator (全局协调器)

**文件**: `study_coordinator.py`

**定位**: 全局研究规划，生成subject_plan并管理跨被试约束

**输入**:

- `design_df`: DataFrame，包含所有候选刺激
- `n_subjects`: int，被试数量 (默认10)
- `total_budget`: int，总试验预算 (默认350)
- `n_batches`: int，批次数量 (默认3)
- `seed`: Optional[int]

**输出**:

- `subject_plan`: Dict，单个被试的规划（通过`allocate_subject_plan()`方法）
- `global_summary`: Dict，全局研究汇总（通过`summarize_global()`方法）

**核心方法**:

```python
coordinator = StudyCoordinator(design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42)
coordinator.fit_initial_plan()

# 为每个被试分配计划
for subject_id in range(10):
    subject_plan = coordinator.allocate_subject_plan(subject_id)
    # 可将subject_plan传递给WarmupSubjectGenerator

# 全局汇总
global_summary = coordinator.summarize_global()
```

**特点**:

- 管理全局Core-1候选集
- 基于方差启发式选择交互对
- 生成边界库
- 使用最大余数法进行预算分配
- 规划桥接被试
- 检查Core-1重复策略的可行性

## 使用示例

### 示例1: 独立运行单被试生成器

```python
import pandas as pd
import numpy as np
from warmup_subject_generator import WarmupSubjectGenerator

# 准备设计数据
design_data = {
    'f1': np.random.rand(100),
    'f2': np.random.rand(100),
    'f3': np.random.rand(100),
}
design_df = pd.DataFrame(design_data)

# 定义被试计划（通常由协调器生成，这里手动指定）
subject_plan = {
    "subject_id": "S001",
    "quotas": {
        "core1": 8,
        "main": 20,
        "inter": 10,
        "boundary": 8,
        "lhs": 14,
    },
    "constraints": {
        "must_include_design_ids": [0, 10, 20],
        "per_factor_min_counts": {"f1": 5},
        "interactions": [{"pair": [0, 1], "quadrants": 3, "strategy": "balanced"}],
    },
}

# 生成试验
generator = WarmupSubjectGenerator(design_df, subject_plan, seed=42)
trials_df = generator.generate_trials()
summary = generator.summarize(trials_df)

print(f"Generated {len(trials_df)} trials")
print(f"Coverage: {summary['coverage_rate']:.3f}")
print(f"Gini: {summary['gini']:.3f}")
```

### 示例2: 独立运行全局协调器

```python
import pandas as pd
import numpy as np
from study_coordinator import StudyCoordinator

# 准备设计数据
design_data = {
    'f1': np.random.rand(300),
    'f2': np.random.rand(300),
    'f3': np.random.rand(300),
    'f4': np.random.rand(300),
    'f5': np.random.rand(300),
}
design_df = pd.DataFrame(design_data)

# 创建协调器
coordinator = StudyCoordinator(
    design_df=design_df,
    n_subjects=10,
    total_budget=350,
    n_batches=3,
    seed=42
)

# 拟合全局计划
coordinator.fit_initial_plan()

# 为每个被试分配计划
subject_plans = []
for subject_id in range(10):
    plan = coordinator.allocate_subject_plan(subject_id)
    subject_plans.append(plan)
    print(f"Subject {subject_id}: quotas={plan['quotas']}, bridge={plan['is_bridge']}")

# 全局汇总
global_summary = coordinator.summarize_global()
print(f"\nExpected coverage: {global_summary['expected_coverage']:.3f}")
print(f"Warnings: {global_summary['warnings']}")
```

### 示例3: 联合使用（建议的集成模式）

```python
from study_coordinator import StudyCoordinator
from warmup_subject_generator import WarmupSubjectGenerator

# 全局规划阶段
coordinator = StudyCoordinator(design_df, n_subjects=10, total_budget=350, seed=42)
coordinator.fit_initial_plan()

# 为每个被试生成试验
all_trials = []
all_summaries = []

for subject_id in range(10):
    # 1. 获取被试计划
    subject_plan = coordinator.allocate_subject_plan(subject_id)
    
    # 2. 生成被试试验
    generator = WarmupSubjectGenerator(design_df, subject_plan, seed=42)
    trials_df = generator.generate_trials()
    summary = generator.summarize(trials_df)
    
    # 3. 保存结果
    all_trials.append(trials_df)
    all_summaries.append(summary)

# 合并所有被试的试验
combined_trials = pd.concat(all_trials, ignore_index=True)
print(f"Total trials: {len(combined_trials)}")
```

## 后续联动的建议数据协议

为了支持模块间协调和结果追踪，建议采用以下文件协议：

### 1. subject_plan_{sid}.json

协调器生成的被试计划

```json
{
    "subject_id": "S001",
    "batch_id": 0,
    "is_bridge": false,
    "quotas": {
        "core1": 8,
        "main": 20,
        "inter": 10,
        "boundary": 8,
        "lhs": 14
    },
    "constraints": {
        "must_include_design_ids": [0, 10, 20],
        "per_factor_min_counts": {"f1": 5, "f2": 5},
        "interactions": [
            {"pair": [0, 1], "quadrants": 3, "strategy": "balanced"}
        ]
    }
}
```

### 2. subject_trials_{sid}.csv

单被试生成器输出的试验列表

```csv
subject_id,batch_id,block_type,trial_index,f1,f2,f3,design_row_id
S001,0,must_include,0,0.5,0.3,0.7,0
S001,0,core1,1,0.1,0.2,0.3,45
...
```

### 3. subject_results_{sid}.csv

实验执行后的结果（外部系统填充）

```csv
subject_id,trial_index,response,timestamp
S001,0,1,2024-01-01T10:00:00
S001,1,0,2024-01-01T10:00:15
...
```

### 4. study_state.json

全局研究状态（协调器维护）

```json
{
    "study_id": "EXP001",
    "status": "planned|running|completed",
    "global_plan": {
        "n_subjects": 10,
        "n_batches": 3,
        "total_budget": 350
    },
    "global_core1_candidates": [0, 10, 20, 30, 40, 50, 60, 70],
    "interaction_pairs": [[0, 1], [1, 2], [0, 2]],
    "bridge_subjects": [0, 1, 2, 3, 4, 5],
    "completed_subjects": ["S001", "S002"],
    "warnings": []
}
```

## 独立运行

### 测试WarmupSubjectGenerator

```bash
pixi run python extensions/custom_generators/SCOUT\ -ICM\ WarmStart/warmup_subject_generator.py
```

预期输出：

- 自动生成示例design_df (100个候选，4个因子)
- 创建示例subject_plan
- 生成约60条试验
- 显示覆盖率、Gini系数等指标

### 测试StudyCoordinator

```bash
pixi run python extensions/custom_generators/SCOUT\ -ICM\ WarmStart/study_coordinator.py
```

预期输出：

- 自动生成示例design_df (300个候选，5个因子)
- 拟合全局计划
- 为前3个被试分配计划
- 显示全局汇总信息
- 检查Core-1重复策略可行性

## 模块职责对比表

| 维度 | WarmupSubjectGenerator | StudyCoordinator |
|------|----------------------|------------------|
| **范围** | 单个被试 | 整个研究 |
| **输入** | design_df + subject_plan | design_df + 研究参数 |
| **输出** | 试验DataFrame | subject_plan字典 |
| **Core-1** | 接收must_include列表 | 生成全局Core-1候选集 |
| **主效应** | D-optimal局部选择 | 计算全局预算比例 |
| **交互** | 根据约束选择点 | 基于方差选择交互对 |
| **边界** | 使用边界库选择 | 生成边界库 |
| **LHS** | 距离优化采样 | 计算LHS预算 |
| **桥接** | 不处理 | 规划桥接被试 |
| **批次** | 不处理 | 分配批次 |
| **验证** | 配额闭合、去重 | 全局覆盖期望、Core-1策略 |

## 关键设计决策

### 1. 职责分离

- **协调器**: "WHAT"和"WHO" - 决定做什么、谁来做
- **生成器**: "HOW" - 具体如何选择点、如何优化距离

### 2. 接口设计

- `subject_plan`作为两模块间的唯一数据契约
- 协调器不依赖生成器的输出（单向依赖）
- 生成器不知道其他被试的存在

### 3. 配额分配

- 使用最大余数法确保整数预算准确分配
- 支持子预算的递归分配（core2→main+inter, individual→boundary+lhs）

### 4. 随机性控制

- 两模块均支持seed参数
- 协调器使用seed生成全局组件
- 生成器使用seed生成试验序列

## 扩展建议

### 短期扩展

1. 在`WarmupSubjectGenerator`中添加试验顺序优化（如block randomization）
2. 在`StudyCoordinator`中添加自适应预算调整（基于前期被试表现）
3. 支持动态Core-1点集更新

### 长期扩展

1. **实时协调**: 协调器根据已完成被试的结果动态调整后续被试计划
2. **分阶段生成**: 支持被试内的自适应试验生成（Phase-2/3）
3. **结果整合**: 协调器收集所有被试结果并计算全局指标（ICC、批次效应等）
4. **持久化**: 支持中断恢复，保存/加载study_state.json

## 典型工作流

```python
# 步骤1: 全局规划（研究启动前）
coordinator = StudyCoordinator(design_df, n_subjects=10, total_budget=350)
coordinator.fit_initial_plan()

# 保存全局计划
import json
global_plan = coordinator.summarize_global()
with open("study_state.json", "w") as f:
    json.dump(global_plan, f)

# 步骤2: 为每个被试生成计划
for subject_id in range(10):
    subject_plan = coordinator.allocate_subject_plan(subject_id)
    
    # 保存被试计划
    with open(f"subject_plan_{subject_id}.json", "w") as f:
        json.dump(subject_plan, f)

# 步骤3: 实验执行时，为每个被试生成试验
# (可以在不同机器/进程中并行执行)
for subject_id in range(10):
    # 加载计划
    with open(f"subject_plan_{subject_id}.json", "r") as f:
        subject_plan = json.load(f)
    
    # 生成试验
    generator = WarmupSubjectGenerator(design_df, subject_plan)
    trials_df = generator.generate_trials()
    
    # 保存试验列表
    trials_df.to_csv(f"subject_trials_{subject_id}.csv", index=False)
    
    # 执行实验...
    # (AEPsych server 运行试验并收集响应)

# 步骤4: 结果分析（研究完成后）
# 加载所有被试的试验和响应
all_trials = []
for subject_id in range(10):
    trials = pd.read_csv(f"subject_trials_{subject_id}.csv")
    results = pd.read_csv(f"subject_results_{subject_id}.csv")
    merged = trials.merge(results, on=["subject_id", "trial_index"])
    all_trials.append(merged)

combined_data = pd.concat(all_trials, ignore_index=True)

# 计算全局指标
# - ICC (组内相关系数)
# - 批次效应
# - 主效应估计
# - 交互效应筛选
# ...
```

## 注意事项

### 当前版本限制

1. **无联动**: 两模块尚未实现I/O联动，需手动传递subject_plan
2. **静态规划**: 协调器生成的计划是静态的，不根据执行结果调整
3. **简化验证**: 单被试生成器的验证是简化的，不保证全局约束

### 未来联动实现

要实现完整联动，建议添加：

1. `SubjectPlanManager`类管理subject_plan的序列化/反序列化
2. `ResultsAggregator`类收集和分析所有被试结果
3. `AdaptiveCoordinator`类支持自适应调整

## 技术细节

### 预算分配算法

使用最大余数法（Largest Remainder Method）确保：

```
sum(quotas.values()) == per_subject_budget
```

步骤：

1. 按百分比计算浮点预算
2. 向下取整得到基础整数配额
3. 计算每项的余数
4. 按余数大小排序，分配剩余预算

### 点选择策略

**Maximin**: 最大化最小距离

- 用于Core-1和边界点选择
- 确保空间均匀覆盖

**D-optimal**: 最大化设计矩阵的行列式

- 用于主效应覆盖
- 优化参数估计精度

**LHS**: 拉丁超立方采样

- 用于填充点
- 同时优化空间覆盖和距离特性

### 因子类型检测

- 离散因子: ≤10个唯一值
- 连续因子: >10个唯一值
- 连续因子按p10/p50/p90分箱用于覆盖度计算

## 相关文件

- `scout_warmup_generator.py`: 原始单体实现（保留用于参考）
- `warmup_subject_generator.py`: 单被试生成器（新）
- `study_coordinator.py`: 全局协调器（新）
- `test/`: 测试脚本目录
- `README.md`: 原始文档

## 维护指南

### 修改单被试逻辑

编辑`warmup_subject_generator.py`中的点选择方法：

- `_select_core1_local()`
- `_plan_main_effects_local()`
- `_plan_interactions_local()`
- `_build_boundary_local()`
- `_plan_lhs_local()`

### 修改全局规划逻辑

编辑`study_coordinator.py`中的规划方法：

- `_generate_global_core1()`
- `_build_interaction_pairs_heuristic()`
- `_compute_global_budget_split()`
- `_allocate_subject_quotas()`

### 添加新的约束类型

1. 在`study_coordinator.py`的`_generate_subject_constraints()`中添加新约束
2. 在`warmup_subject_generator.py`的`generate_trials()`中处理新约束
3. 更新`subject_plan`的文档说明

## 版本历史

### v1.0 (Current)

- 拆分为两个独立模块
- 实现基本的规划和生成功能
- 支持独立运行和自测
- 暂无I/O联动

### v0.9 (Legacy)

- 单体`WarmupAEPsychGenerator`实现
- 包含所有功能但职责混合

## 贡献者

本模块基于SCOUT-ICM WarmStart项目开发，专为AEPsych Phase-1 warmup设计。

## 许可证

遵循AEPsych项目许可证。
