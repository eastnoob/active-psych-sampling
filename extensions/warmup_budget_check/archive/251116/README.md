# 两阶段实验规划工具

用于EUR-ANOVA主动学习的预热采样和Phase 2参数生成工具。

## 快速开始

### 方法1：使用快速启动脚本（推荐）

```bash
# 1. 编辑 quick_start.py 中的配置参数
# 2. 选择运行模式（step1/step2/both）
# 3. 运行
python quick_start.py
```

### 方法2：分步运行

#### Step 1: 生成预热采样方案

```bash
python warmup_sampler.py
```

**输入**：
- 设计空间CSV（只包含自变量）
- 被试数和每人trials数

**输出**：
- `sample/subject_N.csv` - 采样方案文件

#### Step 2: 分析数据并生成Phase 2参数

```bash
# 执行实验，收集响应数据后...
python analyze_phase1.py
```

**输入**：
- 实验数据CSV（包含响应列）

**输出**：
- `phase1_analysis_output/*.json` - Phase 2配置
- `phase1_analysis_output/*.npz` - λ/γ动态调度
- `phase1_analysis_output/*.txt` - 分析报告

## 核心文件

```
warmup_budget_check/
├── quick_start.py              # 快速启动脚本（推荐）
├── warmup_sampler.py           # 步骤1：生成采样方案
├── analyze_phase1.py           # 步骤2：分析数据
├── warmup_budget_estimator.py  # 预算评估工具（依赖）
├── phase1_analyzer.py          # 数据分析工具（依赖）
├── README.md                   # 本文档
└── archive/                    # 历史版本和测试文件
```

## 使用流程

```
设计空间CSV (只有自变量)
    ↓
[Step 1: warmup_sampler.py]
    ↓
subject_N.csv (采样方案)
    ↓
【用户执行实验】
    ↓
实验数据CSV (添加响应列)
    ↓
[Step 2: analyze_phase1.py]
    ↓
Phase 2配置 + λ/γ调度
    ↓
【Phase 2: EUR-ANOVA】
```

## 参数说明

### quick_start.py 配置

```python
# 选择运行模式
MODE = "step1"  # "step1", "step2", 或 "both"

# Step 1 配置
STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",  # 设计空间CSV
    "n_subjects": 14,                       # 被试数
    "trials_per_subject": 25,               # 每人trials数
    "skip_interaction": False,              # 是否跳过交互探索
    "output_dir": "sample",                 # 输出目录
    "merge": False,                         # 是否合并为单个CSV
}

# Step 2 配置
STEP2_CONFIG = {
    "data_csv_path": "warmup_data.csv",     # 实验数据CSV
    "subject_col": "subject_id",            # 被试列名
    "response_col": "response",             # 响应列名
    "max_pairs": 5,                         # 最多交互对数
    "min_pairs": 3,                         # 最少交互对数
    "selection_method": "elbow",            # 选择方法
    "phase2_n_subjects": 18,                # Phase 2被试数
    "phase2_trials_per_subject": 25,        # Phase 2每人trials
    "lambda_adjustment": 1.2,               # λ调整系数
}
```

### 关键参数

- **selection_method**: 交互对选择方法
  - `elbow` - 肘部法则（推荐）
  - `bic_threshold` - BIC阈值法
  - `top_k` - 固定top-K

- **lambda_adjustment**: λ调整系数
  - `1.0` - 不调整
  - `1.2` - 增强20%交互探索（默认）
  - `1.5` - 大幅增强

- **skip_interaction**: 是否跳过交互探索
  - `False` - 包含交互（推荐）
  - `True` - 仅主效应

## Phase 2集成

```python
import json
import numpy as np
from eur_anova_pair import EURAnovaPairAcqf

# 加载配置
with open('phase1_analysis_output/phase1_phase2_config.json') as f:
    config = json.load(f)

data = np.load('phase1_analysis_output/phase1_phase2_schedules.npz')
lambda_schedule = data['lambda_schedule']
gamma_schedule = data['gamma_schedule']

# 初始化EUR-ANOVA
acqf = EURAnovaPairAcqf(
    model=your_gp_model,
    gamma=config['gamma_init'],
    interaction_pairs=config['interaction_pairs'],
    tau_n_max=config['total_budget'],
    gamma_min=config['gamma_end']
)

# 主动学习循环
for trial in range(1, config['total_budget'] + 1):
    lambda_t = lambda_schedule[trial - 1, 1]
    gamma_t = gamma_schedule[trial - 1, 1]

    scores = acqf(X_candidates)
    X_next = X_candidates[scores.argmax()]
    y_next = conduct_experiment(X_next)
    your_gp_model.update(X_next, y_next)
```

## 常见问题

**Q: 预算评估显示"不足"怎么办？**
- 增加被试数或每人trials数
- 如果只关注主效应，设置 `skip_interaction=True`

**Q: 如何选择合并或分文件模式？**
- 分文件：每个被试独立一个CSV，便于按被试分配任务
- 合并：所有样本在一个CSV，方便批量处理

**Q: Phase 1必须选出3-5个交互对吗？**
- 不一定。使用 `elbow` 方法时会自动确定最优数量

**Q: 如何调整λ/γ衰减策略？**
- 修改 `analyze_phase1.py` 中的 `_compute_lambda_schedule()` 和 `_compute_gamma_schedule()` 方法

## 详细文档

- [README_STANDALONE.md](README_STANDALONE.md) - 完整使用指南
- `archive/` - 历史版本和测试文件

## 许可

MIT License
