# Warmup Budget Check - 两阶段实验规划工具

用于EUR-ANOVA主动学习的预热采样、模拟应答和Phase 2参数生成工具。

## 快速开始

### 使用快速启动脚本（推荐）

```bash
# 1. 编辑 quick_start.py 中的配置参数
# 2. 选择运行模式（step1/step2/step3/all）
# 3. 运行
python quick_start.py
```

### 三步工作流

```
Step 1: 生成采样方案
    ↓
Step 2: 模拟被试作答（可选，用于测试）
    ↓
Step 3: 分析数据并生成Phase 2配置
```

## 目录结构

```
warmup_budget_check/
├── quick_start.py              # 快速启动脚本（主入口）
├── core/                       # 核心功能模块
│   ├── warmup_sampler.py       # Step 1: 生成采样方案
│   ├── simulation_runner.py    # Step 2: 模拟被试作答
│   ├── single_output_subject.py # 被试模拟类
│   ├── analyze_phase1.py       # Step 3: 分析数据
│   ├── phase1_analyzer.py      # 数据分析工具
│   ├── warmup_budget_estimator.py # 预算评估
│   ├── warmup_api.py           # API接口
│   └── config_models.py        # 配置模型
├── docs/                       # 文档
│   ├── README.md               # 完整使用指南
│   ├── README_API.md           # API文档
│   └── ...                     # 其他文档
├── tests/                      # 测试文件
├── examples/                   # 示例代码
├── sample/                     # 采样输出目录
└── archive/                    # 历史版本
```

## 使用步骤

### Step 1: 生成采样方案

输入：设计空间CSV（只包含自变量）

```python
# 编辑 quick_start.py
MODE = "step1"

STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",
    "n_subjects": 5,
    "trials_per_subject": 25,
    "skip_interaction": False,
    "output_dir": "sample",
}
```

输出：`sample/subject_N.csv` - 采样方案文件

### Step 2: 模拟被试作答（可选）

用于测试流程，无需真实被试实验。

```python
MODE = "step1.5"

STEP1_5_CONFIG = {
    "input_dir": "sample/202511172026",  # Step 1输出目录
    "output_type": "likert",
    "likert_levels": 5,
    "interaction_pairs": [(0,1), (3,4)],
    "population_std": 0.4,
    # 模型显示与保存 ⭐新增
    "print_model": True,           # 控制台打印模型规格
    "save_model_summary": True,    # 保存模型摘要文件
    "model_summary_format": "txt", # txt/md/both
}
```

输出：
- `sample/*/result/combined_results.csv` - 带响应的数据
- `sample/*/result/MODEL_SUMMARY.txt` - 模型规格摘要（可选）

### Step 3: 分析数据生成Phase 2配置

输入：实验数据CSV（包含响应列）

```python
MODE = "step3"

STEP3_CONFIG = {
    "data_csv_path": "sample/*/result/combined_results.csv",
    "subject_col": "subject",
    "response_col": "y",
    "max_pairs": 5,
    "selection_method": "elbow",
}
```

输出：
- `phase1_analysis_output/*.json` - Phase 2配置
- `phase1_analysis_output/*.npz` - λ/γ动态调度

### 一键运行全流程

```python
MODE = "all"  # 运行Step 1 → Step 2 → Step 3
```

## 关键参数

### Step 1 配置
- **n_subjects**: 被试数量
- **trials_per_subject**: 每人trials数
- **skip_interaction**: 是否跳过交互探索

### Step 2 配置（模拟应答）
- **output_type**: 输出类型 (continuous/likert)
- **interaction_pairs**: 交互对 [(i,j), ...]
- **population_std**: 群体权重标准差
- **individual_std_percent**: 个体差异比例

### Step 3 配置
- **selection_method**: 交互对选择方法
  - `elbow` - 肘部法则（推荐）
  - `bic_threshold` - BIC阈值法
  - `top_k` - 固定top-K

## Phase 2集成示例

```python
import json
import numpy as np
from core.warmup_api import WarmupAPI

# 加载配置
api = WarmupAPI()
config = api.load_phase2_config("phase1_analysis_output/phase1_phase2_config.json")

# 初始化EUR-ANOVA
acqf = EURAnovaMultiAcqf(
    model=your_gp_model,
    interaction_pairs=config['interaction_pairs'],
    gamma=config['gamma_init'],
    tau_n_max=config['total_budget'],
)

# 主动学习循环
for trial in range(1, config['total_budget'] + 1):
    scores = acqf(X_candidates)
    X_next = X_candidates[scores.argmax()]
    y_next = conduct_experiment(X_next)
    your_gp_model.update(X_next, y_next)
```

## 详细文档

- [完整使用指南](docs/README.md)
- [API文档](docs/README_API.md)
- [增强功能说明](docs/ENHANCEMENT_SUMMARY.md)

## 许可

MIT License
