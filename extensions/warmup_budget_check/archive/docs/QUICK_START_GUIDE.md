# 快速开始指南

## 最简单的使用方式

### 1. 编辑 `quick_start.py` 配置参数

```python
# 选择运行模式
MODE = "step1"  # 第一次运行选择 "step1"

# Step 1 配置
STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",  # 你的设计空间CSV文件
    "n_subjects": 14,                       # 被试数量
    "trials_per_subject": 25,               # 每人测试次数
    "skip_interaction": False,              # 是否跳过交互探索
    "output_dir": "sample",                 # 输出目录
    "merge": False,                         # False=每个被试一个文件
}
```

### 2. 运行步骤1：生成采样方案

```bash
python quick_start.py
```

会生成：
```
sample/
├── subject_1.csv
├── subject_2.csv
├── ...
└── subject_14.csv
```

### 3. 执行实验，收集数据

按照生成的CSV文件执行实验，将响应值添加到数据中。

### 4. 编辑配置，运行步骤2

修改 `quick_start.py`：

```python
MODE = "step2"  # 改为 "step2"

STEP2_CONFIG = {
    "data_csv_path": "warmup_data.csv",     # 实验数据CSV
    "subject_col": "subject_id",
    "response_col": "response",
    "phase2_n_subjects": 18,
    "phase2_trials_per_subject": 25,
}
```

### 5. 运行步骤2：分析数据

```bash
python quick_start.py
```

会生成：
```
phase1_analysis_output/
├── phase1_phase2_config.json       # Phase 2配置
├── phase1_phase2_schedules.npz     # λ/γ动态调度
├── phase1_analysis_report.txt      # 分析报告
└── PHASE2_USAGE_GUIDE.txt          # 使用指南
```

## 在Phase 2中使用

```python
import json
import numpy as np

# 加载配置
with open('phase1_analysis_output/phase1_phase2_config.json') as f:
    config = json.load(f)

data = np.load('phase1_analysis_output/phase1_phase2_schedules.npz')

# 使用参数
interaction_pairs = config['interaction_pairs']  # 例如: [[0, 2], [1, 5]]
lambda_init = config['lambda_init']              # 例如: 0.883
gamma_init = config['gamma_init']                # 例如: 0.300

# 在EUR-ANOVA中使用这些参数...
```

## 常见配置示例

### 示例1：小规模实验（每人20次）

```python
STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",
    "n_subjects": 10,
    "trials_per_subject": 20,
    "skip_interaction": False,
    "merge": False,
}

STEP2_CONFIG = {
    "phase2_n_subjects": 15,
    "phase2_trials_per_subject": 20,
}
```

### 示例2：只关注主效应（跳过交互）

```python
STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",
    "n_subjects": 12,
    "trials_per_subject": 22,
    "skip_interaction": True,  # 跳过交互探索
    "merge": False,
}
```

### 示例3：合并为单个CSV

```python
STEP1_CONFIG = {
    "design_csv_path": "design_space.csv",
    "n_subjects": 14,
    "trials_per_subject": 25,
    "skip_interaction": False,
    "merge": True,  # 合并为单个CSV
    "subject_col_name": "subject_id",
}
```

## 预算评估标准

运行步骤1时，系统会自动评估预算充足性：

- **刚好** - 预算满足需求，推荐配置
- **充足** - 预算充裕，有额外探索空间
- **过度充足** - 被试或预算过多，可以优化
- **勉强** - 预算略有不足，但可以接受
- **不足** - 预算不足，建议增加被试或trials
- **严重不足** - 预算严重不足，无法满足基本需求

## 需要帮助？

查看详细文档：
- [README.md](README.md) - 完整功能说明
- [README_STANDALONE.md](README_STANDALONE.md) - 详细使用指南
- `phase1_analysis_output/PHASE2_USAGE_GUIDE.txt` - Phase 2集成指南（运行步骤2后生成）
