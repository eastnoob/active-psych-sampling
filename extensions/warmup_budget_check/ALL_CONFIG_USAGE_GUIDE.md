# ALL_CONFIG 使用指南

## 🎯 快速上手

在 `quick_start.py` 中，只需修改 `ALL_CONFIG` 区域的参数，即可完成全流程配置！

### 最简使用流程

```python
# 1. 打开 quick_start.py
# 2. 找到 ALL_CONFIG（约第 83 行）
# 3. 修改关键参数（见下方）
# 4. 运行：python quick_start.py
```

---

## 📋 关键参数说明

### **全局配置**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_output_dir` | `phase1_analysis_output` | 所有结果的根目录 |
| `run_step1_5` | `True` | 是否运行模拟应答（推荐True） |

### **Step 1: 预热采样**

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `design_csv` | `...csv` | - | 设计空间文件路径 |
| `n_subjects` | `5` | 5-10 | Phase 1 被试数量 |
| `trials_per_subject` | `30` | 20-40 | 每个被试的测试次数 |
| `skip_interaction` | `False` | False | 是否探索交互（False=探索） |

**预算建议**：
- 轻量测试：5人 × 20次 = 100 trials
- 标准配置：5人 × 30次 = 150 trials
- 充分探索：10人 × 30次 = 300 trials

---

### **Step 1.5: 模拟应答**

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `simulation_seed` | `42` | - | 随机种子（保证可复现） |
| `output_type` | `"likert"` | likert/continuous | 输出类型 |
| `likert_levels` | `5` | 5/7 | Likert量表级别 |
| `population_std` | `0.4` | 0.3-0.5 | 群体权重标准差 |
| `individual_std_percent` | `0.3` | 0.2-0.5 | 个体差异比例 |
| `interaction_pairs` | `[(3,4), (0,1)]` | - | 预设的交互对 |
| `interaction_scale` | `0.25` | 0.2-0.4 | 交互效应强度 |

**模拟真实性调整**：
- `population_std` 越大 → 被试间差异越大
- `individual_std_percent` 越大 → 同一被试内部噪声越大
- `interaction_scale` 越大 → 交互效应越明显

---

### **Step 2: Phase 1 数据分析**

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `max_pairs` | `5` | 3-5 | 最多选择的交互对数 |
| `min_pairs` | `2` | 1-3 | 最少选择的交互对数 |
| `selection_method` | `"elbow"` | elbow/bic | 交互对选择方法 |
| `phase2_n_subjects` | `20` | 15-25 | Phase 2 被试数 |
| `phase2_trials_per_subject` | `25` | 20-30 | Phase 2 每人trials数 |
| `lambda_adjustment` | `1.2` | 1.0-1.5 | λ调整系数 |

**Phase 2 预算建议**：
- 总预算 = `phase2_n_subjects` × `phase2_trials_per_subject`
- 推荐：20人 × 25次 = 500 trials

---

### **Step 3: Base GP 训练**

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `max_iters` | `200` | 100-300 | GP训练最大迭代次数 |
| `learning_rate` | `0.05` | 0.01-0.1 | 学习率 |
| `use_cuda` | `False` | - | 是否使用GPU（需要CUDA） |

---

## 📝 常见使用场景

### 场景1：快速测试（小预算）

```python
ALL_CONFIG = {
    # ...
    "n_subjects": 3,
    "trials_per_subject": 20,
    "phase2_n_subjects": 10,
    "phase2_trials_per_subject": 20,
    # ...
}
```

### 场景2：标准实验（推荐配置）

```python
ALL_CONFIG = {
    # ...
    "n_subjects": 5,
    "trials_per_subject": 30,
    "phase2_n_subjects": 20,
    "phase2_trials_per_subject": 25,
    # ...
}
```

### 场景3：充分探索（高预算）

```python
ALL_CONFIG = {
    # ...
    "n_subjects": 10,
    "trials_per_subject": 40,
    "phase2_n_subjects": 30,
    "phase2_trials_per_subject": 30,
    # ...
}
```

### 场景4：调整交互强度

```python
ALL_CONFIG = {
    # ...
    "interaction_scale": 0.4,  # 增强交互效应（更明显）
    "max_pairs": 3,            # 减少交互对数量
    # ...
}
```

---

## 🔧 高级用法

### 使用真实数据（跳过模拟）

如果已有真实被试数据，可以跳过Step 1.5：

```python
ALL_CONFIG = {
    # ...
    "run_step1_5": False,  # 不运行模拟
    "step2_data_csv": r"F:\path\to\real_data.csv",  # 指定真实数据路径
    # ...
}
```

### 为不同步骤使用不同设计空间

```python
# 在 ALL_CONFIG 定义后添加：
ALL_CONFIG['step3_design_space_csv'] = r'F:\path\to\another_design_space.csv'
```

---

## 📊 输出结果

运行完成后，所有结果保存在：

```
phase1_analysis_output/{timestamp}/
├── step1/               # 预热采样方案
├── step1_5/             # 模拟应答数据
│   └── result/
├── step2/               # Phase 1 分析结果
│   ├── phase1_phase2_config.json    ← Phase 2 配置
│   └── phase1_analysis_report.md    ← 分析报告
├── step3/               # Base GP 模型
│   ├── base_gp_state.pth
│   └── base_gp_report.md
└── ALL_MODE_SUMMARY.md  ← 总结报告（从这里开始看！）
```

---

## ❓ 常见问题

### Q1: 如何调整Phase 1预算？

修改 `n_subjects` 和 `trials_per_subject`：
- 总预算 = `n_subjects` × `trials_per_subject`
- 推荐：5人 × 30次 = 150 trials

### Q2: 如何让交互效应更明显？

增大 `interaction_scale`（从0.25提高到0.4）

### Q3: 如何使用自己的设计空间文件？

修改 `design_csv` 路径：
```python
ALL_CONFIG['design_csv'] = r'F:\my_project\design_space.csv'
```

### Q4: 所有STEP配置还需要改吗？

**不需要！** 只改 `ALL_CONFIG` 即可，系统会自动应用到各个步骤。

---

## 🚀 完整示例

```python
# 打开 quick_start.py，找到 ALL_CONFIG 区域

ALL_CONFIG = {
    # 全局
    "base_output_dir": str(Path(__file__).parent / "my_experiment_results"),
    "run_step1_5": True,

    # Step 1: 我想要 8 个被试，每人做 25 次
    "n_subjects": 8,
    "trials_per_subject": 25,
    "skip_interaction": False,

    # Step 1.5: 让模拟更真实一点
    "population_std": 0.5,      # 增大群体差异
    "interaction_scale": 0.3,    # 中等交互强度

    # Step 2: Phase 2 要 25 个被试
    "phase2_n_subjects": 25,
    "phase2_trials_per_subject": 25,
    "max_pairs": 4,

    # Step 3: GPU加速（如果有CUDA）
    "use_cuda": True,
    "max_iters": 300,
}

# 然后运行：python quick_start.py
```

---

**就这么简单！享受全流程自动化吧！** 🎉
