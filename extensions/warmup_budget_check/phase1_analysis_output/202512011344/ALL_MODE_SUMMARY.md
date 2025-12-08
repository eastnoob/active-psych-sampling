# All 模式执行总结

**执行时间**: 2025-12-01 13:44:22

---

## 输出目录结构

所有结果保存在: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344`

```
202512011344/
├── step1/           # Step 1: 预热采样方案
│   ├── subject_1.csv
│   ├── subject_2.csv
│   └── ...
├── step1_5/         # Step 1.5: 模拟被试作答
│   ├── subject_*.csv
│   └── result/      # 带响应的数据
│       ├── subject_*.csv
│       ├── combined_results.csv
│       └── MODEL_SUMMARY.*
├── step2/           # Step 2: Phase 1 数据分析
│   ├── phase1_analysis_report.md
│   ├── phase1_phase2_config.json
│   ├── phase1_phase2_schedules.npz
│   └── phase1_usage_guide.md
├── step3/           # Step 3: Base GP 训练
│   ├── base_gp_state.pth
│   ├── base_gp_key_points.json
│   ├── base_gp_lengthscales.json
│   ├── design_space_scan.csv
│   └── base_gp_report.md
└── ALL_MODE_SUMMARY.md  # 本总结文件
```

## 各阶段成果

### Step 1: 预热采样方案

- 生成了 **5** 个被试的采样方案
- 位置: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step1`

### Step 1.5: 模拟被试作答

- 模拟了 **5** 个被试的响应数据
- 位置: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step1_5\result`

### Step 2: Phase 1 数据分析

- 筛选了 **5** 个交互对
- Phase 2 总预算: **500** 次
- λ: 0.100 → 0.287
- γ: 0.300 → 0.060
- 详细报告: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step2\phase1_analysis_report.md`

### Step 3: Base GP 训练

- 训练完成 Base GP 模型
- Best Prior 预测: 1.529
- Worst Prior 预测: -1.759
- 模型文件: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step3\base_gp_state.pth`
- 详细报告: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step3\base_gp_report.md`

## 下一步操作建议

### 1. 查看各阶段详细报告

- **Step 2 分析报告**: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step2\phase1_analysis_report.md`
- **Step 3 GP 报告**: `F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step3\base_gp_report.md`

### 2. 使用 Phase 2 配置

```python
import json
config_path = r'F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step2\phase1_phase2_config.json'
with open(config_path) as f:
    phase2_config = json.load(f)
```

### 3. 加载 Base GP 模型

```python
import torch
model_path = r'F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202512011344\step3\base_gp_state.pth'
state_dict = torch.load(model_path)
# 在 Phase 2 模型中作为先验使用
```

---

*自动生成于 All 模式执行流程*
