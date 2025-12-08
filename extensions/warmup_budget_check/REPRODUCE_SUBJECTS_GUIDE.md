# 批量创建同一群体的模拟被试 - 使用指南

> **从已有的Phase 1数据，批量复制出更多同群体的被试**

---

## 🎯 使用场景

你已经完成了Phase 1实验（5个被试），现在需要：
1. **为Phase 2创建20个同群体的被试**
2. **测试被试在完整设计空间上的响应**
3. **保持群体一致性，只有个体差异**

---

## 🚀 快速开始

### **场景1：基础用法 - 创建20个被试**

```bash
cd extensions/warmup_budget_check

python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20
```

**输出：**
- `phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/`
  - `subject_cluster_specs.json` - 所有被试的完整参数
  - `subject_cluster_summary.txt` - 可读性摘要

---

### **场景2：自定义个体差异**

```bash
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20 \
    --individual_std 0.10  # 降低个体差异（推荐：0.10-0.15）
```

---

### **场景3：在设计空间上测试被试响应**

```bash
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20 \
    --test_design_space ../../../data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv
```

**输出：**
- `design_space_responses.csv` - 包含所有被试在设计空间上的响应

---

## 📋 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--base_dir` | **必填** | Phase 1 result目录（包含`fixed_weights_auto.json`） |
| `--n_subjects` | 20 | 要创建的被试数量 |
| `--individual_std` | 0.125 | 个体差异标准差（推荐：0.10-0.15） |
| `--base_seed` | 100 | 基础随机种子（避免与原始数据重复） |
| `--output_dir` | `{base_dir}/reproduced_subjects` | 输出目录 |
| `--test_design_space` | None | 设计空间CSV路径（用于测试） |
| `--likert_levels` | 5 | Likert量表等级数 |
| `--likert_sensitivity` | 2.0 | Likert转换灵敏度 |

---

## 🔧 工作原理

### **1. 群体参数（所有被试共享）**

从 `fixed_weights_auto.json` 中读取：

```json
{
  "global": [[0.124, -0.035, 0.162, 0.381, -0.059, -0.059]],
  "interactions": {"3,4": 0.12, "0,1": -0.02},
  "bias": -0.218
}
```

这些是**群体固定效应**，代表这个群体的整体特征。

### **2. 个体差异（每个被试独特）**

为每个被试生成随机偏差：

```python
individual_deviation = N(0, individual_std)
subject_weights = population_weights + individual_deviation
```

**示例：**
```
Population weights: [0.124, -0.035, 0.162, 0.381, -0.059, -0.059]

Subject 1 deviation: [+0.02, -0.01, +0.03, -0.02, +0.01, +0.02]
→ Subject 1 weights: [0.144, -0.045, 0.192, 0.361, -0.049, -0.039]

Subject 2 deviation: [-0.03, +0.02, -0.01, +0.01, -0.02, +0.03]
→ Subject 2 weights: [0.094, -0.015, 0.152, 0.391, -0.079, -0.029]
```

### **3. 响应生成**

```python
y_continuous = bias + Σ(weights[i] × x[i]) + Σ(interaction_weights × x[i] × x[j])
y_likert = tanh_transform(y_continuous, sensitivity)
```

---

## 📊 控制被试间差异

### **individual_std 的影响**

| `individual_std` | 被试间SD | 特征 |
|-----------------|---------|------|
| 0.05 | ~0.3-0.4 | 非常相似 |
| 0.10 | ~0.5-0.6 | 较相似 ✅ **推荐** |
| 0.125 | ~0.6-0.7 | 适中差异 |
| 0.15 | ~0.7-0.8 | 较大差异 |
| 0.20 | ~0.9-1.1 | 很大差异 |

**建议：**
- Phase 2实验：`individual_std = 0.10-0.12` （保持一致性）
- 探索性研究：`individual_std = 0.15` （允许更多差异）

---

## 💡 使用示例

### **完整流程：创建Phase 2被试并测试**

```bash
# 1. 创建20个Phase 2被试
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20 \
    --individual_std 0.10 \
    --output_dir phase2_subjects

# 2. 在设计空间上测试
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20 \
    --individual_std 0.10 \
    --output_dir phase2_subjects \
    --test_design_space data/only_independences/.../design.csv
```

**结果：**
- `phase2_subjects/subject_cluster_specs.json` - 被试参数
- `phase2_subjects/design_space_responses.csv` - 响应数据
- `phase2_subjects/subject_cluster_summary.txt` - 摘要

---

## 🔍 验证被试质量

### **检查被试间差异**

```bash
python -c "
import json
import numpy as np

with open('phase2_subjects/subject_cluster_specs.json') as f:
    data = json.load(f)

# 提取每个被试的个体偏差
deviations = [np.array(s['individual_deviation']) for s in data['subjects']]

# 计算偏差的标准差
dev_stds = [np.std(d) for d in deviations]

print(f'Average individual deviation std: {np.mean(dev_stds):.3f}')
print(f'Range: {min(dev_stds):.3f} - {max(dev_stds):.3f}')
"
```

### **查看设计空间响应**

```bash
python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('phase2_subjects/design_space_responses.csv')

# 提取所有被试的响应列
response_cols = [col for col in df.columns if col.startswith('y_subject_')]

# 计算每个被试的平均响应
subject_means = [df[col].mean() for col in response_cols]

print(f'Between-subject SD: {np.std(subject_means, ddof=1):.3f}')
print(f'Mean range: {min(subject_means):.2f} - {max(subject_means):.2f}')
"
```

---

## 🎨 高级用法

### **1. 创建多个群体**

```bash
# 群体A：低个体差异
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 10 \
    --individual_std 0.08 \
    --base_seed 100 \
    --output_dir cluster_A

# 群体B：高个体差异
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 10 \
    --individual_std 0.15 \
    --base_seed 200 \
    --output_dir cluster_B
```

### **2. 使用不同的Likert灵敏度**

```bash
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20 \
    --likert_sensitivity 1.5  # 允许更多极端响应
```

---

## 📦 输出文件说明

### **subject_cluster_specs.json**

```json
{
  "population_params": {
    "weights": [0.124, -0.035, ...],
    "bias": -0.218,
    "interactions": {"3,4": 0.12}
  },
  "individual_std": 0.125,
  "base_seed": 100,
  "n_subjects": 20,
  "subjects": [
    {
      "subject_id": 1,
      "seed": 100,
      "population_weights": [...],
      "individual_deviation": [0.02, -0.01, ...],
      "subject_weights": [0.144, -0.045, ...],
      ...
    },
    ...
  ]
}
```

**用途：**
- 重现被试（使用seed）
- 分析个体差异模式
- 导出到其他实验平台

### **subject_cluster_summary.txt**

人类可读的摘要，包含：
- 群体参数
- 各被试的偏差向量
- 快速诊断信息

### **design_space_responses.csv**

包含：
- 原始设计空间的所有特征列
- `y_subject_1`, `y_subject_2`, ... - 各被试的响应

**用途：**
- 可视化被试响应分布
- 计算被试间差异
- 验证模型质量

---

## ⚠️ 注意事项

1. **seed的选择**
   - 使用不同的 `base_seed` 避免与原始数据重复
   - 推荐范围：100-1000

2. **individual_std的设置**
   - 太小（<0.05）：被试几乎相同
   - 太大（>0.20）：可能偏离群体特征
   - **推荐：0.10-0.15**

3. **验证步骤**
   - 创建后务必检查被试间SD
   - 与Phase 1的被试间差异对比
   - 确保没有极端被试（全1分或全5分）

---

## 🔗 相关文档

- **Phase 1分析报告**: `phase1_analysis_output/{timestamp}/step2/phase1_analysis_report.md`
- **ALL_CONFIG配置指南**: `ALL_CONFIG_GUIDE.md`
- **批次效应解释**: Phase 1报告中的"数据质量指标"部分

---

## 🆘 故障排查

### **问题1：找不到fixed_weights_auto.json**

**解决：**
```bash
# 确认文件存在
ls phase1_analysis_output/202512011547/step1_5/result/

# 如果不存在，重新运行Step1.5
python quick_start.py  # MODE="all"
```

### **问题2：被试间差异太大**

**解决：**
```bash
# 降低individual_std
python reproduce_subject_cluster.py \
    --base_dir ... \
    --individual_std 0.08  # 从0.125降到0.08
```

### **问题3：ImportError**

**解决：**
```bash
# 确认tools路径
ls tools/subject_simulator_v2/linear.py

# 或使用绝对路径
export PYTHONPATH="/path/to/aepsych-source:$PYTHONPATH"
```

---

**最后更新**: 2025-12-01
**版本**: 1.0
**作者**: Claude Code
