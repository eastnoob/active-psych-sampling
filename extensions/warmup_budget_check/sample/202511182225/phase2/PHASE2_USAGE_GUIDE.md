# Phase 2 EUR-ANOVA 使用指南

> 本指南说明如何在自适应采样中正确使用Phase 1的分析结果

## 🚀 快速开始

Phase 2使用EUR-ANOVA进行自适应采样。主要思想是：
- **根据Phase 1的发现**，智能地选择下一个采样点
- **平衡探索和精化**，最大化信息获取
- **动态调整参数**，适应实验进展

## 📁 生成的文件说明

| 文件 | 格式 | 用途 |
|------|------|------|
| `phase1_phase2_config.json` | JSON | 被程序读取的配置（λ、γ初始值等） |
| `phase1_phase2_schedules.npz` | NumPy | λ和γ的动态衰减表（每个trial一行） |
| `phase1_analysis_report.md` | Markdown | 分析结果总结（给人看的） |
| `PHASE2_USAGE_GUIDE.md` | Markdown | 本指南 |

## 1️⃣ 第1步：加载配置

**为什么要这样做？**
- Phase 1分析生成的参数需要被EUR-ANOVA采样器读取
- JSON文件保存了交互对列表、λ和γ的初始值
- NPZ文件保存了整个Phase 2期间的参数衰减表

**代码实现：**

```python
import numpy as np
import json

# 读取Phase 1的分析结果
with open('phase1_phase2_config.json') as f:
    config = json.load(f)

# 交互对列表
interaction_pairs = [(1, 3), (3, 4), (2, 3), (1, 4), (4, 5)]
print(f"要探索的交互对: {interaction_pairs}")

# 加载λ和γ的动态衰减表
schedules = np.load('phase1_phase2_schedules.npz')
lambda_schedule = schedules['lambda_schedule']  # 500行2列：(trial_idx, lambda_value)
gamma_schedule = schedules['gamma_schedule']    # 500行2列：(trial_idx, gamma_value)
```

## 2️⃣ 第2步：初始化EUR-ANOVA采集函数

**为什么要这样做？**
- EUR-ANOVA是一种主动学习算法，能根据数据自动选择最有价值的采样点
- 通过交互对信息（从Phase 1），它能优先探索有交互效应的因子组合
- λ参数告诉它"多重视这些交互"，γ参数告诉它"探索多大范围"

**代码实现：**

```python
from eur_anova_pair import EURAnovaPairAcqf

# 初始化采集函数
# 注意：这假设你已经有一个GP模型
acqf = EURAnovaPairAcqf(
    model=your_gp_model,          # 你训练的高斯过程
    lambda_init=0.240,  # 初始λ（交互权重）
    gamma_init=0.300,    # 初始γ（探索程度）
    interaction_pairs=[(1, 3), (3, 4), (2, 3), (1, 4), (4, 5)],  # 要探索的交互
    n_trials=500,  # 总共500个trial
)
```

## 3️⃣ 第3步：主采样循环

**为什么要这样做？**
- λ和γ不是固定不变的，而是根据进度逐步衰减的
- 前期：λ高 → 积极探索交互；γ高 → 广泛探索设计空间
- 后期：λ低 → 专注主效应；γ低 → 集中在高价值区域
- 这样能充分利用500个试验的预算

**代码实现：**

```python
total_budget = 500

for trial in range(total_budget):
    # 【关键】从衰减表查询当前trial的λ和γ
    current_lambda = lambda_schedule[trial, 1]  # 第trial行，第1列（值）
    current_gamma = gamma_schedule[trial, 1]    # 第trial行，第1列（值）
    
    # 【重要】更新采集函数的参数
    # 这样EUR-ANOVA才知道当前应该有多重视交互
    acqf.set_lambda(current_lambda)
    acqf.set_gamma(current_gamma)
    
    # 【核心】用EUR-ANOVA选择下一个最有价值的采样点
    x_candidates = # ... 从设计空间生成候选点
    scores = acqf(x_candidates)  # 评分每个候选点
    x_next = x_candidates[np.argmax(scores)]  # 选分数最高的
    
    # 执行实验
    y_next = conduct_experiment(x_next)
    
    # 更新GP模型
    your_gp_model.update(x_next, y_next)
    
    # 可选：在中期进行诊断
    if trial == 335:
        print("🔍 中期诊断时刻！检查是否需要调整策略...")
```

## 4️⃣ 第4步：中期诊断（可选但推荐）

**在第 335 次trial进行诊断，检查：**

✅ **主效应**
- 主效应的估计是否与Phase 1一致？
- 是否有因子的效应变化很大（可能有非线性）？

✅ **交互效应**
- 筛选出的交互对是否确实有预期的效应？
- 有没有其他意外的强交互出现？

✅ **参数调整**
- λ和γ的衰减速度是否合适？
- 需不需要手动调整后续的参数？

**如何调整（如果需要）：**
```python
# 如果发现要加强交互探索
acqf.set_lambda(0.5)  # 手动提高λ

# 如果发现应该更聚焦探索
acqf.set_gamma(0.1)   # 手动降低γ
```

## 📚 关键参数详解

### λ（Lambda）：交互权重

| 含义 | λ值 | 采样行为 |
|------|--------|----------|
| 只关注主效应 | 0.0 | 均匀探索所有点，忽略交互信息 |
| 你的Phase 2初始值 | 0.240 | **平衡模式**：既探索交互也精化主效应 |
| 平衡权重 | 0.5 | 交互和主效应同等重要 |
| 完全关注交互 | 1.0 | 集中探索交互对，忽视主效应 |

**实例：**
- 如果λ=0.36，EUR-ANOVA会36%的力气探索选定的交互对，64%探索其他
- Phase 2后期λ衰减到0.2，意味着逐步转向主效应精化

### γ（Gamma）：覆盖权重

| 含义 | γ值 | 采样行为 |
|------|--------|----------|
| 完全精化 | 0.0 | 聚焦在已知最优点附近，不探索新区域 |
| 你的Phase 2终点值 | 0.060 | 精化阶段：主要精化已发现的好点 |
| 你的Phase 2初始值 | 0.300 | 探索阶段：广泛探索设计空间 |
| 完全探索 | 1.0 | 随机探索所有点，不利用已有信息 |

**实例：**
- 如果γ=0.3，采样器会在"已知好的点"和"新颖点"之间平衡
- Phase 2后期γ衰减到0.06，意味着逐步聚焦到最有希望的区域

## 🔧 高级用法（可选）

**动态调整λ**（如果发现某些交互特别重要）
```python
# 手动提高特定交互对的权重
acqf.increase_interaction_weight((3, 4), factor=2.0)
```

**查看采样历史**（理解EUR-ANOVA的决策）
```python
# 查看每个trial选择的点
import pandas as pd
sampling_history = pd.DataFrame({
    'trial': range(1, total_budget+1),
    'lambda': lambda_schedule[:, 1],
    'gamma': gamma_schedule[:, 1],
    'x_selected': x_history,  # 你保存的采样点
    'y_observed': y_history   # 对应的响应
})
sampling_history.to_csv('phase2_sampling_log.csv', index=False)
```

## ❓ 常见问题

**Q: 为什么要衰减λ和γ？**
A: 早期需要探索新区域，后期需要精化已发现的好点。固定参数会浪费预算。

**Q: 可以不用动态衰减表吗？**
A: 可以，但效率会降低。衰减表是Phase 1分析优化的结果，能最大化信息利用。

**Q: 中期诊断发现了新问题怎么办？**
A: 可以手动调整λ、γ或交互对列表，但要记录变更以便后续分析。

**Q: EUR-ANOVA不收敛怎么办？**
A: 检查GP模型是否训练充分，或尝试调整λ和γ的衰减速度。

---

*本指南由Phase 1数据分析系统自动生成，最后更新于Phase 2开始前*
