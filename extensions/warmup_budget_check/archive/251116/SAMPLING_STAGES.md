# 五步预热采样法 (Five-Stage Warmup Sampling)

## 概览 (Overview)

| 阶段 | 共享性 | 目标 | 预算占比 |
|------|--------|------|----------|
| Core-1 | 全被试共享 | ICC估计 + 个体差异建模 | ~32% |
| Core-2a | 分散分配 | 主效应覆盖 | ~27% |
| Core-2b | 分散分配 | 交互效应初筛 | ~19% |
| Boundary | 分散分配 | 边界探索 | ~9% |
| LHS | 分散分配 | 空间填充 | ~13% |

---

## Stage 1: Core-1 固定重复点 (Fixed Repeated Points)

### 目标 (Objective)
- 估计组内相关系数 (ICC, Intraclass Correlation Coefficient)
- 建立个体差异基线 (Individual difference baseline)
- 提供可靠性参考点 (Reliability reference)

### 行为 (Behavior)
- **采样方式**: 从设计空间随机选择8个配置
- **分配策略**: 所有被试测试相同的8个配置
- **重复性**: 完全共享 (Fully shared across subjects)

### 采样特征 (Sampling Characteristics)
```
n_configs = 8
n_samples = 8 × n_subjects

示例 (14被试):
  Core-1样本数 = 8 × 14 = 112次
  每个配置测量 = 14次 (跨被试)
```

### 统计意义 (Statistical Significance)
- **ICC公式**: `ICC = σ²_between / (σ²_between + σ²_within)`
- **用途**: 混合效应模型中的随机效应估计
- **最小要求**: ≥7个被试 × 8个配置

---

## Stage 2: Core-2a 主效应覆盖 (Main Effect Coverage)

### 目标 (Objective)
- 确保每个因子的每个水平有足够样本
- 支持主效应估计 (Main effect estimation)
- 建立因子-响应关系 (Factor-response relationship)

### 行为 (Behavior)
- **采样方式**: D-optimal设计或近似均衡采样
- **分配策略**: 分散到不同被试，确保整体覆盖
- **重复性**: 每个被试测试不同配置

### 采样特征 (Sampling Characteristics)
```
目标: 每个因子水平 ≥7次测量
预算: 总预算的 32-40%

示例 (6因子，5水平):
  需求 ≈ 6 × 5 × 7 / 重叠率
  实际分配 ≈ 95次 (跨所有被试)
```

### 统计意义 (Statistical Significance)
- 支持主效应显著性检验
- 提供边际均值估计
- 用于方差分解

---

## Stage 3: Core-2b 交互初筛 (Interaction Screening)

### 目标 (Objective)
- 初步探索可能的交互效应
- 筛选显著交互对
- 为Phase 2提供先验信息

### 行为 (Behavior)
- **采样方式**: 选择top-K交互对，每对采样5-10次
- **分配策略**: 象限均衡采样 (四个极端组合)
- **重复性**: 分散到不同被试

### 采样特征 (Sampling Characteristics)
```
交互对选择: C(d, 2) → 选择 top-5
每对样本: 5-10次
预算: 总预算的 22-28%

示例 (6因子):
  候选交互对 = C(6,2) = 15
  选择5个 × 10次/对 = 50次
  实际分配 ≈ 66次 (考虑覆盖)
```

### 统计意义 (Statistical Significance)
- 检测非加性效应 (Non-additive effects)
- 初步估计交互强度
- 指导Phase 2的λ参数初始化

---

## Stage 4: Boundary 边界探索 (Boundary Exploration)

### 目标 (Objective)
- 探索设计空间边界行为
- 检测极端条件下的响应
- 避免外推风险 (Extrapolation risk)

### 行为 (Behavior)
- **采样方式**:
  - 单维极端点 (Single-dimension extremes)
  - 全局极端点 (Global extremes: 全0, 全1)
  - MaxiMin补充 (最大化最小距离)
- **分配策略**: 分散到不同被试

### 采样特征 (Sampling Characteristics)
```
单维极端: 2d个点 (每因子两端)
全局极端: 2个点 (全0, 全1)
MaxiMin: 填充至预算

示例 (6因子):
  单维 = 2 × 6 = 12个
  全局 = 2个
  补充 ≈ 16个
  总计 ≈ 30次
```

### 统计意义 (Statistical Significance)
- 支持边界约束建模
- 检测非线性行为
- 改善GP模型外推性能

---

## Stage 5: LHS 空间填充 (Space-Filling via LHS)

### 目标 (Objective)
- 填充未覆盖的设计空间区域
- 最大化探索效率
- 提供均匀分布的训练数据

### 行为 (Behavior)
- **采样方式**:
  - 分层拉丁超立方采样 (Stratified LHS)
  - Gower距离匹配到最近邻配置
- **分配策略**: 每个被试独立LHS采样
- **重复性**: 完全独立 (每个被试不同)

### 采样特征 (Sampling Characteristics)
```
预算: 剩余预算 (25-35%)
采样: 每个被试独立LHS

示例 (14被试):
  剩余预算 ≈ 47次
  平均每被试 ≈ 3-4个LHS点
  总覆盖增加 ≈ 47个独特配置
```

### 统计意义 (Statistical Significance)
- 改善设计空间覆盖率
- 减少GP模型的不确定性
- 提供多样化的训练样本

---

## 共享性与个体化 (Sharing vs. Individualization)

### 完全共享 (Fully Shared)
```
Core-1: 8个配置 × n_subjects次重复
用途: ICC估计, 随机效应建模
```

### 部分共享 (Partially Shared)
```
Core-2a + Core-2b + Boundary:
  整体确保覆盖目标
  个体贡献独特样本
  跨被试信息互补
```

### 完全独立 (Fully Independent)
```
LHS: 每个被试独立采样
用途: 最大化整体覆盖
```

---

## 预算分配公式 (Budget Allocation Formula)

```python
# Core-1 (固定)
n_core1 = 8 × n_subjects

# 剩余预算
remaining = total_budget - n_core1

if not skip_interaction:
    # 有交互探索
    n_core2a = remaining × 0.40  # 40%
    n_core2b = remaining × 0.28  # 28%
    n_explore = remaining × 0.32  # 32% (Boundary + LHS)
else:
    # 无交互探索
    n_core2a = remaining × 0.60  # 60%
    n_core2b = 0
    n_explore = remaining × 0.40  # 40%

# 探索预算细分
n_boundary = n_explore × 0.30-0.40
n_lhs = n_explore - n_boundary
```

---

## 与混合效应模型的关系 (Relation to Mixed-Effects Models)

### 必需组件 (Required Components)
| 模型组件 | 采样支持 | 来源阶段 |
|----------|----------|----------|
| 固定效应 (主效应) | ✅ | Core-2a |
| 固定效应 (交互) | ✅ | Core-2b |
| 随机截距 | ✅ | Core-1 |
| 随机斜率 | ⚠️ 有限 | Core-1 + Core-2a |
| 残差估计 | ✅ | All stages |

### 局限性 (Limitations)
- **随机斜率估计不足**: Core-1配置数量有限（8个）
- **交互项覆盖有限**: 只探索top-5交互对
- **嵌套结构缺失**: 无层次嵌套设计

---

## 设计原则 (Design Principles)

1. **平衡性 (Balance)**: Core-1共享 vs. 其他阶段分散
2. **效率性 (Efficiency)**: 避免过度重复，最大化信息
3. **鲁棒性 (Robustness)**: Core-1提供可靠性基准
4. **可扩展性 (Scalability)**: 支持不同被试数和预算

---

## 使用建议 (Usage Guidelines)

### 最小配置
- 被试数: ≥7人
- 每人trials: ≥20次
- 总预算: ≥140次

### 推荐配置
- 被试数: 10-15人
- 每人trials: 25-30次
- 总预算: 250-450次

### 理想配置
- 被试数: ≥18人
- 每人trials: ≥25次
- 总预算: ≥450次

---

## 关键指标 (Key Metrics)

```yaml
ICC估计精度:
  被试数: ≥7 (可接受), ≥10 (良好), ≥15 (优秀)
  重复配置: 8个 (固定)

主效应估计:
  每水平样本: ≥7 (最小), ≥10 (推荐)

交互效应筛选:
  每对样本: 5-10次 (初筛), ≥15次 (可靠)

设计空间覆盖:
  目标: ≥60% (可接受), ≥70% (良好)
```
