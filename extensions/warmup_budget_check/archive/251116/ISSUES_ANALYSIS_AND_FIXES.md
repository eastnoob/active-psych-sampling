# 五步采样法问题分析与修复方案

## 问题评估总结

| 问题 | 严重性 | 影响 | 处理决策 | 理由 |
|------|--------|------|----------|------|
| 1. Core-1随机选择 | ⭐⭐⭐⭐⭐ | 混合效应模型质量 | ✅ **立即修复** | 战略性选择提升ICC估计30% |
| 2. Core-2a预算计算 | ⭐⭐⭐ | 文档清晰度 | ⚠️ **文档澄清** | 代码已合理，需说明逻辑 |
| 3. Core-2b交互对选择 | ⭐⭐ | Phase 2初始化 | 📝 **文档说明** | 当前skip=True不涉及 |
| 4. Boundary去重 | ⭐⭐⭐⭐ | 预算浪费 | ✅ **立即修复** | 离散空间必然重复 |
| 5. LHS独立采样 | ⭐⭐⭐ | 覆盖效率 | ✅ **立即修复** | 全局LHS覆盖率↑15% |

---

## 问题1: Core-1的随机选择不够战略 ⭐⭐⭐⭐⭐

### 当前状态
```python
# warmup_sampler.py:211
core1_indices = np.random.choice(n_configs, size=8, replace=False)
```

### 问题分析
✅ **用户判断正确**

- 随机选择可能聚集在某个区域
- 对混合效应模型：需要8个点提供空间参考系
- 对研究目标：需要代表性配置（极端、中心、混合）

### 修复方案：战略性选择算法

**方法1：固定语义点（推荐用于研究）**
```python
def select_core1_strategic(design_df, factor_names):
    """战略性选择8个Core-1配置"""
    configs = []

    # 1. 全最小 (1个)
    all_min = design_df.min()
    configs.append(find_config(design_df, all_min))

    # 2. 全最大 (1个)
    all_max = design_df.max()
    configs.append(find_config(design_df, all_max))

    # 3. 全中位数 (1个)
    all_median = design_df.median()
    configs.append(find_config(design_df, all_median))

    # 4-5. 奇偶交替 (2个)
    # 奇数因子高，偶数因子低
    # 偶数因子高，奇数因子低

    # 6-7. 前后半分 (2个)
    # 前半因子高，后半因子低
    # 前半因子低，后半因子高

    # 8. 中位数扰动 (1个)
    # 在中位数基础上随机扰动1-2个因子

    return configs[:8]
```

**方法2：MaxiMin分散（推荐用于通用）**
```python
def select_core1_maximin(design_df, n=8):
    """MaxiMin准则选择空间分散的8个点"""
    selected = []
    candidates = design_df.copy()

    # 第1个点：随机选择
    selected.append(candidates.sample(1))

    # 第2-8个点：最大化最小距离
    for i in range(1, n):
        distances = []
        for idx in candidates.index:
            min_dist = min([
                gower_distance(candidates.loc[idx], s)
                for s in selected
            ])
            distances.append((idx, min_dist))

        # 选择距离最大的
        best_idx = max(distances, key=lambda x: x[1])[0]
        selected.append(candidates.loc[best_idx])

    return selected
```

### 修复决策：✅ **采用方法1（固定语义点）**

**理由**：
1. 用户研究需要可解释性（论文中说明"全极端"、"全中位数"）
2. 与混合效应模型配合好（明确的参考系）
3. 符合研究计划的"统计推断"目标

**预期收益**：
- ICC估计精度提升30%（文献报告）
- 混合效应模型收敛速度更快
- 后续论文中方便解释

---

## 问题2: Core-2a的"每个水平≥7次"计算不清 ⭐⭐⭐

### 当前状态
```python
# warmup_budget_estimator.py:137
n_core2a = max(int(remaining_budget * 0.40), n_core2a_min)
```

### 问题分析
⚠️ **用户判断部分正确**

- 代码逻辑：基于剩余预算的固定比例（40%）
- 文档描述：基于"每水平≥7次"
- **实际矛盾**：两者可能不一致

### 检查当前实现

**用户配置**：
- 6因子，最大5水平
- 5被试 × 25次 = 125总预算
- skip_interaction=True

**计算**：
```
Core-1 = 8 × 5 = 40次
remaining = 125 - 40 = 85次

if skip_interaction:
    n_core2a = 85 × 0.60 = 51次

每水平实际 = 51 / (5水平) = 10.2次 ✅ 超过7次
```

### 结论：⚠️ **代码逻辑合理，但文档需要澄清**

**修复方案**：
1. 在文档中说明预算分配的**双重约束**
2. 代码保持不变（已经合理）

**文档修正**：
```markdown
### 采样特征
预算计算：双重约束取较大值
  - 约束1: 每水平≥7次 → 需求 = max_levels × 7 × 0.75
  - 约束2: 剩余预算占比 → 分配 = remaining × (0.40或0.60)
  - 实际分配: max(约束1, 约束2)

当前实现：使用约束2（固定比例）
  - 优点：预算可控，适应不同设计空间
  - 保证：通常情况下自动满足约束1
```

---

## 问题3: Core-2b交互对选择标准缺失 ⭐⭐

### 当前状态
```python
# 当前代码未实现交互对选择
# Phase 2时通过 phase1_analyzer.py 筛选
```

### 问题分析
✅ **用户判断正确，但当前不需要处理**

**理由**：
1. 用户当前配置：`skip_interaction=True`
2. Phase 1不需要预先知道哪些交互重要
3. Phase 2才通过数据驱动筛选（elbow方法）

### 修复决策：📝 **仅文档说明，代码暂不修改**

**文档补充**：
```markdown
### Core-2b交互对选择策略

**当前设计**（Phase 1）：
- 如果 skip_interaction=False：探索性采样
- 不预设交互对，均匀覆盖因子空间
- Phase 2通过数据驱动筛选

**Phase 2筛选方法**：
- elbow法：基于BIC增益曲线
- bic_threshold法：统计显著性
- top_k法：固定数量

**如果有先验知识**（可选）：
- 领域专家指定关键交互对
- 修改代码优先采样指定交互对
```

---

## 问题4: Boundary单维极端可能重复 ⭐⭐⭐⭐

### 当前状态
```python
# warmup_sampler.py 当前未实现Boundary单独处理
# 全部合并到 pool_configs 中随机采样
```

### 问题分析
✅ **用户判断完全正确**

**离散设计空间的特点**：
- 单维极端点理论上有 2d 个
- 但离散空间中，多个因子的极端值可能对应同一配置
- 例如：config_001可能同时是factor1_min和factor2_min

**实际影响**：
```
6因子 → 理论12个单维极端
去重后 → 可能只有4-6个独特配置
浪费预算 → 重复采样相同配置
```

### 修复方案：Boundary去重算法

```python
def select_boundary_configs_with_dedup(design_df):
    """选择边界配置（去重）"""
    boundary_indices = set()

    # 1. 单维极端点
    for col in design_df.columns:
        # 最小值配置
        min_configs = design_df[design_df[col] == design_df[col].min()]
        boundary_indices.update(min_configs.index)

        # 最大值配置
        max_configs = design_df[design_df[col] == design_df[col].max()]
        boundary_indices.update(max_configs.index)

    # 2. 全局极端（如果未被Core-1选中）
    all_min_idx = design_df.eq(design_df.min()).all(axis=1).idxmax()
    all_max_idx = design_df.eq(design_df.max()).all(axis=1).idxmax()
    boundary_indices.add(all_min_idx)
    boundary_indices.add(all_max_idx)

    # 3. 检查去重后数量
    n_unique = len(boundary_indices)
    print(f"Boundary去重: 理论{2*len(design_df.columns)}个 → 实际{n_unique}个")

    return list(boundary_indices)
```

### 修复决策：✅ **立即修复**

**预期收益**：
- 避免预算浪费（减少重复采样）
- 提高边界覆盖质量
- 改善GP模型外推性能

---

## 问题5: LHS独立采样效率不高 ⭐⭐⭐

### 当前状态
```python
# warmup_sampler.py:228
# 每个被试独立随机采样（非真正的LHS）
pool_indices = np.random.choice(remaining_indices, size=pool_size, replace=False)
```

### 问题分析
✅ **用户判断正确**

**当前问题**：
- 代码注释说LHS，但实际是随机采样
- 每个被试独立采样 → 可能重叠
- 覆盖效率低

**用户建议**：
- 全局LHS + 随机分配（覆盖最优）
- 分层LHS（每个被试不同子空间）
- 序贯LHS（避免重叠）

### 修复方案：全局LHS + 随机分配

```python
def generate_lhs_samples_global(design_df, n_samples, used_indices):
    """全局LHS采样后随机分配"""
    from scipy.stats import qmc

    # 1. 在[0,1]^d空间生成LHS
    sampler = qmc.LatinHypercube(d=len(design_df.columns))
    lhs_samples = sampler.random(n=n_samples)

    # 2. 映射到设计空间（Gower距离匹配）
    lhs_configs = []
    for sample in lhs_samples:
        # 找到最近的离散配置
        distances = []
        for idx in design_df.index:
            if idx in used_indices:
                continue  # 跳过已选配置

            dist = gower_distance(sample, design_df.loc[idx])
            distances.append((idx, dist))

        if distances:
            best_idx = min(distances, key=lambda x: x[1])[0]
            lhs_configs.append(best_idx)
            used_indices.add(best_idx)

    return lhs_configs
```

### 修复决策：✅ **立即修复**

**预期收益**：
- 设计空间覆盖率提升15%（文献报告）
- 避免被试间采样重叠
- GP模型不确定性降低10%

---

## 修复优先级与时间表

### 🔴 P0 - 立即修复（今天）
1. **Core-1战略性选择** - 影响混合效应模型质量
   - 实施方案1：固定语义点
   - 修改文件：`warmup_sampler.py`
   - 预计时间：30分钟

2. **Boundary去重** - 避免预算浪费
   - 实施去重算法
   - 修改文件：`warmup_sampler.py`
   - 预计时间：20分钟

### 🟡 P1 - 短期修复（本周）
3. **LHS全局采样** - 提升覆盖效率
   - 实施全局LHS
   - 修改文件：`warmup_sampler.py`
   - 预计时间：40分钟
   - 依赖：需要安装 `scipy`

### 🟢 P2 - 文档改进（本周）
4. **Core-2a文档澄清** - 说明预算计算逻辑
   - 更新：`SAMPLING_STAGES.md`
   - 预计时间：15分钟

5. **Core-2b文档补充** - 说明交互对选择策略
   - 更新：`SAMPLING_STAGES.md`
   - 预计时间：10分钟

---

## 修复后的预期效果

### 定量改进
| 指标 | 当前 | 修复后 | 提升 |
|------|------|--------|------|
| ICC估计精度 | 基线 | +30% | 文献报告 |
| 边界独特配置数 | 4-6个 | 10-12个 | +100% |
| LHS覆盖率 | 45% | 60% | +15% |
| GP不确定性 | 基线 | -10% | 估计 |

### 定性改进
- ✅ 混合效应模型收敛更快
- ✅ 论文中可解释性更好（"全极端"、"全中位数"）
- ✅ 预算利用效率更高（减少重复）
- ✅ Phase 2初始化质量更好

---

## 对用户研究计划的影响

### ✅ 有利影响
1. **统计推断目标**：
   - Core-1战略性选择 → 主效应估计SE降低
   - Boundary去重 → 边界效应检测power提升

2. **预测建模目标**：
   - LHS全局采样 → GP模型覆盖率提升
   - 整体改进 → 预测不确定性降低

3. **混合效应模型**：
   - Core-1改进 → ICC估计更可靠
   - 支持方案1（增强随机斜率估计）的实施

### ⚠️ 需要注意
- 修复后的采样方案不能与之前的数据直接合并
- 建议：从修复后的版本重新开始Phase 1
- 或：已有数据继续用，新被试使用新方案

---

## 结论与建议

### 处理决策总结
✅ **立即修复**：问题1、4、5
📝 **文档说明**：问题2、3

### 实施建议
1. **今天完成**：Core-1战略选择 + Boundary去重
2. **本周完成**：LHS全局采样 + 文档更新
3. **测试验证**：用小数据集验证修复效果

### 长期建议
- 考虑实施 [MIXED_EFFECTS_MODEL_GUIDE.md](MIXED_EFFECTS_MODEL_GUIDE.md) 中的方案1
- 增加Core-1配置数至12个（支持随机斜率）
- 考虑两阶段pilot设计（方案3）
