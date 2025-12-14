# 预热采样架构完整分析报告

**分析时间**：2025-12-14
**代码版本**：当前 main + feature/custom-ordinal-parameter
**分析范围**：extensions/warmup_budget_check 完整模块

---

## 📐 五步采样法设计规范

基于 `warmup_budget_estimator.py` 和 `STRUCTURE.md` 的设计定义：

### 整体设计框架

```
预热采样（Warmup Sampling Phase）的目的：
  ↓
通过有策略的配置采样，快速覆盖设计空间的关键区域，为后续主动学习提供初始数据
```

### 五个采样步骤及设计要求

#### Step 1: **Core-1** - 固定语义参考点（8个配置）

**设计目标**：
- 选择设计空间的**战略性关键点**，确保覆盖中心和边界
- 每个被试都要测试这8个配置（重复点，用于估计被试内方差）
- 配置数固定 = 8（warmup_budget_estimator.py:119）

**设计思路**（warmup_sampler.py:360-423）：
1. 中心点（几何中心或中位数）
2. 四分位数点（各维度的Q1/Q3）
3. 中位数扰动点（中位数±1个标准差）
4. MaxiMin补充（确保设计空间的最大最小距离）

**评估**：✅ **实现完全符合设计**
- 代码位置：`_select_core1_strategic()` (warmup_sampler.py:360-423)
- 使用Gower距离处理混合类型变量 ✓
- 覆盖战略性关键点 ✓
- 配置数固定为8 ✓

---

#### Step 2: **Boundary** - 边界极值配置池

**设计目标**：
- 捕捉设计空间的极端行为
- 每个因子的最小值和最大值配置
- 理论数量 = 2×d（如6因子=12个，但可能有重叠）

**设计思路**（warmup_sampler.py:460-497）：
1. 单维极端点：每个因子取min/max
2. 去重：离散空间中极端点可能重叠
3. MaxiMin补充：填充边界空白
4. 预算：min(2×d, budget["boundary_configs"])

**评估**：✅ **实现完全符合设计**
- 代码位置：`_select_boundary_configs()` (warmup_sampler.py:460-497)
- 单维极端点选择 ✓
- 去重处理 ✓
- 避免重复配置 ✓

---

#### Step 3: **Core-2a** - D-optimal主效应配置池

**设计目标**：
- 优化覆盖**每个因子的所有主要水平**
- 使用D-optimal设计原理最小化参数估计方差
- 配置数由预算估算器计算（warmup_budget_estimator.py:146-151）：
  ```
  n_core2a_min_abs = max(15, d * 3 * (0.3 + n_subjects/20.0))
  例如：6因子+5被试 ≈ 6*3*0.55 = 10 (最小15)
  ```

**设计思路**：
- **关键承诺**："D-optimal主效应采样"（warmup_budget_estimator.py:563、STRUCTURE.md:73）
- 应该与Core-2b分离，使用不同的采样策略

**评估**：❌ **实现**严重**不符合设计**

**问题证据**：

1. **分离失败**（warmup_sampler.py:797）：
   ```python
   n_core2_total = budget["core2a_configs"] + budget["core2b_configs"]  # ← 合并！
   core2_indices = self._select_interaction_aware_configs(
       n_configs=n_core2_total,  # ← 作为单一数字传入
       interaction_mode=interaction_mode,
       ...
   )
   ```

2. **没有D-optimal实现**：
   - Core-2a完全没有单独的采样函数
   - 最终都通过 `_select_interaction_aware_configs()` 处理
   - 该函数本质是随机采样（np.random.choice），没有D-optimal逻辑

3. **预算分离被忽视**：
   - 预算估算器计算出16个core2a + 21个core2b （假设值）
   - 但实现中37个配置被混合随机选择
   - Core-2a的"主效应覆盖"承诺完全未兑现

**结论**：❌ **完全失败** - Core-2a 应该是D-optimal，实际是随机

---

#### Step 4: **Core-2b** - 交互对感知配置池

**设计目标**：
- 优化覆盖**指定的交互对**或全部交互对
- 支持三种模式（free/specified_only/hybrid）
- 配置数由预算估算器计算（warmup_budget_estimator.py:156-160）：
  ```
  n_core2b_min_abs = max(15, d * (d-1) * 0.7)
  例如：6因子 ≈ 6*5*0.7 = 21个
  ```

**设计思路**（CORE2B_THREE_MODE_GUIDE.md）：
1. **FREE模式**：随机选择，所有交互对等概率
2. **SPECIFIED_ONLY模式**：仅覆盖指定对，预算均分
3. **HYBRID模式**（推荐）：保护分配指定对 + 自由探索其他对

**评估**：✅ **实现符合设计意图**

- 代码位置：`_select_interaction_aware_configs()` (warmup_sampler.py:615-734)
- 三种模式都实现了 ✓
- Hybrid模式逻辑正确 ✓
- 保护分配和自由探索分离 ✓

**但有一个重大缺陷**：
- **被Core-2a污染**：Core-2a和Core-2b共享同一个函数调用
- 结果：Core-2b的精心设计（三模式）应用到了合并的pool上
- 理想上，应该先分离，再各自采样

**结论**：⚠️ **部分成功** - 逻辑正确，但被架构污染

---

#### Step 5: **LHS** - 全局均匀填充配置池

**设计目标**：
- 使用Latin Hypercube Sampling确保全局均匀覆盖
- 填充Core-1/2a/2b/Boundary的空白
- 配置数由预算估算器计算

**设计思路**（warmup_sampler.py:499-586）：
1. 在[0,1]^d空间生成LHS样本
2. 映射到最近的离散配置（Gower距离）
3. 去重（排除已使用的配置）
4. Fallback：无scipy时退化为随机采样

**评估**：✅ **实现完全符合设计**

- 代码位置：`_select_lhs_global()` (warmup_sampler.py:499-586)
- LHS采样实现 ✓
- Gower距离映射 ✓
- 去重处理 ✓
- Scipy fallback ✓

**结论**：✅ **完全成功**

---

## 📊 模块评估总结表

| 模块 | 设计要求 | 实现状态 | 符合度 | 备注 |
|------|---------|---------|--------|------|
| **Core-1** | 战略性8个参考点，每个被试重复 | `_select_core1_strategic()` | ✅ 100% | 完全符合 |
| **Boundary** | 边界极值 + 去重 | `_select_boundary_configs()` | ✅ 100% | 完全符合 |
| **Core-2a** | **D-optimal主效应** | **随机选择** | ❌ 0% | **严重失败** |
| **Core-2b** | 交互对感知（三模式） | `_select_interaction_aware_configs()` | ⚠️ 70% | 逻辑正确，被2a污染 |
| **LHS** | 全局均匀填充 | `_select_lhs_global()` | ✅ 100% | 完全符合 |

---

## 🔴 关键架构缺陷总结

### 缺陷 1：Core-2a/2b 采样合并问题（P0 - 严重）

**现象**：
```python
# warmup_sampler.py:797-810
n_core2_total = budget["core2a_configs"] + budget["core2b_configs"]  # 合并
core2_indices = self._select_interaction_aware_configs(
    n_configs=n_core2_total,  # 单一参数
    ...
)
```

**后果**：
- 16个D-optimal主效应配置 + 21个交互对配置 → 随机混合为37个
- Core-2a的D-optimal承诺完全未履行
- 预算分离的科学意义完全丧失

**影响范围**：
- ❌ 学术严谨性（发表论文时提不出"D-optimal"设计）
- ❌ 采样效率（应该优先覆盖主效应，实际随机）
- ❌ 预测能力（主效应估计方差可能偏高）

---

### 缺陷 2：Core-2a 缺乏专用采样算法（P0 - 严重）

**现象**：
- Core-2a 没有 `_select_main_effect_configs()` 函数
- 没有D-optimal或分层采样逻辑
- 最终通过 `_select_interaction_aware_configs()` 处理 → 随机选择

**关键代码缺失位置**：warmup_sampler.py 应该有但没有：
```python
def _select_main_effect_configs(self, n_configs, used_indices):
    """D-optimal或分层采样实现主效应覆盖"""
    pass  # ← 不存在！
```

**应该做什么**：
- 为每个因子确保足够的水平覆盖
- 使用D-optimal准则（或分层采样作为退化版本）
- 与Core-2b完全独立的采样逻辑

---

### 缺陷 3：文档与实现脱节（P1 - 中等）

**文档承诺**（multiple locations）：
- STRUCTURE.md:73 - "D-optimal主效应"
- warmup_budget_estimator.py:563 - "D-optimal设计，分配给各被试"
- CORE2B_THREE_MODE_GUIDE.md - 详细的交互对三模式说明

**实现实际**：
- 全是随机采样，没有D-optimal
- 文档完全是虚假承诺

---

## 🔧 修复方案对比

### 方案A：最小改动（分离 + 基础覆盖）

**修改量**：~100 行代码
**复杂度**：低
**风险**：低

```python
# warmup_sampler.py:796 处修改

# 分离：而不是合并
n_core2a = budget["core2a_configs"]
n_core2b = budget["core2b_configs"]

# Step 3a: Core-2a - 确定性主效应覆盖
core2a_indices = self._select_main_effect_configs(
    n_configs=n_core2a,
    used_indices=used_indices
)
used_indices.update(core2a_indices)

# Step 3b: Core-2b - 交互对感知采样（保留原逻辑）
core2b_indices = self._select_interaction_aware_configs(
    n_configs=n_core2b,
    interaction_mode=interaction_mode,
    interaction_pairs_to_explore=interaction_pairs_to_explore,
    min_config_per_pair=min_config_per_pair,
    used_indices=used_indices,
)
used_indices.update(core2b_indices)

core2_indices = core2a_indices + core2b_indices
```

新增方法：
```python
def _select_main_effect_configs(self, n_configs, used_indices):
    """
    分层采样确保主效应覆盖：
    - 确保每个因子的不同水平都有表示
    - 使用贪心策略最大化因子水平覆盖度
    """
    # 实现思路见下面"方案实现代码"
```

**优点**：
- 立即解决"合并"问题 ✓
- 基础分层逻辑确保主效应覆盖 ✓
- 不破坏现有代码 ✓
- 风险最低 ✓

**缺点**：
- 不是真正的D-optimal（需要pyDOE库）
- 分层逻辑仍很简单

---

### 方案B：完整实现（D-optimal设计）

**修改量**：~200 行代码 + 新增依赖
**复杂度**：中等
**风险**：中等

需要引入 `pyDOE2` 库：
```bash
pixi add -c conda-forge pyDOE2
```

```python
def _select_doptimal_configs(self, n_configs, used_indices):
    """
    使用D-optimal准则选择主效应配置
    D-optimality: 最小化|X'X|的倒数（等价于最小化参数估计方差）
    """
    from scipy.linalg import det

    available = list(set(self.design_df.index) - used_indices)
    if len(available) <= n_configs:
        return available

    # 评估候选子集的D-optimality
    best_det = -np.inf
    best_subset = None

    for _ in range(min(1000, len(available))):  # 采样评估
        sample_idx = np.random.choice(len(available), n_configs, replace=False)
        subset = self.design_df.iloc[available[sample_idx]].values

        try:
            XtX = subset.T @ subset
            d = np.abs(det(XtX))
            if d > best_det:
                best_det = d
                best_subset = sample_idx
        except:
            pass

    if best_subset is not None:
        return [available[i] for i in best_subset]
    else:
        # Fallback: 分层采样
        return self._select_main_effect_configs_stratified(n_configs, used_indices)
```

**优点**：
- 科学严谨，符合原承诺 ✓
- D-optimal理论有文献支持 ✓
- 可发表 ✓

**缺点**：
- 需要新依赖 ✓
- 计算成本较高 ✓
- 需要更多测试 ✓

---

### 方案C：混合方案（平衡版，推荐）

结合方案A和B的优点：
- 立即用方案A（分离 + 分层采样）解决P0问题
- 之后扩展到方案B（D-optimal）作为可选高级功能

**实现步骤**：
1. Phase 1（本周）：实施方案A
2. Phase 2（下周）：添加D-optimal作为可选参数
3. Phase 3（后续）：默认使用D-optimal，方案A作为fallback

---

## 📋 修复检查清单

如果决定修复（推荐：选择方案A或C）：

- [ ] **Step 0：准备**
  - [ ] 在feature分支上工作
  - [ ] 为修改添加单元测试

- [ ] **Step 1：分离Core-2a/2b**
  - [ ] 修改 `_generate_five_step_samples()` (warmup_sampler.py:796-811)
  - [ ] 分别调用 Core-2a 和 Core-2b 采样函数
  - [ ] 保持 `used_indices` 的正确更新

- [ ] **Step 2：实现Core-2a采样方法**
  - [ ] 选择方案（A=分层/B=D-optimal）
  - [ ] 新增 `_select_main_effect_configs()` 或 `_select_doptimal_configs()`
  - [ ] 确保处理 `used_indices` 和 `available` 列表

- [ ] **Step 3：文档更新**
  - [ ] 更新 STRUCTURE.md 中对Core-2a的描述
  - [ ] 更新 warmup_budget_estimator.py 中的打印说明
  - [ ] 添加 Core-2a 采样的说明文档
  - [ ] 如果用pyDOE，更新依赖文档

- [ ] **Step 4：测试验证**
  - [ ] 新增 `test_core2a_main_effect_coverage.py`
  - [ ] 验证Core-2a覆盖所有因子水平
  - [ ] 验证Core-2b不受影响
  - [ ] 验证预算分离正确

- [ ] **Step 5：回归测试**
  - [ ] 运行所有existing tests
  - [ ] 测试所有三种交互模式（free/specified_only/hybrid）
  - [ ] 测试不同的被试数和预算配置

---

## 🎯 立即建议

**优先级建议**：

1. **P0 (立即修复)**：Core-2a/2b 合并问题
   - 影响：科学严谨性和预算分离的意义
   - 方案：方案 A（最低成本）
   - 时间：2-3 小时
   - 风险：低

2. **P1 (本周)**：如果有学术发表计划，升级到方案 B
   - 影响：论文中的设计声称必须符合D-optimal
   - 时间：1 天
   - 风险：中

3. **P2 (后续)**：完善文档和验证工具
   - 添加诊断脚本
   - 验证覆盖率
   - 更新所有相关文档

---

## 📚 相关文件引用

| 文件 | 关键行号 | 内容 |
|------|---------|------|
| warmup_budget_estimator.py | 98-105 | 五步采样法设计定义 |
| warmup_budget_estimator.py | 143-199 | 预算分离逻辑 |
| warmup_sampler.py | 796-811 | **合并问题所在** |
| warmup_sampler.py | 360-423 | Core-1战略性选择 ✓ |
| warmup_sampler.py | 460-497 | Boundary边界选择 ✓ |
| warmup_sampler.py | 615-734 | Core-2b交互对采样 ⚠️ |
| warmup_sampler.py | 499-586 | LHS全局采样 ✓ |
| STRUCTURE.md | 71-76 | 五步采样法规范定义 |
| CORE2B_THREE_MODE_GUIDE.md | 全部 | Core-2b三模式详细说明 ✓ |

---

## 🏁 总结

| 模块 | 符合度 | 结论 |
|------|--------|------|
| **Core-1** | ✅ 100% | 完全符合设计 |
| **Boundary** | ✅ 100% | 完全符合设计 |
| **Core-2a** | ❌ 0% | **严重不符，需立即修复** |
| **Core-2b** | ⚠️ 70% | 逻辑正确，被2a污染 |
| **LHS** | ✅ 100% | 完全符合设计 |
| **整体架构** | ❌ 60% | **关键缺陷：五步法退化为四步法** |

---

**下一步**：请确认是否：
1. 采纳建议进行修复？（推荐方案A或C）
2. 继续进行其他代码审查？
3. 实施特定的测试计划？

