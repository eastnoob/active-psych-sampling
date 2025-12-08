# 混合效应模型在预热阶段的应用指南

## 核心问题回答

### Q1: 混合效应模型对后续阶段是否有帮助？

**答案：✅ 非常有帮助**

混合效应模型可以为Phase 2主动学习提供：

| 收益 | 具体帮助 | 影响阶段 |
|------|----------|----------|
| **个体差异建模** | 为每个被试提供个性化预测 | Phase 2采样效率 ↑ |
| **迁移学习基础** | 新被试快速适应（few-shot） | Phase 2冷启动 ↑ |
| **先验分布** | 为GP超参数提供informative prior | Phase 2模型质量 ↑ |
| **方差分解** | 明确个体vs整体的方差占比 | Phase 2策略选择 ↑ |
| **交互识别** | 提前筛选显著交互对 | Phase 2 λ初始化 ✅ |

---

## Q2: 当前采样设计的可行性评估

### ✅ 可行的部分

当前五步采样法**可以**支持以下混合效应模型组件：

#### 1. 基础混合效应模型
```r
# 可拟合的基础模型
lmer(response ~ factor1 + factor2 + ... +  # 固定效应：主效应
              (1 | subject_id),             # 随机效应：随机截距
     data = warmup_data)
```

**支持度**: ✅✅✅ 完全支持
- Core-1提供8个共享配置 → ICC估计可靠
- Core-2a确保主效应覆盖 → 固定效应估计充分
- 所有阶段提供被试间差异 → 随机截距估计稳定

#### 2. 带交互的混合效应模型
```r
lmer(response ~ factor1 * factor2 +         # 固定效应：主效应+交互
              (1 | subject_id),              # 随机效应：随机截距
     data = warmup_data)
```

**支持度**: ✅⚠️ 部分支持
- Core-2b探索top-5交互对 → **仅限筛选出的交互**
- 其他交互对样本不足 → 估计不可靠
- **限制**: 只能包含Phase 1筛选出的3-5个交互项

---

### ⚠️ 局限的部分

当前设计**无法充分**支持：

#### 1. 随机斜率模型 (Random Slopes)
```r
# 理想但当前不可靠的模型
lmer(response ~ factor1 + factor2 + ... +
              (1 + factor1 | subject_id),    # 随机斜率：factor1
     data = warmup_data)
```

**问题**:
- Core-1只有8个配置 → 每个被试只有8个数据点
- 估计随机斜率需要每个被试在该因子上有**多个水平的重复测量**
- 当前设计：8个配置无法保证每个因子水平都有重复

**影响**: 随机斜率估计方差过大，可能不收敛

#### 2. 多个随机斜率
```r
# 完全不可行
lmer(response ~ factor1 + factor2 + ... +
              (1 + factor1 + factor2 | subject_id),
     data = warmup_data)
```

**问题**: 参数过多，自由度不足

#### 3. 交叉随机效应
```r
# 当前不可行
lmer(response ~ ... + (1 | subject_id) + (1 | config_id))
```

**问题**: Config维度的重复测量不足

---

## Q3: 关键认识 (Key Insights) 列表

如果将预热阶段作为pilot study，需要获取以下**关键认识**：

### 1. 个体差异结构 (Individual Difference Structure)

#### 需要识别
- **ICC值**: 个体差异占总方差的比例
  - ICC > 0.3 → 个体差异显著，必须用混合模型
  - ICC < 0.1 → 个体差异小，可用pooled模型
- **随机效应方差**: `σ²_subject`
- **残差方差**: `σ²_residual`

#### 当前支持度: ✅✅✅
- Core-1的8个共享配置足够估计ICC
- 推荐: 增加Core-1配置数至**10-12个**可提高精度

---

### 2. 主效应模式 (Main Effect Patterns)

#### 需要识别
- 每个因子的边际效应大小
- 线性 vs. 非线性模式
- 因子重要性排序

#### 当前支持度: ✅✅
- Core-2a已确保每个因子水平≥7次
- 改进: 增加到≥10次可提高显著性检验power

---

### 3. 交互效应结构 (Interaction Structure)

#### 需要识别
- **显著交互对**: 哪些因子对有交互
- **交互强度**: 交互效应 vs. 主效应的相对大小
- **交互模式**: 正交互 vs. 负交互

#### 当前支持度: ✅⚠️ 部分支持
- Core-2b只探索top-5交互对
- **缺失**: 其他C(d,2)-5个交互对未探索

#### 改进建议: 见下节"改进方案3"

---

### 4. 个体化主效应差异 (Subject-Specific Main Effects)

#### 需要识别
- 不同被试对同一因子的响应是否一致
- 是否需要随机斜率
- 哪些因子存在个体差异

#### 当前支持度: ⚠️⚠️ 不足
- Core-1配置数太少（8个）
- 无法可靠估计随机斜率

#### 关键性: ⭐⭐⭐⭐⭐
- 如果存在显著的个体化主效应，Phase 2必须个性化采样
- 如果不存在，Phase 2可以用统一策略

---

### 5. 响应曲面形状 (Response Surface Shape)

#### 需要识别
- 整体平滑 vs. 高频波动
- 多峰 vs. 单峰
- 边界行为

#### 当前支持度: ✅
- Boundary阶段探索边界
- LHS阶段填充空间

---

### 6. 方差分解 (Variance Decomposition)

#### 需要识别
```
总方差 = σ²_main + σ²_interaction + σ²_subject + σ²_residual
```

- 主效应方差占比
- 交互方差占比
- 个体差异占比

#### 当前支持度: ✅✅
- phase1_analyzer.py已实现
- 输出: `var_decomposition` 字典

#### 用途
- 如果交互方差大 → Phase 2高λ
- 如果个体差异大 → Phase 2需要个性化GP

---

## Q4: 改进方案 (Improved Sampling Design)

### 方案1: 增强随机斜率估计（推荐）

**目标**: 能够拟合 `(1 + factor1 | subject_id)` 形式的模型

**改动**:
```python
# 修改 Core-1 设计
n_core1_configs = 12  # 从8增加到12

# 确保12个配置在关键因子上有充分覆盖
# 例如：每个关键因子的每个水平至少出现2次
```

**额外预算需求**: `+4 × n_subjects` 次
- 示例：14被试 → +56次 → 总预算从350增至406次

**收益**:
- 可估计1-2个关键因子的随机斜率
- 识别个体化效应模式
- 为Phase 2个性化策略提供依据

---

### 方案2: 交互效应全面筛选（可选）

**目标**: 探索所有可能的二阶交互

**改动**:
```python
# Core-2b: 从top-5扩展到全面筛选
n_pairs = C(d, 2)  # 所有交互对
samples_per_pair = 5  # 每对最少5次

# 采用"稀疏采样"策略
for each pair (i, j):
    sample 5 points from 4 quadrants + center
```

**额外预算需求**:
- 6因子：C(6,2) = 15对 × 5次 = 75次
- 当前Core-2b ≈ 66次 → 需额外约10次

**收益**:
- 完整的交互图谱
- 避免遗漏重要交互
- Phase 2的λ初始化更准确

---

### 方案3: 分层pilot设计（元学习视角）⭐⭐⭐

**核心思想**: 将预热分为两个子阶段

#### 子阶段1: 快速筛选（Budget: 40%）
```
目标: 快速识别
  - 重要主效应因子 (top-3)
  - 可能的交互对 (top-5)
  - ICC估计

采样:
  - Core-1: 10个配置 × n_subjects
  - 粗略主效应: 每因子每水平5次
  - 粗略交互: top-10交互对，每对3次
```

#### 子阶段2: 精细探索（Budget: 60%）
```
目标: 基于子阶段1的发现，聚焦关键区域
  - 重要因子: 增加采样密度
  - 显著交互: 精细采样
  - 个体化效应: 为关键因子估计随机斜率

采样:
  - 重要因子: 每水平增至10次
  - 显著交互: 每对增至10次
  - 个体化: 关键因子在Core-1配置中重复
```

**实现**:
```python
# 修改 quick_start.py，增加两阶段模式
MODE = "pilot_two_stage"

PILOT_CONFIG = {
    "stage1_budget_ratio": 0.4,
    "stage2_budget_ratio": 0.6,
    "stage1_goal": "screening",
    "stage2_goal": "refinement",
}
```

**收益**:
- 自适应采样，预算利用率最高
- 避免在不重要因子上浪费样本
- 元学习：从初步数据学习重要性，再聚焦

---

### 方案4: 最小改动方案（立即可用）

**目标**: 不增加预算，优化当前设计

**改动**:
```python
# 1. Core-1从随机选择改为"准D-optimal"
#    确保12个配置在主效应上平衡

def select_core1_configs_optimized(design_df, n_core1=8):
    """选择Core-1配置，优化主效应平衡"""
    # 确保每个因子的每个水平至少出现1次
    # 使用贪心算法逐步添加配置
    pass

# 2. Core-2b优先探索高方差交互对
#    基于因子方差的乘积预排序

def rank_interaction_pairs_by_variance(design_df):
    """预估交互对重要性"""
    variances = design_df.var()
    pairs = []
    for i, j in combinations(range(d), 2):
        score = variances[i] * variances[j]
        pairs.append((i, j, score))
    return sorted(pairs, key=lambda x: x[2], reverse=True)
```

**预算**: 不变
**收益**: 提高采样质量，无额外成本

---

## Q5: 混合效应模型对Phase 2的具体帮助

### 1. 个性化GP建模

**传统方法** (当前):
```python
# 单一GP模型，忽略个体差异
gp_model = SingleTaskGP(X_all, y_all)
```

**改进方法** (基于混合效应):
```python
# 层次GP：共享超参数 + 个体化均值偏移
class HierarchicalGP:
    def __init__(self, mixed_model_params):
        # 从Phase 1混合模型提取
        self.global_mean = mixed_model_params['fixed_effects']
        self.subject_shifts = mixed_model_params['random_effects']

    def predict(self, X_new, subject_id):
        # 全局预测 + 个体修正
        global_pred = self.gp.predict(X_new)
        subject_shift = self.subject_shifts[subject_id]
        return global_pred + subject_shift
```

**收益**:
- 新被试冷启动：用全局均值初始化
- 已有被试：用个体shift提高预测精度
- **采样效率提升**: 约20-30% (文献报告)

---

### 2. λ参数初始化

**当前方法**:
```python
# 基于方差分解
lambda_init = Var_interaction / (Var_main + Var_interaction)
```

**改进方法** (基于混合模型):
```python
# 考虑个体差异
lambda_init = Var_interaction / (
    Var_main + Var_interaction + Var_subject
)

# 如果个体差异大，降低λ（因为部分"交互"实为个体差异）
if ICC > 0.3:
    lambda_init *= 0.8  # 修正因子
```

**收益**: λ初始化更准确，Phase 2早期探索更高效

---

### 3. 迁移学习到新被试

**场景**: Phase 2加入新被试

**方法**:
```python
# 基于Phase 1混合模型的先验
new_subject_prior = {
    'mean': global_fixed_effects,
    'variance': sigma_subject_from_phase1,
}

# Few-shot适应：新被试只需3-5个样本即可个性化
def adapt_to_new_subject(new_data):
    # 贝叶斯更新：prior + likelihood → posterior
    posterior_mean = bayesian_update(
        prior=new_subject_prior,
        data=new_data
    )
    return posterior_mean
```

**收益**:
- 新被试预热减少80% (从25次降至5次)
- Phase 2可以不断加入新被试，扩展样本

---

## 推荐方案总结

### 最小改动（立即采用）
1. 修改Core-1为准D-optimal选择（方案4）
2. 当前预算不变
3. 可拟合基础混合模型：`(1 | subject_id)`

### 中度改进（推荐）
1. Core-1配置数增至12个（方案1）
2. 额外预算：+4次/被试
3. 可拟合随机斜率：`(1 + key_factor | subject_id)`
4. 可识别个体化效应

### 理想方案（研究深入）
1. 采用两阶段pilot设计（方案3）
2. 预算增加20-30%
3. 完整混合模型：多随机斜率 + 全交互探索
4. 元学习能力：自适应聚焦重要因子

---

## 代码实现建议

### 修改 warmup_budget_estimator.py

```python
# 增加混合模型支持选项
def estimate_budget_requirements(
    n_subjects,
    trials_per_subject,
    skip_interaction=False,
    enable_random_slopes=False,  # 新增
    n_random_slopes=1,            # 新增
):
    if enable_random_slopes:
        # 增加Core-1配置数
        n_core1_configs = 10 + n_random_slopes * 2
    else:
        n_core1_configs = 8

    # ... 其余逻辑
```

### 修改 analyze_phase1.py

```python
# 增加混合模型拟合
def fit_mixed_effects_model(X, y, subject_ids):
    """
    拟合混合效应模型并提取参数

    Returns:
        - fixed_effects: 固定效应估计
        - random_effects: 随机效应（被试）
        - ICC: 组内相关系数
        - sigma_subject: 被试间标准差
        - sigma_residual: 残差标准差
    """
    import statsmodels.formula.api as smf

    # 准备数据
    df = prepare_data(X, y, subject_ids)

    # 拟合基础模型
    formula = "response ~ factor1 + factor2 + ... + (1 | subject_id)"
    model = smf.mixedlm(formula, data=df, groups=df["subject_id"])
    result = model.fit()

    # 提取ICC
    var_subject = result.cov_re.iloc[0, 0]
    var_residual = result.scale
    ICC = var_subject / (var_subject + var_residual)

    return {
        'fixed_effects': result.fe_params,
        'random_effects': result.random_effects,
        'ICC': ICC,
        'sigma_subject': np.sqrt(var_subject),
        'sigma_residual': np.sqrt(var_residual),
    }
```

### 在 Phase 2 中使用

```python
# 加载Phase 1混合模型参数
mixed_model_params = load_mixed_model_params('phase1_output/')

# 初始化个性化GP
for subject_id in range(n_subjects):
    subject_gp = create_subject_gp(
        global_params=mixed_model_params['fixed_effects'],
        subject_shift=mixed_model_params['random_effects'][subject_id],
    )

    # EUR-ANOVA采样时使用个性化GP
    scores = eur_anova_acqf(
        model=subject_gp,
        subject_id=subject_id,
    )
```

---

## 结论

**当前采样设计可行性**: ⭐⭐⭐ (3/5)
- ✅ 支持基础混合模型 (随机截距)
- ⚠️ 不支持随机斜率
- ⚠️ 交互探索有限

**推荐行动**:
1. **立即**: 采用方案4（最小改动）
2. **短期**: 采用方案1（增强Core-1）
3. **长期**: 考虑方案3（两阶段pilot）

**混合模型对Phase 2的帮助**: ⭐⭐⭐⭐⭐ (5/5)
- 显著提升采样效率
- 支持迁移学习
- 改进λ/γ初始化
- 实现个性化建模
