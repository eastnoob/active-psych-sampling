# Variance Components 评估模块使用说明

## 功能概述

`evaluate_variance_components()` 用于评估混合效应模型(LMM)的方差成分估计能力,核心用于:

- **ICC (Intraclass Correlation)** 评估: 被试间差异占总变异的比例
- **Between-subject variance (σ²_between)**: 群体水平差异
- **Within-subject variance (σ²_within)**: 个体内测量误差

这对于评估采样策略在群体推断中的表现至关重要。

---

## 快速开始

```python
from modules.evaluation_effect_recovery import evaluate_variance_components
import numpy as np

# 示例: 3个被试,每人5次观测
n_subjects = 3
n_obs_per_subject = 5

# 被试ID
subject_ids = np.repeat([0, 1, 2], n_obs_per_subject)  # [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]

# 采样点 (15, d)
X = np.random.randn(15, 6)

# 观测值 (包含被试间差异)
subject_effects = np.array([0.5, -0.3, 0.2])  # 被试随机效应
y = subject_effects[subject_ids] + np.random.randn(15) * 0.3

# 评估
result = evaluate_variance_components(
    X_train=X,
    y_train=y,
    subject_ids=subject_ids,
    include_predictors=True,  # Conditional ICC (控制预测变量)
)

# 查看结果
print(f"ICC: {result['icc']:.3f}")
print(f"Between-subject variance: {result['sigma2_between']:.4f}")
print(f"Within-subject variance: {result['sigma2_within']:.4f}")
```

---

## 参数说明

### 必需参数

- **X_train**: `np.ndarray (n, d)` - 采样点(可包含重复被试的多次测量)
- **y_train**: `np.ndarray (n,)` - 观测值
- **subject_ids**: `np.ndarray (n,)` - 被试ID,标识每个观测来自哪个被试

### 可选参数

- **true_variance_components**: `Dict` - 真实方差成分,用于评估准确性

  ```python
  {
      "sigma2_between": 0.5,
      "sigma2_within": 0.3
  }
  ```

- **include_predictors**: `bool` (默认`True`)
  - `True`: Conditional ICC (在预测变量X基础上的ICC)
  - `False`: Unconditional ICC (仅随机截距模型)

---

## 返回值结构

```python
{
    "icc": 0.625,  # Intraclass correlation
    "sigma2_between": 0.186,  # Between-subject variance
    "sigma2_within": 0.263,   # Within-subject variance
    
    # 被试统计
    "n_subjects": 5,
    "n_observations": 50,
    "obs_per_subject": {0: 10, 1: 10, 2: 10, 3: 10, 4: 10},
    "mean_obs_per_subject": 10.0,
    "min_obs_per_subject": 10,
    "max_obs_per_subject": 10,
    
    # 估计准确性 (如果提供了真实值)
    "estimation_accuracy": {
        "true_icc": 0.625,
        "estimated_icc": 0.415,
        "icc_error": 0.210,
        "icc_relative_error": 0.336,
        "true_sigma2_between": 0.5,
        "estimated_sigma2_between": 0.186,
        "sigma2_between_error": 0.314,
        "true_sigma2_within": 0.3,
        "estimated_sigma2_within": 0.263,
        "sigma2_within_error": 0.037,
    },
    
    "model_type": "conditional_model"  # or "null_model"
}
```

---

## 应用场景

### 场景1: EUR vs Random采样对比

评估不同采样策略对群体参数估计的影响:

```python
# EUR采样结果
result_eur = evaluate_variance_components(X_eur, y_eur, subject_ids_eur)

# Random采样结果
result_random = evaluate_variance_components(X_random, y_random, subject_ids_random)

# 对比
print(f"EUR ICC估计: {result_eur['icc']:.3f}")
print(f"Random ICC估计: {result_random['icc']:.3f}")
print(f"EUR估计误差: {result_eur['estimation_accuracy']['icc_error']:.3f}")
print(f"Random估计误差: {result_random['estimation_accuracy']['icc_error']:.3f}")
```

### 场景2: 不均衡被试观测检测

检测采样是否均匀分布于不同被试:

```python
result = evaluate_variance_components(X, y, subject_ids)

print(f"被试数: {result['n_subjects']}")
print(f"平均每被试观测数: {result['mean_obs_per_subject']:.1f}")
print(f"观测数范围: [{result['min_obs_per_subject']}, {result['max_obs_per_subject']}]")

# 检测不均衡
if result['max_obs_per_subject'] > 2 * result['min_obs_per_subject']:
    print("⚠️ 采样严重不均衡!")
```

### 场景3: 验证群体模型假设

检验数据是否符合混合效应模型假设:

```python
result = evaluate_variance_components(X, y, subject_ids)

icc = result['icc']
if icc < 0.05:
    print("⚠️ ICC接近0,被试间差异极小,可能不需要混合模型")
elif icc > 0.8:
    print("⚠️ ICC过高,被试间差异极大,需要更多被试或增加被试内观测")
else:
    print(f"✅ ICC={icc:.3f},适合混合效应模型")
```

---

## 技术细节

### 估计方法

1. **优先使用 statsmodels MixedLM**: REML估计,方法为powell优化
2. **降级处理**: 如果MixedLM收敛失败,自动切换到简单ANOVA估计器

### ICC类型

- **Unconditional ICC** (`include_predictors=False`):
  - 模型: `y ~ 1 + (1|subject)`
  - 含义: 总变异中被试间差异的比例
  
- **Conditional ICC** (`include_predictors=True`):
  - 模型: `y ~ X + (1|subject)`
  - 含义: 控制预测变量后,残差变异中被试间差异的比例

### 边界情况

- **ICC ≈ 0**: 被试间无差异,可能不需要混合模型
- **ICC ≈ 1**: 被试间差异极大,需要更多被试
- **小样本 (n_subjects < 5)**: 估计不稳定,结果仅供参考

---

## 注意事项

1. **最小被试数**: 建议至少3个被试,否则方差估计极不稳定
2. **观测数均衡**: 不均衡设计会影响估计精度
3. **收敛警告**: 如果看到 "MixedLM fitting failed",说明使用了ANOVA降级估计
4. **估计误差**: 小样本下(< 50观测),ICC估计误差可能高达20-30%

---

## 完整测试

运行完整测试套件:

```bash
pixi run python KEY_CESHI/20251212/test/test_variance_components.py
```

测试覆盖:

- ✅ Null模型ICC估计
- ✅ Conditional模型(含预测变量)
- ✅ 不均衡设计处理
- ✅ 小样本降级处理
- ✅ 边界情况 (ICC=0, ICC=1)
- ✅ 估计准确性验证

---

## 引用与参考

- Nakagawa & Schielzeth (2013). "A general and simple method for obtaining R² from generalized linear mixed-effects models"
- Bates et al. (2015). "Fitting Linear Mixed-Effects Models Using lme4"
- statsmodels MixedLM文档: <https://www.statsmodels.org/stable/mixed_linear.html>
