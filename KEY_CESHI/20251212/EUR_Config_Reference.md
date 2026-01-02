# EUR Acquisition Config Reference

> 源码: `extensions/dynamic_eur_acquisition/eur_anova_multi.py`  
> 模块: `extensions/dynamic_eur_acquisition/modules/dynamic_weights.py`, `local_sampler.py`

---

## 1. Lambda (交互权重) - `DynamicWeightEngine`

**公式**: r_t > τ1 → λ_min | r_t < τ2 → λ_max | 中间线性插值

| 参数 | 默认 | 推荐(20次) | 说明 |
|------|------|-----------|------|
| `use_dynamic_lambda` | True | True | 启用动态 |
| `tau1` | **0.7** | 0.7 | r_t上阈值(高→低λ) |
| `tau2` | **0.3** | 0.3 | r_t下阈值(低→高λ) |
| `lambda_min` | **0.1** | 0.1 | 初期(未收敛) |
| `lambda_max` | **1.0** | 1.0 | 后期(已收敛) |

**设计意图**: 初期聚焦主效应(λ低), 后期探索交互(λ高)

---

## 2. SPS (骨架预测稳定性) - `SPS_Tracker`

**原理**: 追踪模型在骨架点(中心+极端)的预测变化 → r_t ∈ [0,1]

| 参数 | 默认 | 推荐 | 说明 |
|------|------|------|------|
| `use_sps` | True | True | 启用SPS |
| `sps_sensitivity` | 8.0 | 8.0 | tanh放大系数 |
| `sps_ema_alpha` | **0.7** | 0.5-0.7 | EMA权重(高=更平滑) |

**注意**: alpha=0.7时新值权重30%, 当前0.3导致r_t波动大

---

## 3. Gamma (覆盖权重) - `DynamicWeightEngine`

**公式**: n < τ_n_min → γ_max | n ≥ τ_n_max → γ_min | 中间线性

| 参数 | 默认 | 推荐(20次) | 说明 |
|------|------|-----------|------|
| `use_dynamic_gamma` | True | True | 启用动态 |
| `gamma` | 0.3 | 0.3 | 初始值 |
| `gamma_max` | **0.5** | 0.5 | 早期(高覆盖) |
| `gamma_min` | 0.05 | 0.05 | 后期(精细化) |
| `tau_n_min` | 3 | 3 | 开始衰减的样本数 |
| `tau_n_max` | 25 | **total_budget×0.7** | ⚠️必须<total_budget |
| `total_budget` | None | 20 | 自动配置tau_n_max |

**安全刹车**: r_t > tau_safe时增加gamma惩罚

- `tau_safe` = 0.5, `gamma_penalty_beta` = 0.3

---

## 4. 扰动策略 - `LocalSampler`

| 参数 | 默认 | 推荐 | 说明 |
|------|------|------|------|
| `local_jitter_frac` | 0.1 | 0.1 | 扰动幅度(范围%) |
| `local_num` | 4 | 4-6 | 每点扰动数 |
| `use_hybrid_perturbation` | False | **True** | 低水平变量穷举 |
| `exhaustive_level_threshold` | 3 | 3 | 穷举阈值(≤3水平) |

**变量类型配置**:

```ini
variable_types_list = binary, categorical, categorical, categorical, categorical, binary
```

---

## 5. ARD权重 (覆盖度加权) - `CoverageHelper`

**原理**: Gower距离按维度敏感度加权，高敏感维度权重大

| 参数 | 默认 | 说明 |
|------|------|------|
| `ard_weights` | None | 维度权重数组(自动归一化) |

**从BaseGP lengthscales计算**:

```python
lengthscales = [1.94, 1.47, 3.89, 0.44, 4.24, 3.61]
weights = 1.0 / np.array(lengthscales)
weights = weights / weights.sum()  # 归一化
# → [0.117, 0.155, 0.058, 0.520, 0.054, 0.063]
```

**INI配置**:

```ini
ard_weights = [0.117, 0.155, 0.058, 0.520, 0.054, 0.063]
```

---

## 6. 其他关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `main_weight` | 1.0 | 主效应权重(勿改) |
| `fusion_method` | additive | 融合方式 |
| `coverage_method` | min_distance | Gower距离 |
| `interaction_pairs` | None | 指定二阶交互对 |

---

## 7. 常见错误配置

| 错误 | 后果 | 修复 |
|------|------|------|
| tau_n_max > total_budget | γ永不达最小 | tau_n_max = budget×0.7 |
| tau1≈tau2 (如0.55/0.08) | λ几乎恒定 | 保持tau1-tau2≥0.4 |
| lambda范围窄(0.4-0.7) | 动态失效 | 用0.1-1.0 |
| sps_ema_alpha过低(0.3) | r_t波动大 | 用0.5-0.7 |
| 缺少variable_types_list | 扰动类型错误 | 必须配置 |

---

## 7. 20次预算推荐配置

```ini
[EURAnovaMultiAcqf]
# Lambda
use_dynamic_lambda = True
tau1 = 0.7
tau2 = 0.3
lambda_min = 0.1
lambda_max = 1.0

# SPS
use_sps = True
sps_sensitivity = 8.0
sps_ema_alpha = 0.5

# Gamma
use_dynamic_gamma = True
gamma = 0.3
gamma_max = 0.5
gamma_min = 0.05
tau_n_min = 3
tau_n_max = 14
total_budget = 20
tau_safe = 0.5
gamma_penalty_beta = 0.3

# 扰动
use_hybrid_perturbation = True
exhaustive_level_threshold = 3
local_num = 6
local_jitter_frac = 0.1
variable_types_list = binary, categorical, categorical, categorical, categorical, binary

# ARD权重 (从BaseGP: 1/lengthscales归一化)
ard_weights = [0.117, 0.155, 0.058, 0.520, 0.054, 0.063]

# 交互
interaction_pairs = 3,4; 0,1
fusion_method = additive
coverage_method = min_distance
```

---

## 8. 关键文件路径

| 文件 | 路径 |
|------|------|
| 主采集函数 | `extensions/dynamic_eur_acquisition/eur_anova_multi.py` |
| 动态权重引擎 | `extensions/dynamic_eur_acquisition/modules/dynamic_weights.py` |
| 局部采样器 | `extensions/dynamic_eur_acquisition/modules/local_sampler.py` |
| 覆盖度+Gower | `extensions/dynamic_eur_acquisition/modules/coverage.py`, `gower_distance.py` |
| 推荐配置参考 | `extensions/dynamic_eur_acquisition/recommended_config.ini` |
| 当前测试配置 | `test/is_EUR_work/eur_config_basegp_prior.ini` |
