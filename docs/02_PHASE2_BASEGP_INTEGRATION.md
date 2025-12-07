# 02 Phase 2 - BaseGP 先验集成

**目标**: 使用 Phase 1 训练的 BaseGP 作为先验,进行残差学习 (Residual Learning)

---

## 核心概念

**残差学习**: 新 GP 只学习 `y - BaseGP_mean(x)`,利用 BaseGP 的群体知识,用少量数据(如30点)快速学习个体差异。

---

## 必需的 BaseGP 产出

从 `extensions/warmup_budget_check/phase1_analysis_output/{timestamp}/step3/` 获取:

| 文件 | 必需? | 用途 | 使用位置 |
|------|------|------|---------|
| `base_gp_key_points.json` | ✅ 必需 | 3个黄金初始化点 (best/worst/max_std) | `run_*.py` → `initialize_server()` |
| `design_space_scan.csv` | ✅ 必需 | 设计空间的 BaseGP 预测 (先验均值) | `*.ini` → `[Factory]` → `basegp_scan_csv` |
| `base_gp_lengthscales.json` | 📋 参考 | 因子敏感性排序 | 调整 `ard_weights` / `interaction_pairs` |
| `base_gp_encodings.json` | ⚠️ 重要 | 类别变量编码映射 | 确保 INI 配置的类别顺序一致 |
| `base_gp_subject_stats.json` | 📋 参考 | 被试间变异性统计 | 理解数据结构 |

---

## 配置示例

### 1. Python 脚本配置

```python
# 文件: run_eur_residual.py
basegp_keypoints_path = (
    PROJECT_ROOT / "extensions/warmup_budget_check/phase1_analysis_output"
    / "202512072040/step3/base_gp_key_points.json"
)
```

### 2. INI 配置

```ini
# 文件: eur_config_residual.ini
[CustomBaseGPResidualMixedFactory]
basegp_scan_csv = extensions/warmup_budget_check/phase1_analysis_output/202512072040/step3/design_space_scan.csv
mean_type = pure_residual

[ConfigurableGaussianLikelihood]
# 噪声先验 (可选,从 BaseGP 训练日志提取最终 noise 值)
noise_prior_concentration = 2.0
noise_prior_rate = 1.228
noise_init = 0.814  # 或使用新 BaseGP 的收敛值 (如 0.284)
```

---

## design_space_scan.csv 格式

**列名要求**: 自动检测 `x\d+` 模式 (如 `x1_*`, `x2_*`),**必须包含** `pred_mean`

```csv
x1_CeilingHeight,x2_GridModule,x3_OuterFurniture,...,pred_mean,pred_std
2.8,6.5,2,...,0.99,0.55
4.0,8.0,1,...,1.80,0.56
```

**兼容性**:

- ✅ 任意 `x1_*`, `x2_*` 后缀 (如 `x1_binary`, `x1_CeilingHeight`)
- ✅ 可含 `Condition_ID` 列(会被自动忽略)
- ✅ 特征数量自适应 (自动检测)

---

## 代码实现

**自动列名检测** ([custom_basegp_prior_mean.py](../extensions/custom_mean/custom_basegp_prior_mean.py)):

```python
import re
feature_cols = sorted(
    [col for col in df.columns if re.match(r'^x\d+', col)],
    key=lambda x: int(re.match(r'^x(\d+)', x).group(1))
)
```

**特点**:

- 按数字排序 (避免 x10 排在 x2 前面)
- 无需硬编码列名
- 向后兼容旧格式

---

## 更新 BaseGP 检查清单

切换到新 BaseGP 时:

- [ ] 更新 `run_*.py` 中的 `basegp_keypoints_path`
- [ ] 更新 `*.ini` 中的 `basegp_scan_csv`
- [ ] 验证新 CSV 包含 `pred_mean` 列
- [ ] 验证 `base_gp_encodings.json` 与 INI 的类别顺序一致
- [ ] (可选) 从 `base_gp_report.md` 提取最终 noise 值更新 `noise_init`
- [ ] (可选) 基于 `base_gp_lengthscales.json` 调整 `ard_weights`
- [ ] (可选) 优化 `interaction_pairs` (避免低敏感因子组合)

---

## EUR 采集函数配置 (基于 BaseGP 系统性优化)

### 1️⃣ ARD 权重计算公式

**目标**: 根据 BaseGP lengthscales 分配探索优先级

**公式**:
```python
# 从 base_gp_lengthscales.json 读取
lengthscales = [l1, l2, ..., ln]

# 反向加权 (lengthscale 越小 → 敏感性越高 → 权重越大)
raw_weights = [1/l for l in lengthscales]

# 归一化到 [0, 1]
ard_weights = [w / sum(raw_weights) for w in raw_weights]
```

**实例** (当前 BaseGP):
```
Lengthscales: [0.749, 0.960, 0.134, 0.860, 5.004, 4.336]
Raw weights:  [1.34,  1.04,  7.46,  1.16,  0.20,  0.23 ]
Normalized:   [0.20,  0.15,  0.35,  0.18,  0.02,  0.10 ]
```

**阈值判断**:
- `lengthscale < 1.0`: 高敏感 (权重 > 0.15)
- `1.0 ≤ lengthscale < 3.0`: 中等 (权重 0.05-0.15)
- `lengthscale ≥ 4.0`: 低敏感 (权重 < 0.05, 考虑忽略)

---

### 2️⃣ Lambda_max 计算逻辑

**目标**: 根据主效应强度决定交互探索上限

**诊断指标**:
```python
# 从 base_gp_lengthscales.json 计算
lengthscales_sorted = sorted(lengthscales)

# 主效应信号强度 (最敏感因子的相对敏感度)
main_effect_strength = lengthscales_sorted[-1] / lengthscales_sorted[0]

# 示例: 5.004 / 0.134 = 37.3 (主效应极强)
```

**Lambda_max 调整公式**:
```python
# Phase 1 推荐值 (baseline)
lambda_baseline = phase1_phase2_config["lambda_max"]

# 修正系数 (主效应越强 → lambda 越低)
if main_effect_strength > 20:
    lambda_max = lambda_baseline * 1.2   # 主效应极强,适度提升交互探索
elif main_effect_strength > 10:
    lambda_max = lambda_baseline * 1.5
elif main_effect_strength > 5:
    lambda_max = lambda_baseline * 1.8
else:
    lambda_max = lambda_baseline * 2.0   # 主效应弱,优先探索交互

# 示例: 0.50 * 1.2 = 0.6
```

**经验阈值**:
- BaseGP 主效应强 (`ratio > 20`) → `lambda_max ∈ [0.5, 0.7]`
- BaseGP 主效应中等 (`ratio 5-20`) → `lambda_max ∈ [0.7, 1.0]`
- BaseGP 主效应弱 (`ratio < 5`) → `lambda_max ∈ [1.0, 1.5]`

---

### 3️⃣ Gamma_min 计算逻辑

**目标**: 根据 BaseGP 不确定性决定最终探索需求

**诊断指标**:
```python
# 从 design_space_scan.csv 计算
pred_std_mean = df["pred_std"].mean()
pred_std_cv = df["pred_std"].std() / pred_std_mean

# 示例: mean=0.56, cv=0.02 (不确定性均匀且稳定)
```

**Gamma_min 调整公式**:
```python
# Phase 1 推荐值
gamma_baseline = phase1_phase2_config["gamma_end"]

# 修正系数 (不确定性越低 → 探索需求越小)
if pred_std_cv < 0.05 and pred_std_mean < 0.6:
    gamma_min = gamma_baseline * 1.3   # 先验强,减少探索
elif pred_std_cv < 0.10:
    gamma_min = gamma_baseline * 1.1
else:
    gamma_min = gamma_baseline * 1.0   # 先验弱,保持探索

# 示例: 0.06 * 1.3 ≈ 0.08
```

**经验阈值**:
- BaseGP 不确定性低 (`std_mean < 0.6, cv < 0.05`) → `gamma_min ∈ [0.08, 0.10]`
- BaseGP 不确定性中等 → `gamma_min ∈ [0.10, 0.15]`
- BaseGP 不确定性高 (`std_mean > 1.0`) → `gamma_min ∈ [0.15, 0.20]`

---

### 4️⃣ Interaction_pairs 筛选算法

**目标**: 选择最有价值的交互对,避免无效组合

**算法**:
```python
# Step 1: 计算所有可能交互对的"潜在价值"
scores = {}
for i in range(n):
    for j in range(i+1, n):
        # 价值 = 两因子敏感度的调和平均 (避免一高一低的组合)
        harmonic_mean = 2 / (lengthscales[i] + lengthscales[j])
        scores[(i,j)] = harmonic_mean

# Step 2: 过滤低敏感因子组合
# 规则: 两因子中至少一个 lengthscale < 2.0
valid_pairs = [
    (i,j) for (i,j), score in scores.items()
    if min(lengthscales[i], lengthscales[j]) < 2.0
]

# Step 3: 排序并选择 Top-K
top_k_pairs = sorted(valid_pairs, key=lambda p: scores[p], reverse=True)[:3]

# Phase 1 推荐优先 (如果在 valid_pairs 中)
phase1_pairs = phase1_phase2_config["interaction_pairs"]
final_pairs = phase1_pairs + [p for p in top_k_pairs if p not in phase1_pairs]
```

**实例** (当前 BaseGP):
```
候选对:
  (2,3): 2/(0.134+0.860) = 2.01  ← 最高 (x3*x4)
  (0,2): 2/(0.749+0.134) = 2.27  ← 次高 (x1*x3)
  (0,1): 2/(0.749+0.960) = 1.17  ← Phase 1 推荐
  (2,3): Phase 1 推荐 ✓
  (4,5): 2/(5.004+4.336) = 0.21  ✗ 过滤 (两因子均不敏感)

最终: 2,3; 0,1; 0,2  (或 2,3; 0,1; 1,3)
```

**过滤规则**:
- ❌ 两因子 lengthscales 均 > 3.0
- ❌ 调和平均 < 0.5 (潜在价值过低)
- ✅ 优先保留 Phase 1 推荐的交互对

---

### 5️⃣ Tau_n 预算对齐公式

**目标**: Gamma 衰减区间必须匹配实际 EUR 预算

**公式**:
```python
# 实际 EUR 预算 (扣除 warmup)
actual_budget = total_budget - n_warmup_points

# Gamma 开始衰减点 (30% 进度)
tau_n_min = int(actual_budget * 0.3)

# Gamma 完全衰减点 (80-90% 进度)
tau_n_max = int(actual_budget * 0.85)

# 示例: budget=30, warmup=3 → actual=27
# tau_n_min = 27 * 0.3 = 8
# tau_n_max = 27 * 0.85 = 23
```

**关键检查**:
- ⚠️ `tau_n_max > actual_budget` → 衰减逻辑失效!
- ✅ `tau_n_min < tau_n_max < actual_budget`

---

### 6️⃣ 噪声先验更新 (可选)

**目标**: 使用 BaseGP 收敛噪声值作为 Phase 2 初始化

**提取方法**:
```python
# 从 base_gp_report.md 提取最终训练噪声
final_noise = 0.284  # 示例: Iter 200, Noise = 2.836e-01

# Gamma 先验参数 (匹配 noise 均值和方差)
# 假设 noise ~ Gamma(concentration, rate)
# E[noise] = concentration / rate = final_noise
# Var[noise] = concentration / rate^2 (控制先验强度)

# 经验配置 (中等先验强度)
noise_init = final_noise
noise_prior_concentration = 2.0
noise_prior_rate = noise_prior_concentration / final_noise

# 示例: rate = 2.0 / 0.284 = 7.04
```

---

### 📋 完整配置示例

```ini
[EURAnovaMultiAcqf]
# 交互对: 调和平均 Top-3 (过滤 lengthscale>3 的组合)
interaction_pairs = 2,3; 0,1; 1,3

# Lambda: baseline * correction_factor
lambda_min = 0.1
lambda_max = 0.6        # 0.50 * 1.2 (主效应强,ratio=37.3)
tau1 = 0.7

# Gamma: baseline * uncertainty_factor
gamma = 0.30
gamma_max = 0.40
gamma_min = 0.08        # 0.06 * 1.3 (不确定性低,cv=0.02)
tau_n_min = 8           # 27 * 0.30
tau_n_max = 24          # 27 * 0.85
total_budget = 30

# ARD: 归一化 1/lengthscale
ard_weights = [0.20, 0.15, 0.35, 0.18, 0.02, 0.10]
```

```ini
[ConfigurableGaussianLikelihood]
noise_prior_concentration = 2.0
noise_prior_rate = 7.04             # 2.0 / 0.284
noise_init = 0.284                  # 从 BaseGP 最终噪声提取
```

---

### ⚠️ 常见误区

| 错误做法 | 正确做法 |
|---------|---------|
| `ard_weights` 全设为均匀 | 必须基于 lengthscales 反向加权 |
| `lambda_max = 1.0` (固定值) | 根据主效应强度动态调整 |
| `total_budget = 100` (与脚本不一致) | 必须匹配实际 EUR 预算 |
| 包含 x5*x6 交互对 | 过滤两个低敏感因子的组合 |
| `tau_n_max = 70` (超出预算) | 确保 `tau_n_max < actual_budget` |

---

## Phase 2 分析产出 (Step2)

从 `extensions/warmup_budget_check/phase1_analysis_output/{timestamp}/step2/` 获取:

| 文件 | 用途 |
|------|------|
| `phase1_phase2_config.json` | Phase 2 推荐参数 (λ, γ, 交互对, 预算) |
| `phase1_analysis_report.txt` | 主效应和交互效应分析报告 |

**关键参数**:
- `interaction_pairs`: 筛选出的显著交互对
- `lambda_max`: 交互权重上限
- `gamma_init`: 探索覆盖初始权重
- `phase2_n_subjects`, `phase2_trials_per_subject`: 推荐预算

---

## BaseGP 敏感性参考

**用途**: 指导 `interaction_pairs` 和 `ard_weights` 配置

从 `base_gp_lengthscales.json` 提取:

```json
{
  "x3_OuterFurniture": 0.134,    // 最敏感 ← 优先探索
  "x1_CeilingHeight": 0.749,     // 高敏感
  "x4_VisualBoundary": 0.860,    // 中等
  "x2_GridModule": 0.960,        // 中等
  "x5_PhysicalBoundary": 5.004,  // 最不敏感 ← 低优先级
  "x6_InnerFurniture": 4.336     // 不敏感
}
```

**规律**: lengthscale 越小 → 越敏感 → 越值得探索

---

## 常见问题

**Q: 列名不匹配报错?**
A: 升级代码到支持自动检测的版本 (2024-12-07 之后)

**Q: 是否需要转换参数空间?**
A: 不需要,默认假设 BaseGP 与 Phase 2 使用相同编码

**Q: 多个 BaseGP 版本如何选择?**
A: 对比 `base_gp_report.md` 中的训练损失和预测范围,选择收敛更好的

---

**相关文档**: [01_WARMUP_BUDGET.md](./01_WARMUP_BUDGET.md) (假设存在)
