# 02 Phase 2 - BaseGP 先验集成

**目标**: 残差学习 - 新 GP 学习 `y - BaseGP_mean(x)`,用少量数据(~30点)快速学习个体差异

---

## BaseGP 输出文件

从 `extensions/warmup_budget_check/phase1_analysis_output/{timestamp}/step3/`:

| 文件 | 必需? | 用途 | 配置位置 |
|------|------|------|----------|
| `base_gp_key_points.json` | ✅ | 3黄金点 (best/worst/max_std) | `run_*.py` → `basegp_keypoints_path` <br> `*.ini` → `[ManualGenerator]` → `points` |
| `design_space_scan.csv` | ✅ | 设计空间先验均值 | `*.ini` → `[Factory]` → `basegp_scan_csv` |
| `base_gp_lengthscales.json` | 📋 | 因子敏感性排序 | `*.ini` → `[Acqf]` → `ard_weights`, `interaction_pairs` |
| `base_gp_encodings.json` | ⚠️ | 类别变量编码 | 验证 INI 类别顺序一致 |
| `base_gp_report.md` | 📋 | 训练摘要 (noise等) | `*.ini` → `[Likelihood]` → `noise_init` |

---

## ⚙️ 配置验证与对齐检查

### 必查项 (切换新 BaseGP 时)

| 检查项 | BaseGP 来源 | EUR 配置目标 | 验证方法 |
|--------|------------|--------------|----------|
| **黄金初始化点** | `base_gp_key_points.json` | `*.ini` → `[ManualGenerator]` → `points` | 坐标完全一致 (含x1~x6顺序) |
| **先验均值路径** | `design_space_scan.csv` | `*.ini` → `[Factory]` → `basegp_scan_csv` | 路径存在且有 `pred_mean` 列 |
| **ARD 权重** | `base_gp_lengthscales.json` | `*.ini` → `[Acqf]` → `ard_weights` | `normalize(1/lengthscales)` |
| **交互对筛选** | `base_gp_lengthscales.json` | `*.ini` → `[Acqf]` → `interaction_pairs` | 避免两低敏感因子组合 |
| **Gamma 参数** | `design_space_scan.csv` | `*.ini` → `[Acqf]` → `gamma_min/max` | 基于不确定性调整 |
| **预算对齐** | 脚本 `budget` | `*.ini` → `[Acqf]` → `total_budget`, `tau_n_max` | `tau_n_max < actual_budget` |

### 常见对齐错误

| ❌ 错误 | ✅ 正确 | 影响 |
|---------|---------|------|
| 黄金点坐标 `[2.8, 8.0, 2, 1, 0, 1]` | `[2.8, 6.5, 2, 2, 0, 0]` (从 JSON 提取) | Warmup 初始化错误,降低效率 |
| `ard_weights = [均匀分布]` | `[0.084, 0.106, 0.194, 0.354, ...]` | 忽略因子敏感性,探索低效 |
| `gamma_min = 0.06` (无 BaseGP) | `0.10` (残差 BaseGP) | 探索不足,漏检效应 |
| `tau_n_max = 70` | `< actual_budget` (如 24) | Gamma 衰减失效 |
| 交互对含 `(4,5)` (x5*x6) | 过滤低敏感组合 | 浪费预算探索无用交互 |

### 验证脚本示例

```python
# 验证黄金点对齐
import json
with open('phase1_analysis_output/{timestamp}/step3/base_gp_key_points.json') as f:
    keypoints = json.load(f)

expected = [
    list(keypoints['x_best_prior'].values()),
    list(keypoints['x_worst_prior'].values()),
    list(keypoints['x_max_std'].values())
]
print("INI [ManualGenerator] points 应该是:")
for p in expected:
    print(f"  {p}")
```

---

## CSV 格式要求

**`design_space_scan.csv`** 必须包含 `pred_mean` 列,特征列自动检测 `x\d+` 模式:

```csv
x1_CeilingHeight,x2_GridModule,...,pred_mean,pred_std
2.8,6.5,...,0.99,0.55
```

✅ 支持任意后缀 (`x1_binary`, `x1_CeilingHeight`)
✅ 忽略 `Condition_ID` 列
✅ 自适应特征数量

**代码实现** ([custom_basegp_prior_mean.py](../extensions/custom_mean/custom_basegp_prior_mean.py)):
```python
import re
feature_cols = sorted([col for col in df.columns if re.match(r'^x\d+', col)],
                      key=lambda x: int(re.match(r'^x(\d+)', x).group(1)))
```

---

## EUR 参数计算 (基于 BaseGP)

| 参数 | 公式 | 数据来源 | 经验阈值 |
|------|------|----------|----------|
| **ARD 权重** | `normalize([1/l for l in lengthscales])` | `base_gp_lengthscales.json` | 高敏感(l<1.0): w>0.15 <br> 中等(1-3): w=0.05-0.15 <br> 低敏感(l>4): w<0.05 |
| **Lambda_max** | `baseline * factor` <br> factor = 1.2 (ratio>20) <br> factor = 1.5 (ratio 10-20) <br> factor = 1.8 (ratio 5-10) <br> factor = 2.0 (ratio<5) | `lengthscales_sorted[-1] / [0]` | 主效应强: 0.5-0.7 <br> 中等: 0.7-1.0 <br> 弱: 1.0-1.5 |
| **Gamma_min** | `baseline * factor` <br> factor = 1.3 (cv<0.05, mean<0.6) <br> factor = 1.1 (cv<0.10) <br> factor = 1.0 (其他) | `design_space_scan.csv` <br> `pred_std` 均值/变异系数 | 不确定性低: 0.08-0.10 <br> 中等: 0.10-0.15 <br> 高: 0.15-0.20 |
| **Interaction_pairs** | 调和平均 Top-K: <br> `score = 2/(l[i]+l[j])` <br> 过滤: `min(l[i],l[j]) < 2.0` | `base_gp_lengthscales.json` | ❌ 均 > 3.0 <br> ❌ score < 0.5 <br> ✅ Phase 1 推荐优先 |
| **Tau_n** | `tau_n_min = actual_budget * 0.3` <br> `tau_n_max = actual_budget * 0.85` <br> `actual_budget = total - warmup` | 脚本 `budget` 参数 | ⚠️ 确保 `tau_n_max < actual_budget` |
| **Noise 先验** | `noise_init = final_noise` <br> `rate = concentration / final_noise` | `base_gp_report.md` <br> 最终训练噪声 | 中等先验: concentration=2.0 |

### 计算示例 (202512081445 BaseGP)

```python
# ARD 权重
lengthscales = [5.482, 4.341, 2.367, 1.298, 3.648, 3.365]
raw = [1/l for l in lengthscales]  # [0.182, 0.230, 0.422, 0.770, 0.274, 0.297]
ard_weights = [w/sum(raw) for w in raw]  # [0.084, 0.106, 0.194, 0.354, 0.126, 0.137]

# Lambda_max
ratio = 5.482 / 1.298 = 4.22  # 主效应弱
lambda_max = 0.50 * 2.0 = 1.0

# Gamma_min
pred_std_mean = 0.56, cv = 0.02  # 不确定性低
gamma_min = 0.06 * 1.3 = 0.08

# Interaction_pairs
scores = {(3,2): 2.01, (0,3): 1.53, (0,1): 1.02, (4,5): 0.21}
valid = [(3,2), (0,3), (0,1)]  # 过滤 (4,5)

# Tau_n
actual_budget = 30 - 3 = 27
tau_n_min = 27 * 0.3 = 8
tau_n_max = 27 * 0.85 = 23
```

### 完整 INI 配置

```ini
[EURAnovaMultiAcqf]
interaction_pairs = 3,2; 0,1; 0,3
lambda_min = 0.1
lambda_max = 1.0
gamma = 0.30
gamma_min = 0.08
tau_n_min = 8
tau_n_max = 23
total_budget = 30
ard_weights = [0.084, 0.106, 0.194, 0.354, 0.126, 0.137]

[ConfigurableGaussianLikelihood]
noise_init = 0.568  # 从 base_gp_report.md
noise_prior_concentration = 2.0
noise_prior_rate = 3.52  # 2.0 / 0.568
```

---

## Phase 1 Step2 产出 (可选参考)

从 `extensions/warmup_budget_check/phase1_analysis_output/{timestamp}/step2/`:

- `phase1_phase2_config.json`: 交互对推荐、λ/γ baseline、预算建议
- `phase1_analysis_report.txt`: 主效应和交互效应分析

⚠️ Step2 推荐需结合 Step3 BaseGP lengthscales 调整

---

## 快速问答

| 问题 | 答案 |
|------|------|
| CSV 列名不匹配? | 升级到 2024-12-07 后版本 (自动检测 `x\d+`) |
| 需要转换参数空间? | 否,默认 BaseGP 与 Phase 2 同编码 |
| 多个 BaseGP 版本如何选? | 对比 `base_gp_report.md` 训练 loss,选收敛好的 |
| 黄金点从哪来? | `base_gp_key_points.json` → `[ManualGenerator]` → `points` |
| ARD 权重怎么算? | `normalize([1/l for l in lengthscales])` |
| Gamma 参数调整依据? | `design_space_scan.csv` 的 `pred_std` 均值和变异系数 |

---

## ⚠️ 开发陷阱提醒

### 评估时的数据来源

在 BaseGP 残差学习场景下，进行效应恢复评估时需要注意数据来源一致性：

| 数据来源 | 包含范围 | 适用场景 |
|---------|---------|----------|
| `model.train_inputs[0]` | 仅最近 warmup 数据 (如3个黄金点) | ❌ 不适合完整训练历史评估 |
| `model.train_targets` | 仅最近 warmup 数据 | ❌ 不适合完整训练历史评估 |
| `logs["x_points"]` | warmup + EUR 采样点 (完整历史) | ✅ 效应恢复评估 |
| `logs["y_values"]` | warmup + EUR 采样点 (完整历史) | ✅ 效应恢复评估 |

**常见错误**：
```python
# ❌ 错误：混用不同数据源导致维度不匹配
train_X = model.train_inputs[0].cpu().numpy()  # 形状: (3, 6) - 仅 warmup
train_y = np.array(logs["y_values"])          # 形状: (10,) - 完整历史
# ValueError: Found input variables with inconsistent numbers of samples: [3, 10]
```

**正确做法**：
```python
# ✅ 正确：统一使用 logs 获取完整训练历史
train_X = np.array(logs["x_points"])   # 形状: (10, 6) - 完整历史
train_y = np.array(logs["y_values"])   # 形状: (10,) - 完整历史
# 维度一致: [10, 10]
```

**原因**：在残差学习中，`model.train_targets` 只保存最近的 warmup 数据（如3个黄金点），而不包含后续 EUR 采样点。完整的训练历史存储在采样循环返回的 `logs` 字典中。

**参考实现**：
- [evaluation_v2.py](../tests/is_EUR_work/00_plans/251206/scripts/modules/evaluation_v2.py#L96-L107) - 正确的数据获取方式
- [run_eur_residual.py](../tests/is_EUR_work/00_plans/251206/scripts/run_eur_residual.py#L380-L387) - 传递 logs 参数

---

**相关文档**: [02_INI_CONFIG_PITFALLS.md](./02_INI_CONFIG_PITFALLS.md)
