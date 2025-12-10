# ManualGenerator 坐标系统 & Categorical 参数配置

## 🔑 核心规则 (CRITICAL)

**ManualGenerator points 的坐标系统由 `par_type` 决定，与 choices 是否为 string/numeric 无关**

| par_type | lb/ub 空间 | ManualGenerator points 空间 | 示例 |
|----------|-----------|---------------------------|------|
| `continuous` | 实际值 `[2.8, 8.5]` | 实际值 `[[2.8, 6.5]]` | 离散连续变量 ✓ |
| `categorical` | 索引 `[0, 2]` | 索引 `[[0, 1]]` | 数值或字符串选项 ✓ |

```ini
# ✅ par_type=continuous (discrete values OK)
[x1]
par_type = continuous
values = [2.8, 4.0, 8.5]
lb = 2.8  # actual value space
ub = 8.5

[ManualGenerator]
points = [[2.8, 6.5]]  # actual values ✓

# ✅ par_type=categorical (numeric choices)
[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]
lb = 0  # index space
ub = 2

[ManualGenerator]
points = [[0, 1]]  # indices ✓

# ❌ WRONG: mixed format
points = [[2.8, 1]]  # ERROR: 混合实际值和索引 ✗
```

**结论**:
- `par_type=continuous`: 可使用实际值（即使是离散的 int/float）
- `par_type=categorical`: 必须使用索引 [0, n-1]，无论 choices 是数值还是字符串

---

## 配置规则详解

```ini
# ✅ 正确配置
[common]
lb = [0, 0, ...]        # ← indices (从 0 开始)
ub = [2, 1, ...]        # ← indices (choices 数量 - 1)

[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]  # ← actual values

[ManualGenerator]
points = [[2.8, 6.5, ...]]  # ← actual values
```

```ini
# ❌ 错误配置
[common]
lb = [2.8, 6.5, ...]    # ❌ 不能用 actual values
ub = [8.5, 8.0, ...]    # ❌ 会导致 17.0, 51.2 等错误值
```

---

## 完整示例

```ini
[common]
parnames = ['x1_CeilingHeight', 'x2_GridModule', 'x3_Type']
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, opt_strat]

# ⚠️ 关键：所有 categorical 参数用 indices
lb = [0, 0, 0]  # x1 有 3 个选项 (0,1,2), x2 有 2 个 (0,1), x3 有 3 个 (0,1,2)
ub = [2, 1, 2]  # len(choices) - 1

# Categorical numeric parameter #1
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]  # 3 个选项 → ub=2
lb = 0
ub = 2

# Categorical numeric parameter #2
[x2_GridModule]
par_type = categorical
choices = [6.5, 8.0]  # 2 个选项 → ub=1
lb = 0
ub = 1

# Categorical string parameter
[x3_Type]
par_type = categorical
choices = ['Chaos', 'Rotated', 'Strict']  # 3 个选项 → ub=2
lb = 0
ub = 2

[init_strat]
generator = ManualGenerator

[ManualGenerator]
# ⚠️ CRITICAL: ALL categorical parameters MUST use indices
# par_type=categorical → use indices [0, n-1], regardless of numeric/string choices
points = [
    [0, 0, 2],  # x1=2.8(idx0), x2=6.5(idx0), x3=Strict(idx2)
    [1, 1, 0],  # x1=4.0(idx1), x2=8.0(idx1), x3=Chaos(idx0)
    [2, 0, 1]   # x1=8.5(idx2), x2=6.5(idx0), x3=Rotated(idx1)
]

[opt_strat]
generator = CustomPoolBasedGenerator

[CustomPoolBasedGenerator]
acqf = EURAnovaMultiAcqf
# pool_points 由 server_manager.py 动态注入

[CustomBaseGPResidualMixedFactory]
continuous_params = []
discrete_params = {'x1_CeilingHeight': 3, 'x2_GridModule': 2, 'x3_Type': 3}
basegp_scan_csv = extensions/warmup_budget_check/.../design_space_scan.csv
mean_type = pure_residual
lengthscale_prior = lognormal
ls_loc = []
ls_scale = []

[EURAnovaMultiAcqf]
variable_types_list = categorical, categorical, categorical
```

---

## 配置对照表

| 配置项 | par_type=continuous | par_type=categorical | 注意 |
|--------|-------------------|---------------------|------|
| `choices/values` | `[2.8, 4.0, 8.5]` | `[2.8, 4.0, 8.5]` 或 `['A','B']` | 实际值 |
| `[common] lb/ub` | 实际值 `[2.8, 8.5]` | 索引 `[0, 2]` | ⚠️ 关键区别 |
| `[x*] lb/ub` | 实际值 `2.8 / 8.5` | 索引 `0 / 2` | 与 common 一致 |
| `ManualGenerator points` | 实际值 `[[2.8, 6.5]]` | 索引 `[[0, 1]]` | ⚠️ 必须与 lb/ub 匹配 |
| `discrete_params` | N/A | `{'x1': 3}` (len) | 用于 Model |

---

## 快速诊断

### 问题：AEPsych 返回错误的值

**症状**:
```
AEPsych 返回: x1_CeilingHeight = 17.0
期望值范围: [2.8, 4.0, 8.5]
```

**检查步骤**:

1. **查看 `[common]` 的 lb/ub**
   ```ini
   # 如果是这样 → 错误！
   lb = [2.8, 6.5, ...]
   ub = [8.5, 8.0, ...]

   # 应该是这样 → 正确
   lb = [0, 0, ...]
   ub = [2, 1, ...]
   ```

2. **检查验证日志**
   ```bash
   cat debug/aepsych_validation.log
   ```
   - ✅ 正确: `x1_CeilingHeight: 2.8` (或 4.0, 8.5)
   - ❌ 错误: `x1_CeilingHeight: 17.0` (或其他超出范围的值)

3. **应用修复**
   ```bash
   # 如果已安装修复补丁
   pixi run python tools/repair/parameter_transform_skip/verify_fix.py
   ```

---

## 计算 ub 值

```python
# 快速计算公式
ub_value = len(choices) - 1

# 示例
choices = [2.8, 4.0, 8.5]  # 3 个选项
ub = 3 - 1 = 2  # ✅

choices = [6.5, 8.0]  # 2 个选项
ub = 2 - 1 = 1  # ✅

choices = ['Chaos', 'Rotated', 'Strict', 'Grid']  # 4 个选项
ub = 4 - 1 = 3  # ✅
```

---

## 相关文档

- 详细错误排查: [02_INI_CONFIG_PITFALLS.md](02_INI_CONFIG_PITFALLS.md#错误-10-categorical-numeric-parameters-double-transformation-️-重要)
- 修复补丁: `tools/repair/parameter_transform_skip/`
- 实际配置: `tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini`

---

**最后更新**: 2025-12-10
