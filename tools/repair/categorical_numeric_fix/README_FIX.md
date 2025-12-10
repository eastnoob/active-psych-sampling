# 修复说明：AEPsych Categorical Numeric Parameters Bug

**状态**: ✅ 已修复并验证 (双保险方案)

---

## 快速修复指南

### 1. 验证是否需要修复

运行验证脚本：

```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/categorical_numeric_fix/verify_issue.py
```

**预期结果**：
- ✅ 如果显示 `[SUCCESS] No fix needed` - 已修复，无需操作
- ❌ 如果显示 `[FAIL] Bug present` - 需要修复

---

## 2. 应用修复

### 方法 1: 自动应用（推荐）

运行自动修复脚本：

```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/categorical_numeric_fix/apply_fix.py
```

脚本会自动应用**双保险方案**：
- **方案A**: 修复AEPsych的categorical.py (外层防护)
- **方案B**: 在CustomPoolBasedGenerator中添加fallback mapping (内层防护)

### 方法 2: 手动应用

#### 方案A: 修复AEPsych源码

**步骤 1**: 备份原文件

```bash
cp .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py \
   .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py.backup
```

**步骤 2**: 打开文件

```
.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py
```

**步骤 3**: 找到第95-98行并替换为：

```python
if "categories" not in options:
    idx = options["indices"][0]  # There should only be one index

    # ========== Fix for numeric categorical parameters ==========
    # Try to parse choices as numeric (float) first, fallback to string
    # This allows numeric choices like [2.8, 4.0, 8.5] to be stored as floats
    # instead of strings ['2.8', '4.0', '8.5']
    try:
        # Try numeric parsing
        choices_values = config.getlist(name, "choices", element_type=float)
    except (ValueError, TypeError):
        # Fallback to string parsing (remove quotes if present)
        choices_values = config.getlist(name, "choices", element_type=str)
        choices_values = [s.strip("'\"") for s in choices_values]
    # ========== End of fix ==========

    cat_dict = {idx: choices_values}
    options["categories"] = cat_dict
```

#### 方案B: 已自动集成到CustomPoolBasedGenerator

方案B的fallback映射逻辑已经集成到`extensions/custom_generators/custom_pool_based_generator.py`中，无需手动操作。

---

## 3. 验证修复

再次运行验证脚本：

```bash
pixi run python tools/repair/categorical_numeric_fix/verify_issue.py
```

应该显示：`[SUCCESS] Both fixes working correctly!`

---

## 问题根因

**位置**: AEPsych `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py:97`

**原始代码**:
```python
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                 ^^^^^^^^^^^^^^^^
```

**问题**: `element_type=str` 强制所有choices解析为字符串，包括数值型choices如 `[2.8, 4.0, 8.5]`

**影响**:
- Numeric categorical: Server应返回 `2.8`,实际返回 `0.0` (index) 或 `'2.8'` (string)
- String categorical: Server应返回 `'Chaos'`,实际返回 `0.0` (index)

**证据**: 参见[tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/ROOT_CAUSE_FINAL.md](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/ROOT_CAUSE_FINAL.md)

---

## 修复原理

### 双保险架构

```
┌─────────────────────────────────────────────┐
│  外层防护 (方案A: 修复AEPsych transform)      │
│  Pool[0,1,2] → GP → Transform → 2.8 → Oracle│
└─────────────────────────────────────────────┘
                ↓ (如果失效)
┌─────────────────────────────────────────────┐
│  内层防护 (方案B: Generator fallback mapping)│
│  Pool[0,1,2] → GP → gen() → 2.8 → Oracle    │
└─────────────────────────────────────────────┘
```

### 方案A: 智能解析

修改 `get_config_options()` 方法：
1. 首先尝试 `element_type=float` 解析choices
2. 如果失败（string categorical），fallback到 `element_type=str`
3. 结果：numeric categorical存储为float，string categorical存储为string

### 方案B: Fallback Mapping

在 `CustomPoolBasedGenerator` 中：
1. 在pool generation时存储 `{param_idx: {0: 2.8, 1: 4.0, 2: 8.5}}` 映射
2. 在 `gen()` 返回点时，检查是否需要应用映射
3. 如果检测到indices (0, 1, 2)，自动映射到actual values (2.8, 4.0, 8.5)

---

## 影响范围

修复后，EUR实验中的categorical参数将正常工作：
- Server返回正确的actual values而非indices
- Oracle接收到正确的物理/心理参数值
- 不再需要param_validator修正参数

---

## 文件列表

- `README_FIX.md` - 本文件
- `DIAGNOSIS_REPORT.md` - 完整诊断报告
- `apply_fix.py` - 自动修复脚本
- `verify_issue.py` - 验证脚本
- `categorical.py.patch` - AEPsych修复代码片段
- `generator_fallback.py.patch` - Generator修复代码片段

---

## 回滚方案

如果修复后出现问题：

### 回滚方案A

```bash
cp .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py.backup \
   .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py
```

### 方案B自动生效

即使回滚方案A，方案B的fallback机制仍然会自动生效，确保系统正常运行。

---

## 相关文档

- [完整诊断报告](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/DIAGNOSIS_AND_TREATMENT_PLAN.md)
- [Root Cause分析](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/ROOT_CAUSE_FINAL.md)
- [Pool Generation Fix记录](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/VERIFICATION_RESULTS.md)
