# Bug 描述：AEPsych Categorical Numeric Parameters

**发现日期**: 2025-12-10
**严重程度**: 🔴 高 (影响实验数据正确性)
**状态**: ✅ 已修复 (双保险方案)

---

## 问题摘要

AEPsych 的 `Categorical` transform 无法正确处理数值型 categorical 参数，导致 Server 返回 indices 而非 actual values，影响下游系统(如 Oracle)接收到错误的参数值。

---

## 重现步骤

### 1. 配置文件定义

```ini
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]  # 数值型 choices

[x2_GridModule]
par_type = categorical
choices = [6.5, 8.0]
```

### 2. 实际行为

Server 返回：
```python
{
  'x1_CeilingHeight': [0.0],  # ❌ 返回 index 而非 2.8
  'x2_GridModule': [0.0]      # ❌ 返回 index 而非 6.5
}
```

### 3. 预期行为

Server 应返回：
```python
{
  'x1_CeilingHeight': [2.8],  # ✅ actual value
  'x2_GridModule': [6.5]      # ✅ actual value
}
```

---

## 根本原因

### 代码位置

`.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py:97`

### 问题代码

```python
def get_config_options(cls, config, name, options):
    ...
    if "categories" not in options:
        idx = options["indices"][0]
        cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                         ^^^^^^^^^^^^^^^^
                                                         # ❌ 强制解析为 string!
        options["categories"] = cat_dict
```

### 问题机制

1. **Config 定义**:
   ```ini
   choices = [2.8, 4.0, 8.5]  # 数值列表
   ```

2. **被错误解析为**:
   ```python
   categories = {0: ['2.8', '4.0', '8.5']}  # ❌ 字符串列表!
   ```

3. **Transform 行为**:
   - **如果 string_map 存在**: 返回字符串 `'2.8'` (类型错误)
   - **如果 string_map = None**: 返回 index `0.0` (值错误)

4. **最终结果**:
   - Server 返回 `0.0` 而非 `2.8`
   - Oracle 用 `0.0` 计算，而非正确的 `2.8`

---

## 影响范围

### 受影响的参数类型

- ✅ **Numeric categorical**: 如 `[2.8, 4.0, 8.5]`, `[6.5, 8.0]`
- ✅ **String categorical**: 如 `['Chaos', 'Rotated', 'Strict']` (也受影响)

### 受影响的组件

1. **AEPsych Server**: 返回错误的参数值
2. **Downstream Systems**: 接收错误值
   - Oracle 计算
   - 数据记录
   - 结果分析
3. **Param Validator**: 需要手动修正参数 (性能损失)

### 实验数据影响

- ❌ **实验记录错误**: 记录的参数值不是实际使用的值
- ❌ **结果分析错误**: 基于错误参数的分析会得出错误结论
- ❌ **可重复性问题**: 无法基于记录重现实验

---

## 证据

### Validation Log

来源: `tests/is_EUR_work/00_plans/251206/scripts/results/20251210_103045/debug/aepsych_validation.log`

```
[VALIDATION CHECK] iter 3 - AEPsych返回的原始值:
  x1_CeilingHeight: [np.float64(2.0)]  # ❌ index 2, 应该是 8.5
  x2_GridModule: [np.float64(0.0)]     # ❌ index 0, 应该是 6.5

[参数修正 3] x1_CeilingHeight: 2.0 -> 2.8
[参数修正 3] x2_GridModule: 0.0 -> 6.5
```

### 直接测试

运行 `test_direct_categorical.py` 证明：
- ✅ **Test 2**: Categorical 可以用 float list `[2.8, 4.0, 8.5]` (0 → 2.8 ✅)
- ❌ **Test 3**: 但 AEPsych 创建的是 string list `['2.8', '4.0', '8.5']` (0 → '2.8' ❌)
- ❌ **Test 4**: string_map=None 时完全不转换 (0 → 0.0 ❌)

---

## 修复方案

### 🛡️ 双保险架构

```
┌─────────────────────────────────────────────┐
│  外层防护 (方案A: 修复 AEPsych)               │
│  智能解析：优先 float，fallback 到 string     │
└─────────────────────────────────────────────┘
                ↓ (如果失效)
┌─────────────────────────────────────────────┐
│  内层防护 (方案B: Generator fallback)        │
│  Fallback mapping: indices → actual values  │
└─────────────────────────────────────────────┘
```

### 方案 A: 修复 AEPsych (推荐)

**优点**:
- ✅ 根本解决问题
- ✅ 惠及所有 AEPsych 用户
- ✅ 符合设计原则

**修复**: 参见 `categorical.py.patch`

### 方案 B: Generator Fallback (已集成)

**优点**:
- ✅ 不依赖 AEPsych 版本
- ✅ 自动生效
- ✅ 持久有效

**状态**: 已自动集成到 `CustomPoolBasedGenerator`

---

## 验证测试

修复后运行测试：

```bash
# 验证 AEPsych 修复
pixi run python tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/test_direct_categorical.py

# 验证 Generator fallback
pixi run python tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/test_generator_fallback_mapping.py
```

---

## 相关文档

- [完整诊断报告](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/DIAGNOSIS_AND_TREATMENT_PLAN.md)
- [Root Cause 分析](../../../tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/ROOT_CAUSE_FINAL.md)
- [修复指南](README_FIX.md)
