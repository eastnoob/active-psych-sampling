# AEPsych Categorical Transform - 完整查找结果汇总

**生成时间**: 2025-12-11  
**源文件位置**: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py`  
**工作区**: `d:\ENVS\active-psych-sampling`

---

## 📋 生成的文档清单

本次查询已为你生成以下 3 个完整的文档文件：

1. **AEPsych_Categorical_Transform_Analysis.md** - 详细分析文档
   - Categorical 类的完整 `__init__` 实现
   - `_transform` 和 `_untransform` 的详细说明
   - `get_config_options` 的完整流程
   - Bounds 设置原理
   - 特殊配置逻辑
   - 核心问题总结

2. **AEPsych_Categorical_Complete_Source.py** - 完整源代码及注释
   - Categorical 类的全部代码
   - 每个方法都有详细中文注释
   - 问题分析和修复建议
   - 完整的数据流分析

3. **AEPsych_Categorical_QuickRef.md** - 快速参考表
   - 所有方法的简洁总结
   - 参数和返回值表格
   - 关键代码片段
   - 问题快速查询表

4. **AEPsych_Categorical_Problems_and_Fixes.md** - 问题对比与修复
   - 三个核心问题的详细演示
   - 多种修复方案对比
   - 完整的测试用例
   - 实现检查清单

---

## ✅ 查询需求对应表

| 需求 | 位置 | 文档 | 行数 |
|------|------|------|------|
| **1. Categorical 类的完整 __init__ 和主要方法** | 源文件行 23-43 | Analysis.md | 1-37 |
| **2. _transform 的实现** | 源文件行 45-58 | Analysis.md | 42-73 |
| **2. _untransform 的实现** | 源文件行 60-68 | Analysis.md | 75-116 |
| **3. get_config_options 的实现** | 源文件行 70-102 | Analysis.md | 118-194 |
| **4. bounds 的设置方式** | 源文件行 104-165 | Analysis.md | 196-272 |
| **5. 特殊配置逻辑** | 源文件行 1-23 | Analysis.md | 274-349 |

---

## 🎯 核心发现

### Categorical 类的关键点

**源代码路径**:
```
.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py
```

**类定义**:
```python
class Categorical(Transform, StringParameterMixin):
    def __init__(self, indices: list[int], categories: dict[int, list[str]]):
        self.indices = indices              # 分类参数的列位置
        self.categories = categories        # 分类值映射
        self.string_map = self.categories   # 用于 indices_to_str()
```

**关键特性**:
- `indices`: 指定哪些列是分类型的，如 `[0, 2]`
- `categories`: 分类值字典，格式 `{0: ['val1', 'val2', ...]}`
- `string_map`: 直接指向 `categories`，用于 indices → strings 映射

### 三个核心问题

**问题 1** 📍 `get_config_options` 第 97 行
```python
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                  # ^^^^^^^^^^^^^^^^
```
**影响**: 数值型分类如 `[2.8, 4.0, 8.5]` 被错误转换为 `['2.8', '4.0', '8.5']`

**问题 2** 📍 `_transform` 和 `_untransform` 方法
```python
def _untransform(self, X):
    return X.round()  # 仅做四舍五入，无实际映射！
```
**影响**: 假设输入是 indices，但实际可能是实际值

**问题 3** 📍 `ParameterTransformedGenerator` 中的无条件 untransform
```python
return self.transforms.untransform(x)  # 无条件调用！
```
**影响**: 实际值被重复处理，导致 2.8 → 5.6 → 17.0

### Bounds 转换原理

对于 3 个分类选项，boundaries 的设置方式：

```
Index 0 ↔ -0.5 to 0.5
Index 1 ↔  0.5 to 1.5  
Index 2 ↔  1.5 to 2.5
```

**实现**:
```python
if bound == "lb":
    X[0, indices] -= 0.5        # 下界向后
elif bound == "ub":
    X[0, indices] += (0.5 - ε)  # 上界向前
else:  # 完整边界
    X[0, indices] -= 0.5
    X[1, indices] += (0.5 - ε)
```

---

## 📊 方法对应表

| 方法 | 行号 | 功能 | 关键点 |
|------|------|------|--------|
| `__init__` | 23-43 | 初始化 | 设置 indices, categories, string_map |
| `_transform` | 45-58 | 前向转换 | 仅做四舍五入 |
| `_untransform` | 60-68 | 反向转换 | 假设输入是 indices (问题!) |
| `get_config_options` | 70-102 | 配置解析 | element_type=str (问题!) |
| `transform_bounds` | 104-117 | bounds 入口 | 调用 _transform_bounds |
| `_transform_bounds` | 119-165 | bounds 实现 | ±0.5 偏移 |
| `indices_to_str` | (继承) | indices→str | 依赖 categories 类型 |

---

## 🔧 推荐修复优先级

### 🔴 优先级 1: 修复 element_type (必须)

**位置**: `categorical.py:97`

**当前**:
```python
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
```

**修复**:
```python
choices_raw = config.getlist(name, "choices")
try:
    choices = [float(c) for c in choices_raw]
except ValueError:
    choices = choices_raw
cat_dict = {idx: choices}
```

**影响**: ⭐⭐⭐ 高（所有数值分类参数）

---

### 🟠 优先级 2: 幂等化 _untransform (推荐)

**位置**: `categorical.py:60-68`

**当前**:
```python
def _untransform(self, X):
    return X.round()
```

**修复**:
```python
def _untransform(self, X):
    for idx in self.indices:
        if X[0, idx] not in self.categories[idx]:
            # 进行 indices → values 映射
            int_val = int(round(X[0, idx]))
            X[0, idx] = self.categories[idx][int_val]
    return X.round()
```

**影响**: ⭐⭐ 中等（防止双重转换）

---

### 🟡 优先级 3: 修复 ParameterTransformedGenerator (可选)

**位置**: `.pixi/envs/default/Lib/site-packages/aepsych/parameters.py:410`

**方案**: 添加条件性 untransform，检测是否需要应用

**影响**: ⭐ 低（可通过其他方式规避）

---

## 📚 配套文件参考

| 工作区文件 | 说明 |
|-----------|------|
| `AEPsych_Categorical_Transform_Analysis.md` | 完整分析 (464 行) |
| `AEPsych_Categorical_Complete_Source.py` | 完整源代码注释 (500 行) |
| `AEPsych_Categorical_QuickRef.md` | 快速参考 (300 行) |
| `AEPsych_Categorical_Problems_and_Fixes.md` | 问题对比 (550 行) |
| `tools/repair/categorical_numeric_fix/README_FIX.md` | 修复说明 |
| `tools/repair/parameter_transform_skip/README_FIX.md` | 参数转换修复 |
| `extensions/handoff/20251210_categorical_transform_root_issue.md` | 根本问题分析 |

---

## 🎓 学习路径建议

### 快速了解 (10 分钟)
1. 阅读本文档的「核心发现」部分
2. 查看 `AEPsych_Categorical_QuickRef.md` 的表格

### 深入理解 (30 分钟)
1. 阅读 `AEPsych_Categorical_Transform_Analysis.md`
2. 关注各个问题部分的分析

### 完整掌握 (1 小时)
1. 读完 `AEPsych_Categorical_Complete_Source.py` 的代码注释
2. 学习 `AEPsych_Categorical_Problems_and_Fixes.md` 的修复方案
3. 查看修复文件中的测试用例

### 实施修复 (2-3 小时)
1. 按优先级依次实施修复
2. 运行 `Problems_and_Fixes.md` 中的测试用例
3. 验证数值和字符串分类都正常工作

---

## 🔍 快速查询索引

### 如果你想知道...

**"__init__ 方法做了什么?"**
- 见 Analysis.md 的「1. Categorical 类的完整 __init__」部分
- 或 QuickRef.md 的「1. __init__ 方法」部分

**"_transform 和 _untransform 有什么区别?"**
- 见 Complete_Source.py 的 `_transform` 和 `_untransform` 注释
- 核心: 都只做四舍五入，没有实际的索引映射

**"配置中的 choices 怎么被解析的?"**
- 见 Analysis.md 的「3. get_config_options 的实现」
- 问题在 97 行的 `element_type=str`

**"Bounds 怎么被转换的?"**
- 见 Analysis.md 的「4. Bounds 的设置方式」
- 简单总结: indices ±0.5 的偏移

**"为什么会出现数值型分类返回错误的值?"**
- 见 Problems_and_Fixes.md 的「问题 1」和「问题 2」
- 两个问题叠加导致的

**"怎么修复这些问题?"**
- 见 Problems_and_Fixes.md 的「完整对比表」和「修复优先级」
- 三个修复方案，分别针对三个问题

---

## ✨ 关键代码片段速查

### Categorical 的完整初始化
```python
def __init__(self, indices: list[int], categories: dict[int, list[str]]):
    super().__init__()
    self.indices = indices
    self.categories = categories
    self.string_map = self.categories
```

### 配置解析中的问题位置
```python
# 第 97 行，问题在此
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                            # ^^^^^^^^^^^^^^^^
```

### Bounds 转换的核心
```python
if bound == "lb":
    X[0, self.indices] -= 0.5
elif bound == "ub":
    X[0, self.indices] += (0.5 - epsilon)
else:
    X[0, self.indices] -= 0.5
    X[1, self.indices] += (0.5 - epsilon)
```

### 继承的 indices_to_str
```python
def indices_to_str(self, X: np.ndarray) -> np.ndarray:
    obj_arr = X.astype("O")
    if self.string_map is not None:
        for idx, cats in self.string_map.items():
            obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]
    return obj_arr
```

---

## 🎯 查询完成

✅ 已提供的信息：
- [x] Categorical 类的完整 `__init__` 和主要方法
- [x] `_transform` 和 `_untransform` 的完整实现
- [x] `get_config_options` 的完整实现和问题分析
- [x] bounds 的设置方式和原理
- [x] 特殊的配置逻辑（element_type, string_map 等）

✅ 额外提供：
- [x] 三个核心问题的详细分析
- [x] 修复方案和代码示例
- [x] 完整的测试用例
- [x] 优先级排序和影响评估
- [x] 相关文件参考和学习路径

---

## 📞 相关资源

**本工作区的修复文档**:
- `tools/repair/categorical_numeric_fix/` - 数值型分类修复
- `tools/repair/parameter_transform_skip/` - 参数转换跳过修复
- `extensions/handoff/20251210_categorical_transform_root_issue.md` - 根本问题分析

**已集成的修复**:
- `extensions/custom_generators/custom_pool_based_generator.py` - 包含 Fallback 机制

**测试文件**:
- `tests/test_categorical_transform.py` - Categorical 测试
- `test_real_config.py` - 真实配置测试

---

**生成文档**:
1. AEPsych_Categorical_Transform_Analysis.md
2. AEPsych_Categorical_Complete_Source.py  
3. AEPsych_Categorical_QuickRef.md
4. AEPsych_Categorical_Problems_and_Fixes.md
