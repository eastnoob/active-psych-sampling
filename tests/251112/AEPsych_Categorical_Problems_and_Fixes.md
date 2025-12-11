# AEPsych Categorical Transform - 问题对比与修复方案

## 核心问题演示

### 问题 1: element_type=str 强制转换

**位置**: `categorical.py:97`

#### 原始代码
```python
def get_config_options(cls, config, name=None, options=None):
    # ...
    if "categories" not in options:
        idx = options["indices"][0]
        cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                    # ^^^^^^^^^^^^^^^^
                                                    # 问题在这里！
        options["categories"] = cat_dict
```

#### 问题演示

**配置输入**:
```ini
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]
```

**处理过程**:
```
config.getlist(name, "choices", element_type=str)
  ↓
['2.8', '4.0', '8.5']  # 字符串列表！
```

**期望 vs 实际**:

| 期望 | 实际 | 类型 |
|------|------|------|
| `[2.8, 4.0, 8.5]` | `['2.8', '4.0', '8.5']` | ❌ 字符串 |
| `float` | `str` | ❌ 错误 |

#### 修复方案 A: 自动检测

```python
def get_config_options(cls, config, name=None, options=None):
    # ...
    if "categories" not in options:
        idx = options["indices"][0]
        choices_raw = config.getlist(name, "choices")  # 不强制类型
        
        # 尝试转换为浮点数
        try:
            choices = [float(c) for c in choices_raw]
        except (ValueError, TypeError):
            # 如果失败，保持为字符串
            choices = [str(c) for c in choices_raw]
        
        cat_dict = {idx: choices}  # 保留原始类型
        options["categories"] = cat_dict
```

**修复后**:
```
choices_raw = ['2.8', '4.0', '8.5']
choices = [float(c) for c in choices_raw]
  ↓
choices = [2.8, 4.0, 8.5]  # 数值列表！✓
```

#### 修复方案 B: 类型标记

```python
def get_config_options(cls, config, name=None, options=None):
    # ...
    par_type = config.get(name, "par_type", "continuous")
    
    if par_type == "categorical":
        choices_raw = config.getlist(name, "choices")
        
        # 检查配置中的类型提示
        element_type_hint = config.get(name, "element_type", 
                                      default=None)
        
        if element_type_hint == "float":
            choices = [float(c) for c in choices_raw]
        elif element_type_hint == "int":
            choices = [int(c) for c in choices_raw]
        else:  # 自动检测
            try:
                choices = [float(c) for c in choices_raw]
            except ValueError:
                choices = [str(c) for c in choices_raw]
        
        cat_dict = {idx: choices}
        options["categories"] = cat_dict
```

---

### 问题 2: indices_to_str 返回错误类型

**位置**: `base.py:StringParameterMixin.indices_to_str`

#### 原始代码
```python
def indices_to_str(self, X: np.ndarray) -> np.ndarray:
    obj_arr = X.astype("O")
    
    if self.string_map is not None:
        for idx, cats in self.string_map.items():
            obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]
    
    return obj_arr
```

#### 问题演示

**场景 1: 字符串分类 (正常)**
```python
categories = {0: ['Chaos', 'Rotated', 'Strict']}
X = np.array([[0, 1, 2]])

result = indices_to_str(X)
# result[0] = ['Chaos', 'Rotated', 'Strict']  ✓ 正确
```

**场景 2: 数值分类 (错误)**
```python
# 被 get_config_options 错误地转换为字符串
categories = {0: ['2.8', '4.0', '8.5']}  # 应该是 [2.8, 4.0, 8.5]
X = np.array([[0, 1, 2]])

result = indices_to_str(X)
# result[0] = ['2.8', '4.0', '8.5']  # 字符串！
# 期望:     = [2.8, 4.0, 8.5]  # 数值

# 下游系统期望浮点，收到字符串 ❌
oracle.process(result[0, 0])  # 接收 '2.8' (str)，期望 2.8 (float)
```

#### 修复方案

**修复依赖于问题 1 的修复**。一旦 `categories` 包含正确的类型：

```python
# 修复后的 categories
categories = {0: [2.8, 4.0, 8.5]}  # 数值

# indices_to_str 会正确返回
result = indices_to_str(X)
# result[0] = [2.8, 4.0, 8.5]  # 数值！✓
```

---

### 问题 3: _transform/_untransform 双重转换

**位置**: `parameters.py` (ParameterTransformedGenerator)

#### 问题演示

**调用链**:
```
ParameterTransformedGenerator.gen()
  ↓
x = self._base_obj.gen(...)  # 返回实际值 [2.8, ...]
  ↓
return self.transforms.untransform(x)  # ❌ 无条件调用！
  ↓
Categorical._untransform([2.8])
  ↓
return X.round()  # 只做四舍五入，没有映射
  ↓
最终返回 [2.8]  # 实际上应该返回索引！
```

**具体例子**:

假设配置：
```ini
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]
```

**Generator 返回实际值**:
```python
# CustomPoolBasedGenerator.gen()
return [[2.8, 6.5, 0, 0, ...]]  # x1=2.8 (actual value)
```

**经过 ParameterTransformedGenerator 的处理**:
```python
# ParameterTransformedGenerator.gen()
x = [[2.8, ...]]  # 从 base generator 获得
untransformed = self.transforms.untransform(x)
  ↓
# Categorical._untransform([2.8])
return [2.8].round()  # → [2.8] (没做任何映射！)
```

**问题**:
- 如果后续调用 `indices_to_str([2.8])`
- 会尝试 `categories[int(2.8)]` → 索引 2
- 返回 `categories[0][2]` → `[2.8, 4.0, 8.5][2]` → `'8.5'`
- 实际值变成了错误的值！❌

#### 修复方案 A: 检测输入类型 (幂等化)

```python
def _untransform(self, X: torch.Tensor) -> torch.Tensor:
    """检测输入是否已经是实际值，避免重复映射"""
    
    for idx in self.indices:
        val = X[0, idx].item()
        
        # 检查是否已经是实际值
        if val in self.categories[idx]:
            continue  # 已是实际值，跳过
        
        # 否则，进行 indices → values 映射
        int_val = int(round(val))
        if 0 <= int_val < len(self.categories[idx]):
            X[0, idx] = torch.tensor(self.categories[idx][int_val])
    
    return X.round()
```

#### 修复方案 B: 检查 ParameterTransformedGenerator

```python
# 在 ParameterTransformedGenerator.gen() 中
def gen(self, num_points, model, **kwargs):
    x = self._base_obj.gen(num_points, model, **kwargs)
    
    # 检查 generator 是否已经处理了 transforms
    if hasattr(self._base_obj, 'handles_transforms'):
        if self._base_obj.handles_transforms:
            # 跳过转换，直接返回
            return x
    
    # 否则，应用转换
    return self.transforms.untransform(x)
```

#### 修复方案 C: Generator Fallback (已实现)

在 `CustomPoolBasedGenerator` 中：

```python
def _ensure_actual_values(self, points):
    """检测并修正双重转换导致的错误值"""
    
    for param_idx, mapping in self.categorical_mappings.items():
        actual_values = list(mapping.values())
        
        for i, point in enumerate(points):
            val = point[param_idx]
            
            # 如果值不在实际值列表中
            if val not in actual_values:
                # 尝试作为索引进行映射
                try:
                    int_idx = int(round(val))
                    if int_idx in mapping:
                        points[i][param_idx] = mapping[int_idx]
                except:
                    pass
    
    return points
```

---

## 完整对比表

| 问题 | 原始代码 | 问题描述 | 修复 | 影响范围 |
|------|---------|---------|------|---------|
| **element_type=str** | `get_config_options:97` | 数值被强制转换为字符串 | 自动检测类型 | 所有数值分类参数 |
| **indices_to_str** | `base.py` | 返回错误的类型 | 修复 element_type 问题 | 所有分类参数 |
| **双重 untransform** | `parameters.py` | Generator 输出被重复转换 | 检测输入类型或跳过 | ParameterTransformedGenerator 包装的 generators |
| **Bounds 映射** | `categorical.py:139-148` | 边界转换可能不准确 | 当前实现已可接受 | 模型优化阶段 |

---

## 修复优先级

### 🔴 优先级 1: 修复 element_type=str (必须)

**为什么**: 这是根本原因，影响所有下游处理

**修复位置**: `categorical.py:97`

**复杂度**: 低 (3-5 行代码)

**影响**: 高 (所有数值分类参数)

```python
# 当前
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}

# 修复后
choices_raw = config.getlist(name, "choices")
try:
    choices = [float(c) for c in choices_raw]
except ValueError:
    choices = choices_raw
cat_dict = {idx: choices}
```

### 🟠 优先级 2: 幂等化 _untransform (推荐)

**为什么**: 防止双重转换问题

**修复位置**: `categorical.py:54-68`

**复杂度**: 中等 (10-15 行代码)

**影响**: 中等 (ParameterTransformedGenerator 用户)

```python
def _untransform(self, X):
    # 检查值是否已是实际值
    for idx in self.indices:
        if X[0, idx] not in self.categories[idx]:
            # 进行映射
            pass
    return X.round()
```

### 🟡 优先级 3: ParameterTransformedGenerator 修复 (可选)

**为什么**: 避免在源头就无条件调用 untransform

**修复位置**: `.pixi/envs/default/Lib/site-packages/aepsych/parameters.py:410`

**复杂度**: 中等

**影响**: 低 (可通过其他方式解决)

---

## 测试用例

### 测试 1: 数值分类配置解析

```python
from aepsych.config import Config
from aepsych.transforms.ops import Categorical

config_str = """
[common]
parnames = ['x1']
strategy_names = [test_strat]

[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[test_strat]
min_asks = 1
generator = ManualGenerator
"""

config = Config()
config.update(config_str=config_str)

# 获取 Categorical transform
from aepsych.transforms.parameters import ParameterTransforms
transforms = ParameterTransforms.from_config(config)
cat = transforms._modules['x1']

# 测试：categories 应该包含数值
assert isinstance(cat.categories[0][0], float), \
    f"Expected float, got {type(cat.categories[0][0])}"

# 期望: {'2.8', '4.0', '8.5'} → [2.8, 4.0, 8.5]
print("✓ Test 1 passed: Numeric categories preserved")
```

### 测试 2: 字符串分类配置解析

```python
config_str = """
[x1]
par_type = categorical
choices = [Chaos, Rotated, Strict]
"""

# 期望: {0: ['Chaos', 'Rotated', 'Strict']}
cat.categories[0]  # ['Chaos', 'Rotated', 'Strict']

print("✓ Test 2 passed: String categories work")
```

### 测试 3: indices_to_str 返回正确类型

```python
import numpy as np

# 数值分类
X = np.array([[0, 1, 2]], dtype=object)
result = cat.indices_to_str(X)

# 应该返回 [2.8, 4.0, 8.5]
assert isinstance(result[0, 0], float), \
    f"Expected float, got {type(result[0, 0])}"

print("✓ Test 3 passed: indices_to_str returns correct types")
```

### 测试 4: _untransform 幂等性

```python
import torch

# 测试 untransform 是否幂等
x1 = torch.tensor([[2.8]])  # 实际值
x2 = torch.tensor([[0.0]])   # 索引

result1 = cat._untransform(x1)
result2 = cat._untransform(x2)

# 不应该改变已是实际值的输入
assert torch.allclose(result1, torch.tensor([[2.8]]))

print("✓ Test 4 passed: _untransform is idempotent")
```

---

## 相关代码文件

| 文件 | 行号 | 内容 |
|------|------|------|
| `categorical.py` | 20-41 | `__init__` 方法 |
| `categorical.py` | 43-58 | `_transform` 方法 |
| `categorical.py` | 60-68 | `_untransform` 方法 |
| `categorical.py` | 70-102 | `get_config_options` 方法 |
| `categorical.py` | 104-165 | `transform_bounds` 和 `_transform_bounds` |
| `base.py` | 75-96 | `StringParameterMixin.indices_to_str` |
| `parameters.py` | 394-428 | `ParameterTransformedGenerator.gen()` |

---

## 实现检查清单

- [ ] **问题 1 修复**: 替换 `get_config_options` 中的 `element_type=str`
  - [ ] 自动检测浮点/字符串类型
  - [ ] 添加类型检测逻辑
  - [ ] 添加单元测试
  
- [ ] **问题 2 修复**: 幂等化 `_untransform`
  - [ ] 检测值是否已是实际值
  - [ ] 添加条件映射逻辑
  - [ ] 添加单元测试
  
- [ ] **问题 3 修复**: 修复 ParameterTransformedGenerator（可选）
  - [ ] 添加 `handles_transforms` 标记
  - [ ] 条件性应用转换
  
- [ ] **验证**: 运行所有测试
  - [ ] 数值分类参数
  - [ ] 字符串分类参数
  - [ ] 混合参数配置
  - [ ] 双重转换检测
