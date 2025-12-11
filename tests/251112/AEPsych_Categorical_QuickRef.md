# AEPsych Categorical Transform - 快速参考

## 源代码位置

```
.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py
```

---

## 1. `__init__` 方法

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `indices` | `list[int]` | 分类参数的列位置 | `[0, 2]` |
| `categories` | `dict[int, list[str]]` | 分类值映射 | `{0: ['2.8', '4.0', '8.5']}` |

**关键赋值**:
```python
self.indices = indices
self.categories = categories
self.string_map = self.categories  # 指向同一对象
```

---

## 2. `_transform` 方法

```python
@subset_transform
def _transform(self, X: torch.Tensor) -> torch.Tensor:
    return X.round()
```

**功能**: 仅对 `indices` 指定的列进行四舍五入（基本上是恒等变换）

**输入**: 实际值或 indices
**输出**: 四舍五入后的输入

---

## 3. `_untransform` 方法

```python
@subset_transform
def _untransform(self, X: torch.Tensor) -> torch.Tensor:
    return X.round()
```

**功能**: 同上

**⚠️ 问题**: 假设输入是 indices，但实际可能已经是实际值

**示例 BUG**:
```
输入: [2.8] (actual value)
假设: 这是 index 2.8
执行: X.round() → 3.0
输出: 3.0 ❌ (应该输出 2.8)
```

---

## 4. `get_config_options` 方法

```python
@classmethod
def get_config_options(cls, config, name=None, options=None):
    options = super().get_config_options(config=config, name=name, options=options)
    
    if name is None:
        raise ValueError(...)
    
    if "categories" not in options:
        idx = options["indices"][0]
        # ⚠️ 问题在这里！
        cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                         # 强制为字符串！
        options["categories"] = cat_dict
    
    if "bounds" in options:
        del options["bounds"]
    
    return options
```

**执行流程**:

1. 调用父类方法 → 设置 `indices`
2. 验证 `name` 非空
3. 提取 `choices` → **转换为字符串**
4. 删除 `bounds`

**🐛 核心问题**:

```
配置：choices = [2.8, 4.0, 8.5]
处理：config.getlist(..., element_type=str)
结果：['2.8', '4.0', '8.5']  # 字符串！

期望：[2.8, 4.0, 8.5]  # 数值
```

---

## 5. Bounds 设置

### `transform_bounds` 入口

```python
def transform_bounds(self, X, bound=None, **kwargs):
    epsilon = kwargs.get("epsilon", 1e-6)
    return self._transform_bounds(X, bound=bound, epsilon=epsilon)
```

### `_transform_bounds` 实现

```python
def _transform_bounds(self, X, bound=None, epsilon=1e-6):
    X = X.clone()
    
    if bound == "lb":
        # 下界：减去 0.5
        X[0, self.indices] -= 0.5
    elif bound == "ub":
        # 上界：加上 (0.5 - epsilon)
        X[0, self.indices] += (0.5 - epsilon)
    else:  # 完整边界
        X[0, self.indices] -= 0.5
        X[1, self.indices] += (0.5 - epsilon)
    
    return X
```

### Bounds 转换示例

对于 3 个分类选项（indices = [0, 1, 2]）：

| Index | Actual Value | Original | After transform_bounds |
|-------|--------------|----------|----------------------|
| 0 | 2.8 | [0, ..., 2] | [-0.5, ..., 1.5] |
| 1 | 4.0 | [0, ..., 2] | [-0.5, ..., 1.5] |
| 2 | 8.5 | [0, ..., 2] | [-0.5, ..., 1.5] |

**原理**:
- 每个 index 占据 1 个单位的空间
- 下界向后偏移 0.5（包含 index）
- 上界向前偏移 0.5-ε（不包含下一个 index）

---

## 6. 继承的特殊方法

### `indices_to_str` (来自 StringParameterMixin)

```python
def indices_to_str(self, X: np.ndarray) -> np.ndarray:
    obj_arr = X.astype("O")  # 转换为 object 类型
    
    if self.string_map is not None:
        for idx, cats in self.string_map.items():
            # 关键：用 int(i) 索引 cats 列表
            obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]
    
    return obj_arr
```

**功能**: indices → strings 的映射

**示例**:

```python
# 字符串分类
categories = {0: ['Chaos', 'Rotated', 'Strict']}
X = np.array([[0, 1, 2]])

indices_to_str(X)
# 返回: [['Chaos', 'Rotated', 'Strict']]

# 数值分类（被错误转换）
categories = {0: ['2.8', '4.0', '8.5']}  # 字符串！
X = np.array([[0, 1, 2]])

indices_to_str(X)
# 返回: [['2.8', '4.0', '8.5']]  # 字符串！❌
# 期望: [[2.8, 4.0, 8.5]]  # 数值 ✓
```

---

## 7. 类属性

```python
class Categorical(Transform, StringParameterMixin):
    is_one_to_many = False          # 非一对多转换
    transform_on_train = True       # 训练时应用
    transform_on_eval = True        # 评估时应用
    transform_on_fantasize = True   # fantasize 时应用
    reverse = False                 # 无反向转换
```

---

## 问题总结表

| 问题 | 位置 | 原因 | 影响 |
|------|------|------|------|
| **元素类型强制** | `get_config_options:97` | `element_type=str` | 数值分类变字符串 |
| **未映射 indices** | `_untransform:62-68` | 仅做四舍五入 | 没有实际的 indices→values 映射 |
| **双重转换** | `ParameterTransformedGenerator` | 无条件调用 `untransform` | 2.8 → 5.6 → 17.0 ❌ |

---

## 推荐修复方案

### 方案 1: 自动检测类型 ⭐ 推荐

在 `get_config_options` 第 97 行：

```python
if "categories" not in options:
    idx = options["indices"][0]
    choices_raw = config.getlist(name, "choices")
    
    # 尝试转换为浮点
    try:
        choices = [float(c) for c in choices_raw]
    except (ValueError, TypeError):
        choices = choices_raw  # 保持为字符串
    
    cat_dict = {idx: choices}
    options["categories"] = cat_dict
```

### 方案 2: 使 untransform 幂等

```python
def _untransform(self, X: torch.Tensor) -> torch.Tensor:
    for idx in self.indices:
        if X[0, idx] in self.categories[idx]:
            continue  # 已是实际值
        # 否则进行映射
    return X.round()
```

### 方案 3: Generator Fallback（已集成）

在 `CustomPoolBasedGenerator` 中检测并自动映射 indices

---

## 关键代码片段查询

### 如何找到配置解析位置?

```bash
# 搜索 choices 关键字
grep -n "choices" .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py

# 结果: Line 97
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
```

### 如何查看完整类定义?

```python
from aepsych.transforms.ops import Categorical
import inspect

print(inspect.getsource(Categorical))
```

### 如何测试修复?

```python
from aepsych.transforms.ops import Categorical
import torch

# 创建数值分类
cat = Categorical(
    indices=[0],
    categories={0: [2.8, 4.0, 8.5]}  # 应该是这样
)

# 测试
x = torch.tensor([[0.0, 1.0, 2.0]])
transformed = cat.transform(x)
untransformed = cat.untransform(transformed)

print(untransformed)  # 应该能正确恢复
```

---

## 相关文档参考

| 文件 | 内容 |
|------|------|
| `tools/repair/categorical_numeric_fix/README_FIX.md` | 详细修复说明 |
| `tools/repair/parameter_transform_skip/README_FIX.md` | 参数转换跳过修复 |
| `extensions/handoff/20251210_categorical_transform_root_issue.md` | 根本问题分析 |
| `tests/test_categorical_transform.py` | 测试脚本 |
| `AEPsych_Categorical_Complete_Source.py` | 完整源代码（本工作区） |
