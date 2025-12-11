# AEPsych Categorical Transform 实现分析

**源文件路径**: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py`

---

## 1. Categorical 类的完整 `__init__` 和主要方法

### `__init__` 方法

```python
def __init__(
    self,
    indices: list[int],
    categories: dict[int, list[str]],
) -> None:
    """Initialize a categorical transform. The transform itself does not
    change the tensors. Instead, this class allows passing in NumPy object
    arrays where the categorical values are stored as strings. This provides
    a convenient API to turn mixed categorical/continuous data into the
    expected form for models.

    Args:
        indices (list[int]): The indices of the inputs that are categorical.
        categories (dict[int, list[str]]): A dictionary mapping indices to
            the list of categories for that input. There must be a list for
            each index in `indices`.
    """
    super().__init__()
    self.indices = indices
    self.categories = categories
    self.string_map = self.categories
```

**关键点**:
- `indices`: 哪些参数位置是分类型的（例如 `[0, 2]` 表示第0和第2列是分类）
- `categories`: 字典，格式 `{index: [category_list]}`，例如 `{0: ['2.8', '4.0', '8.5']}`
- `string_map`: 直接指向 `categories`，用于 `StringParameterMixin.indices_to_str()` 方法

---

## 2. `_transform` 和 `_untransform` 的实现

### `_transform` 方法

```python
@subset_transform
def _transform(self, X: torch.Tensor) -> torch.Tensor:
    r"""This basically does nothing but round the nputs to the nearest
    integer.

    Args:
        X (torch.Tensor): A `batch_shape x n x d`-dim tensor of inputs.

    Returns:
        torch.Tensor: The input tensor.
    """
    return X.round()
```

**功能**:
- 仅对指定 `indices` 的维度进行四舍五入
- `@subset_transform` 装饰器自动处理只对相关维度应用转换
- 实际上是 **恒等转换**（做的工作很少）

### `_untransform` 方法

```python
@subset_transform
def _untransform(self, X: torch.Tensor) -> torch.Tensor:
    r"""This basically does nothing but round the inputs to the nearest
    integer.

    Args:
        X (torch.Tensor): A `batch_shape x n x d`-dim tensor of transformed inputs.

    Returns:
        torch.Tensor: The input tensor.
    """
    return X.round()
```

**功能**:
- 与 `_transform` 完全相同
- 仅进行四舍五入
- **问题所在**：假设输入是 indices（0, 1, 2），但当输入是实际值（2.8, 4.0, 8.5）时，不进行任何映射

---

## 3. `get_config_options` 的实现

```python
@classmethod
def get_config_options(
    cls,
    config: Config,
    name: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a dictionary of the relevant options to initialize a Fixed parameter
    transform for the named parameter within the config.

    Args:
        config (Config): Config to look for options in.
        name (str, optional): Parameter to find options for.
        options (Dict[str, Any], optional): Options to override from the config.

    Returns:
        Dict[str, Any]: A dictionary of options to initialize this class with,
            including the transformed bounds.
    """
    options = super().get_config_options(config=config, name=name, options=options)

    if name is None:
        raise ValueError(f"{name} must be set to initialize a transform.")

    if "categories" not in options:
        idx = options["indices"][0]  # There should only be one index
        cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
        options["categories"] = cat_dict

    if "bounds" in options:
        del options["bounds"]  # Remove bounds if present

    return options
```

**处理流程**:

1. **调用父类方法**: `super().get_config_options()` 从 `Transform` 基类继承
   - 父类会设置 `indices` 列表

2. **验证参数名**: 确保 `name` 不为 None

3. **提取 categories**:
   - 检查是否已在 options 中
   - 如果没有，从配置文件读取 `choices` 参数
   - **关键问题**: `element_type=str` **强制所有 choices 转换为字符串**
   - 例如: `choices = [2.8, 4.0, 8.5]` 被解析为 `['2.8', '4.0', '8.5']`

4. **删除 bounds**: 分类参数不需要连续的边界

**根本问题**: 数值型分类参数（如 `[2.8, 4.0, 8.5]`）被错误地转换为字符串列表

---

## 4. Bounds 的设置方式

### `transform_bounds` 方法

```python
def transform_bounds(
    self, X: torch.Tensor, bound: Literal["lb", "ub"] | None = None, **kwargs
) -> torch.Tensor:
    r"""Return the bounds X transformed.

    Args:
        X (torch.Tensor): Either a `[1, dim]` or `[2, dim]` tensor of parameter
            bounds.
        bound (Literal["lb", "ub"], optional): The bound that this is, if None, we
            will assume the input is both bounds with a `[2, dim]` X.
        **kwargs: passed to _transform_bounds
            epsilon: will modify the offset for the rounding to ensure each discrete
                value has equal space in the parameter space.

    Returns:
        torch.Tensor: A transformed set of parameter bounds.
    """
    epsilon = kwargs.get("epsilon", 1e-6)
    return self._transform_bounds(X, bound=bound, epsilon=epsilon)
```

### `_transform_bounds` 方法

```python
def _transform_bounds(
    self,
    X: torch.Tensor,
    bound: Literal["lb", "ub"] | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""Return the bounds X transformed.

    Args:
        X (torch.Tensor): Either a `[1, dim]` or `[2, dim]` tensor of parameter
            bounds.
        bound (Literal["lb", "ub"], optional): The bound that this is, if None, we
            will assume the input is both bounds with a `[2, dim]` X.
        epsilon:
        **kwargs: other kwargs

    Returns:
        torch.Tensor: A transformed set of parameter bounds.
    """
    X = X.clone()

    if bound == "lb":
        X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
    elif bound == "ub":
        X[0, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))
    else:  # Both bounds
        X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
        X[1, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))

    return X
```

**Bounds 调整原理**:

假设有 3 个分类选项，indices 为 [0, 1, 2]，需要映射到连续空间：

| Index | Actual Value | Transformed Bounds |
|-------|--------------|-------------------|
| 0     | 2.8          | [-0.5, 0.5)       |
| 1     | 4.0          | [0.5, 1.5)        |
| 2     | 8.5          | [1.5, 2.5)        |

**设置方式**:
- **下界 (lb)**: 减去 0.5，使得 index 0 映射到 [-0.5, ...)
- **上界 (ub)**: 加上 (0.5 - epsilon)，使得最大 index 映射到 (..., 2.5)
- **epsilon**: 确保边界之间有细微差异

---

## 5. 特殊的配置逻辑

### 类属性

```python
class Categorical(Transform, StringParameterMixin):
    # These attributes do nothing here but ensures compat.
    is_one_to_many = False
    transform_on_train = True
    transform_on_eval = True
    transform_on_fantasize = True
    reverse = False
```

**说明**:
- `is_one_to_many`: 表示这不是一对多转换
- `transform_on_train/eval/fantasize`: 所有情况下都启用转换
- `reverse`: 不反向转换

### 继承的 `indices_to_str` 方法 (来自 StringParameterMixin)

```python
def indices_to_str(self, X: np.ndarray) -> np.ndarray:
    r"""Return a NumPy array of objects where the parameter values that can be
    represented as a string is changed to a string.

    Args:
        X (np.ndarray): A mixed type NumPy array with some
            indices that will be turned into strings.

    Returns:
        np.ndarray: An array with the object type where the relevant parameters are
            converted to strings.
    """
    obj_arr = X.astype("O")

    if self.string_map is not None:
        for idx, cats in self.string_map.items():
            obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]

    return obj_arr
```

**功能**:
- 将数值 indices 转换为字符串类别
- 例如: `[0, 1, 2]` → `['2.8', '4.0', '8.5']`（如果 categories 是这些）
- 假设输入是 **indices**，不处理已经是实际值的情况

---

## 核心问题总结

### ❌ 问题 1: 强制 element_type=str

在 `get_config_options` 第 97 行：
```python
cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
```

**问题**: 数值型分类 `[2.8, 4.0, 8.5]` 被转换为字符串 `['2.8', '4.0', '8.5']`

### ❌ 问题 2: Double Transform Bug

`ParameterTransformedGenerator` 无条件调用 `untransform()`：
```python
# ParameterTransformedGenerator.gen()
x = self._base_obj.gen(...)  # 返回实际值 [2.8, 8.0, ...]
return self.transforms.untransform(x)  # ❌ 再次 untransform!
```

`Categorical.untransform()` 假设输入是 indices：
```python
# 输入: 2.8 (actual value, NOT an index!)
X.round()  # 2.8 → 2.8 (no mapping!)
# 如果有 indices_to_str，会尝试 cats[int(2.8)] → 错误!
```

---

## 推荐修复方案

### 方案 1: 修复 element_type（根本解决）

```python
# get_config_options 中，自动检测类型
choices_raw = config.getlist(name, "choices")
try:
    choices = [float(c) for c in choices_raw]
except ValueError:
    choices = choices_raw  # 保持为字符串

cat_dict = {idx: choices}
```

### 方案 2: 使 untransform 幂等

```python
def _untransform(self, X: torch.Tensor) -> torch.Tensor:
    # 如果已经是实际值，返回原样
    for idx in self.indices:
        if X[0, idx] in self.categories[idx]:
            continue  # 已经是实际值
        # 否则进行 indices → values 映射
    return X.round()
```

---

## 文件引用

- **主文件**: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py`
- **基类**: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/base.py`
- **本工作区修复文档**:
  - `tools/repair/categorical_numeric_fix/README_FIX.md`
  - `tools/repair/parameter_transform_skip/README_FIX.md`
  - `extensions/handoff/20251210_categorical_transform_root_issue.md`
