# ParameterTransformedModel `train_inputs` 遮蔽 Bug 修复报告

**日期**: 2025-11-29  
**状态**: ✅ 已修复

---

## 1. 问题现象

EUR 动态权重 (`lambda_t`, `gamma_t`) 在使用官方 AEPsych init_strat + opt_strat 流程时不更新。

```
# 期望行为: n_train 随试验递增
n_train: 3 → 4 → 5 → 6 → 7

# 实际行为: n_train 卡死在初始值
n_train: 3 → 3 → 3 → 3 → 3
```

---

## 2. 诊断过程

### 2.1 对比测试

| 测试场景 | 结果 | 说明 |
|---------|------|------|
| 直接 Python API | ✅ 通过 | `model.train_inputs` 正确更新 |
| AEPsych Server | ❌ 失败 | `model.train_inputs` 卡住 |

**关键差异**: Server 路径使用 `ParameterTransformedModel` 包装 `GPRegressionModel`。

### 2.2 追踪日志发现

```python
# test_server_model_tracking.py 输出
[After ask] base._train_inputs[0].shape: torch.Size([4, 2])   # ✅ 底层模型正确
[After ask] model.train_inputs[0].shape: torch.Size([3, 2])   # ❌ 包装器卡住！
```

**结论**: `GPRegressionModel._train_inputs` 正确更新，但 `ParameterTransformedModel.train_inputs` 返回陈旧值。

### 2.3 根因定位

```python
# test_train_inputs_resolution.py 验证
wrapped_model.__dict__['_train_inputs']  # 存在！遮蔽了 base_model._train_inputs
```

---

## 3. 根本原因

### 3.1 代码路径

```
ParameterTransformedModel.__init__
    ↓
self.__class__ = type(f"ParameterTransformed{...}", (self.__class__, _base_obj.__class__), {})
    ↓
# 动态继承 AEPsychModelMixin，获得 train_inputs property
    ↓
GPRegressionModel.fit() → set_train_data() → self.train_inputs = (inputs,)
    ↓
# train_inputs.setter 执行: self._train_inputs = inputs
# 但 self 是 wrapped_model，不是 _base_obj！
    ↓
wrapped_model.__dict__['_train_inputs'] 被创建，遮蔽 _base_obj._train_inputs
```

### 3.2 Bug 机制图解

```
┌─────────────────────────────────────────────────────────────┐
│ ParameterTransformedModel (wrapped_model)                   │
│  ├─ __class__ = ParameterTransformedGPRegressionModel      │
│  │              (继承 AEPsychModelMixin.train_inputs)       │
│  ├─ _base_obj → GPRegressionModel                          │
│  ├─ __getattr__ → 委托给 _base_obj ✅                       │
│  ├─ __setattr__ → 未重写 ❌                                 │
│  └─ __dict__['_train_inputs'] = 初始值 (遮蔽!)             │
│                        ↑                                    │
│     train_inputs setter 写入这里，而不是 _base_obj          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 关键代码位置

**`temp_aepsych/aepsych/models/base.py`** (lines 62-74, 197-199):

```python
@train_inputs.setter
def train_inputs(self, train_inputs):
    ...
    self._train_inputs = inputs  # 写入调用者的 __dict__

def set_train_data(self, inputs, targets, strict=False):
    ...
    self.train_inputs = (inputs,)  # 触发 setter
```

**`temp_aepsych/aepsych/transforms/parameters.py`** (lines 597-603):

```python
self.__class__ = type(
    f"ParameterTransformed{_base_obj.__class__.__name__}",
    (self.__class__, _base_obj.__class__),  # 继承 train_inputs property
    {},
)
# 缺少 __setattr__ 或 train_inputs 委托
```

---

## 4. 修复方案

**文件**: `temp_aepsych/aepsych/transforms/parameters.py`  
**位置**: `ParameterTransformedModel` 类，`fit()` 方法之前  
**修改**: 添加 `train_inputs` 属性委托

```python
# ========== Fix for _train_inputs shadowing bug ==========
@property
def train_inputs(self) -> tuple[torch.Tensor, ...] | None:
    """Delegate train_inputs to the underlying model."""
    return self._base_obj.train_inputs

@train_inputs.setter
def train_inputs(self, value: tuple[torch.Tensor, ...] | None) -> None:
    """Delegate train_inputs setting to the underlying model."""
    self._base_obj.train_inputs = value
# ========== End of fix ==========
```

---

## 5. 验证结果

### 修复前

```
[After ask] base._train_inputs[0].shape: torch.Size([4, 2])
[After ask] model.train_inputs[0].shape: torch.Size([3, 2])  ❌ 不一致
```

### 修复后

```
[After ask] base._train_inputs[0].shape: torch.Size([4, 2])
[After ask] model.train_inputs[0].shape: torch.Size([4, 2])  ✅ 一致

# WeightEngine 正确更新
[WeightEngine] n_train increased: 0 -> 5
[WeightEngine] n_train increased: 0 -> 6
[WeightEngine] n_train increased: 0 -> 7
```

---

## 6. 修改文件清单

| 文件 | 修改类型 | 说明 |
|-----|---------|------|
| `temp_aepsych/aepsych/transforms/parameters.py` | **核心修复** | 添加 `train_inputs` property 委托 |
| `temp_aepsych/aepsych/models/gp_regression.py` | 调试日志 | 可删除 |
| `temp_aepsych/aepsych/strategy/strategy.py` | 调试日志 | 可删除 |
| `extensions/custom_generators/pool_based_generator.py` | 调试日志 | 可删除 |

---

## 7. 技术总结

**问题本质**: Python 动态类继承 + 属性遮蔽

- `__getattr__` 只处理**不存在**的属性查找
- `__setattr__` 未重写，写入操作在 wrapper 上创建新属性
- 一旦 `wrapper.__dict__['_train_inputs']` 存在，后续读取不再触发 `__getattr__`

**修复原理**: 显式定义 `train_inputs` property，拦截所有读写并委托给 `_base_obj`。
