# 修复说明：ParameterTransformedModel train_targets 属性委托缺失

**状态**: ✅ 已修复并验证

---

## 问题描述

`ParameterTransformedModel` 类中存在 `train_inputs` 属性委托修复（见第746-756行），但**缺少对应的 `train_targets` 属性委托**。这导致在使用 `custom_ordinal_mono` 等参数类型时，模型的 `train_targets` 数据丢失。

### 症状

- EUR 采样过程中，`model.train_inputs` 正常增长（如9个样本）
- 但 `model.train_targets` 仅保留部分样本（如3个样本）
- 导致 EUR 动态权重计算异常，影响采样效率

### 根本原因

`ParameterTransformedModel` 是一个包装器类，在参数变换场景下包装底层 GP 模型。现有代码修复了 `train_inputs` 的属性遮蔽问题，但遗漏了对 `train_targets` 的相同修复。

---

## 快速修复指南

### 应用到你的 AEPsych Fork

**文件路径**: `aepsych/transforms/parameters.py`

**修改位置**: 第756行后（在 `train_inputs` 属性修复代码之后）

**插入以下代码**：

```python
# ========== Fix for train_targets shadowing bug ==========
@property
def train_targets(self) -> torch.Tensor | None:
    """Delegate train_targets to the underlying model."""
    return self._base_obj.train_targets

@train_targets.setter
def train_targets(self, value: torch.Tensor | None) -> None:
    """Delegate train_targets setting to the underlying model."""
    self._base_obj.train_targets = value
# ========== End of fix ==========
```

**插入后代码结构应该是**：

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

# ========== Fix for train_targets shadowing bug ==========
@property
def train_targets(self) -> torch.Tensor | None:
    """Delegate train_targets to the underlying model."""
    return self._base_obj.train_targets

@train_targets.setter
def train_targets(self, value: torch.Tensor | None) -> None:
    """Delegate train_targets setting to the underlying model."""
    self._base_obj.train_targets = value
# ========== End of fix ==========

@_promote_1d
def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
    """Fit underlying model."""
    ...
```

---

## 验证修复

修复后，运行 EUR 采样测试：

```bash
pixi run python scripts/run_eur_residual.py --budget 10
```

**预期结果**：

```
[DEBUG after tell 5] Model: train_inputs=9, train_targets=9  ✅
```

两者应该保持一致。

---

## 文件列表

- `README.md` - 本文件
- `ISSUE_DESCRIPTION.md` - 问题详细分析
- `parameters.py.patch` - 修复补丁

---

## 修复原理

与现有的 `train_inputs` 修复相同：显式定义 `train_targets` property，将所有读写操作委托给 `_base_obj`，避免属性遮蔽导致的数据不一致。

---

## 影响范围

- 修复后，EUR 动态权重计算将使用完整的训练数据
- 统计功效（statistical power）计算将更准确
- 交互效应恢复性能将提升
