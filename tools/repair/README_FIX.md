# 修复说明：ParameterTransformedModel train_inputs 遮蔽 Bug

**状态**: ✅ 已修复并验证

---

## 快速修复指南

### 1. 验证是否需要修复

运行验证脚本：

```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/verify_issue_reproduction.py
```

**预期结果**：
- ✅ 如果显示 `[ :) ] Fix effective` - 已修复，无需操作
- ❌ 如果显示 `[ :( ] BUG present` - 需要修复

---

## 2. 应用修复

### 方法 1: 自动应用（推荐）

运行修复脚本：

```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/apply_fix.py
```

### 方法 2: 手动应用

**步骤 1**: 打开文件

```
.pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py
```

**步骤 2**: 找到位置

搜索这一行（大约在第 723 行）：

```python
return self._base_obj.posterior(X=X, **kwargs)
```

**步骤 3**: 在下一行（空行之后，`@_promote_1d` 之前）插入以下代码：

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

**插入后应该看起来像这样**：

```python
        return self._base_obj.posterior(X=X, **kwargs)

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

    @_promote_1d
    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
```

---

## 3. 验证修复

再次运行验证脚本：

```bash
pixi run python tools/repair/verify_issue_reproduction.py
```

应该显示：`[ :) ] Fix effective`

---

## 文件列表

- `ISSUE_ParameterTransformedModel_train_inputs_shadowing.md` - Bug 详细描述
- `TRAIN_INPUTS_SHADOWING_BUG_FIX.md` - 修复报告（中文）
- `verify_issue_reproduction.py` - 验证脚本
- `parameters.py.patch` - 修复代码片段
- `apply_fix.py` - 自动修复脚本（如果存在）
- `README_FIX.md` - 本文件

---

## 问题根因

`ParameterTransformedModel` 通过动态继承获得 `train_inputs` property，但 setter 将 `_train_inputs` 写入 wrapper 的 `__dict__`，导致后续读取获得陈旧数据。

## 修复原理

显式定义 `train_inputs` property，拦截所有读写操作并委托给 `_base_obj`，确保始终访问底层模型的最新数据。

---

## 影响范围

修复后，EUR 实验中的动态权重更新（`lambda_t`, `gamma_t`）将正常工作。
