# 修复摘要：train_targets 属性委托

## 修改的系统文件

### 1. aepsych/transforms/parameters.py

**修改位置**: 第756行后（紧接 `train_inputs` 属性修复代码）

**修改内容**: 添加 `train_targets` 属性委托

**代码**:
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

---

## 在你的 AEPsych Fork 中应用

### 方法 1: 使用 patch 文件

```bash
cd /path/to/your/aepsych/fork
patch -p1 < parameters.py.patch
```

### 方法 2: 手动编辑

1. 打开 `aepsych/transforms/parameters.py`
2. 找到第756行（`# ========== End of fix ==========` 标记，属于 `train_inputs` 修复）
3. 在该行后添加上述代码块
4. 保存文件

### 方法 3: 直接复制修改后的文件

如果你有访问修改后的完整文件，可以直接替换：

```bash
# 从本地环境复制修改后的文件
cp .pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py \
   /path/to/your/aepsych/fork/aepsych/transforms/parameters.py
```

**注意**: 确保文件的其他部分与你的 fork 版本兼容。

---

## 验证修复

在你的 fork 中运行测试：

```python
import torch
from aepsych.models import OrdinalGPModel
from aepsych.config import Config

config_str = """
[common]
parnames = [x1]
outcome_type = single_ordinal

[x1]
par_type = continuous
lower_bound = 0
upper_bound = 1

[OrdinalGPModel]
n_levels = 5
"""

config = Config()
config.update(config_str=config_str)
model = OrdinalGPModel.from_config(config)

X = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

model.fit(X, y)

assert model.train_inputs[0].shape[0] == len(y), "train_inputs shape mismatch!"
assert model.train_targets.shape[0] == len(y), "train_targets shape mismatch!"

print("✅ Verification passed!")
print(f"   train_inputs: {model.train_inputs[0].shape}")
print(f"   train_targets: {model.train_targets.shape}")
```

**预期输出**:
```
✅ Verification passed!
   train_inputs: torch.Size([5, 1])
   train_targets: torch.Size([5])
```

---

## 提交 PR 建议

### Commit Message

```
Fix: Add train_targets property delegation in ParameterTransformedModel

The ParameterTransformedModel class has an existing fix for train_inputs
property delegation but was missing the corresponding fix for train_targets.
This caused training target data loss when using parameter transformations,
affecting EUR dynamic weight calculations and statistical metrics.

This commit adds property delegation for train_targets, mirroring the
existing pattern used for train_inputs.

Fixes: #<issue_number>
```

### PR Checklist

- [ ] Code added to `aepsych/transforms/parameters.py:756+`
- [ ] Verification test passes
- [ ] No regression in existing tests
- [ ] Documentation updated (if needed)

---

## 相关文件

- `README.md` - 快速修复指南
- `ISSUE_DESCRIPTION.md` - 问题详细分析
- `parameters.py.patch` - Git patch 文件
- `apply_fix.py` - 自动应用脚本（用于本地环境）
- `FIX_SUMMARY.md` - 本文件
