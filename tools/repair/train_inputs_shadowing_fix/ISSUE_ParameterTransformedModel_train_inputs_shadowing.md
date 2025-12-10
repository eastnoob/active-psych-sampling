# Bug: `ParameterTransformedModel.train_inputs` returns stale data due to attribute shadowing

## Summary

`ParameterTransformedModel` wraps a model and delegates attribute access via `__getattr__`. However, `train_inputs` is a property with a setter in `AEPsychModelMixin`. Due to dynamic class inheritance, the setter writes `_train_inputs` to the **wrapper's** `__dict__`, shadowing the base model's `_train_inputs`. Subsequent reads return stale data.

## Reproduction

```python
from aepsych.models import GPRegressionModel
from aepsych.transforms.parameters import ParameterTransformedModel
from botorch.models.transforms.input import ChainedInputTransform
import torch

# Setup
base_model = GPRegressionModel(dim=2)
wrapped = ParameterTransformedModel(model=base_model, transforms=ChainedInputTransform(**{}))

# Initial fit
x1 = torch.rand(3, 2)
y1 = torch.rand(3)
wrapped.fit(x1, y1)

print(f"After fit #1: wrapped.train_inputs[0].shape = {wrapped.train_inputs[0].shape}")
# Expected: torch.Size([3, 2]) ✓

# Second fit with more data
x2 = torch.rand(5, 2)
y2 = torch.rand(5)
wrapped.fit(x2, y2)

print(f"After fit #2: wrapped.train_inputs[0].shape = {wrapped.train_inputs[0].shape}")
# Expected: torch.Size([5, 2])
# Actual (before fix):   torch.Size([3, 2])  <-- BUG: stale value
# Actual (after fix):    torch.Size([5, 2])  <-- FIXED

print(f"After fit #2: wrapped._base_obj.train_inputs[0].shape = {wrapped._base_obj.train_inputs[0].shape}")
# Always:   torch.Size([5, 2])  <-- base model is always correct
```

## Root Cause

In `ParameterTransformedModel.__init__` ([parameters.py#L597-603](https://github.com/facebookresearch/aepsych/blob/main/aepsych/transforms/parameters.py#L597-L603)):

```python
self.__class__ = type(
    f"ParameterTransformed{_base_obj.__class__.__name__}",
    (self.__class__, _base_obj.__class__),
    {},
)
```

This makes `wrapped` inherit `train_inputs` property from `AEPsychModelMixin`.

In `AEPsychModelMixin.train_inputs` setter ([base.py#L70-74](https://github.com/facebookresearch/aepsych/blob/main/aepsych/models/base.py#L70-L74)):

```python
@train_inputs.setter
def train_inputs(self, train_inputs):
    ...
    self._train_inputs = inputs  # writes to caller's __dict__
```

When `set_train_data()` calls `self.train_inputs = (inputs,)`, it writes to `wrapped.__dict__['_train_inputs']`, not `_base_obj._train_inputs`.

Since `_train_inputs` now exists in `wrapped.__dict__`, `__getattr__` is never invoked for subsequent reads.

## Suggested Fix

Add explicit property delegation in `ParameterTransformedModel`:

```python
@property
def train_inputs(self):
    return self._base_obj.train_inputs

@train_inputs.setter
def train_inputs(self, value):
    self._base_obj.train_inputs = value
```

## Impact

Any code relying on `ParameterTransformedModel.train_inputs` to reflect current training data will get stale values. This affects:

- Acquisition function cache invalidation
- Any logic using `model.train_inputs[0].shape[0]` to detect data changes

## Environment

- AEPsych version: main branch (as of 2025-11-29)
- Python: 3.10+
