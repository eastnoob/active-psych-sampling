# Issue: ParameterTransformedModel Missing train_targets Property Delegation

**Date**: 2025-12-14
**Status**: ✅ Fixed and Verified
**Severity**: High - Data Loss Bug

---

## Executive Summary

`ParameterTransformedModel` in `aepsych/transforms/parameters.py` has a property delegation fix for `train_inputs` (lines 746-756) but is **missing the corresponding fix for `train_targets`**. This asymmetry causes training target data loss when using parameter transformations (e.g., `custom_ordinal_mono` parameter types), severely impacting EUR sampling efficiency.

---

## Observed Symptoms

### Symptom 1: Shape Mismatch

```python
[DEBUG after tell 4] Strategy: x=5, y=5      # Strategy layer: OK
[DEBUG after tell 4] Model: train_inputs=9, train_targets=3  # Model layer: BROKEN ❌
```

- `train_inputs` grows correctly (9 samples)
- `train_targets` only retains partial data (3 samples)

### Symptom 2: EUR Dynamic Weights Failure

EUR acquisition function relies on `model.train_targets` to compute:
- `lambda_t` (exploitation-exploration balance)
- `gamma_t` (residual weight)
- Statistical power metrics

With incomplete data, these calculations become unreliable.

---

## Root Cause Analysis

### Background

`ParameterTransformedModel` is a wrapper class that applies parameter transformations before passing data to the underlying GP model. It inherits from the base model class and overrides key methods.

### The Asymmetry

**Existing Fix for `train_inputs`** (lines 746-756):

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

**Missing Fix for `train_targets`**: ❌ No corresponding property exists!

### Why This Causes Data Loss

1. **During `fit()`**:
   - `ParameterTransformedModel.fit()` calls `self._base_obj.fit(train_x, train_y)`
   - Base model's `fit()` calls `self.set_train_data(train_x, train_y)`
   - `set_train_data()` sets both `self.train_inputs` and `self.train_targets`

2. **The Problem**:
   - `train_inputs` is delegated via property → writes to `_base_obj.train_inputs` ✅
   - `train_targets` is NOT delegated → writes to wrapper's `__dict__` ❌
   - Later reads of `wrapper.train_targets` get stale/partial data from wrapper's namespace

3. **Result**:
   - The underlying model has correct data
   - But the wrapper's `train_targets` attribute shadows it with outdated data

---

## Impact Assessment

### Affected Scenarios

- ✅ **Continuous parameters only**: No impact (no wrapper used)
- ❌ **With `custom_ordinal_mono` parameters**: High impact
- ❌ **With categorical parameters**: High impact
- ❌ **Any scenario using `ParameterTransformedModel`**: High impact

### Metrics Affected

1. **EUR Dynamic Weights**:
   - `lambda_t` calculation uses `len(train_targets)` → wrong sample size
   - `gamma_t` residual weight → based on incomplete data

2. **Statistical Power**:
   - Effect size calculations use wrong N
   - Power estimates become unreliable

3. **Effect Recovery**:
   - Main effect correlation degraded
   - Interaction detection accuracy reduced

---

## Fix Implementation

### Solution

Add property delegation for `train_targets`, mirroring the existing `train_inputs` fix:

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

### Location

**File**: `aepsych/transforms/parameters.py`
**Insert After**: Line 756 (after the `train_inputs` property fix)
**Before**: Line 758 (before the `@_promote_1d` decorator on `fit()` method)

---

## Verification

### Test Case

```bash
cd /path/to/active-psych-sampling
pixi run python scripts/run_eur_residual.py --budget 10
```

### Expected Output

**Before Fix**:
```
[DEBUG after tell 4] Model: train_inputs=9, train_targets=3  ❌
```

**After Fix**:
```
[DEBUG after tell 4] Model: train_inputs=9, train_targets=9  ✅
```

### Additional Verification

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

print(f"train_inputs: {model.train_inputs[0].shape if model.train_inputs else None}")
print(f"train_targets: {model.train_targets.shape if model.train_targets is not None else None}")

# Expected: Both should be torch.Size([5, 1]) and torch.Size([5])
```

---

## Upstream Submission

### Recommendation

This fix should be submitted as a PR to the official AEPsych repository:

**PR Title**: Fix train_targets property delegation in ParameterTransformedModel

**PR Description**:
> The `ParameterTransformedModel` class has an existing fix for `train_inputs` property delegation (lines 746-756) but is missing the corresponding fix for `train_targets`. This causes training target data loss in scenarios using parameter transformations, affecting EUR dynamic weight calculations and statistical power metrics.
>
> This PR adds property delegation for `train_targets`, mirroring the existing pattern used for `train_inputs`.

**Related Issue**: Link to the existing `train_inputs` fix commit/PR if available

---

## References

- Original `train_inputs` fix: `aepsych/transforms/parameters.py:746-756`
- Related: EUR dynamic weights module
- Test suite: `scripts/run_eur_residual.py`

---

## Change History

- **2025-12-14**: Issue identified and fixed
- **2025-12-14**: Verification completed (EUR test with budget=10)
