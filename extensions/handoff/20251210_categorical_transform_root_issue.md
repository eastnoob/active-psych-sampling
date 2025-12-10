# Categorical Transform Root Issue Analysis

**Date**: 2025-12-10
**Status**: 🔴 CRITICAL - Transform pipeline is double/triple-transforming actual values

---

## Problem Summary

AEPsych's **ParameterTransformedGenerator wrapper** is applying `untransform()` to CustomPoolBasedGenerator's output, causing **multiple layers of incorrect transformations** on categorical parameters.

---

## Evidence from Integration Tests

### Test 1: Original Config (lb/ub = indices [0,0,...])
**Config**: `lb = [0, 0, 0, 0, 0, 0]`, `ub = [2, 1, 2, 2, 1, 2]`

**Result**:
- Generator outputs correct actual values: `[2.8, 8.0, 1.0, ...]`
- Transform.untransform **doubles** them: `[5.6, 8.0, 2.0, ...]`
- AEPsych returns wrong values: `x1=5.6, 17.0, 8.0` (should be `[2.8, 4.0, 8.5]`)

**Log Evidence**:
```
[BaseGen.gen] Output: [2.799999952316284, 8.0, 1.0, 0.0, 1.0, 1.0]
[Transforms.untransform] Input: [2.799999952316284, 8.0, 1.0, 0.0, 1.0, 1.0]
[Transforms.untransform] Output: [5.599999904632568, 8.0, 2.0, 0.0, 1.0, 2.0]  # ❌ DOUBLED!
```

### Test 2: With actual value lb/ub
**Config**: `lb = [2.8, 6.5, 0, ...]`, `ub = [8.5, 8.0, 2, ...]`

**Result**: Even worse! Values now **tripled/quadrupled**:
- `x1 = 51.2, 18.76, 25.6` (correct: [2.8, 4.0, 8.5])
- `x2 = 16.25, 18.5` (correct: [6.5, 8.0])

**Validation Log**:
```
[VALIDATION CHECK] iter 6:
  x1_CeilingHeight: 51.249996185302734 -> corrected to 8.5
  x2_GridModule: 16.25 -> corrected to 8.0
```

---

## Root Cause

### Transform Pipeline Architecture

```
Pool[actual values]
  → CustomPoolBasedGenerator.gen() returns actual values [2.8, 8.0, ...]
  → ParameterTransformedGenerator wrapper calls untransform()
  → Categorical.untransform() treats actual values as indices
  → Multiplies by the "step size" calculated from lb/ub
  → Returns WRONG doubled/tripled values
```

### Why untransform() is Wrong

**Categorical.untransform() Logic**:
```python
# Assumes input is INDEX (0, 1, 2)
# Multiplies by (choices_max - choices_min) to get actual value
# But our input is ALREADY an actual value!

# Example with choices=[2.8, 4.0, 8.5]:
input = 2.8  # Already actual value!
untransform thinks: "this is index 2.8, multiply by step"
step = (8.5 - 2.8) / 2 = 2.85
output = 2.8 * 2.85 = 7.98 ≈ 8.0 ❌

# With lb=2.8, ub=8.5:
step = (8.5 - 2.8) / 1 = 5.7
output = 2.8 * 5.7 ≈ 16.0 ❌ WORSE!
```

---

## Attempted Fixes (ALL FAILED)

### ❌ Fix Attempt 1: Modify AEPsych categorical.py
**Goal**: Make `element_type=float` work for numeric choices
**Status**: FAILED
**Reason**: Even if categories are stored as floats, the transform pipeline still applies

### ❌ Fix Attempt 2: Add `transform_options = skip_transform` to config
**Goal**: Disable transform via config option
**Status**: FAILED
**Reason**: `skip_transform` is not a valid AEPsych config option

### ❌ Fix Attempt 3: Use actual values in lb/ub
**Goal**: Make everything work in actual value space
**Status**: FAILED (made it worse!)
**Reason**: Transform still applied, but now with wrong scaling factor

### ✅ Fix Attempt 4: Generator Fallback Mapping (Solution B)
**Goal**: Post-process untransform output to fix wrong values
**Status**: WORKING (but shouldn't be needed!)
**Location**: `CustomPoolBasedGenerator._ensure_actual_values()`

---

## Current Workaround

**方案B (Generator Fallback)** is currently the only working solution:

1. Generator outputs correct actual values
2. Transform pipeline corrupts them
3. `_ensure_actual_values()` detects corrupted values (e.g., 5.6 not in [2.8, 4.0, 8.5])
4. Maps them back using stored categorical_mappings
5. Returns correct values to Oracle

**Problem**: This is a **band-aid fix**. We're correcting AEPsych's mistakes after they happen.

---

## Proper Solution Needed

###人 Option 1: Bypass Transform Pipeline Completely
**Approach**: Make CustomPoolBasedGenerator NOT wrapped by ParameterTransformedGenerator

**How**: Modify AEPsych's generator instantiation logic to:
- Detect when generator already handles transforms
- Skip ParameterTransformedGenerator wrapping

**File**: `.pixi/envs/default/Lib/site-packages/aepsych/generators/__init__.py` or wherever generator wrapping happens

### Option 2: Make Categorical Transform Idempotent
**Approach**: Fix Categorical.untransform() to detect when input is already an actual value

**How**: Check if input value is in choices list:
```python
def untransform(self, x):
    # If x is already in choices, return as-is (idempotent)
    if x in self.choices:
        return x
    # Otherwise, treat as index and map
    return self.choices[int(x)]
```

**File**: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py`

### Option 3: Disable par_type=categorical, Use Ordinal Instead
**Approach**: Treat all parameters as ordinal (no transform)

**Config Change**:
```ini
[x1_CeilingHeight]
par_type = continuous  # NOT categorical!
lb = 2.8
ub = 8.5
# Generator ensures only [2.8, 4.0, 8.5] are selected via pool
```

**Problem**: Loses semantic meaning of "categorical", GP model might interpolate

---

## Recommendation

**Short term**: Keep using **Solution B (Generator Fallback)** - it works reliably

**Long term**: Implement **Option 2 (Idempotent Transform)** - this is the cleanest fix:
- Fixes the root cause
- Makes transforms safe regardless of input
- Backward compatible
- Benefits all AEPsych users

---

## Related Files

- Generator: `extensions/custom_generators/custom_pool_based_generator.py`
- Transform Bug: `.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py`
- Config: `tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini`
- Repair Package: `tools/repair/categorical_numeric_fix/`

---

## Test Results

| Test | Config | Result | Note |
|------|--------|--------|------|
| v1 | lb/ub=indices, no skip_transform | ❌ Doubled values | 2.8→5.6, 8.5→17.0 |
| v2 | lb/ub=indices, skip_transform | ❌ Doubled values | Config option ignored |
| v3 | lb/ub=actual values | ❌ Tripled values | 2.8→18.76, 8.5→51.25 |
| Fallback | All configs + fallback mapping | ✅ Working | Corrects after corruption |

---

**Next Steps**:
1. Document current fallback solution as permanent workaround
2. Consider contributing Option 2 fix to AEPsych upstream
3. Update repair package documentation with findings
