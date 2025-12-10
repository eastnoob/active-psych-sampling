# Parameter Transform Skip Issue

**Problem**: AEPsych ParameterTransformedGenerator double/triple transforms categorical numeric parameters
**Status**: ✅ FIXED
**Date**: 2025-12-10
**Priority**: 🔴 HIGH

---

## Problem Description

AEPsych's `ParameterTransformedGenerator` wrapper unconditionally applies `untransform()` to all generator outputs, causing **double/triple transformation** of categorical numeric parameters when generators output actual values directly.

### Symptoms

**Observed Behavior**:
- Categorical numeric choices: `[2.8, 4.0, 8.5]`
- Generator outputs correct values: `[2.8, 8.0, ...]`
- **AEPsych returns wrong values**: `[5.6, 17.0, 51.2, ...]` ❌

**Validation Log Evidence** (before fix):
```
[VALIDATION CHECK] iter 3 - AEPsych返回的原始值:
  x1_CeilingHeight: [np.float64(17.0)]           ❌ WRONG! (should be 8.5)
  x1_CeilingHeight: [np.float64(5.599999904632568)]  ❌ WRONG! (should be 4.0)

[VALIDATION CHECK] iter 6 - AEPsych返回的原始值:
  x1_CeilingHeight: [np.float64(51.249996185302734)] ❌ WRONG! (should be 8.5)
  x2_GridModule: [np.float64(16.25)]                 ❌ WRONG! (should be 8.0)
```

**After Fix**:
```
[VALIDATION CHECK] iter 3 - AEPsych返回的原始值:
  x1_CeilingHeight: [np.float64(2.799999952316284)] ✅ CORRECT! (minor float precision)
  x2_GridModule: [np.float64(8.0)]                  ✅ CORRECT!
  x3_OuterFurniture: [np.float64(0.0)]              ✅ CORRECT!
```

---

## Root Cause Analysis

### Transform Pipeline

```python
# Expected flow for pool-based generators:
Pool → [2.8, 8.0, ...] (actual values)
  ↓
Generator.gen() → [2.8, 8.0, ...] (actual values)
  ↓
??? ParameterTransformedGenerator wrapper ???
  ↓
Oracle receives values
```

### The Bug

**ParameterTransformedGenerator.gen()** (`.pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py:394-428`):

```python
def gen(self, num_points, model, **kwargs):
    x = self._base_obj.gen(...)  # Generator outputs [2.8, 8.0, ...]
    return self.transforms.untransform(x)  # ❌ UNCONDITIONAL!
```

**Categorical.untransform()** assumes input is **indices**:
```python
# Input: 2.8 (actual value, NOT an index!)
Categorical.untransform(2.8)
  → treats 2.8 as index
  → multiplies by step size
  → returns 5.6 ❌ WRONG!

# With lb=2.8, ub=8.5 config:
step = (8.5 - 2.8) / 1 = 5.7
output = 2.8 * 5.7 ≈ 16.0 ❌ EVEN WORSE!
```

### Why This Happens

AEPsych's design assumes:
1. All generators work in **transformed/normalized space** (indices [0, 1, 2])
2. `ParameterTransformedGenerator` always needs to call `untransform()` to convert back to actual values

**However**, pool-based generators (`ManualGenerator`, `CustomPoolBasedGenerator`) **store and output actual values directly** because:
- `ManualGenerator`: `points` config contains actual values: `[[2.8, 6.5, ...], [4.0, 6.5, ...]]`
- `CustomPoolBasedGenerator`: Pool is generated from `choices` via cartesian product

This architectural mismatch causes the bug.

---

## Impact

### Affected Components
1. **ManualGenerator** - warmup points are corrupted
2. **CustomPoolBasedGenerator** - all pool-based sampling corrupted
3. **Any custom generator** that outputs actual values directly

### Experimental Consequences
- ❌ Oracle receives wrong parameter values
- ❌ GP model trains on wrong X coordinates
- ❌ Acquisition function optimizes wrong parameter space
- ❌ Results completely invalid

### Workaround Used (Before Fix)
**param_validator** in sampling loop manually corrected values before passing to Oracle. This masked the problem but:
- ✅ Prevented Oracle from receiving wrong values
- ❌ GP model still trained on corrupted data
- ❌ Acquisition function still optimized wrong space
- ❌ Not a real solution, just damage control

---

## Solution

### Design Decision

**Option A**: Modify transform system (rejected - too complex, fragile)
**Option B**: Modify Generator behavior (rejected - doesn't address root cause)
**✓ Option C**: Add opt-out mechanism for generators

### Implementation

Add `_skip_untransform` flag mechanism:

1. **Generators** declare they output actual values:
   ```python
   class ManualGenerator(AEPsychGenerator):
       _skip_untransform = True  # Signal actual value output
   ```

2. **ParameterTransformedGenerator** checks flag before untransform:
   ```python
   def gen(self, num_points, model, **kwargs):
       x = self._base_obj.gen(...)

       # Check if generator outputs actual values
       if hasattr(self._base_obj, '_skip_untransform') and self._base_obj._skip_untransform:
           return x  # Skip untransform

       return self.transforms.untransform(x)
   ```

### Modified Files

1. **ParameterTransformedGenerator** - adds skip check
2. **ManualGenerator** - adds `_skip_untransform = True`
3. **CustomPoolBasedGenerator** - adds `_skip_untransform = True`

---

## Verification

### Unit Test
```bash
pixi run python tools/repair/parameter_transform_skip/verify_fix.py
```

**Expected Output**:
```
[1/3] Verifying AEPsych ParameterTransformedGenerator patch...
  ✓ _skip_untransform check
  ✓ Early return on skip
  ✓ Patch comment marker

[2/3] Verifying AEPsych ManualGenerator patch...
  ✓ _skip_untransform flag
  ✓ Comment explanation

[3/3] Verifying CustomPoolBasedGenerator patch...
  ✓ _skip_untransform flag
  ✓ Comment explanation

[SUCCESS] All patches verified!
```

### Integration Test
```bash
cd tests/is_EUR_work/00_plans/251206/scripts
pixi run python run_eur_residual.py --budget 15 --tag verification_test
```

**Check validation log**:
```bash
cat results/*/debug/aepsych_validation.log
```

**Expected**: No double/triple values, only minor float precision corrections

---

## Related Files

- **Patch files**: `tools/repair/parameter_transform_skip/*.patch`
- **Apply script**: `tools/repair/parameter_transform_skip/apply_fix.py`
- **Verify script**: `tools/repair/parameter_transform_skip/verify_fix.py`
- **Quick guide**: `tools/repair/parameter_transform_skip/README_FIX.md`
- **Investigation notes**: `extensions/handoff/20251210_categorical_transform_root_issue.md`

---

## Technical Details

### Why Not Fix in Categorical Transform?

We considered making `Categorical.untransform()` idempotent:
```python
def untransform(self, x):
    if x in self.choices:  # Already actual value?
        return x
    return self.choices[int(x)]  # Treat as index
```

**Rejected because**:
- Numeric choices like `[2.8, 4.0, 8.5]` are stored as floats with precision issues
- `2.799999952316284 in [2.8, 4.0, 8.5]` → False (float comparison failure)
- Would need fuzzy matching → fragile, unpredictable
- Doesn't address architectural mismatch

### Why `_skip_untransform` Works

✅ **Clear semantics**: Generator explicitly declares output space
✅ **Minimal changes**: 3 lines total
✅ **Backward compatible**: Default behavior unchanged
✅ **No float comparison**: Boolean flag, no precision issues
✅ **Addresses root cause**: Fixes architectural mismatch

---

## Maintenance

### When AEPsych Updates
1. Check if `ParameterTransformedGenerator.gen()` changed
2. Re-run `verify_fix.py`
3. Re-apply patch if necessary

### For New Custom Generators
- If generator outputs actual values → add `_skip_untransform = True`
- If generator outputs normalized/index space → don't add flag

---

**Created**: 2025-12-10
**Author**: Active Psych Sampling Team
**Fix Status**: ✅ Verified Working
