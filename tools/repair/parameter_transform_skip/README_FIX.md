# Parameter Transform Skip Patch

**Status**: ✅ Verified Working
**Date**: 2025-12-10
**Priority**: 🔴 HIGH - Fixes parameter value corruption

---

## Quick Fix

```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/parameter_transform_skip/apply_fix.py
```

---

## Problem Summary

AEPsych's `ParameterTransformedGenerator` wrapper unconditionally calls `untransform()` on all generator outputs, causing **double/triple transformation** of categorical numeric parameters.

### Impact

**Without Fix**:
```python
# Generator outputs correct actual values
Generator.gen() → [2.8, 8.0, ...]

# ParameterTransformedGenerator corrupts them
untransform() treats 2.8 as index → returns 5.6 ❌
untransform() treats 8.5 as index → returns 17.0 ❌

# Oracle receives WRONG values
Oracle receives: [5.6, 17.0, 51.2, ...] ❌
```

**With Fix**:
```python
# Generator signals it outputs actual values
Generator._skip_untransform = True

# ParameterTransformedGenerator skips untransform
ParameterTransformedGenerator.gen() checks flag → returns [2.8, 8.0, ...] ✅

# Oracle receives CORRECT values
Oracle receives: [2.8, 4.0, 8.5, ...] ✅
```

---

## What This Fix Does

Adds a `_skip_untransform` flag mechanism to allow generators to signal they output actual values directly.

### Modified Files

1. **AEPsych ParameterTransformedGenerator** (`.pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py`)
   - Adds check for `_skip_untransform` flag before calling `untransform()`
   - Lines: ~428-434

2. **AEPsych ManualGenerator** (`.pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py`)
   - Adds `_skip_untransform = True` class attribute
   - Line: ~22

3. **CustomPoolBasedGenerator** (`extensions/custom_generators/custom_pool_based_generator.py`)
   - Adds `_skip_untransform = True` class attribute
   - Line: ~81

---

## Verification

```bash
# Check if patches applied
pixi run python tools/repair/parameter_transform_skip/verify_fix.py

# Run integration test
cd tests/is_EUR_work/00_plans/251206/scripts
pixi run python run_eur_residual.py --budget 15 --tag transform_skip_verified

# Check validation log (should show correct values)
cat results/*/debug/aepsych_validation.log
```

**Expected Result**:
- ✅ Values are `[2.8, 4.0, 8.5]` (not `[5.6, 17.0, 51.2]`)
- ✅ Only minor float precision corrections (`2.799999... → 2.8`)
- ✅ No `[PoolGen FALLBACK]` warnings about AEPsych transform failures

---

## Rollback

```bash
# Find backup files
ls -lh .pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py.backup_*
ls -lh .pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py.backup_*

# Restore from backup
cp .pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py.backup_YYYYMMDD_HHMMSS \
   .pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py

cp .pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py.backup_YYYYMMDD_HHMMSS \
   .pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py

# CustomPoolBasedGenerator is in version control, use git:
cd extensions/custom_generators
git checkout custom_pool_based_generator.py
```

---

## Technical Details

### Root Cause

`ParameterTransformedGenerator.gen()` assumes ALL generators output transformed/normalized values and **unconditionally** calls `untransform()` to convert back to actual value space.

However, pool-based generators (ManualGenerator, CustomPoolBasedGenerator) store and output **actual values directly** because:
- ManualGenerator: `points` config contains actual values `[2.8, 6.5, ...]`
- CustomPoolBasedGenerator: Pool is generated from actual values via cartesian product of `choices`

When `untransform()` is called on actual values:
```python
Categorical.untransform(2.8)  # Treats 2.8 as index
→ choices[int(2.8)] → choices[2] → 8.5  # WRONG!
```

### Solution Design

Instead of modifying AEPsych's transform system (complex, brittle), we add an **opt-out mechanism**:

1. Generators set `_skip_untransform = True` to signal they output actual values
2. `ParameterTransformedGenerator.gen()` checks this flag before calling `untransform()`
3. If flag is True, returns generator output directly

**Benefits**:
- ✅ Minimal invasive changes (3 lines total)
- ✅ Backward compatible (default behavior unchanged)
- ✅ Clear semantic (generator explicitly declares output space)
- ✅ No changes to transform logic

---

## Related Issues

- [ISSUE_DESCRIPTION.md](./ISSUE_DESCRIPTION.md) - Full technical analysis
- `extensions/handoff/20251210_categorical_transform_root_issue.md` - Investigation notes
- Previous attempt: `tools/repair/categorical_numeric_fix/` - Worked around the issue in Generator, this fix solves at root

---

## Maintenance Notes

**If AEPsych updates**:
- Check if `ParameterTransformedGenerator.gen()` implementation changed
- Re-apply patch if necessary
- Verify with `verify_fix.py`

**For new custom generators**:
- If generator outputs actual values directly, add `_skip_untransform = True`
- Otherwise, let AEPsych handle transforms normally

---

**Maintained by**: Active Psych Sampling Team
**Last Updated**: 2025-12-10
