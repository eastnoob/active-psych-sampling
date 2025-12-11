# Per-Variable Exhaustive Search Configuration

**Date**: 2025-12-11
**Status**: Design Proposal

## Problem Statement

Current limitation: `exhaustive_level_threshold=3` is a global parameter in `LocalSampler`. All variables with ≤3 levels use exhaustive enumeration, all variables with >3 levels use sampling.

**User Need**:
- Force a 10-level ordinal variable to use exhaustive mode
- Prevent a 3-level categorical from exhausting (override default)
- Per-variable control without changing global threshold

## Solution

Add per-variable exhaustive configuration via parameter names in config file:

```ini
[EURAnovaPairAcqf]
exhaustive_params = [x1_height, x2_grid]  # Force these to exhaust
non_exhaustive_params = [x3_category]     # Never exhaust these
exhaustive_level_threshold = 3            # Default threshold (unchanged)
```

**Logic Priority**:
1. If variable in `exhaustive_params` → always exhaust
2. Else if variable in `non_exhaustive_params` → never exhaust
3. Else if `n_levels <= exhaustive_level_threshold` → exhaust
4. Else → use sampling

## Background: Exhaustive vs Sampling Strategies

### Exhaustive Mode (Complete Enumeration)
**When**: `n_levels <= exhaustive_level_threshold` (default: 3)

```python
# For 3-level variable [0.0, 0.5, 1.0]
samples = np.tile(unique_vals, (B, n_repeats))
# Result: [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, ...]
```

- **Categorical & Ordinal**: Identical strategy (enumerate all values)
- **Advantage**: Complete coverage of discrete space
- **Cost**: O(n_levels) samples per base point

### Non-Exhaustive Mode (Sampling)
**When**: `n_levels > exhaustive_level_threshold`

#### Categorical Sampling
```python
# Uniform random sampling from unique values
samples = rng.choice(unique_vals, size=(B, local_num))
# No center preference, no order assumption
```

#### Ordinal Sampling (DIFFERENT)
```python
# Gaussian perturbation + nearest neighbor constraint
center_vals = base[:, :, k]  # Current normalized value
noise = rng.normal(0, sigma, size=(B, local_num))
perturbed = center_vals + noise

# Snap to nearest valid normalized value
for i, j in product(range(B), range(local_num)):
    closest_idx = np.argmin(np.abs(unique_vals - perturbed[i, j]))
    samples[i, j] = unique_vals[closest_idx]
```

**Key Differences**:
- **Categorical**: No center preference, uniform exploration
- **Ordinal**: Gaussian centered at current value, local exploration
- **Why Different**: Ordinal preserves order and spacing information for ANOVA

### Design Rationale

**Exhaustive is better when feasible** (complete coverage), but:
- Non-exhaustive categorical: loses no information (unordered)
- Non-exhaustive ordinal: preserves local structure (ordered + spacing)

This hybrid approach balances completeness with computational cost.

## Implementation Plan

### Step 1: Extend LocalSampler
**File**: `extensions/dynamic_eur_acquisition/modules/local_sampler.py`

```python
class LocalSampler:
    def __init__(
        self,
        ...,
        exhaustive_dims: Optional[Set[int]] = None,      # NEW
        non_exhaustive_dims: Optional[Set[int]] = None,  # NEW
        exhaustive_level_threshold: int = 3,
    ):
        self.exhaustive_dims = exhaustive_dims or set()
        self.non_exhaustive_dims = non_exhaustive_dims or set()
        self.exhaustive_level_threshold = exhaustive_level_threshold
```

**Modified Exhaustive Logic** (lines ~423):

```python
def _perturb_ordinal(self, base, k, B):
    unique_vals = self._unique_vals_dict.get(k)
    n_levels = len(unique_vals)

    # NEW: Priority-based exhaustive decision
    use_exhaustive = (
        k in self.exhaustive_dims  # Explicit override
        or (k not in self.non_exhaustive_dims  # Not explicitly excluded
            and self.use_hybrid_perturbation
            and n_levels <= self.exhaustive_level_threshold)
    )

    if use_exhaustive:
        # Exhaustive mode...
    else:
        # Gaussian perturbation mode...
```

**Same logic applies to** `_perturb_categorical` (lines ~301-345).

### Step 2: Add Config Parser
**File**: `extensions/dynamic_eur_acquisition/modules/config_parser.py`

```python
def parse_exhaustive_params(
    config: Config,
    section: str = "EURAnovaPairAcqf"
) -> Tuple[List[str], List[str]]:
    """Parse exhaustive/non-exhaustive parameter lists from config.

    Returns:
        (exhaustive_params, non_exhaustive_params): Lists of parameter names
    """
    exhaustive = config.getlist(section, "exhaustive_params", fallback=[])
    non_exhaustive = config.getlist(section, "non_exhaustive_params", fallback=[])

    # Validation: no overlap
    overlap = set(exhaustive) & set(non_exhaustive)
    if overlap:
        raise ValueError(
            f"Parameters cannot be both exhaustive and non-exhaustive: {overlap}"
        )

    return exhaustive, non_exhaustive
```

### Step 3: Integrate in EURAnovaPairAcqf
**File**: `extensions/dynamic_eur_acquisition/eur_anova_pair.py`

```python
class EURAnovaPairAcqf:
    @classmethod
    def from_config(cls, config, name, acqf_kwargs=None):
        # ... existing code ...

        # NEW: Parse exhaustive configuration
        from modules.config_parser import parse_exhaustive_params
        exhaustive_params, non_exhaustive_params = parse_exhaustive_params(
            config, section=name
        )

        # NEW: Map parameter names to dimension indices
        parnames = config.getlist("common", "parnames", fallback=[])
        exhaustive_dims = {parnames.index(p) for p in exhaustive_params}
        non_exhaustive_dims = {parnames.index(p) for p in non_exhaustive_params}

        # Pass to LocalSampler
        local_sampler_kwargs.update({
            "exhaustive_dims": exhaustive_dims,
            "non_exhaustive_dims": non_exhaustive_dims,
        })

        local_sampler = LocalSampler(**local_sampler_kwargs)
        # ...
```

## Key Files

### Core Implementation
1. **LocalSampler**: `extensions/dynamic_eur_acquisition/modules/local_sampler.py`
   - Lines 36-103: `__init__` (add exhaustive_dims/non_exhaustive_dims)
   - Lines 301-345: `_perturb_categorical` (modify exhaustive logic)
   - Lines 396-460: `_perturb_ordinal` (modify exhaustive logic)

2. **Config Parser**: `extensions/dynamic_eur_acquisition/modules/config_parser.py`
   - Add `parse_exhaustive_params()` function

3. **EURAnovaPairAcqf**: `extensions/dynamic_eur_acquisition/eur_anova_pair.py`
   - `from_config()`: Parse config + map param names to dims + pass to LocalSampler

### Design Reference
4. **Ordinal Design Doc**: `handoff/ordinal_normalized_design.md`
   - Lines 540-596: `_perturb_ordinal` design specification
   - Context: normalized value space perturbation

## Configuration Examples

### Example 1: Force 10-level ordinal to exhaust

```ini
[common]
parnames = [x1_height, x2_intensity, x3_category]

[x1_height]
par_type = custom_ordinal
values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]  # 10 levels

[EURAnovaPairAcqf]
exhaustive_params = [x1_height]  # Override threshold for this variable
exhaustive_level_threshold = 3   # Other variables: ≤3 levels exhaust
```

**Result**: `x1_height` uses exhaustive mode despite 10 levels.

### Example 2: Prevent 3-level categorical from exhausting

```ini
[common]
parnames = [x1_grid, x2_category]

[x2_category]
par_type = categorical
values = [A, B, C]  # 3 levels

[EURAnovaPairAcqf]
non_exhaustive_params = [x2_category]  # Force sampling mode
exhaustive_level_threshold = 3         # Other 3-level vars still exhaust
```

**Result**: `x2_category` uses sampling mode despite 3 levels.

### Example 3: Mixed configuration

```ini
[common]
parnames = [x1_height, x2_grid, x3_category, x4_continuous]

[EURAnovaPairAcqf]
exhaustive_params = [x1_height, x2_grid]  # Always exhaust
non_exhaustive_params = [x3_category]     # Never exhaust
exhaustive_level_threshold = 4            # Default: ≤4 levels exhaust
```

**Behavior**:
- `x1_height`: exhaustive (explicit override)
- `x2_grid`: exhaustive (explicit override)
- `x3_category`: sampling (explicit override)
- `x4_continuous`: N/A (continuous variable, no exhaustive mode)
- Any other 4-level variable: exhaustive (default threshold)
- Any other 5+ level variable: sampling (default threshold)

## Testing Strategy

### Unit Tests

**File**: `tests/test_local_sampler_exhaustive_config.py`

```python
def test_exhaustive_dims_override():
    """Test explicit exhaustive_dims forces exhaustive mode"""
    # 10-level ordinal, threshold=3, but in exhaustive_dims
    sampler = LocalSampler(
        exhaustive_dims={0},
        exhaustive_level_threshold=3,
        use_hybrid_perturbation=True,
    )
    # Verify exhaustive mode triggered for dim 0

def test_non_exhaustive_dims_override():
    """Test explicit non_exhaustive_dims prevents exhaustive mode"""
    # 3-level categorical, threshold=3, but in non_exhaustive_dims
    sampler = LocalSampler(
        non_exhaustive_dims={0},
        exhaustive_level_threshold=3,
        use_hybrid_perturbation=True,
    )
    # Verify sampling mode triggered for dim 0

def test_exhaustive_priority_logic():
    """Test priority: explicit > threshold"""
    # exhaustive_dims takes priority over threshold
    # non_exhaustive_dims takes priority over threshold
    # threshold applies when neither override present
```

### Integration Test

**File**: `tests/test_eur_exhaustive_config_e2e.py`

```python
def test_config_exhaustive_override():
    """Test config parsing and LocalSampler integration"""
    config_content = """
    [common]
    parnames = [x1_10level, x2_3level]

    [x1_10level]
    par_type = custom_ordinal
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    [EURAnovaPairAcqf]
    exhaustive_params = [x1_10level]
    exhaustive_level_threshold = 3
    """
    # Create config, init acqf, verify x1 uses exhaustive
```

### Validation Test

**File**: `tests/test_config_parser_exhaustive.py`

```python
def test_exhaustive_params_overlap_error():
    """Test validation: parameter cannot be in both lists"""
    # Config with x1 in both exhaustive_params and non_exhaustive_params
    # Should raise ValueError

def test_exhaustive_params_nonexistent_warning():
    """Test warning for nonexistent parameter names"""
    # Config with exhaustive_params = [nonexistent_var]
    # Should raise ValueError or warning
```

## Migration Notes

### Backward Compatibility

**Fully backward compatible**:
- Existing configs without `exhaustive_params`/`non_exhaustive_params` work unchanged
- Default behavior: `exhaustive_level_threshold=3` applies to all variables
- New parameters are optional with empty defaults

### Migration Path

No migration needed for existing configs. Users can opt-in to per-variable control:

```ini
# Old config (still works)
[EURAnovaPairAcqf]
exhaustive_level_threshold = 3

# New config (opt-in per-variable control)
[EURAnovaPairAcqf]
exhaustive_params = [x1_height]
non_exhaustive_params = [x2_category]
exhaustive_level_threshold = 3
```

## Implementation Checklist

- [ ] Extend `LocalSampler.__init__` with `exhaustive_dims`/`non_exhaustive_dims`
- [ ] Modify `_perturb_ordinal` exhaustive logic (priority-based)
- [ ] Modify `_perturb_categorical` exhaustive logic (priority-based)
- [ ] Add `parse_exhaustive_params()` to config_parser.py
- [ ] Integrate in `EURAnovaPairAcqf.from_config()` (name → dim mapping)
- [ ] Write unit tests for LocalSampler override logic
- [ ] Write integration test for config parsing
- [ ] Write validation test for config errors
- [ ] Update user documentation with examples
- [ ] Test backward compatibility with existing configs

## References

### Related Documents
- `handoff/ordinal_normalized_design.md`: Ordinal parameter design specification
- `handoff/20251207_ordinal_parameter_support.md`: Original ordinal implementation handoff

### Related Issues
- User request: "如果有这个功能我希望更易用,比如穷举的设定是基于变量的名字之类的" (2025-12-11)
