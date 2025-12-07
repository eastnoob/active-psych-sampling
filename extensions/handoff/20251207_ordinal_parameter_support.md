# Ordinal Parameter Support Implementation Plan

**Date**: 2025-12-07
**Task**: Add ordinal categorical parameter support (ordered categories with integer distance metric)
**Priority**: Optional enhancement (not required for current project)
**Estimated Effort**: 1-2 days (150 LOC)

---

## 📌 Core Problem

AEPsych treats all categorical parameters as **unordered** using Hamming distance (0 or 1):

```python
# Current behavior for choices = [2.8, 4.0, 8.5]
dist(2.8, 4.0) = 1  # Adjacent values
dist(2.8, 8.5) = 1  # Far values
# CategoricalKernel: k(x1, x2) = exp(-dist / lengthscale)
```

**Desired behavior for ordinal parameters**:

```python
# Ordinal mapping: [2.8, 4.0, 8.5] → [0, 1, 2]
dist(0, 1) = 1   # Adjacent
dist(0, 2) = 2   # Far apart
# RBF/Matern kernel: k(x1, x2) = exp(-||x1-x2||^2 / (2*ls^2))
```

---

## 🎯 Design Goals

1. **Minimal core changes**: Do NOT modify `aepsych/config.py` parameter type validation
2. **Reuse categorical infrastructure**: Config still uses `par_type = categorical`
3. **Custom kernel selection**: Factory decides RBF vs CategoricalKernel per parameter
4. **Clear user intent**: Explicit `ordinal_params` vs `discrete_params` in config

---

## 🏗️ Architecture Design

### Approach: Custom Factory Extension (Recommended)

```
User Config:
  par_type = categorical (for all discrete params)
  choices = [2.8, 4.0, 8.5]
       ↓
Factory reads config:
  ordinal_params = [x1, x2]      # User specifies: treat as ordered
  discrete_params = [x3, x4, x5, x6]  # Treat as unordered
       ↓
Transform layer:
  Categorical transform: [2.8, 4.0, 8.5] → [0, 1, 2]
  (No changes needed - already works)
       ↓
Kernel selection:
  Ordinal dims → RBFKernel(active_dims=[0, 1])
  Discrete dims → CategoricalKernel(active_dims=[2, 3, 4, 5])
       ↓
Final covar_module:
  ProductKernel(RBF_ordinal, Categorical_unordered)
```

---

## 📂 Files to Modify/Create

### File 1: `extensions/custom_factory/custom_basegp_residual_mixed_factory.py` (MODIFY)

**Current state**: Supports `continuous_params` and `discrete_params`

**Changes needed**: Add `ordinal_params` support

**Additions**: ~100 lines

**Key changes**:

```python
class CustomBaseGPResidualMixedFactory(MeanCovarFactory):
    def __init__(
        self,
        dim: int,
        continuous_params: list = None,   # Existing
        discrete_params: dict = None,     # Existing (categorical)
        ordinal_params: dict = None,      # NEW: ordinal categorical
        basegp_scan_csv: str = None,
        mean_type: str = "pure_residual",
        ...
    ):
        """
        Args:
            continuous_params: List of continuous parameter names
            discrete_params: Dict {param_name: n_categories} for unordered categorical
            ordinal_params: Dict {param_name: n_levels} for ordered categorical
                           Will use RBF kernel instead of CategoricalKernel

        Example:
            continuous_params = []
            ordinal_params = {'x1_CeilingHeight': 3, 'x2_GridModule': 2}
            discrete_params = {'x3_OuterFurniture': 3, 'x4_VisualBoundary': 3}
        """
        self.continuous_params = continuous_params or []
        self.ordinal_params = ordinal_params or {}
        self.discrete_params = discrete_params or {}

        # Validate: no overlap between param types
        all_param_sets = [
            set(self.continuous_params),
            set(self.ordinal_params.keys()),
            set(self.discrete_params.keys())
        ]
        # Check pairwise disjoint
        for i in range(len(all_param_sets)):
            for j in range(i+1, len(all_param_sets)):
                overlap = all_param_sets[i] & all_param_sets[j]
                if overlap:
                    raise ValueError(f"Parameter overlap detected: {overlap}")

    def _make_covar_module(self, ...):
        """Build covariance kernel with ordinal support"""
        kernels = []

        # 1. Continuous parameters → MaternKernel
        if self.continuous_params:
            cont_dims = self._get_dims_for_params(self.continuous_params)
            continuous_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                active_dims=cont_dims,
                ard_num_dims=len(cont_dims),
                lengthscale_prior=...
            )
            kernels.append(continuous_kernel)

        # 2. Ordinal parameters → RBFKernel (treat as integers)
        if self.ordinal_params:
            ordinal_dims = self._get_dims_for_params(list(self.ordinal_params.keys()))
            ordinal_kernel = gpytorch.kernels.RBFKernel(
                active_dims=ordinal_dims,
                ard_num_dims=len(ordinal_dims),
                lengthscale_prior=...  # Same prior as continuous
            )
            kernels.append(ordinal_kernel)

        # 3. Discrete parameters → CategoricalKernel (unordered)
        if self.discrete_params:
            discrete_dims = self._get_dims_for_params(list(self.discrete_params.keys()))
            categorical_kernel = botorch.models.kernels.CategoricalKernel(
                active_dims=tuple(discrete_dims),
                ard_num_dims=len(discrete_dims),
                lengthscale_constraint=...
            )
            kernels.append(categorical_kernel)

        # Combine kernels
        if len(kernels) == 0:
            raise ValueError("Must have at least one parameter type")
        elif len(kernels) == 1:
            base_kernel = kernels[0]
        else:
            base_kernel = gpytorch.kernels.ProductKernel(*kernels)

        # Apply scaling if needed
        if not self.fixed_kernel_amplitude:
            base_kernel = gpytorch.kernels.ScaleKernel(
                base_kernel,
                outputscale_prior=...
            )

        return base_kernel

    def _get_dims_for_params(self, param_names: list) -> list[int]:
        """Map parameter names to dimension indices"""
        # Assumes dimension order: continuous, ordinal, discrete
        dim_mapping = {}
        idx = 0
        for p in self.continuous_params:
            dim_mapping[p] = idx
            idx += 1
        for p in self.ordinal_params.keys():
            dim_mapping[p] = idx
            idx += 1
        for p in self.discrete_params.keys():
            dim_mapping[p] = idx
            idx += 1

        return [dim_mapping[p] for p in param_names]

    @classmethod
    def get_config_options(cls, config: Config, name: str, options: dict):
        """Parse ordinal_params from config"""
        options = super().get_config_options(config, name, options)

        # Parse ordinal_params from config
        if config.has_option(name, "ordinal_params"):
            ordinal_dict = config.getobj(name, "ordinal_params")
            # ordinal_dict format: {'x1_CeilingHeight': 3, 'x2_GridModule': 2}
            options["ordinal_params"] = ordinal_dict

        return options
```

---

### File 2: Unit Tests (NEW)

**File**: `extensions/custom_factory/test_ordinal_support.py`

**Size**: ~100 lines

**Test cases**:

```python
def test_ordinal_only_parameters():
    """Test factory with only ordinal parameters"""
    factory = CustomBaseGPResidualMixedFactory(
        dim=2,
        ordinal_params={'x1': 3, 'x2': 2},
        basegp_scan_csv="...",
    )
    covar = factory._make_covar_module(...)
    # Assert: Uses RBFKernel
    # Assert: active_dims = [0, 1]

def test_mixed_ordinal_and_categorical():
    """Test mixed ordinal + unordered categorical"""
    factory = CustomBaseGPResidualMixedFactory(
        dim=4,
        ordinal_params={'x1': 3, 'x2': 2},
        discrete_params={'x3': 3, 'x4': 3},
        basegp_scan_csv="...",
    )
    covar = factory._make_covar_module(...)
    # Assert: ProductKernel(RBFKernel, CategoricalKernel)
    # Assert: RBF active_dims=[0,1], Cat active_dims=[2,3]

def test_ordinal_kernel_distance_metric():
    """Verify ordinal parameters use Euclidean distance"""
    X1 = torch.tensor([[0.0, 0.0]])  # [x1=0, x2=0]
    X2 = torch.tensor([[1.0, 0.0]])  # [x1=1, x2=0] - adjacent in x1
    X3 = torch.tensor([[2.0, 0.0]])  # [x1=2, x2=0] - far in x1

    kernel = create_ordinal_kernel(active_dims=[0, 1])

    k12 = kernel(X1, X2).item()  # Adjacent
    k13 = kernel(X1, X3).item()  # Far

    # Assert: k12 > k13 (adjacent more similar than far)
    assert k12 > k13

def test_no_parameter_overlap():
    """Ensure continuous/ordinal/discrete params don't overlap"""
    with pytest.raises(ValueError, match="overlap"):
        factory = CustomBaseGPResidualMixedFactory(
            dim=3,
            continuous_params=['x1'],
            ordinal_params={'x1': 3},  # Duplicate!
        )

def test_backward_compatibility():
    """Old configs without ordinal_params still work"""
    factory = CustomBaseGPResidualMixedFactory(
        dim=2,
        discrete_params={'x1': 3, 'x2': 2},  # Old style
        basegp_scan_csv="...",
    )
    # Should not crash - ordinal_params defaults to {}
```

---

## 📝 Configuration Example

### Before (current - all categorical treated as unordered):

```ini
[common]
parnames = [x1_CeilingHeight, x2_GridModule, x3_OuterFurniture, x4_VisualBoundary]

[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[x2_GridModule]
par_type = categorical
choices = [6.5, 8.0]

[x3_OuterFurniture]
par_type = categorical
choices = [Chaos, Rotated, Strict]

[x4_VisualBoundary]
par_type = categorical
choices = [Color, Solid, Translucent]

[CustomBaseGPResidualMixedFactory]
continuous_params = []
discrete_params = {'x1_CeilingHeight': 3, 'x2_GridModule': 2, 'x3_OuterFurniture': 3, 'x4_VisualBoundary': 3}
```

### After (with ordinal support):

```ini
[common]
parnames = [x1_CeilingHeight, x2_GridModule, x3_OuterFurniture, x4_VisualBoundary]

# Parameter definitions stay the same
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[x2_GridModule]
par_type = categorical
choices = [6.5, 8.0]

[x3_OuterFurniture]
par_type = categorical
choices = [Chaos, Rotated, Strict]

[x4_VisualBoundary]
par_type = categorical
choices = [Color, Solid, Translucent]

# Factory distinguishes ordinal vs unordered
[CustomBaseGPResidualMixedFactory]
continuous_params = []
ordinal_params = {'x1_CeilingHeight': 3, 'x2_GridModule': 2}  # NEW: Ordered
discrete_params = {'x3_OuterFurniture': 3, 'x4_VisualBoundary': 3}  # Unordered
basegp_scan_csv = extensions/warmup_budget_check/phase1_analysis_output/202512070056/step3/design_space_scan.csv
mean_type = pure_residual
```

---

## ✅ Implementation Checklist

### Phase 1: Core Functionality (4 hours)

- [ ] Modify `CustomBaseGPResidualMixedFactory.__init__()` to accept `ordinal_params`
- [ ] Add parameter overlap validation
- [ ] Implement `_get_dims_for_params()` helper method
- [ ] Modify `_make_covar_module()` to build RBF kernel for ordinal params
- [ ] Update `get_config_options()` to parse `ordinal_params` from config

### Phase 2: Testing (3 hours)

- [ ] Create `test_ordinal_support.py`
- [ ] Test ordinal-only configuration
- [ ] Test mixed ordinal + categorical configuration
- [ ] Test kernel distance metric correctness
- [ ] Test parameter overlap validation
- [ ] Test backward compatibility

### Phase 3: Documentation (1 hour)

- [ ] Add docstring examples for ordinal_params
- [ ] Document dimension ordering convention
- [ ] Add ordinal example to README
- [ ] Update config template with ordinal example

---

## 🚨 Important Decisions

### Why Not Create New Parameter Type?

**Rejected**: Adding `par_type = ordinal` to AEPsych core

**Reasons**:
1. Requires modifying AEPsych source code (~285 lines across multiple files)
2. Increases maintenance burden
3. Limited benefit: config-level distinction not necessary

**Chosen**: Factory-level distinction

**Advantages**:
1. Zero changes to AEPsych core
2. User explicitly declares semantic intent in Factory config
3. Easy to maintain as extension

### Why RBF Instead of Matern for Ordinal?

**Recommendation**: Use `RBFKernel` for ordinal parameters

**Reasoning**:
- RBF (squared exponential): Infinitely differentiable, smooth
- Matern-2.5: Twice differentiable, less smooth
- For ordinal integers [0, 1, 2], smoothness matters less
- RBF is simpler and standard for discrete inputs

**Alternative**: Can use Matern if preferred, just match continuous param kernel type

### Dimension Ordering Convention

**Critical**: Establish clear dimension ordering to avoid index confusion

**Chosen convention**:
```
train_X dimensions:
  [0 ... n_cont-1]:           Continuous parameters
  [n_cont ... n_cont+n_ord-1]: Ordinal parameters
  [n_cont+n_ord ... n_total-1]: Discrete parameters
```

**Why this order**:
- Natural progression: continuous → semi-continuous (ordinal) → discrete
- Easy to slice: `X[:, :n_cont]` gets continuous, etc.

---

## 📊 Expected Benefits

### Quantitative Improvements (for ordinal parameters)

| Metric | Categorical (0/1 distance) | Ordinal (integer distance) | Improvement |
|--------|---------------------------|---------------------------|-------------|
| **Data efficiency** | Baseline | ~10-15% fewer samples | Moderate |
| **Interpolation** | Equal similarity | Smooth gradient | Better |
| **Prior inductive bias** | No structure | Assumes ordering | Stronger |

### When Benefits Are Highest

1. **No population priors**: Learning relationships from scratch
2. **Continuous optimization**: Smooth acquisition function gradients matter
3. **Many ordinal levels**: 5+ levels benefit more than 2-3 levels

### When Benefits Are Limited

1. **BaseGP + residual learning**: Population prior already knows relationships
2. **Pool-based generator**: Discrete search, no smooth gradients
3. **Few ordinal levels**: 2 levels (binary) - no benefit over categorical

---

## 🔧 Alternative: Integer Workaround (Zero-Effort Option)

If implementation is deprioritized, use this immediate workaround:

### Configuration

```ini
[x1_CeilingHeight]
par_type = integer
lb = 0
ub = 2
# Mapping: 0→2.8m, 1→4.0m, 2→8.5m (document in comment)

[x2_GridModule]
par_type = integer
lb = 0
ub = 1
# Mapping: 0→6.5m, 1→8.0m
```

### External Mapping Code

```python
# design_space.py
ORDINAL_MAPPINGS = {
    'x1_CeilingHeight': {0: 2.8, 1: 4.0, 2: 8.5},
    'x2_GridModule': {0: 6.5, 1: 8.0},
}

def map_indices_to_values(X_indices: np.ndarray) -> np.ndarray:
    """Convert integer indices back to original values"""
    X_values = X_indices.copy()
    for param_name, mapping in ORDINAL_MAPPINGS.items():
        col_idx = PARAM_NAME_TO_COL[param_name]
        X_values[:, col_idx] = [mapping[int(x)] for x in X_indices[:, col_idx]]
    return X_values
```

**Trade-offs**:
- ✅ Immediate: Works now (0 implementation time)
- ✅ Functional: Gets 90% of ordinal benefits
- ⚠️ Manual: Requires external mapping management
- ⚠️ Error-prone: Mapping mistakes possible
- ⚠️ Less clear: Config doesn't show actual values

---

## 🎯 Success Criteria

### Functional Requirements

- [ ] Ordinal parameters use RBF/Matern kernel (not CategoricalKernel)
- [ ] Kernel distance is integer-based: `dist(0,1) < dist(0,2)`
- [ ] Mixed ordinal + categorical configurations work correctly
- [ ] Backward compatible: old configs without `ordinal_params` still work

### Testing Requirements

- [ ] 6+ unit tests covering key scenarios
- [ ] Test coverage > 85% for modified code
- [ ] No breaking changes to existing BaseGP functionality

### Documentation Requirements

- [ ] Clear docstring for `ordinal_params` parameter
- [ ] Example configuration showing ordinal usage
- [ ] Dimension ordering convention documented
- [ ] Migration guide: categorical → ordinal conversion

---

## 📈 Implementation Priority

### Priority: LOW (Optional Enhancement)

**Context from current project**:
- Using BaseGP population priors (main relationships already learned)
- Pool-based generator (324 discrete points, not continuous optimization)
- Only 2 numeric categorical variables (x1, x2)
- Project timeline: Need results soon

**Recommendation**:
1. **For current project**: Use integer workaround (0 effort, 90% benefit)
2. **For future projects**: Implement full ordinal support if:
   - Multiple experiments need ordinal parameters
   - Learning without population priors
   - Team prefers explicit config semantics

**ROI Calculation**:
- Implementation cost: 1-2 days
- Per-project benefit: ~0.5 day saved (config + mapping)
- Break-even: ~3-4 projects

---

## 🔄 Migration Path

### For existing projects using categorical:

**Step 1**: Identify ordinal parameters
```python
# Review parameter semantics
x1_CeilingHeight: [2.8, 4.0, 8.5]  → Ordinal (clear ordering)
x2_GridModule: [6.5, 8.0]          → Ordinal (numeric)
x3_OuterFurniture: [Chaos, Strict] → Categorical (no inherent order)
```

**Step 2**: Update factory config
```ini
# Before
discrete_params = {'x1': 3, 'x2': 2, 'x3': 3}

# After
ordinal_params = {'x1': 3, 'x2': 2}
discrete_params = {'x3': 3}
```

**Step 3**: Verify kernel selection
```python
# Check that RBF kernel is used for ordinal dims
print(model.covar_module)
# Expected: ProductKernel(RBFKernel(active_dims=[0,1]), CategoricalKernel(active_dims=[2]))
```

---

## 💡 Design Rationale Summary

| Choice | Rationale |
|--------|-----------|
| **Factory-level distinction** | No AEPsych core changes, minimal LOC |
| **Reuse categorical config** | No new parameter type, simpler validation |
| **RBF kernel for ordinal** | Standard for discrete inputs, smooth |
| **Explicit ordinal_params** | Clear user intent, self-documenting |
| **Product kernel combination** | Standard practice, acquisition compatible |

---

## 📚 Reference Materials

### Related AEPsych Components

- `aepsych/config.py`: Parameter type validation (lines 358-414)
- `aepsych/transforms/ops/categorical.py`: Categorical transform (string→integer)
- `aepsych/factory/mixed.py`: Mixed kernel factory reference (lines 116-133)
- `botorch/models/kernels/categorical.py`: CategoricalKernel implementation

### Key Concepts

- **Categorical kernels**: Hamming distance (0/1) for unordered categories
- **Ordinal encodings**: Integer mapping preserves order
- **GP with mixed inputs**: Gaussian Processes for Machine Learning (Rasmussen & Williams, 2006)
- **Product kernels**: Multiplicative combination of kernels for different dimensions

---

**Status**: Implementation plan ready, awaiting prioritization decision

**Recommendation**: Use integer workaround for current project, implement full support if needed for future work

**Next Step If Implementing**: Start with Phase 1 (4 hours core functionality modification)
