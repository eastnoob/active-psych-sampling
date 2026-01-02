# KEY_CESHI 1:1 Replication Status

## ✅ Completed

1. **EUR Role Configuration** ([roles/eur_role.py](roles/eur_role.py))
   - ✅ ManualGenerator for warmup
   - ✅ CustomPoolBasedGenerator for EUR sampling
   - ✅ EURAnovaMultiAcqf acquisition function with all parameters
   - ✅ GPRegressionModel with CustomBaseGPResidualFactory
   - ✅ All KEY_CESHI INI parameters replicated

2. **Custom Components Import** ([main.py](main.py))
   - ✅ CustomPoolBasedGenerator imported
   - ✅ EURAnovaMultiAcqf imported
   - ✅ CustomBaseGPResidualFactory imported

3. **INI Builder Fix** ([utils/ini_builder.py](utils/ini_builder.py))
   - ✅ Parnames now quoted: `['x1', 'x2', 'x3']`

## ⚠️ Remaining Issues

### 1. Parameter Definitions
**Issue**: Current behaviors only support continuous parameters. KEY_CESHI uses mixed types:
- x1, x2: `custom_ordinal_mono` with specific values
- x3-x6: `categorical` with string choices

**Solution Needed**:
- Behaviors need to parse parameter definitions from TOML
- INI builder needs to generate proper parameter sections
- Oracle needs to work with mixed parameter types

### 2. Outcome Type
**Issue**: Current oracle defaults to binary. KEY_CESHI uses continuous.

**Solution**: Already supported in oracle, just needs proper configuration flow.

### 3. ManualGenerator Warmup Points
**Issue**: ManualGenerator requires warmup points to be provided programmatically.

**Solution**: Behaviors need to inject warmup points into ManualGenerator section.

## Current Test Result

**Command**: `pixi run python main.py --config config/eur_minimal_test.toml`

**Error**: `ValueError: Missing ub or lb in [common] with incomplete parameter-specific bounds`

**Cause**: No parameter sections ([x1], [x2], etc.) in generated INI.

## Simplified Test Approach

For quick verification that KEY_CESHI components work, I recommend:

1. **Use continuous parameters only** (simpler, no mixed types)
2. **Skip ManualGenerator** (use SobolGenerator for warmup)
3. **Test CustomPoolBasedGenerator + EURAnovaMultiAcqf** (core EUR functionality)

This would verify the core EUR replication without the complexity of mixed parameter types.

## Full 1:1 Replication Requirements

To achieve complete 1:1 replication, need to implement:

1. **Mixed Parameter Type Support**
   - Parse parameter definitions from TOML
   - Generate proper INI sections for each parameter type
   - Handle ordinal and categorical parameters in oracle

2. **ManualGenerator Integration**
   - Inject BaseGP keypoints programmatically
   - Or use SobolGenerator as simplified alternative

3. **Design Space Integration**
   - Load design space CSV
   - Inject pool_points into CustomPoolBasedGenerator
   - Handle categorical mappings

## Recommendation

**Option A: Simplified EUR Test** (Quick verification)
- Use continuous parameters
- Use SobolGenerator for warmup
- Test CustomPoolBasedGenerator + EURAnovaMultiAcqf
- Verify EUR acquisition works

**Option B: Full 1:1 Replication** (Complete but complex)
- Implement mixed parameter type support
- Integrate ManualGenerator with warmup points
- Full design space integration
- Requires significant additional work

**My Recommendation**: Start with Option A to verify the core EUR components work, then incrementally add mixed parameter support if needed.
