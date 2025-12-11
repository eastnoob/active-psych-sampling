# Ordinal Implementation Evaluation Report

Date: 2025-12-11

Summary: point-by-point evaluation of handoff/ordinal plan vs repository implementation.

1) Transform implementation
- Status: Implemented (extension)
- Details: `extensions/dynamic_eur_acquisition/transforms/ops/custom_ordinal.py` implements normalized mapping, transform/untransform, transform_bounds, config parsing, and arithmetic sequence calculation using `linspace`.
- Notes: Matches design principle for normalized space; robust and well-tested in `tests/test_custom_ordinal_transform.py`.

2) LocalSampler integration
- Status: Implemented (extension)
- Details: `extensions/dynamic_eur_acquisition/modules/local_sampler.py` updated. `_precompute_categorical_values` now supports `ordinal`/`custom_ordinal`/`custom_ordinal_mono`. `_perturb_ordinal` implements normalized-space Gaussian jitter + nearest neighbor; hybrid/exhaustive mode included.
- Notes: Well aligned with plan and covered by `extensions/dynamic_eur_acquisition/test/test_ordinal_integration.py`.

3) Variable type parsing
- Status: Implemented
- Details: `extensions/dynamic_eur_acquisition/modules/config_parser.py` recognizes `custom_ordinal` and `custom_ordinal_mono` and maps them correctly.

4) EUR/ANOVA integration and variable inference
- Status: Partially implemented
- Details: eur_anova_pair/_maybe_infer_variable_types attempts to detect AEPsych core `Ordinal` transform but not extension's `CustomOrdinal`. Thus automatic inference of ordinal types may fail if model uses `CustomOrdinal` not `aepsych.Ordinal`.
- Recommendation: Extend inference to detect extension `CustomOrdinal` (check transform class name or presence of `normalized_values`/`physical_to_normalized`) or use config-supplied `variable_types` preferentially.

5) Pool generator integration (extensions/custom_generators)
- Status: Not implemented / Partial
- Details: `extensions/custom_generators/custom_pool_based_generator.py` currently only supports categorical parameters and raises an error for non-categorical; it does not include `custom_ordinal` support.
- Recommendation: Add `par_type` branch for `custom_ordinal`/`custom_ordinal_mono` that uses `CustomOrdinal.get_config_options` to compute normalized values and append to `param_choices_values`. Add unit and integration tests for pool generation.

6) AEPsych core integration (patches)
- Status: Prepared but not applied
- Details: `tools/repair/ordinal_parameter_extension` contains patch files for AEPsych (aepsych/transforms/ops/ordinal.py, parameters.py changes, config.py change, ops/__init__ import), with `apply_fix.py` and `verify_fix.py` automation. The patch creates `Ordinal` class matching extension `CustomOrdinal` behavior.
- Recommendation: Apply patches in an environment with aepsych installed, run verify_fix.py, then add tests that depend on aepsych core patches. Consider adding a note to `apply_fix.py` for supported aepsych versions.

7) Tests
- Status: Mostly implemented (extension-level)
- Details: Unit tests exist for `CustomOrdinal` and LocalSampler ordinal behavior; `test_ordinal_e2e.py` covers end-to-end with `SobolGenerator`. Missing tests for `CustomPoolBasedGenerator` ordinal support and for aepsych core Ordinal class (patches) integration.
- Recommendation: Add `custom_generator` tests for ordinal pool generation; add patch application CI test or a manual verification step if required.

8) Documentation & PR artifacts
- Status: Implemented
- Details: `handoff/ordinal` includes design docs, patch guides, and a `tools/repair` README to apply AEPsych patches. Many docs explain design & rationale.
- Notes: Well documented. Consider adding small `CHANGELOG` stating whether patch applied to AEPsych or left as extension-only.

9) Misc / Edge cases
- Observations:
  - `eur_anova_pair._maybe_infer_variable_types` import of `aepsych.Ordinal` for inference means inference may not work when only `CustomOrdinal` exists; rely on config `variable_types` in that case.
  - `CustomPoolBasedGenerator` adoption of ordinal not implemented; fallback may be to manually provide `pool_points`.
  - Patches for AEPsych core make breaking changes (imports & new transform); patch application should be tested on supported aepsych versions.

10) Suggested next steps
- 1. Implement `custom_ordinal` support in `extensions/custom_generators/custom_pool_based_generator.py` (pool building branch).
- 2. Add unit + integration tests for `custom_generator` ordinal support.
- 3. Improve `_maybe_infer_variable_types` to detect `CustomOrdinal` by transform attributes or class names to cover extension transforms.
- 4. Test core aepsych patches using `apply_fix.py` and ensure `verify_fix.py` passes; add CI job to run verification on an environment with aepsych installed.
- 5. Add a `README` note summarizing which changes are extension-only vs core patch and instructions for maintainers.

Report compiled by: automation check on workspace (2025-12-11)
