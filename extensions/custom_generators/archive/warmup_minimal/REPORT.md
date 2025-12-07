# Warmup Minimal Generator - Build Report

- Date: 2025-10-30
- Location: `extensions/custom_generators/warmup_minimal`

## Summary

Implemented a high-information-density warmup generator focusing on main effects, with minimal design sizes and optional center point. The generator is integrated into the extensions package and registered with AEPsych's `Config` registry.

## Files

- `warmup_minimal_generator.py`: Implementation with L4, 8-run 2^3 with derived columns, PB(12), PB(20), and 16-run Hadamard options; linear mapping to [lb, ub].
- `__init__.py`: Export symbol.
- `README.md`: Usage and design documentation.
- `tests/run_tests.py`: Lightweight verification script.

## Test Run (via Pixi)

To run tests:

```powershell
pixi run -- python extensions/custom_generators/warmup_minimal/tests/run_tests.py
```

Observed output:

```text
ALL TESTS PASSED
```

## Notes

- Design rules implemented as requested; for k=8 we promote to PB(12) to keep main-effect full rank.
- Supports up to k=19 (via PB 20-run). Extendable by adding larger PB tables or OA generators if needed.
- Mixed categorical factors are not currently supported; treat as continuous or pre-encode/fix upstream.
