#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Script for Parameter Transform Skip Patch

Checks if all patches have been correctly applied.
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Target files
AEPSYCH_PARAMETERS = Path(".pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py")
AEPSYCH_MANUAL_GEN = Path(".pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py")
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CUSTOM_POOL_GEN = PROJECT_ROOT / "extensions/custom_generators/custom_pool_based_generator.py"

def verify_parameters_patch() -> bool:
    """Verify ParameterTransformedGenerator patch."""
    if not AEPSYCH_PARAMETERS.exists():
        print(f"❌ File not found: {AEPSYCH_PARAMETERS}")
        return False

    content = AEPSYCH_PARAMETERS.read_text(encoding="utf-8")

    checks = [
        ("hasattr(self._base_obj, '_skip_untransform')", "_skip_untransform check"),
        ("return x  # Already in actual value space", "Early return on skip"),
        ("# ========== PATCH: Allow generators to skip untransform", "Patch comment marker"),
    ]

    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_passed = False

    return all_passed

def verify_manual_generator_patch() -> bool:
    """Verify ManualGenerator patch."""
    if not AEPSYCH_MANUAL_GEN.exists():
        print(f"❌ File not found: {AEPSYCH_MANUAL_GEN}")
        return False

    content = AEPSYCH_MANUAL_GEN.read_text(encoding="utf-8")

    checks = [
        ("_skip_untransform = True", "_skip_untransform flag"),
        ("# Signal that points are already in actual value space", "Comment explanation"),
    ]

    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_passed = False

    return all_passed

def verify_custom_pool_generator_patch() -> bool:
    """Verify CustomPoolBasedGenerator patch."""
    if not CUSTOM_POOL_GEN.exists():
        print(f"❌ File not found: {CUSTOM_POOL_GEN}")
        return False

    content = CUSTOM_POOL_GEN.read_text(encoding="utf-8")

    checks = [
        ("_skip_untransform = True", "_skip_untransform flag"),
        ("# Signal that we output actual values directly", "Comment explanation"),
    ]

    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_passed = False

    return all_passed

def main():
    print("="*80)
    print("Parameter Transform Skip Patch - Verification")
    print("="*80)
    print()

    print("[1/3] Verifying AEPsych ParameterTransformedGenerator patch...")
    param_ok = verify_parameters_patch()
    print()

    print("[2/3] Verifying AEPsych ManualGenerator patch...")
    manual_ok = verify_manual_generator_patch()
    print()

    print("[3/3] Verifying CustomPoolBasedGenerator patch...")
    custom_ok = verify_custom_pool_generator_patch()
    print()

    print("="*80)
    print("Verification Results")
    print("="*80)
    print(f"  ParameterTransformedGenerator: {'[OK] PATCHED' if param_ok else '[FAIL] NOT PATCHED'}")
    print(f"  ManualGenerator:               {'[OK] PATCHED' if manual_ok else '[FAIL] NOT PATCHED'}")
    print(f"  CustomPoolBasedGenerator:      {'[OK] PATCHED' if custom_ok else '[FAIL] NOT PATCHED'}")
    print()

    if param_ok and manual_ok and custom_ok:
        print("[SUCCESS] All patches verified!")
        print()
        print("Expected behavior:")
        print("  - Generators with _skip_untransform=True will skip untransform")
        print("  - Categorical numeric parameters return actual values (2.8, 4.0, 8.5)")
        print("  - No more double/triple transformation (5.6, 17.0, 51.2)")
        print()
        print("Next step:")
        print("  Run integration test to confirm fix works in real scenario")
        return 0
    else:
        print("[FAIL] Some patches missing or incomplete")
        print()
        print("Fix:")
        print("  pixi run python tools/repair/parameter_transform_skip/apply_fix.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
