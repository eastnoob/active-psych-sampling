#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Fix Application Script for Parameter Transform Skip Patch

This script applies patches to:
1. AEPsych ParameterTransformedGenerator (adds _skip_untransform check)
2. AEPsych ManualGenerator (adds _skip_untransform flag)
3. CustomPoolBasedGenerator (adds _skip_untransform flag)
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Determine project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Target files
AEPSYCH_PARAMETERS = Path(".pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py")
AEPSYCH_MANUAL_GEN = Path(".pixi/envs/default/Lib/site-packages/aepsych/generators/manual_generator.py")
CUSTOM_POOL_GEN = PROJECT_ROOT / "extensions/custom_generators/custom_pool_based_generator.py"

def backup_file(file_path: Path) -> Path:
    """Create backup of file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
    shutil.copy2(file_path, backup_path)
    print(f"✓ Backed up: {file_path.name} -> {backup_path.name}")
    return backup_path

def apply_parameters_patch(file_path: Path) -> bool:
    """Apply patch to ParameterTransformedGenerator.gen()."""
    content = file_path.read_text(encoding="utf-8")

    # Check if already patched
    if "_skip_untransform" in content:
        print(f"⚠ {file_path.name} already patched, skipping")
        return False

    # Find and replace
    old_code = """        x = self._base_obj.gen(
            num_points, model, fixed_features=transformed_fixed, **kwargs
        )
        return self.transforms.untransform(x)"""

    new_code = """        x = self._base_obj.gen(
            num_points, model, fixed_features=transformed_fixed, **kwargs
        )

        # ========== PATCH: Allow generators to skip untransform ==========
        # Check if generator explicitly signals it outputs actual values
        if hasattr(self._base_obj, '_skip_untransform') and self._base_obj._skip_untransform:
            return x  # Already in actual value space, skip untransform
        # ========== End of patch ==========

        return self.transforms.untransform(x)"""

    if old_code not in content:
        print(f"❌ {file_path.name}: Cannot find target code to patch")
        return False

    backup_file(file_path)
    content = content.replace(old_code, new_code)
    file_path.write_text(content, encoding="utf-8")
    print(f"✓ Patched: {file_path.name}")
    return True

def apply_manual_generator_patch(file_path: Path) -> bool:
    """Apply patch to ManualGenerator class."""
    content = file_path.read_text(encoding="utf-8")

    # Check if already patched
    if "_skip_untransform" in content:
        print(f"⚠ {file_path.name} already patched, skipping")
        return False

    # Find and replace
    old_code = """class ManualGenerator(AEPsychGenerator):
    \"\"\"Generator that generates points from a predefined list.\"\"\"

    _requires_model = False"""

    new_code = """class ManualGenerator(AEPsychGenerator):
    \"\"\"Generator that generates points from a predefined list.\"\"\"

    _requires_model = False
    _skip_untransform = True  # Signal that points are already in actual value space"""

    if old_code not in content:
        print(f"❌ {file_path.name}: Cannot find target code to patch")
        return False

    backup_file(file_path)
    content = content.replace(old_code, new_code)
    file_path.write_text(content, encoding="utf-8")
    print(f"✓ Patched: {file_path.name}")
    return True

def apply_custom_pool_generator_patch(file_path: Path) -> bool:
    """Apply patch to CustomPoolBasedGenerator class."""
    content = file_path.read_text(encoding="utf-8")

    # Check if already patched
    if "_skip_untransform" in content:
        print(f"⚠ {file_path.name} already patched, skipping")
        return False

    # Find and replace
    old_code = """    _requires_model = True  # Changed to True since we use acquisition functions

    def __init__("""

    new_code = """    _requires_model = True  # Changed to True since we use acquisition functions
    _skip_untransform = True  # Signal that we output actual values directly

    def __init__("""

    if old_code not in content:
        print(f"❌ {file_path.name}: Cannot find target code to patch")
        return False

    backup_file(file_path)
    content = content.replace(old_code, new_code)
    file_path.write_text(content, encoding="utf-8")
    print(f"✓ Patched: {file_path.name}")
    return True

def main():
    print("="*80)
    print("Parameter Transform Skip Patch - Application Script")
    print("="*80)
    print()

    success_count = 0

    # Apply patches
    print("[1/3] Patching AEPsych ParameterTransformedGenerator...")
    if apply_parameters_patch(AEPSYCH_PARAMETERS):
        success_count += 1
    print()

    print("[2/3] Patching AEPsych ManualGenerator...")
    if apply_manual_generator_patch(AEPSYCH_MANUAL_GEN):
        success_count += 1
    print()

    print("[3/3] Patching CustomPoolBasedGenerator...")
    if apply_custom_pool_generator_patch(CUSTOM_POOL_GEN):
        success_count += 1
    print()

    print("="*80)
    if success_count == 3:
        print("✓ All patches applied successfully!")
        print()
        print("Next steps:")
        print("  1. Run verification: pixi run python tools/repair/parameter_transform_skip/verify_fix.py")
        print("  2. Test with integration test: pixi run python tests/is_EUR_work/00_plans/251206/scripts/run_eur_residual.py --budget 10")
    elif success_count > 0:
        print(f"⚠ Partial success: {success_count}/3 patches applied")
        print("Some files were already patched or could not be patched.")
    else:
        print("❌ No patches applied (all files already patched or patch failed)")
    print("="*80)

    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
