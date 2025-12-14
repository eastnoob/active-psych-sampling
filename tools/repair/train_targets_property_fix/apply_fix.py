#!/usr/bin/env python3
"""Apply train_targets property delegation fix to ParameterTransformedModel."""

import sys
from pathlib import Path

def find_aepsych_parameters_file():
    """Locate the parameters.py file in the installed aepsych package."""
    # Try common locations
    candidates = [
        Path(".pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py"),
        Path("venv/lib/python3.11/site-packages/aepsych/transforms/parameters.py"),
        Path("venv/lib/python3.10/site-packages/aepsych/transforms/parameters.py"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None

def check_if_fix_needed(content: str) -> bool:
    """Check if the fix is needed (not already applied)."""
    # Look for the train_targets property fix marker
    return "Fix for train_targets shadowing bug" not in content

def apply_fix(file_path: Path) -> bool:
    """Apply the fix to the file."""
    print(f"Reading {file_path}...")
    content = file_path.read_text(encoding='utf-8')

    if not check_if_fix_needed(content):
        print("✅ Fix already applied!")
        return True

    # Find insertion point (after train_inputs fix)
    marker = "# ========== End of fix =========="
    if marker not in content:
        print("❌ Could not find insertion point (train_inputs fix marker not found)")
        return False

    # Find the FIRST occurrence (train_inputs fix)
    insert_pos = content.find(marker) + len(marker)

    # Prepare the fix code
    fix_code = '''

    # ========== Fix for train_targets shadowing bug ==========
    @property
    def train_targets(self) -> torch.Tensor | None:
        """Delegate train_targets to the underlying model."""
        return self._base_obj.train_targets

    @train_targets.setter
    def train_targets(self, value: torch.Tensor | None) -> None:
        """Delegate train_targets setting to the underlying model."""
        self._base_obj.train_targets = value
    # ========== End of fix =========='''

    # Insert the fix
    new_content = content[:insert_pos] + fix_code + content[insert_pos:]

    # Write back
    print(f"Writing fixed content to {file_path}...")
    file_path.write_text(new_content, encoding='utf-8')

    print("✅ Fix applied successfully!")
    return True

def main():
    print("="*80)
    print("ParameterTransformedModel train_targets Property Fix")
    print("="*80)

    # Find the file
    file_path = find_aepsych_parameters_file()
    if file_path is None:
        print("❌ Could not locate aepsych/transforms/parameters.py")
        print("   Please specify the file path manually.")
        return 1

    print(f"Found: {file_path}")

    # Apply the fix
    if apply_fix(file_path):
        print("\n" + "="*80)
        print("Next steps:")
        print("  1. Run verification: pixi run python scripts/run_eur_residual.py --budget 10")
        print("  2. Check for: [DEBUG] Model: train_inputs=N, train_targets=N (matching)")
        print("="*80)
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
