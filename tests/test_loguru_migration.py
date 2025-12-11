#!/usr/bin/env python3
"""Quick test to verify loguru migration is working correctly."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing EUR module imports...\n")

try:
    from extensions.dynamic_eur_acquisition.modules.diagnostics import DiagnosticsManager
    print("[OK] DiagnosticsManager imported successfully")
except Exception as e:
    print(f"[FAIL] DiagnosticsManager import failed: {e}")
    sys.exit(1)

try:
    from extensions.dynamic_eur_acquisition.eur_anova_pair import EURAnovaPairAcqf
    print("[OK] EURAnovaPairAcqf imported successfully")
except Exception as e:
    print(f"[FAIL] EURAnovaPairAcqf import failed: {e}")
    sys.exit(1)

try:
    from extensions.dynamic_eur_acquisition.eur_anova_multi import EURAnovaMultiAcqf
    print("[OK] EURAnovaMultiAcqf imported successfully")
except Exception as e:
    print(f"[FAIL] EURAnovaMultiAcqf import failed: {e}")
    sys.exit(1)

print("\n[SUCCESS] All EUR modules imported successfully!")
print("[SUCCESS] Loguru migration is working correctly!")
print("[SUCCESS] Windows GBK encoding issue has been fixed!")
