#!/usr/bin/env python3
"""
Test: New warmup_adapter implementation with interaction-as-features
Verify that the refactored code produces the expected distribution
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
from collections import Counter

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("Test: New Adapter Implementation (Interaction-as-Features)")
print("=" * 80)
print()

# Design space CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# Create temporary test directory
test_dir = Path(__file__).parent / "temp_test_new_adapter"
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

# Copy full design space as subject_1.csv (testing with 1 subject on full space)
df_full = pd.read_csv(design_space_csv)
subject_1_csv = test_dir / "subject_1.csv"
df_full.to_csv(subject_1_csv, index=False)

print(f"Created test directory: {test_dir}")
print(f"Subject will answer on {len(df_full)} points")
print()

# Import the NEW adapter (will be modified)
# For now, this will fail until we implement the changes
print("=" * 80)
print("PLACEHOLDER: Will test new adapter implementation here")
print("=" * 80)
print()

print("Expected configuration (Config #1):")
print("  Main effects: N(0.0, 0.3)")
print("  Interaction x3*x4: 0.12 (fixed)")
print("  Interaction x0*x1: -0.02 (fixed)")
print()

print("Expected distribution:")
print("  Likert 1:  93 (28.7%)")
print("  Likert 2:  45 (13.9%)")
print("  Likert 3:  41 (12.7%)")
print("  Likert 4:  56 (17.3%)")
print("  Likert 5:  89 (27.5%)")
print("  Mean: 3.01")
print()

print("=" * 80)
print("Implementation Steps:")
print("=" * 80)
print()
print("1. Add 'interaction_as_features' parameter to run()")
print("2. When True:")
print("   a. Extend design_space with interaction features (x3*x4, x0*x1)")
print("   b. Sample main weights: N(0.0, 0.3) for original 6 features")
print("   c. Set interaction weights: [0.12, -0.02] for 2 interaction features")
print("   d. Combine into 8-element weight vector")
print("   e. Calculate bias using extended design space")
print("3. Modify ClusterGenerator or use extended design space directly")
print("4. Update fixed_weights format to save 8 weights")
print()

print("Next: Implement these changes in warmup_adapter.py")

# Cleanup
shutil.rmtree(test_dir)

print()
print("=" * 80)
print("Test placeholder completed!")
print("=" * 80)
