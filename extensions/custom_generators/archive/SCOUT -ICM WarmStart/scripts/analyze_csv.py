#!/usr/bin/env python3
import pandas as pd
import sys

csv_path = r"d:\WORKSPACE\python\aepsych-source\extensions\custom_generators\SCOUT -ICM WarmStart\results\251112_003508\subject_0.csv"
csv = pd.read_csv(csv_path)

print(f"Total rows: {len(csv)}")
print(f"Rows with NaN f1: {csv['f1'].isna().sum()}")
print(f"Rows with NaN design_row_id: {csv['design_row_id'].isna().sum()}")
print(
    f"Rows with non-empty interaction_pair_id: {csv['interaction_pair_id'].notna().sum()}"
)

print("\nBlock type distribution:")
print(csv["block_type"].value_counts())

print("\nCore2 rows with interaction_pair_id (first 5):")
core2_inter = csv[(csv["block_type"] == "core2") & (csv["interaction_pair_id"].notna())]
print(
    core2_inter[
        ["trial_number", "block_type", "interaction_pair_id", "design_row_id", "f1"]
    ].head(5)
)

print("\nSample rows with NaN f-values (if any):")
nan_rows = csv[csv["f1"].isna()]
if len(nan_rows) > 0:
    print(nan_rows[["trial_number", "block_type", "design_row_id", "f1", "f2"]].head(3))
else:
    print("No rows with NaN f-values found!")
