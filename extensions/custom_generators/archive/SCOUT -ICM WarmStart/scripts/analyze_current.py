#!/usr/bin/env python3
import pandas as pd

csv = pd.read_csv(
    r"d:\WORKSPACE\python\aepsych-source\extensions\custom_generators\SCOUT -ICM WarmStart\results\251112_004016\trial_schedule.csv"
)
print("Subject distribution:")
print(csv["subject_id"].value_counts().sort_index())
print()
print("Total trials:", len(csv))
print()
print("First 30 rows:")
print(csv[["subject_id", "batch_id", "block_type"]].head(30))
