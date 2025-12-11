#!/usr/bin/env python3
import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "test_data.csv"
df = pd.read_csv(csv_path)

x_cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']

print("="*80)
print("检查重复的x点")
print("="*80)

# 检查重复
duplicates = df[x_cols].duplicated(keep=False)

if duplicates.any():
    print(f"\n[WARNING] 发现 {duplicates.sum()} 个重复点:\n")
    dup_df = df[duplicates][['iteration', 'phase', 'y_value'] + x_cols]
    print(dup_df.to_string(index=False))

    # 按组显示重复点
    print("\n重复组详情:")
    for vals, group in df[duplicates].groupby(x_cols, dropna=False):
        print(f"\n  x = {list(vals)}")
        for idx, row in group.iterrows():
            print(f"    iteration {int(row['iteration'])}: y={row['y_value']}")
else:
    print("\n[OK] 没有重复的x点")

print(f"\n总共 {len(df)} 个样本，{len(df[x_cols].drop_duplicates())} 个不同的x点")
