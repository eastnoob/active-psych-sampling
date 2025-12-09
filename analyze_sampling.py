#!/usr/bin/env python3
"""分析最新采样数据中的重复点"""

import pandas as pd
import numpy as np


def analyze_sampling_data(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)

    print("=== 采样数据分析 ===")
    print(f"总采样点数: {len(df)}")
    print(f'Golden warmup阶段: {len(df[df["phase"] == "golden_warmup"])} 个点')
    print(f'EUR optimization阶段: {len(df[df["phase"] == "eur_optimization"])} 个点')

    # 提取坐标列
    coords_cols = ["x0", "x1", "x2", "x3", "x4", "x5"]
    coords = df[coords_cols]

    print("\n=== 重复点检查 ===")

    # 检查完全重复的点
    duplicates = coords.duplicated(keep=False)
    if duplicates.any():
        print("❌ 发现重复点:")
        dup_points = df[duplicates]
        for idx, row in dup_points.iterrows():
            print(
                f'  迭代{row["iteration"]} ({row["phase"]}): [{row["x0"]}, {row["x1"]}, {row["x2"]}, {row["x3"]}, {row["x4"]}, {row["x5"]}]'
            )
    else:
        print("✅ 没有发现完全重复的点")

    # 分阶段检查
    print("\n=== 分阶段分析 ===")
    warmup = df[df["phase"] == "golden_warmup"]
    eur = df[df["phase"] == "eur_optimization"]

    print("Golden warmup点:")
    for idx, row in warmup.iterrows():
        print(
            f'  迭代{row["iteration"]}: [{row["x0"]}, {row["x1"]}, {row["x2"]}, {row["x3"]}, {row["x4"]}, {row["x5"]}]'
        )

    print("\nEUR阶段是否重用了warmup点:")
    warmup_coords = warmup[coords_cols]
    eur_coords = eur[coords_cols]

    reuse_found = False
    for w_idx, w_row in warmup_coords.iterrows():
        w_point = w_row.values
        for e_idx, e_row in eur_coords.iterrows():
            e_point = e_row.values
            if np.allclose(w_point, e_point, atol=1e-6):
                w_iter = df.loc[w_idx, "iteration"]
                e_iter = df.loc[e_idx, "iteration"]
                print(f"  ❌ Warmup点{w_iter}被EUR迭代{e_iter}重复使用: {w_point}")
                reuse_found = True

    if not reuse_found:
        print("  ✅ EUR阶段没有重复使用warmup点")

    print("\n=== 距离分析 ===")
    min_distances = df["min_distance"].dropna()
    if len(min_distances) > 0:
        print(f"最小距离统计:")
        print(f"  平均: {min_distances.mean():.3f}")
        print(f"  最小: {min_distances.min():.3f}")
        print(f"  最大: {min_distances.max():.3f}")
        zero_dist_count = (min_distances == 0.0).sum()
        if zero_dist_count > 0:
            print(f"  ⚠️  零距离点数: {zero_dist_count} (可能是重复点)")
            # 显示零距离点的详情
            zero_dist_rows = df[df["min_distance"] == 0.0]
            for _, row in zero_dist_rows.iterrows():
                print(
                    f'    迭代{row["iteration"]}: [{row["x0"]}, {row["x1"]}, {row["x2"]}, {row["x3"]}, {row["x4"]}, {row["x5"]}]'
                )
        else:
            print(f"  ✅ 没有零距离点")

    return not duplicates.any() and not reuse_found


if __name__ == "__main__":
    csv_path = "tests/is_EUR_work/00_plans/251206/scripts/results/20251209_172709/data_files/test_data.csv"
    success = analyze_sampling_data(csv_path)

    if success:
        print("\n🎉 历史点排除功能工作正常，没有发现重复采样！")
    else:
        print("\n⚠️  发现了重复采样问题。")
