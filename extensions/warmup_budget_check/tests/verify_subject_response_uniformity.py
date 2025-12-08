#!/usr/bin/env python3
"""
验证模拟被试对给定自变量空间的因变量分布均匀性

读取自变量空间CSV，用模拟被试生成因变量，检查分布是否均匀
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 添加路径以导入 MixedEffectsLatentSubject
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "tools" / "archive" / "simulate_subject" / "models"))

from subject_models.MixedEffectsLatentSubject import MixedEffectsLatentSubject


def convert_row_to_features_auto(row, df):
    """
    自动检测CSV格式并转换为数值特征向量

    支持两种格式：
    1. i9csy65bljq14ovww2v91 格式：
       - x1_CeilingHeight, x2_GridModule, x3_OuterFurniture,
         x4_VisualBoundary, x5_PhysicalBoundary, x6_InnerFurniture

    2. 6vars 格式：
       - x1_binary, x2_5level_discrete, x3_5level_decimal,
         x4_4level_categorical, x5_3level_categorical, x6_binary
    """
    columns = df.columns.tolist()

    # 检测格式类型
    if 'x1_CeilingHeight' in columns:
        # 格式1：i9csy65bljq14ovww2v91
        x1 = float(row['x1_CeilingHeight'])
        x2 = float(row['x2_GridModule'])

        x3_map = {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
        x3 = x3_map.get(row['x3_OuterFurniture'], 0)

        x4_map = {'Solid': 0, 'Translucent': 1, 'Color': 2}
        x4 = x4_map.get(row['x4_VisualBoundary'], 0)

        x5_map = {'Closed': 0, 'Open': 1}
        x5 = x5_map.get(row['x5_PhysicalBoundary'], 0)

        x6_map = {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
        x6 = x6_map.get(row['x6_InnerFurniture'], 0)

    elif 'x1_binary' in columns:
        # 格式2：6vars
        x1 = int(row['x1_binary'])
        x2 = float(row['x2_5level_discrete'])
        x3 = float(row['x3_5level_decimal'])

        x4_map = {'low': 0, 'mid': 1, 'high': 2, 'max': 3}
        x4_val = row['x4_4level_categorical']
        if isinstance(x4_val, str):
            x4 = x4_map.get(x4_val, 0)
        else:
            x4 = float(x4_val)

        x5_map = {'A': 0, 'B': 1, 'C': 2}
        x5_val = row['x5_3level_categorical']
        if isinstance(x5_val, str):
            x5 = x5_map.get(x5_val, 0)
        else:
            x5 = float(x5_val)

        x6_val = row['x6_binary']
        if isinstance(x6_val, str):
            x6 = 1 if x6_val.strip().lower() in {'true', '1', 't', 'yes'} else 0
        else:
            x6 = 1 if bool(x6_val) else 0
    else:
        raise ValueError(f"无法识别CSV格式，列名：{columns}")

    return np.array([x1, x2, x3, x4, x5, x6], dtype=float)


def test_response_uniformity(
    csv_path,
    n_subjects=10,
    seed=42,
    use_latent=False,
    output_type='likert',
    likert_levels=5,
    likert_mode='percentile',
    population_std=0.4,
    individual_std_percent=1.0,
    output_dir=None,
):
    """
    测试模拟被试对给定自变量空间的响应均匀性

    Args:
        csv_path: 自变量空间CSV路径
        n_subjects: 模拟被试数量
        seed: 随机种子
        use_latent: 是否使用潜变量模型
        output_type: 'continuous' 或 'likert'
        likert_levels: Likert档位数
        likert_mode: 'tanh' 或 'percentile'
        population_std: 群体权重标准差
        individual_std_percent: 个体偏差百分比
        output_dir: 输出目录（默认为当前脚本所在目录）
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在读取自变量空间: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"自变量空间大小: {len(df)} 条件")
    print(f"变量列: {df.columns.tolist()}")

    # 生成所有被试的响应
    all_responses = []

    for subj_idx in range(1, n_subjects + 1):
        subject_seed = seed + subj_idx

        # 创建模拟被试
        subject = MixedEffectsLatentSubject(
            num_features=6,
            num_observed_vars=1,
            seed=subject_seed,
            population_mean=0.0,
            population_std=population_std,
            individual_std_percent=individual_std_percent,
            use_latent=use_latent,
            noise_std=0.1,
            bias=0.0,
        )

        print(f"\n被试 {subj_idx} (seed={subject_seed}):")

        # 对每个条件生成响应
        responses = []
        for idx, row in df.iterrows():
            X = convert_row_to_features_auto(row, df)
            y = subject(X)
            # 如果是列表，取第一个值
            if isinstance(y, list):
                y = y[0]
            responses.append(float(y))

        # 显示该被试的响应统计
        print(f"  连续响应 - 均值: {np.mean(responses):.4f}, "
              f"标准差: {np.std(responses):.4f}, "
              f"范围: [{np.min(responses):.4f}, {np.max(responses):.4f}]")

        all_responses.extend(responses)

    # 转换为数组
    all_responses = np.array(all_responses)

    print(f"\n总计响应数: {len(all_responses)}")
    print(f"连续响应统计:")
    print(f"  均值: {np.mean(all_responses):.4f}")
    print(f"  标准差: {np.std(all_responses):.4f}")
    print(f"  中位数: {np.median(all_responses):.4f}")
    print(f"  范围: [{np.min(all_responses):.4f}, {np.max(all_responses):.4f}]")

    # 如果需要 Likert 转换
    if output_type == 'likert':
        if likert_mode == 'percentile':
            # 百分位数映射
            percentiles = np.linspace(0, 100, likert_levels + 1)
            bins = [np.percentile(all_responses, p) for p in percentiles]
            likert_responses = np.digitize(all_responses, bins[1:-1]) + 1
            likert_responses = np.clip(likert_responses, 1, likert_levels)
        else:  # tanh
            # tanh 映射
            normalized = 0.5 * (np.tanh(all_responses / 0.5) + 1.0)
            likert_responses = np.round(normalized * (likert_levels - 1)).astype(int) + 1
            likert_responses = np.clip(likert_responses, 1, likert_levels)

        # 统计 Likert 分布
        counter = Counter(likert_responses)
        print(f"\nLikert 分布 (模式={likert_mode}):")
        print(f"{'档位':<8} {'计数':<10} {'百分比':<10} {'柱状图'}")
        print("-" * 60)

        for level in range(1, likert_levels + 1):
            count = counter.get(level, 0)
            pct = 100 * count / len(likert_responses)
            bar = '█' * int(pct / 2)
            print(f"{level:<8} {count:<10} {pct:>6.2f}%    {bar}")

        # 计算均匀性指标
        expected_count = len(likert_responses) / likert_levels
        chi_square = sum((counter.get(i, 0) - expected_count)**2 / expected_count
                        for i in range(1, likert_levels + 1))

        print(f"\n均匀性检验:")
        print(f"  期望每档计数: {expected_count:.2f}")
        print(f"  卡方统计量: {chi_square:.4f}")
        print(f"  自由度: {likert_levels - 1}")

        # 判断均匀性（粗略）
        # 卡方临界值（df=4, α=0.05）约为 9.49
        if chi_square < 9.49:
            print(f"  结论: 分布较为均匀 (p > 0.05)")
        else:
            print(f"  结论: 分布不均匀 (p < 0.05)")

        # 绘制分布图
        plt.figure(figsize=(10, 6))
        levels = list(range(1, likert_levels + 1))
        counts = [counter.get(i, 0) for i in levels]
        plt.bar(levels, counts, alpha=0.7, edgecolor='black')
        plt.axhline(y=expected_count, color='r', linestyle='--',
                   label=f'Expected uniform ({expected_count:.1f})')
        plt.xlabel('Likert Level')
        plt.ylabel('Frequency')
        plt.title(f'Response Distribution (n={n_subjects} subjects, {len(df)} conditions, {likert_mode} mode)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        csv_name = Path(csv_path).stem
        output_path = output_dir / f"{csv_name}_distribution_{likert_mode}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分布图已保存至: {output_path}")
        plt.close()

    else:
        # 连续响应的直方图
        plt.figure(figsize=(10, 6))
        plt.hist(all_responses, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Continuous Response Value')
        plt.ylabel('Frequency')
        plt.title(f'Continuous Response Distribution (n={n_subjects} subjects, {len(df)} conditions)')
        plt.grid(axis='y', alpha=0.3)

        csv_name = Path(csv_path).stem
        output_path = output_dir / f"{csv_name}_distribution_continuous.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分布图已保存至: {output_path}")
        plt.close()

    return {
        'all_responses': all_responses,
        'n_conditions': len(df),
        'n_subjects': n_subjects,
        'mean': np.mean(all_responses),
        'std': np.std(all_responses),
    }


if __name__ == "__main__":
    # 设置输出目录
    output_dir = Path(__file__).parent

    # 测试文件1：i9csy65bljq14ovww2v91 (324 条件)
    csv1_path = Path(__file__).parents[3] / "data" / "only_independences" / "data" / "only_independences" / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

    # 测试文件2：6vars (1200 条件)
    csv2_path = Path(__file__).parents[3] / "data" / "only_independences" / "data" / "only_independences" / "6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

    print("=" * 80)
    print("验证模拟被试响应分布均匀性")
    print("=" * 80)

    # 测试文件1
    if csv1_path.exists():
        print("\n\n【测试文件1】i9csy65bljq14ovww2v91 (324条件)")
        print("=" * 80)

        print("\n【1.1】Likert 5档 - percentile 模式")
        print("-" * 80)
        test_response_uniformity(
            csv_path=csv1_path,
            n_subjects=10,
            seed=42,
            use_latent=False,
            output_type='likert',
            likert_levels=5,
            likert_mode='percentile',
            population_std=0.4,
            individual_std_percent=1.0,
            output_dir=output_dir,
        )

    # 测试文件2
    if csv2_path.exists():
        print("\n\n【测试文件2】6vars_1200combinations (1200条件)")
        print("=" * 80)

        print("\n【2.1】Likert 5档 - percentile 模式")
        print("-" * 80)
        test_response_uniformity(
            csv_path=csv2_path,
            n_subjects=10,
            seed=42,
            use_latent=False,
            output_type='likert',
            likert_levels=5,
            likert_mode='percentile',
            population_std=0.4,
            individual_std_percent=1.0,
            output_dir=output_dir,
        )

        print("\n【2.2】Likert 5档 - tanh 模式")
        print("-" * 80)
        test_response_uniformity(
            csv_path=csv2_path,
            n_subjects=10,
            seed=42,
            use_latent=False,
            output_type='likert',
            likert_levels=5,
            likert_mode='tanh',
            population_std=0.4,
            individual_std_percent=1.0,
            output_dir=output_dir,
        )

        print("\n【2.3】连续响应（无 Likert 转换）")
        print("-" * 80)
        test_response_uniformity(
            csv_path=csv2_path,
            n_subjects=10,
            seed=42,
            use_latent=False,
            output_type='continuous',
            population_std=0.4,
            individual_std_percent=1.0,
            output_dir=output_dir,
        )

    print("\n\n" + "=" * 80)
    print("验证完成！")
    print(f"所有结果已保存至: {output_dir}")
    print("=" * 80)
