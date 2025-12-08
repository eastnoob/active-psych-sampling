#!/usr/bin/env python3
"""
测试不同配置以获得松散的正态分布（中间多两头少，但比例适中）

测试策略：
1. 使用 tanh 模式（保持正态分布形态）
2. 调整 likert_sensitivity 参数（控制分布宽度）
   - sensitivity 越大，分布越集中在中间
3. 目标：找到使分布温和正态的参数组合
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 添加路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "tools" / "archive" / "simulate_subject" / "models"))

from subject_models.MixedEffectsLatentSubject import MixedEffectsLatentSubject


def convert_row_to_features_auto(row, df):
    """自动检测CSV格式并转换为数值特征向量"""
    columns = df.columns.tolist()

    if 'x1_CeilingHeight' in columns:
        # 格式1
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
        # 格式2
        x1 = int(row['x1_binary'])
        x2 = float(row['x2_5level_discrete'])
        x3 = float(row['x3_5level_decimal'])
        x4_map = {'low': 0, 'mid': 1, 'high': 2, 'max': 3}
        x4 = x4_map.get(row['x4_4level_categorical'], 0) if isinstance(row['x4_4level_categorical'], str) else float(row['x4_4level_categorical'])
        x5_map = {'A': 0, 'B': 1, 'C': 2}
        x5 = x5_map.get(row['x5_3level_categorical'], 0) if isinstance(row['x5_3level_categorical'], str) else float(row['x5_3level_categorical'])
        x6 = 1 if (isinstance(row['x6_binary'], str) and row['x6_binary'].strip().lower() in {'true', '1', 't', 'yes'}) else (1 if bool(row['x6_binary']) else 0)
    else:
        raise ValueError(f"无法识别CSV格式")

    return np.array([x1, x2, x3, x4, x5, x6], dtype=float)


def test_sensitivity_config(
    csv_path,
    sensitivity,
    n_subjects=10,
    seed=42,
    population_std=0.4,
    individual_std_percent=1.0,
    output_dir=None,
):
    """
    测试特定 sensitivity 配置的分布

    Args:
        csv_path: CSV路径
        sensitivity: likert_sensitivity 参数
        n_subjects: 被试数量
        seed: 随机种子
        population_std: 群体标准差
        individual_std_percent: 个体差异百分比
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    df = pd.read_csv(csv_path)
    all_responses = []

    for subj_idx in range(1, n_subjects + 1):
        subject_seed = seed + subj_idx
        subject = MixedEffectsLatentSubject(
            num_features=6,
            num_observed_vars=1,
            seed=subject_seed,
            population_mean=0.0,
            population_std=population_std,
            individual_std_percent=individual_std_percent,
            use_latent=False,
            noise_std=0.1,
            bias=0.0,
            likert_levels=5,
            likert_sensitivity=sensitivity,
        )

        for idx, row in df.iterrows():
            X = convert_row_to_features_auto(row, df)
            y = subject(X)
            if isinstance(y, list):
                y = y[0]
            all_responses.append(float(y))

    all_responses = np.array(all_responses)

    # 统计分布
    counter = Counter(all_responses)
    total = len(all_responses)

    # 计算统计量
    counts = [counter.get(i, 0) for i in range(1, 6)]
    expected = total / 5
    chi_square = sum((c - expected)**2 / expected for c in counts)

    # 计算理想正态分布的偏离度
    # 理想正态：1档15%, 2档20%, 3档30%, 4档20%, 5档15%
    ideal_normal = [0.15, 0.20, 0.30, 0.20, 0.15]
    actual_props = [counter.get(i, 0) / total for i in range(1, 6)]
    deviation = sum(abs(a - i) for a, i in zip(actual_props, ideal_normal))

    return {
        'sensitivity': sensitivity,
        'distribution': counter,
        'proportions': actual_props,
        'chi_square': chi_square,
        'deviation_from_ideal': deviation,
        'mean': np.mean(all_responses),
        'std': np.std(all_responses),
    }


def run_sensitivity_tests(csv_path, output_dir=None):
    """运行一系列 sensitivity 测试"""
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    print("=" * 80)
    print("测试目标：找到产生松散正态分布的最佳 likert_sensitivity")
    print("=" * 80)
    print()

    # 测试不同的 sensitivity 值
    sensitivities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []

    for sens in sensitivities:
        print(f"测试 likert_sensitivity = {sens}")
        result = test_sensitivity_config(
            csv_path=csv_path,
            sensitivity=sens,
            n_subjects=10,
            seed=42,
            population_std=0.4,
            individual_std_percent=1.0,
            output_dir=output_dir,
        )
        results.append(result)

        # 打印结果
        print(f"  分布: ", end="")
        for i in range(1, 6):
            count = result['distribution'].get(i, 0)
            pct = result['proportions'][i-1] * 100
            print(f"{i}档={pct:5.1f}%  ", end="")
        print()
        print(f"  偏离理想正态: {result['deviation_from_ideal']:.4f}")
        print(f"  卡方统计量: {result['chi_square']:.2f}")
        print()

    # 找到最接近理想正态分布的配置
    best_result = min(results, key=lambda r: r['deviation_from_ideal'])
    print("=" * 80)
    print(f"最佳配置: likert_sensitivity = {best_result['sensitivity']}")
    print("=" * 80)
    print()
    print("分布详情:")
    for i in range(1, 6):
        count = best_result['distribution'].get(i, 0)
        pct = best_result['proportions'][i-1] * 100
        ideal_pct = [15, 20, 30, 20, 15][i-1]
        diff = pct - ideal_pct
        print(f"  档位{i}: {pct:5.1f}% (理想={ideal_pct}%, 偏差={diff:+5.1f}%)")
    print()
    print(f"偏离度: {best_result['deviation_from_ideal']:.4f} (越小越好)")
    print(f"卡方统计量: {best_result['chi_square']:.2f}")
    print()

    # 绘制对比图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]
        levels = list(range(1, 6))
        counts = [result['distribution'].get(i, 0) for i in levels]
        proportions = [c / sum(counts) * 100 for c in counts]

        # 绘制柱状图
        bars = ax.bar(levels, proportions, alpha=0.7, edgecolor='black')

        # 标记最接近理想的配置
        if result['sensitivity'] == best_result['sensitivity']:
            for bar in bars:
                bar.set_color('green')
            ax.set_title(f"sensitivity={result['sensitivity']} ⭐ BEST", fontweight='bold', color='green')
        else:
            ax.set_title(f"sensitivity={result['sensitivity']}")

        # 添加理想分布参考线
        ideal_line = [15, 20, 30, 20, 15]
        ax.plot(levels, ideal_line, 'r--', label='Ideal Normal', linewidth=2)

        ax.set_xlabel('Likert Level')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 80)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

        # 添加偏离度文本
        ax.text(0.95, 0.95, f'Dev={result["deviation_from_ideal"]:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    csv_name = Path(csv_path).stem
    output_path = output_dir / f"{csv_name}_sensitivity_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()

    # 绘制最佳配置的详细图
    plt.figure(figsize=(10, 6))
    levels = list(range(1, 6))
    counts = [best_result['distribution'].get(i, 0) for i in levels]
    proportions = [c / sum(counts) * 100 for c in counts]

    bars = plt.bar(levels, proportions, alpha=0.7, edgecolor='black', color='green')
    ideal_line = [15, 20, 30, 20, 15]
    plt.plot(levels, ideal_line, 'r--', label='Ideal Normal Distribution', linewidth=3)

    plt.xlabel('Likert Level', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title(f'Best Configuration: likert_sensitivity={best_result["sensitivity"]}\n'
              f'Loose Normal Distribution (Deviation={best_result["deviation_from_ideal"]:.4f})',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=11)

    # 添加数值标签
    for i, (bar, pct, ideal) in enumerate(zip(bars, proportions, ideal_line)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(ideal={ideal}%)',
                ha='center', va='bottom', fontsize=9)

    output_path = output_dir / f"{csv_name}_best_loose_normal.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"最佳配置详图已保存: {output_path}")
    plt.close()

    # 保存推荐配置
    config_path = output_dir / "RECOMMENDED_CONFIG.txt"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("松散正态分布推荐配置\n")
        f.write("=" * 80 + "\n\n")
        f.write("目标：中间多两头少，但比例差距适中\n\n")
        f.write("推荐配置（用于 STEP1_5_CONFIG）：\n")
        f.write("-" * 80 + "\n")
        f.write(f'"likert_mode": "tanh",\n')
        f.write(f'"likert_sensitivity": {best_result["sensitivity"]},\n')
        f.write(f'"population_std": 0.4,\n')
        f.write(f'"individual_std_percent": 1.0,\n')
        f.write("-" * 80 + "\n\n")
        f.write("预期分布：\n")
        for i in range(1, 6):
            pct = best_result['proportions'][i-1] * 100
            ideal_pct = [15, 20, 30, 20, 15][i-1]
            f.write(f"  档位{i}: {pct:5.1f}% (理想={ideal_pct}%)\n")
        f.write("\n")
        f.write(f"统计指标：\n")
        f.write(f"  - 偏离理想正态: {best_result['deviation_from_ideal']:.4f}\n")
        f.write(f"  - 卡方统计量: {best_result['chi_square']:.2f}\n")
        f.write(f"  - 连续响应均值: {best_result['mean']:.4f}\n")
        f.write(f"  - 连续响应标准差: {best_result['std']:.4f}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"推荐配置已保存: {config_path}")
    print()

    return best_result


if __name__ == "__main__":
    # 测试两个CSV文件
    csv1 = Path(__file__).parents[3] / "data" / "only_independences" / "data" / "only_independences" / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    csv2 = Path(__file__).parents[3] / "data" / "only_independences" / "data" / "only_independences" / "6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

    output_dir = Path(__file__).parent

    if csv1.exists():
        print("\n" + "=" * 80)
        print("测试文件1: i9csy65bljq14ovww2v91 (324条件)")
        print("=" * 80 + "\n")
        result1 = run_sensitivity_tests(csv1, output_dir)
        print("\n")

    if csv2.exists():
        print("\n" + "=" * 80)
        print("测试文件2: 6vars_1200combinations (1200条件)")
        print("=" * 80 + "\n")
        result2 = run_sensitivity_tests(csv2, output_dir)
        print("\n")

    print("=" * 80)
    print("所有测试完成！")
    print("=" * 80)
