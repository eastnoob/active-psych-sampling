#!/usr/bin/env python3
"""
精细调整：找到更均衡的松散正态分布配置
目标：减少2和4档位的比例，增加5档位，使分布更接近理想

理想目标：
- 档位1: ~15-18%
- 档位2: ~20-23%  (当前26%太多，需要降低)
- 档位3: ~25-30%
- 档位4: ~20-23%  (当前23.7%可以)
- 档位5: ~12-15%  (当前10.6%太少，需要增加)
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

    if 'x1_binary' in columns:
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


def test_sensitivity_config(csv_path, sensitivity, n_subjects=10, seed=42):
    """测试特定 sensitivity 配置的分布"""
    df = pd.read_csv(csv_path)
    all_responses = []

    for subj_idx in range(1, n_subjects + 1):
        subject_seed = seed + subj_idx
        subject = MixedEffectsLatentSubject(
            num_features=6,
            num_observed_vars=1,
            seed=subject_seed,
            population_mean=0.0,
            population_std=0.4,
            individual_std_percent=1.0,
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
    counter = Counter(all_responses)
    total = len(all_responses)

    counts = [counter.get(i, 0) for i in range(1, 6)]
    proportions = [c / total for c in counts]

    # 计算与理想分布的偏离
    # 修正的理想分布：更强调均衡性
    ideal_target = [0.15, 0.21, 0.28, 0.21, 0.15]

    # 计算偏离度
    deviation = sum(abs(a - i) for a, i in zip(proportions, ideal_target))

    # 额外惩罚：2档位超过23%和5档位低于12%
    penalty = 0
    if proportions[1] > 0.23:  # 档位2
        penalty += (proportions[1] - 0.23) * 2
    if proportions[4] < 0.12:  # 档位5
        penalty += (0.12 - proportions[4]) * 2

    total_score = deviation + penalty

    return {
        'sensitivity': sensitivity,
        'distribution': counter,
        'proportions': proportions,
        'deviation': deviation,
        'penalty': penalty,
        'total_score': total_score,
        'mean': np.mean(all_responses),
        'std': np.std(all_responses),
    }


def run_fine_tuning_tests(csv_path, output_dir=None):
    """运行精细调整测试"""
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    print("=" * 80)
    print("精细调整：寻找更均衡的松散正态分布配置")
    print("=" * 80)
    print()
    print("目标约束：")
    print("  - 档位1: 15-18%")
    print("  - 档位2: ≤23% (当前26%太多)")
    print("  - 档位3: 25-30%")
    print("  - 档位4: 20-23%")
    print("  - 档位5: ≥12% (当前10.6%太少)")
    print()

    # 精细测试：1.5-2.5之间，步长0.1
    sensitivities = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    results = []

    for sens in sensitivities:
        print(f"测试 likert_sensitivity = {sens:.1f}...", end=" ")
        result = test_sensitivity_config(
            csv_path=csv_path,
            sensitivity=sens,
            n_subjects=10,
            seed=42,
        )
        results.append(result)

        # 打印结果
        props = result['proportions']
        p1, p2, p3, p4, p5 = [p * 100 for p in props]
        print(f"[{p1:4.1f}% | {p2:4.1f}% | {p3:4.1f}% | {p4:4.1f}% | {p5:4.1f}%]  ", end="")
        print(f"Score={result['total_score']:.4f}", end="")

        # 标记是否满足约束
        meets_constraints = (
            15 <= p1 <= 18 and
            p2 <= 23 and
            25 <= p3 <= 30 and
            20 <= p4 <= 23 and
            p5 >= 12
        )
        if meets_constraints:
            print(" [OK] 满足所有约束")
        else:
            warnings = []
            if p2 > 23:
                warnings.append(f"2档过多({p2:.1f}%)")
            if p5 < 12:
                warnings.append(f"5档过少({p5:.1f}%)")
            if warnings:
                print(f" [!] {', '.join(warnings)}")
            else:
                print()

    print()
    print("=" * 80)

    # 找到最佳配置（总分最低）
    best_result = min(results, key=lambda r: r['total_score'])

    print(f"最佳配置: likert_sensitivity = {best_result['sensitivity']:.1f}")
    print("=" * 80)
    print()
    print("分布详情:")
    ideal = [15, 21, 28, 21, 15]
    for i in range(1, 6):
        pct = best_result['proportions'][i-1] * 100
        ideal_pct = ideal[i-1]
        diff = pct - ideal_pct
        status = "[OK]" if abs(diff) <= 3 else "[!]"
        print(f"  {status} 档位{i}: {pct:5.1f}% (目标≈{ideal_pct}%, 偏差={diff:+5.1f}%)")
    print()
    print(f"偏离度: {best_result['deviation']:.4f}")
    print(f"惩罚分: {best_result['penalty']:.4f}")
    print(f"总分: {best_result['total_score']:.4f} (越低越好)")
    print()

    # 绘制详细对比图
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]
        levels = list(range(1, 6))
        proportions = [p * 100 for p in result['proportions']]

        # 绘制柱状图
        bars = ax.bar(levels, proportions, alpha=0.7, edgecolor='black')

        # 标记最佳配置
        if result['sensitivity'] == best_result['sensitivity']:
            for bar in bars:
                bar.set_color('green')
            ax.set_title(f"sensitivity={result['sensitivity']:.1f} ⭐ BEST",
                        fontweight='bold', color='green', fontsize=11)
        else:
            ax.set_title(f"sensitivity={result['sensitivity']:.1f}", fontsize=11)

        # 添加理想分布参考线
        ideal_line = [15, 21, 28, 21, 15]
        ax.plot(levels, ideal_line, 'r--', label='Target', linewidth=2)

        # 添加约束区域
        ax.axhline(y=23, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=12, color='orange', linestyle=':', alpha=0.5, linewidth=1)

        ax.set_xlabel('Likert Level', fontsize=9)
        ax.set_ylabel('Percentage (%)', fontsize=9)
        ax.set_ylim(0, 35)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7)

        # 添加评分
        ax.text(0.95, 0.95, f'Score={result["total_score"]:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / "balanced_sensitivity_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()

    # 绘制最佳配置详图
    plt.figure(figsize=(12, 7))
    levels = list(range(1, 6))
    proportions = [p * 100 for p in best_result['proportions']]

    bars = plt.bar(levels, proportions, alpha=0.7, edgecolor='black', color='green', width=0.6)
    ideal_line = [15, 21, 28, 21, 15]
    plt.plot(levels, ideal_line, 'r--', label='Target Distribution', linewidth=3, marker='o', markersize=8)

    # 添加约束线
    plt.axhline(y=23, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='Max for Level 2/4')
    plt.axhline(y=12, color='blue', linestyle=':', alpha=0.6, linewidth=2, label='Min for Level 5')

    plt.xlabel('Likert Level', fontsize=13)
    plt.ylabel('Percentage (%)', fontsize=13)
    plt.title(f'Balanced Configuration: likert_sensitivity={best_result["sensitivity"]:.1f}\n'
              f'More Balanced Distribution (Score={best_result["total_score"]:.4f})',
              fontsize=15, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=11, loc='upper left')

    # 添加数值标签
    for i, (bar, pct, ideal) in enumerate(zip(bars, proportions, ideal_line)):
        height = bar.get_height()
        color = 'green' if abs(pct - ideal) <= 3 else 'red'
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(target≈{ideal}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    output_path = output_dir / "balanced_best_configuration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"最佳配置详图已保存: {output_path}")
    plt.close()

    # 保存推荐配置
    config_path = output_dir / "BALANCED_RECOMMENDED_CONFIG.txt"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("更均衡的松散正态分布推荐配置\n")
        f.write("=" * 80 + "\n\n")
        f.write("优化目标：\n")
        f.write("  - 减少档位2的比例（从26%降低）\n")
        f.write("  - 增加档位5的比例（从10.6%提升）\n")
        f.write("  - 保持整体松散正态形态\n\n")
        f.write("推荐配置（用于 STEP1_5_CONFIG）：\n")
        f.write("-" * 80 + "\n")
        f.write(f'"likert_mode": "tanh",\n')
        f.write(f'"likert_sensitivity": {best_result["sensitivity"]:.1f},\n')
        f.write(f'"population_std": 0.4,\n')
        f.write(f'"individual_std_percent": 1.0,\n')
        f.write("-" * 80 + "\n\n")
        f.write("实际分布：\n")
        ideal = [15, 21, 28, 21, 15]
        for i in range(1, 6):
            pct = best_result['proportions'][i-1] * 100
            ideal_pct = ideal[i-1]
            diff = pct - ideal_pct
            status = "[OK]" if abs(diff) <= 3 else "[!]"
            f.write(f"  {status} 档位{i}: {pct:5.1f}% (目标≈{ideal_pct}%, 偏差={diff:+5.1f}%)\n")
        f.write("\n")
        f.write(f"评分指标：\n")
        f.write(f"  - 偏离度: {best_result['deviation']:.4f}\n")
        f.write(f"  - 惩罚分: {best_result['penalty']:.4f}\n")
        f.write(f"  - 总分: {best_result['total_score']:.4f} (越低越好)\n")
        f.write(f"  - 连续响应均值: {best_result['mean']:.4f}\n")
        f.write(f"  - 连续响应标准差: {best_result['std']:.4f}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"推荐配置已保存: {config_path}")
    print()

    return best_result


if __name__ == "__main__":
    csv_path = Path(__file__).parents[3] / "data" / "only_independences" / "data" / "only_independences" / "6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"
    output_dir = Path(__file__).parent

    if csv_path.exists():
        result = run_fine_tuning_tests(csv_path, output_dir)
    else:
        print(f"错误：CSV文件不存在 - {csv_path}")

    print("=" * 80)
    print("精细调整完成！")
    print("=" * 80)
