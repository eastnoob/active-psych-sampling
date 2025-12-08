#!/usr/bin/env python3
"""
快速复现脚本：生成与当前模拟被试统计上类似的新被试

使用方法:
    python reproduce_similar_subjects.py --method [exact|similar|improved]

方法说明:
    exact    - 完全相同复现（相同seed）
    similar  - 统计上类似（使用fixed_weights，不同seed）
    improved - 调优版本（更接近理想分布）
"""

import sys
import argparse
from pathlib import Path

# 添加路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from subject_simulator_v2.adapters.warmup_adapter import run


def reproduce_exact(new_sampling_dir: Path):
    """方法1: 完全相同复现"""
    print("=" * 80)
    print("方法1: 完全相同复现（Exact Clone）")
    print("=" * 80)
    print()

    design_space_csv = PROJECT_ROOT / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

    run(
        input_dir=new_sampling_dir,
        seed=42,  # ⭐ 相同种子 → 完全相同的被试
        output_mode="combined",
        clean=True,
        # V3方法参数
        interaction_as_features=True,
        interaction_x3x4_weight=0.12,
        interaction_x0x1_weight=-0.02,
        # 模型参数
        output_type="likert",
        likert_levels=5,
        likert_mode="tanh",
        likert_sensitivity=2.0,
        population_mean=0.0,
        population_std=0.4,
        individual_std_percent=0.3,
        noise_std=0.0,
        design_space_csv=str(design_space_csv),
        print_model=True,
        save_model_summary=True,
    )


def reproduce_similar(new_sampling_dir: Path):
    """方法2: 统计上类似（推荐）"""
    print("=" * 80)
    print("方法2: 统计上类似（Similar Population）")
    print("=" * 80)
    print()

    design_space_csv = PROJECT_ROOT / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    fixed_weights_file = SCRIPT_DIR / "fixed_weights_auto.json"

    run(
        input_dir=new_sampling_dir,
        seed=99,  # ⭐ 不同种子 → 新的个体偏差
        output_mode="combined",
        clean=True,
        # 使用固定权重文件
        fixed_weights_file=str(fixed_weights_file),
        # V3方法参数
        interaction_as_features=True,
        interaction_x3x4_weight=0.12,
        interaction_x0x1_weight=-0.02,
        # 模型参数
        output_type="likert",
        likert_levels=5,
        likert_mode="tanh",
        likert_sensitivity=2.0,
        population_mean=0.0,
        population_std=0.4,
        individual_std_percent=0.3,
        noise_std=0.0,
        design_space_csv=str(design_space_csv),
        print_model=True,
        save_model_summary=True,
    )


def reproduce_improved(new_sampling_dir: Path):
    """方法3: 调优版本（更接近理想分布）"""
    print("=" * 80)
    print("方法3: 调优版本（Improved Distribution）")
    print("=" * 80)
    print()

    design_space_csv = PROJECT_ROOT / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

    run(
        input_dir=new_sampling_dir,
        seed=99,  # ⭐ seed=99（测试时的完美分布）
        output_mode="combined",
        clean=True,
        # V3方法参数
        interaction_as_features=True,
        interaction_x3x4_weight=0.12,
        interaction_x0x1_weight=-0.02,
        # 调优的模型参数
        output_type="likert",
        likert_levels=5,
        likert_mode="tanh",
        likert_sensitivity=2.0,
        population_mean=0.0,
        population_std=0.3,  # ⭐ 降低到0.3（更稳定的分布）
        individual_std_percent=0.3,
        noise_std=0.0,
        design_space_csv=str(design_space_csv),
        print_model=True,
        save_model_summary=True,
    )


def compare_distributions(original_csv: Path, new_csv: Path):
    """对比原始和新生成的分布"""
    import pandas as pd
    from collections import Counter

    print()
    print("=" * 80)
    print("分布对比")
    print("=" * 80)
    print()

    df_old = pd.read_csv(original_csv)
    df_new = pd.read_csv(new_csv)

    print("原始分布 (202511302204):")
    counter_old = Counter(df_old['y'])
    for level in range(1, 6):
        count = counter_old.get(level, 0)
        pct = count / len(df_old) * 100
        bar = '#' * int(pct / 5)
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"  Mean: {df_old['y'].mean():.2f}, Std: {df_old['y'].std():.2f}")

    print()
    print("新生成分布:")
    counter_new = Counter(df_new['y'])
    for level in range(1, 6):
        count = counter_new.get(level, 0)
        pct = count / len(df_new) * 100
        bar = '#' * int(pct / 5)
        print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"  Mean: {df_new['y'].mean():.2f}, Std: {df_new['y'].std():.2f}")

    # KS检验
    try:
        from scipy.stats import ks_2samp
        statistic, pvalue = ks_2samp(df_old['y'], df_new['y'])
        print()
        print(f"KS检验: statistic={statistic:.3f}, p-value={pvalue:.3f}")
        if pvalue > 0.05:
            print("  ✅ 分布在统计上相似 (p > 0.05)")
        else:
            print("  ⚠ 分布存在统计差异 (p <= 0.05)")
    except ImportError:
        print("\n[提示] 安装 scipy 可进行统计检验: pip install scipy")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="复现与当前模拟被试类似的新被试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完全相同复现
    python reproduce_similar_subjects.py --method exact --input path/to/new/sampling/dir

    # 统计上类似（推荐）
    python reproduce_similar_subjects.py --method similar --input path/to/new/sampling/dir

    # 调优版本
    python reproduce_similar_subjects.py --method improved --input path/to/new/sampling/dir
        """
    )

    parser.add_argument(
        "--method",
        choices=["exact", "similar", "improved"],
        default="similar",
        help="复现方法 (默认: similar)"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="新的采样方案目录（包含subject_*.csv）。如果不提供，将使用测试目录。"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="生成后对比分布"
    )

    args = parser.parse_args()

    # 确定输入目录
    if args.input:
        new_sampling_dir = Path(args.input)
    else:
        # 默认使用原始目录（用于测试）
        new_sampling_dir = SCRIPT_DIR.parent
        print(f"[提示] 未指定输入目录，使用测试目录: {new_sampling_dir}")

    if not new_sampling_dir.exists():
        print(f"[错误] 输入目录不存在: {new_sampling_dir}")
        sys.exit(1)

    # 检查是否有subject_*.csv
    subject_csvs = list(new_sampling_dir.glob("subject_*.csv"))
    if not subject_csvs:
        print(f"[错误] 输入目录中没有找到 subject_*.csv 文件: {new_sampling_dir}")
        sys.exit(1)

    print(f"找到 {len(subject_csvs)} 个被试CSV文件")
    print()

    # 执行复现
    if args.method == "exact":
        reproduce_exact(new_sampling_dir)
    elif args.method == "similar":
        reproduce_similar(new_sampling_dir)
    elif args.method == "improved":
        reproduce_improved(new_sampling_dir)

    # 对比分布
    if args.compare:
        original_csv = SCRIPT_DIR / "combined_results.csv"
        new_csv = new_sampling_dir / "result" / "combined_results.csv"

        if original_csv.exists() and new_csv.exists():
            compare_distributions(original_csv, new_csv)
        else:
            print("[警告] 无法对比分布：找不到combined_results.csv")


if __name__ == "__main__":
    main()
