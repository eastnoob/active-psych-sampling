#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid模式自由探索覆盖度改进对比测试（排除 Condition_ID）

测试改进前后的覆盖度差异：
- 改进前：纯随机采样
- 改进后：覆盖度优化采样

关键修复：排除 Condition_ID（配置编号）这个非实验因子
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Set
from collections import defaultdict

# 设置编码 - Windows 兼容性修复
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent / "core"))

from warmup_sampler import WarmupSampler

# 设计空间路径
DESIGN_CSV = (
    Path(__file__).parent.parent.parent
    / "data"
    / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
)


def compute_single_factor_coverage(design_df: pd.DataFrame, selected_indices: List[int],
                                   exclude_condition_id: bool = True) -> dict:
    """
    计算单因子水平覆盖度

    Args:
        exclude_condition_id: 是否排除 Condition_ID

    Returns:
        dict: {factor_name: coverage_ratio}
    """
    if not selected_indices:
        return {}

    df_sub = design_df.iloc[selected_indices]
    coverage = {}

    for col in df_sub.columns:
        # 排除 Condition_ID
        if exclude_condition_id and col == 'Condition_ID':
            continue

        n_unique_in_subset = df_sub[col].nunique()
        n_unique_total = design_df[col].nunique()
        coverage[col] = n_unique_in_subset / n_unique_total if n_unique_total > 0 else 0.0

    return coverage


def compute_pairwise_coverage(design_df: pd.DataFrame, selected_indices: List[int],
                              exclude_condition_id: bool = True) -> dict:
    """
    计算因子对的值组合覆盖度

    Args:
        exclude_condition_id: 是否排除 Condition_ID

    Returns:
        dict: {(factor_i, factor_j): coverage_ratio}
    """
    if not selected_indices:
        return {}

    df_sub = design_df.iloc[selected_indices]
    coverage = {}

    # 确定要分析的因子范围
    start_idx = 1 if exclude_condition_id else 0
    n_factors = len(design_df.columns)

    for i in range(start_idx, n_factors):
        for j in range(i + 1, n_factors):
            col_i = design_df.columns[i]
            col_j = design_df.columns[j]

            # 全部可能的值组合
            all_combinations = set(design_df[[col_i, col_j]].itertuples(index=False, name=None))
            # 实际出现的值组合
            actual_combinations = set(df_sub[[col_i, col_j]].itertuples(index=False, name=None))

            coverage[(i, j)] = len(actual_combinations) / len(all_combinations) if all_combinations else 0.0

    return coverage


def simulate_random_sampling(design_df: pd.DataFrame, n_free: int, protected_configs: List[int],
                            used_indices: Set[int], n_trials: int = 100,
                            exclude_condition_id: bool = True) -> dict:
    """
    模拟纯随机采样（改进前的方法）

    Args:
        n_trials: 重复试验��数，用于评估随机性带来的波动

    Returns:
        dict: 统计结果
    """
    remaining_available = list(set(design_df.index) - set(protected_configs) - used_indices)

    single_factor_coverages = []
    pairwise_coverages = []

    for _ in range(n_trials):
        free_exploration = np.random.choice(remaining_available, size=n_free, replace=False).tolist()

        # 计算覆盖度
        single_cov = compute_single_factor_coverage(design_df, free_exploration, exclude_condition_id)
        pairwise_cov = compute_pairwise_coverage(design_df, free_exploration, exclude_condition_id)

        single_factor_coverages.append(np.mean(list(single_cov.values())))
        pairwise_coverages.append(list(pairwise_cov.values()))

    # 统计结果
    avg_single_factor = np.mean(single_factor_coverages)
    std_single_factor = np.std(single_factor_coverages)

    # 展平pairwise_coverages以计算统计量
    all_pairwise = []
    for trial_cov in pairwise_coverages:
        all_pairwise.extend(trial_cov)
    avg_pairwise = np.mean(all_pairwise)
    std_pairwise = np.std(all_pairwise)

    return {
        "single_factor_mean": avg_single_factor,
        "single_factor_std": std_single_factor,
        "pairwise_mean": avg_pairwise,
        "pairwise_std": std_pairwise,
        "n_trials": n_trials
    }


def run_comparison_test():
    """运行对比测试"""

    print("=" * 80)
    print("Hybrid模式自由探索覆盖度改进对比测试（排除 Condition_ID）")
    print("=" * 80)
    print()

    # 读取设计空间
    design_df = pd.read_csv(DESIGN_CSV)

    # 检查并识别 Condition_ID
    print("设计空间检查：")
    for i, col in enumerate(design_df.columns):
        n_levels = design_df[col].nunique()
        is_id = (n_levels == len(design_df))
        marker = " [配置ID - 将被排除]" if is_id else ""
        print(f"  因子{i} ({col}): {n_levels} 个水平{marker}")
    print()

    # 排除 Condition_ID 后的因子数量
    n_real_factors = len(design_df.columns) - 1  # 排除因子0
    n_configs = len(design_df)
    n_pairs = n_real_factors * (n_real_factors - 1) // 2

    print(f"实验因子信息（排除 Condition_ID）：")
    print(f"  实验因子数量: {n_real_factors}")
    print(f"  配置总数: {n_configs}")
    print(f"  有意义的因子对数量: {n_pairs}")
    print()

    # 初始化采样器
    sampler = WarmupSampler(str(DESIGN_CSV))

    # 设置测试参数
    n_subjects = 5
    trials_per_subject = 20
    # 使用真实的实验因子作为交互对（排除因子0）
    interaction_pairs = [(3, 4), (1, 2)]  # x3_OuterFurniture × x4_VisualBoundary, x1_CeilingHeight × x2_GridModule
    min_config_per_pair = 2

    print(f"测试参数：")
    print(f"  被试数量: {n_subjects}")
    print(f"  每人trials: {trials_per_subject}")
    print(f"  指定保护的交互对（索引）: {interaction_pairs}")
    print(f"  指定保护的交互对（名称）:")
    for i, j in interaction_pairs:
        print(f"    ({i}, {j}) = ({design_df.columns[i]}, {design_df.columns[j]})")
    print(f"  每对保护配置数: {min_config_per_pair}")
    print()

    # 评估预算
    adequacy, budget = sampler.evaluate_budget(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
        skip_interaction=False,
    )

    n_core2b = budget.get('core2b_configs', 0)
    print(f"预算分配：")
    print(f"  Core-2b总配置数: {n_core2b}")
    print()

    # 模拟hybrid模式的保护阶段
    print("=" * 80)
    print("第1步：模拟保护配置选择（对指定交互对）")
    print("=" * 80)
    print()

    df_cols = list(design_df.columns)
    available = list(design_df.index)
    used_indices = set()

    # Step 1: 为每个指定对分配保护配置
    protected_configs = []
    for pair_idx, (i, j) in enumerate(interaction_pairs):
        col_i, col_j = df_cols[i], df_cols[j]
        n_protected = min_config_per_pair

        pair_configs = []
        for idx in available:
            if idx not in protected_configs:
                row = design_df.loc[idx]
                pair_configs.append((idx, (row[col_i], row[col_j])))

        if pair_configs:
            pair_configs.sort(key=lambda x: x[1])
            selected_for_pair = [
                pair_configs[k][0]
                for k in np.linspace(0, len(pair_configs) - 1,
                                   num=min(n_protected, len(pair_configs))).astype(int)
            ]
            protected_configs.extend(selected_for_pair)

    protected_configs = list(set(protected_configs))
    print(f"保护配置数量: {len(protected_configs)}")

    # 计算保护配置对指定交互对的覆盖度
    protected_pairwise_cov = compute_pairwise_coverage(design_df, protected_configs, exclude_condition_id=True)
    for pair in interaction_pairs:
        cov = protected_pairwise_cov.get(tuple(sorted(pair)), 0.0)
        col_i, col_j = df_cols[pair[0]], df_cols[pair[1]]
        print(f"  指定对 {pair} ({col_i} × {col_j}) 覆盖度: {cov*100:.1f}%")
    print()

    # Step 2: 自由探索预算
    n_free = n_core2b - len(protected_configs)
    print(f"自由探索预算: {n_free} 个配置")
    print()

    # ========== 改进前：纯随机采样 ==========
    print("=" * 80)
    print("第2步：改进前 - 纯随机采样 (100次模拟)")
    print("=" * 80)
    print()

    random_results = simulate_random_sampling(
        design_df, n_free, protected_configs, used_indices, n_trials=100, exclude_condition_id=True
    )

    print(f"随机采样结果（排除 Condition_ID）：")
    print(f"  单因子水平覆盖度: {random_results['single_factor_mean']*100:.2f}% (±{random_results['single_factor_std']*100:.2f}%)")
    print(f"  因子对值组合覆盖度: {random_results['pairwise_mean']*100:.2f}% (±{random_results['pairwise_std']*100:.2f}%)")
    print()

    # ========== 改进后：覆盖度优化采样 ==========
    print("=" * 80)
    print("第3步：改进后 - 综合覆盖度优化采样（单因子 + 因子对）")
    print("=" * 80)
    print()

    # 使用当前实现的_select_covering_configs方法
    free_exploration_improved = sampler._select_covering_configs(
        n_configs=n_free,
        used_indices=set(protected_configs) | used_indices,
        target_coverage=0.85
    )

    # 计算覆盖度
    improved_single_cov = compute_single_factor_coverage(design_df, free_exploration_improved, exclude_condition_id=True)
    improved_pairwise_cov = compute_pairwise_coverage(design_df, free_exploration_improved, exclude_condition_id=True)

    improved_single_mean = np.mean(list(improved_single_cov.values()))
    improved_pairwise_mean = np.mean(list(improved_pairwise_cov.values()))

    print(f"覆盖度优化采样结果（排除 Condition_ID）：")
    print(f"  单因子水平覆盖度: {improved_single_mean*100:.2f}%")
    print(f"  因子对值组合覆盖度: {improved_pairwise_mean*100:.2f}%")
    print()

    # ========== 对比总结 ==========
    print("=" * 80)
    print("对比总结")
    print("=" * 80)
    print()

    single_improvement = (improved_single_mean - random_results['single_factor_mean']) * 100
    pairwise_improvement = (improved_pairwise_mean - random_results['pairwise_mean']) * 100

    print(f"{'指标':<30} {'改进前 (随机)':<20} {'改进后 (优化)':<20} {'提升':<15}")
    print("-" * 85)
    print(f"{'单因子水平覆盖度':<30} {random_results['single_factor_mean']*100:>6.2f}% (±{random_results['single_factor_std']*100:.2f}%) {improved_single_mean*100:>6.2f}%           {single_improvement:>+6.2f}%")
    print(f"{'因子对值组合覆盖度':<30} {random_results['pairwise_mean']*100:>6.2f}% (±{random_results['pairwise_std']*100:.2f}%) {improved_pairwise_mean*100:>6.2f}%           {pairwise_improvement:>+6.2f}%")
    print()

    # 详细的因子对覆盖度对比
    print("=" * 80)
    print("未保护交互对的详细覆盖度对比（仅真实实验因子，排除 Condition_ID）")
    print("=" * 80)
    print()

    # 模拟一次随机采样作为对比
    remaining_available = list(set(design_df.index) - set(protected_configs) - used_indices)
    free_exploration_random = np.random.choice(remaining_available, size=n_free, replace=False).tolist()
    random_pairwise_cov = compute_pairwise_coverage(design_df, free_exploration_random, exclude_condition_id=True)

    # 找出未保护的交互对（排除指定的保护对）
    protected_pairs_set = set(tuple(sorted(pair)) for pair in interaction_pairs)

    print(f"{'交互对':<25} {'因子名称':<45} {'改进前':<12} {'改进后':<12} {'提升':<15}")
    print("-" * 110)

    unprotected_improvements = []
    for i in range(1, len(design_df.columns)):  # 从1开始，跳过 Condition_ID
        for j in range(i + 1, len(design_df.columns)):
            pair = (i, j)
            if pair not in protected_pairs_set:
                col_i = design_df.columns[i]
                col_j = design_df.columns[j]
                random_cov = random_pairwise_cov.get(pair, 0.0)
                improved_cov = improved_pairwise_cov.get(pair, 0.0)
                improvement = (improved_cov - random_cov) * 100
                unprotected_improvements.append(improvement)

                factor_names = f"{col_i} × {col_j}"
                print(f"{str(pair):<25} {factor_names:<45} {random_cov*100:>6.2f}%    {improved_cov*100:>6.2f}%    {improvement:>+6.2f}%")

    print()
    print(f"未保护交互对平均提升: {np.mean(unprotected_improvements):>+6.2f}%")
    print(f"未保护交互对数量: {len(unprotected_improvements)}")
    print(f"改进后达到100%覆盖的交互对数量: {sum(1 for pair, cov in improved_pairwise_cov.items() if pair not in protected_pairs_set and cov >= 0.999)}")
    print()

    # ========== 结论 ==========
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print()

    if single_improvement > 5:
        print("✅ 单因子水平覆盖度有显著提升")
    elif single_improvement > 0:
        print("⚠️  单因子水平覆盖度有小幅提升")
    else:
        print("❌ 单因子水平覆盖度无明显提升")

    if pairwise_improvement > 5:
        print("✅ 因子对覆盖度有显著提升")
    elif pairwise_improvement > 0:
        print("⚠️  因子对覆盖度有小幅提升")
    else:
        print("❌ 因子对覆盖度无明显提升")

    if np.mean(unprotected_improvements) > 5:
        print("✅ 未保护交互对的探索能力显著增强")
    elif np.mean(unprotected_improvements) > 0:
        print("⚠️  未保护交互对的探索能力小幅增强")
    else:
        print("❌ 未保护交互对的探索能力无明显改善")

    print()
    print("改进建议的评估：")
    if single_improvement > 10 or pairwise_improvement > 5:
        print("  ✅ 强烈推荐：改进效果明显，应该采用")
    elif single_improvement > 5 or pairwise_improvement > 2:
        print("  ⚠️  建议采用：有一定改进，但效果中等")
    else:
        print("  ❌ 效果不明显：可能需要重新评估改进方案")
    print()


if __name__ == "__main__":
    run_comparison_test()
