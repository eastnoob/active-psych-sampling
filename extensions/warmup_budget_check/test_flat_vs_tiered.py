#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test: Flat vs Tiered Budget Allocation Strategy for Core-2b
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Windows encoding fix
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "core"))
from warmup_sampler import WarmupSampler

DESIGN_CSV = (
    Path(__file__).parent.parent.parent
    / "data"
    / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
)


def compute_pairwise_coverage(design_df: pd.DataFrame, selected_indices: list) -> dict:
    """Compute pairwise value combination coverage"""
    if not selected_indices:
        return {}
    df_sub = design_df.iloc[selected_indices]
    coverage = {}
    n_factors = len(design_df.columns)
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            col_i, col_j = design_df.columns[i], design_df.columns[j]
            all_combinations = set(design_df[[col_i, col_j]].itertuples(index=False, name=None))
            actual_combinations = set(df_sub[[col_i, col_j]].itertuples(index=False, name=None))
            coverage[(i, j)] = len(actual_combinations) / len(all_combinations) if all_combinations else 0.0
    return coverage


def test_flat_vs_tiered():
    """Test flat allocation vs tiered allocation"""

    print("=" * 80)
    print("Budget Allocation Strategy Comparison Test")
    print("=" * 80)
    print()

    design_df = pd.read_csv(DESIGN_CSV)
    sampler = WarmupSampler(str(DESIGN_CSV))

    # Test configuration
    total_budget = 21
    protected_pairs = [(1, 2), (3, 4)]
    min_config_per_pair = 2

    print(f"Test Configuration:")
    print(f"  Core-2b Total Budget: {total_budget} configs")
    print(f"  Protected Pairs: {protected_pairs}")
    print(f"  Min Configs per Protected Pair: {min_config_per_pair}")
    print()

    # ========== Strategy 1: Flat Allocation (Current Implementation) ==========
    print("=" * 80)
    print("Strategy 1: Flat Allocation (Current Implementation)")
    print("=" * 80)
    print()

    df_cols = list(design_df.columns)
    available = list(design_df.index)

    # Simulate protected config selection
    protected_configs = []
    for i, j in protected_pairs:
        col_i, col_j = df_cols[i], df_cols[j]
        pair_configs = []
        for idx in available:
            if idx not in protected_configs:
                row = design_df.loc[idx]
                pair_configs.append((idx, (row[col_i], row[col_j])))
        if pair_configs:
            pair_configs.sort(key=lambda x: x[1])
            selected_for_pair = [
                pair_configs[k][0]
                for k in np.linspace(0, len(pair_configs) - 1, num=min(min_config_per_pair, len(pair_configs))).astype(int)
            ]
            protected_configs.extend(selected_for_pair)

    protected_configs = list(set(protected_configs))
    n_free = total_budget - len(protected_configs)

    print(f"  Protected configs: {len(protected_configs)}")
    print(f"  Free exploration: {n_free}")
    print()

    # Free exploration (coverage optimization)
    free_exploration = sampler._select_covering_configs(
        n_configs=n_free,
        used_indices=set(protected_configs),
        target_coverage=0.85
    )

    all_configs_flat = protected_configs + free_exploration
    cov_flat = compute_pairwise_coverage(design_df, all_configs_flat)

    # Calculate coverage for unprotected pairs
    protected_pairs_set = set(tuple(sorted(pair)) for pair in protected_pairs)
    unprotected_cov_flat = [v for k, v in cov_flat.items() if k not in protected_pairs_set]

    print(f"Results:")
    print(f"  Overall pairwise coverage: {np.mean(list(cov_flat.values()))*100:.2f}%")
    print(f"  Unprotected pairs avg coverage: {np.mean(unprotected_cov_flat)*100:.2f}%")
    print(f"  Unprotected pairs min coverage: {min(unprotected_cov_flat)*100:.2f}%")
    print(f"  Unprotected pairs at 100%: {sum(1 for v in unprotected_cov_flat if v >= 0.999)}/{len(unprotected_cov_flat)}")
    print()

    # ========== Strategy 2: Tiered Allocation ==========
    print("=" * 80)
    print("Strategy 2: Tiered Allocation")
    print("=" * 80)
    print()

    # High priority: protected pairs get 4 configs each
    high_priority_configs = []
    for i, j in protected_pairs:
        col_i, col_j = df_cols[i], df_cols[j]
        pair_configs = []
        for idx in available:
            if idx not in high_priority_configs:
                row = design_df.loc[idx]
                pair_configs.append((idx, (row[col_i], row[col_j])))
        if pair_configs:
            pair_configs.sort(key=lambda x: x[1])
            n_for_high = 4  # 4 configs per protected pair
            selected_for_pair = [
                pair_configs[k][0]
                for k in np.linspace(0, len(pair_configs) - 1, num=min(n_for_high, len(pair_configs))).astype(int)
            ]
            high_priority_configs.extend(selected_for_pair)

    high_priority_configs = list(set(high_priority_configs))

    # Medium priority: select some unprotected pairs
    unprotected_pairs = []
    for i in range(len(design_df.columns)):
        for j in range(i+1, len(design_df.columns)):
            if (i, j) not in protected_pairs_set:
                unprotected_pairs.append((i, j))

    # Use first 3 unprotected pairs as medium priority
    mid_priority_pairs = unprotected_pairs[:3]
    n_mid_budget = 6

    mid_priority_configs = []
    configs_per_mid_pair = n_mid_budget // len(mid_priority_pairs)
    for i, j in mid_priority_pairs:
        col_i, col_j = df_cols[i], df_cols[j]
        pair_configs = []
        for idx in available:
            if idx not in high_priority_configs and idx not in mid_priority_configs:
                row = design_df.loc[idx]
                pair_configs.append((idx, (row[col_i], row[col_j])))
        if pair_configs:
            pair_configs.sort(key=lambda x: x[1])
            selected_for_pair = [
                pair_configs[k][0]
                for k in np.linspace(0, len(pair_configs) - 1, num=min(configs_per_mid_pair, len(pair_configs))).astype(int)
            ]
            mid_priority_configs.extend(selected_for_pair)

    mid_priority_configs = list(set(mid_priority_configs))

    # Low priority: remaining budget
    n_low_budget = total_budget - len(high_priority_configs) - len(mid_priority_configs)
    low_priority_configs = sampler._select_covering_configs(
        n_configs=n_low_budget,
        used_indices=set(high_priority_configs) | set(mid_priority_configs),
        target_coverage=0.85
    )

    print(f"  High priority (protected pairs): {len(high_priority_configs)} configs")
    print(f"  Medium priority (some unprotected pairs): {len(mid_priority_configs)} configs")
    print(f"  Low priority (remaining): {len(low_priority_configs)} configs")
    print()

    all_configs_tier = high_priority_configs + mid_priority_configs + low_priority_configs
    cov_tier = compute_pairwise_coverage(design_df, all_configs_tier)

    unprotected_cov_tier = [v for k, v in cov_tier.items() if k not in protected_pairs_set]

    print(f"Results:")
    print(f"  Overall pairwise coverage: {np.mean(list(cov_tier.values()))*100:.2f}%")
    print(f"  Unprotected pairs avg coverage: {np.mean(unprotected_cov_tier)*100:.2f}%")
    print(f"  Unprotected pairs min coverage: {min(unprotected_cov_tier)*100:.2f}%")
    print(f"  Unprotected pairs at 100%: {sum(1 for v in unprotected_cov_tier if v >= 0.999)}/{len(unprotected_cov_tier)}")
    print()

    # ========== Comparison Summary ==========
    print("=" * 80)
    print("Strategy Comparison Summary")
    print("=" * 80)
    print()

    print(f"{'Metric':<35} {'Strategy 1 (Flat)':<22} {'Strategy 2 (Tiered)':<22} {'Difference':<15}")
    print("-" * 95)

    flat_avg = np.mean(unprotected_cov_flat)*100
    tier_avg = np.mean(unprotected_cov_tier)*100
    flat_min = min(unprotected_cov_flat)*100
    tier_min = min(unprotected_cov_tier)*100
    flat_perfect = sum(1 for v in unprotected_cov_flat if v >= 0.999)
    tier_perfect = sum(1 for v in unprotected_cov_tier if v >= 0.999)

    print(f"{'Unprotected pairs avg coverage':<35} {flat_avg:>6.2f}%              {tier_avg:>6.2f}%              {tier_avg-flat_avg:>+6.2f}%")
    print(f"{'Unprotected pairs min coverage':<35} {flat_min:>6.2f}%              {tier_min:>6.2f}%              {tier_min-flat_min:>+6.2f}%")
    print(f"{'Unprotected pairs at 100%':<35} {flat_perfect:>2}/{len(unprotected_cov_flat):<2}                {tier_perfect:>2}/{len(unprotected_cov_tier):<2}                {tier_perfect-flat_perfect:>+2}")
    print()

    # Detailed pair-by-pair comparison
    print("=" * 80)
    print("Detailed Pair-by-Pair Coverage Comparison")
    print("=" * 80)
    print()

    print(f"{'Pair':<15} {'Factor Names':<45} {'Flat':<12} {'Tiered':<12} {'Diff':<10}")
    print("-" * 95)

    for i in range(len(design_df.columns)):
        for j in range(i+1, len(design_df.columns)):
            pair = (i, j)
            col_i, col_j = design_df.columns[i], design_df.columns[j]
            factor_names = f"{col_i} x {col_j}"

            flat_cov = cov_flat.get(pair, 0.0) * 100
            tier_cov = cov_tier.get(pair, 0.0) * 100
            diff = tier_cov - flat_cov

            marker = "[PROTECTED]" if pair in protected_pairs_set else "[MID-PRI]" if pair in [(p[0], p[1]) for p in mid_priority_pairs] else ""

            print(f"{str(pair):<15} {factor_names:<45} {flat_cov:>6.2f}%    {tier_cov:>6.2f}%    {diff:>+6.2f}% {marker}")

    print()

    # Conclusion
    if tier_avg > flat_avg + 1:
        print("CONCLUSION: Tiered allocation is better - improves coverage of medium-priority pairs")
    elif flat_avg > tier_avg + 1:
        print("CONCLUSION: Flat allocation is better - more balanced exploration of all unprotected pairs")
    else:
        print("CONCLUSION: Both strategies perform similarly, difference not significant")
    print()


if __name__ == "__main__":
    test_flat_vs_tiered()
